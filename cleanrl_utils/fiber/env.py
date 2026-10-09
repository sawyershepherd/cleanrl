"""Gymnasium environment: keep an optical pulse clean across a multi-span fiber link.

A Gaussian pulse travels through ``n_spans`` spans of standard single-mode fiber. Before each
span an amplifier sets the launch power (adding a fixed amount of amplifier noise), and after
each span a lumped dispersion compensator is applied. The agent picks both per span:

* ``action[0]`` in [-1, 1] -> launch peak power from ``power_dbm_range`` (linear in dBm).
  Too little power and amplifier noise dominates; too much and self-phase modulation
  distorts the pulse.
* ``action[1]`` in [-1, 1] -> compensating dispersion ``action[1] * max_compensation`` ps².
  Perfect compensation of a span of length L is ``-beta2 * L``.

Span lengths are drawn at random each episode, so the agent has to infer how much dispersion
to undo from what the received pulse looks like. The reward after each span is the intensity
fidelity (0 to 1) between the received and the transmitted pulse.

Usage with CleanRL, no registration step needed::

    python cleanrl/ppo_continuous_action.py --env-id cleanrl_utils.fiber:FiberDispersionComp-v0
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cleanrl_utils.fiber.simulator import (
    Fiber,
    apply_dispersion,
    gaussian_pulse,
    intensity_fidelity,
    propagate,
    rms_width,
    time_grid,
)


class FiberDispersionCompEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_spans: int = 4,
        span_length_range: Tuple[float, float] = (40.0, 80.0),
        pulse_width: float = 25.0,
        power_dbm_range: Tuple[float, float] = (-10.0, 20.0),
        noise_power: float = 1e-5,
        n_points: int = 512,
        window_widths: float = 40.0,
        steps_per_span: int = 40,
        obs_bins: int = 64,
        fiber: Optional[Fiber] = None,
    ):
        self.n_spans = n_spans
        self.span_length_range = span_length_range
        self.pulse_width = pulse_width
        self.power_dbm_range = power_dbm_range
        self.noise_power = noise_power  # complex noise variance per sample added by each amplifier (W)
        self.steps_per_span = steps_per_span
        self.obs_bins = obs_bins
        self.fiber = fiber or Fiber()
        self.max_compensation = abs(self.fiber.beta2) * 100.0  # undoes up to 100 km of fiber

        if n_points % obs_bins:
            raise ValueError("n_points must be a multiple of obs_bins")
        self.t = time_grid(n_points, window_widths * pulse_width)
        self.reference = gaussian_pulse(self.t, pulse_width, peak_power=1.0)
        self.reference_energy = np.sum(np.abs(self.reference) ** 2)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Downsampled intensity profile (normalised to its peak), RMS width ratio, link progress.
        self.observation_space = spaces.Box(0.0, np.inf, shape=(obs_bins + 2,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        intensity = np.abs(self.field) ** 2
        profile = intensity.reshape(self.obs_bins, -1).mean(axis=1)
        profile = profile / max(profile.max(), 1e-30)
        width_ratio = rms_width(self.field, self.t) / rms_width(self.reference, self.t)
        progress = self.span / self.n_spans
        return np.concatenate([profile, [width_ratio, progress]]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.span = 0
        self.span_lengths = self.np_random.uniform(*self.span_length_range, size=self.n_spans)
        self.field = self.reference.copy()
        return self._obs(), {"span_lengths": self.span_lengths.copy()}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        low, high = self.power_dbm_range
        launch_dbm = low + (action[0] + 1.0) / 2.0 * (high - low)
        launch_power = 1e-3 * 10.0 ** (launch_dbm / 10.0)
        compensation = action[1] * self.max_compensation

        # Amplifier: scale the pulse energy to that of a clean pulse with the chosen peak power, then add noise.
        gain = launch_power * self.reference_energy / np.sum(np.abs(self.field) ** 2)
        field = self.field * np.sqrt(gain)
        noise = self.np_random.normal(size=field.size) + 1j * self.np_random.normal(size=field.size)
        field = field + np.sqrt(self.noise_power / 2.0) * noise

        span = Fiber(
            length=float(self.span_lengths[self.span]),
            alpha_db=self.fiber.alpha_db,
            beta2=self.fiber.beta2,
            beta3=self.fiber.beta3,
            gamma=self.fiber.gamma,
        )
        field = propagate(field, self.t, span, n_steps=self.steps_per_span)
        self.field = apply_dispersion(field, self.t, compensation)
        self.span += 1

        fidelity = intensity_fidelity(self.field, self.reference)
        terminated = self.span >= self.n_spans
        info = {
            "fidelity": fidelity,
            "launch_dbm": launch_dbm,
            "compensation": compensation,
            "ideal_compensation": -span.beta2 * span.length,
        }
        return self._obs(), fidelity, terminated, False, info
