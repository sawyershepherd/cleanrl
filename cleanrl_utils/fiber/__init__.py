"""Optical fiber pulse propagation simulator and a Gymnasium environment built on it."""

import gymnasium as gym

from cleanrl_utils.fiber.simulator import (  # noqa: F401
    Fiber,
    apply_dispersion,
    energy,
    gaussian_pulse,
    intensity_fidelity,
    peak_power,
    propagate,
    rms_width,
    sech_pulse,
    soliton_peak_power,
    spectrum,
    time_grid,
)

if "FiberDispersionComp-v0" not in gym.registry:
    gym.register(
        id="FiberDispersionComp-v0",
        entry_point="cleanrl_utils.fiber.env:FiberDispersionCompEnv",
        max_episode_steps=None,
    )
