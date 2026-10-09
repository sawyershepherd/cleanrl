import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from cleanrl_utils.fiber import (
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
    time_grid,
)
from cleanrl_utils.fiber.env import FiberDispersionCompEnv

T0 = 20.0  # ps
t = time_grid(2048, 2000.0)


def test_attenuation_matches_db_loss():
    fiber = Fiber(length=50.0, alpha_db=0.2)
    pulse = gaussian_pulse(t, T0, peak_power=1e-3)
    out = propagate(pulse, t, fiber)
    assert np.isclose(energy(out, t) / energy(pulse, t), 10 ** (-0.2 * 50.0 / 10), rtol=1e-6)


def test_gaussian_broadening_matches_theory():
    fiber = Fiber(length=60.0, alpha_db=0.0, gamma=0.0)
    pulse = gaussian_pulse(t, T0)
    out = propagate(pulse, t, fiber, n_steps=10)
    expected = np.sqrt(1 + (fiber.length / fiber.dispersion_length(T0)) ** 2)
    assert np.isclose(rms_width(out, t) / rms_width(pulse, t), expected, rtol=1e-6)


def test_lumped_compensation_undoes_dispersion():
    fiber = Fiber(length=80.0, alpha_db=0.0, gamma=0.0)
    pulse = gaussian_pulse(t, T0)
    out = apply_dispersion(propagate(pulse, t, fiber, n_steps=10), t, -fiber.beta2 * fiber.length)
    assert np.allclose(out, pulse, atol=1e-10)


def test_fundamental_soliton_keeps_its_shape():
    fiber = Fiber(alpha_db=0.0)
    fiber.length = 5 * fiber.dispersion_length(T0)
    pulse = sech_pulse(t, T0, soliton_peak_power(fiber, T0))
    out = propagate(pulse, t, fiber, n_steps=500)
    assert intensity_fidelity(out, pulse) > 0.9999
    assert np.isclose(peak_power(out), peak_power(pulse), rtol=1e-3)


def test_self_phase_modulation_broadens_spectrum_not_pulse():
    fiber = Fiber(length=20.0, alpha_db=0.0, beta2=0.0)
    pulse = gaussian_pulse(t, T0, peak_power=1.0)
    out = propagate(pulse, t, fiber)
    assert np.allclose(np.abs(out), np.abs(pulse))
    spec_in = np.abs(np.fft.fft(pulse)) ** 2
    spec_out = np.abs(np.fft.fft(out)) ** 2
    w = np.fft.fftfreq(t.size, t[1] - t[0])
    rms_spec = lambda s: np.sqrt(np.sum(w**2 * s) / np.sum(s))  # noqa: E731
    assert rms_spec(spec_out) > 5 * rms_spec(spec_in)


def test_snapshots():
    z, fields = propagate(gaussian_pulse(t, T0), t, Fiber(length=10.0), n_steps=20, n_snapshots=4)
    assert z.shape == (5,) and fields.shape == (5, t.size)
    assert z[-1] == 10.0


def test_env_passes_gymnasium_checks():
    check_env(FiberDispersionCompEnv(), skip_render_check=True)


def test_env_rewards_good_compensation():
    env = gym.make("cleanrl_utils.fiber:FiberDispersionComp-v0")
    max_comp = env.unwrapped.max_compensation

    def episode_return(policy):
        _, info = env.reset(seed=0)
        total, done, k = 0.0, False, 0
        while not done:
            _, reward, done, _, _ = env.step(policy(info["span_lengths"][k]))
            total += reward
            k += 1
        return total

    ideal = episode_return(lambda length: np.array([0.0, 21.7 * length / max_comp], dtype=np.float32))
    uncompensated = episode_return(lambda length: np.array([0.0, 0.0], dtype=np.float32))
    assert ideal > 3.8  # close to the maximum of 4 (one per span)
    assert ideal > uncompensated + 1.0
