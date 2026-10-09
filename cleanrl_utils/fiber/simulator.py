"""Single-mode optical fiber pulse propagation.

Solves the nonlinear Schrödinger equation (NLSE) for the slowly varying
envelope ``A(z, T)`` of an optical pulse with the symmetric split-step
Fourier method:

    dA/dz = -alpha/2 A - i beta2/2 d²A/dT² + beta3/6 d³A/dT³ + i gamma |A|² A

Units used throughout: time in ps, distance in km, power in W,
``beta2`` in ps²/km, ``beta3`` in ps³/km, ``gamma`` in 1/(W km) and
attenuation in dB/km. ``|A|²`` is the instantaneous power in W.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

DB_PER_NEPER = 10.0 * np.log10(np.e)  # ~4.343


@dataclass
class Fiber:
    """Physical parameters of a fiber span. Defaults are standard single-mode fiber (SMF-28) at 1550 nm."""

    length: float = 50.0  # km
    alpha_db: float = 0.2  # dB/km
    beta2: float = -21.7  # ps²/km
    beta3: float = 0.0  # ps³/km
    gamma: float = 1.3  # 1/(W km)

    @property
    def alpha(self) -> float:
        """Power attenuation coefficient in 1/km."""
        return self.alpha_db / DB_PER_NEPER

    @property
    def effective_length(self) -> float:
        """Length over which nonlinearity acts, accounting for loss (km)."""
        if self.alpha == 0:
            return self.length
        return (1.0 - np.exp(-self.alpha * self.length)) / self.alpha

    def dispersion_length(self, t0: float) -> float:
        """Distance over which a pulse of half-width ``t0`` (ps) broadens noticeably (km)."""
        return t0**2 / abs(self.beta2) if self.beta2 != 0 else np.inf

    def nonlinear_length(self, peak_power: float) -> float:
        """Distance over which self-phase modulation becomes significant at ``peak_power`` (W) (km)."""
        return 1.0 / (self.gamma * peak_power) if self.gamma * peak_power != 0 else np.inf


def time_grid(n_points: int = 1024, window: float = 400.0) -> np.ndarray:
    """Uniform time grid of ``n_points`` samples centred on zero, spanning ``window`` ps."""
    return (np.arange(n_points) - n_points // 2) * (window / n_points)


def angular_frequencies(t: np.ndarray) -> np.ndarray:
    """Angular frequency grid (rad/ps) matching ``numpy.fft`` ordering for time grid ``t``."""
    return 2.0 * np.pi * np.fft.fftfreq(t.size, d=t[1] - t[0])


def gaussian_pulse(t: np.ndarray, t0: float, peak_power: float = 1e-3, chirp: float = 0.0) -> np.ndarray:
    """Gaussian pulse ``sqrt(P0) exp(-(1 + iC) T² / (2 T0²))`` with half-width at 1/e intensity ``t0`` (ps)."""
    return np.sqrt(peak_power) * np.exp(-(1.0 + 1j * chirp) * t**2 / (2.0 * t0**2))


def sech_pulse(t: np.ndarray, t0: float, peak_power: float = 1e-3) -> np.ndarray:
    """Hyperbolic-secant pulse ``sqrt(P0) sech(T / T0)``, the shape of an optical soliton."""
    return np.sqrt(peak_power) / np.cosh(t / t0)


def soliton_peak_power(fiber: Fiber, t0: float, order: int = 1) -> float:
    """Peak power (W) of an order-``order`` soliton of width ``t0`` in ``fiber`` (needs beta2 < 0)."""
    return order**2 * abs(fiber.beta2) / (fiber.gamma * t0**2)


def apply_dispersion(field: np.ndarray, t: np.ndarray, accumulated_beta2: float, accumulated_beta3: float = 0.0) -> np.ndarray:
    """Apply a lumped, lossless dispersive element (e.g. a compensating fiber or grating).

    ``accumulated_beta2`` is the total group-delay dispersion in ps² (``beta2 * length``). Passing
    ``-fiber.beta2 * fiber.length`` exactly undoes the linear dispersion of ``fiber``.
    """
    w = angular_frequencies(t)
    phase = 0.5 * accumulated_beta2 * w**2 - accumulated_beta3 / 6.0 * w**3
    return np.fft.ifft(np.fft.fft(field) * np.exp(1j * phase))


def propagate(
    field: np.ndarray,
    t: np.ndarray,
    fiber: Fiber,
    n_steps: int = 200,
    n_snapshots: int = 0,
):
    """Propagate ``field`` through ``fiber`` with the symmetric split-step Fourier method.

    Returns the output field. If ``n_snapshots > 0`` it instead returns ``(z, fields)``, where
    ``z`` has ``n_snapshots + 1`` positions (km, including the input) and ``fields`` has shape
    ``(n_snapshots + 1, len(t))``. ``n_steps`` must then be a multiple of ``n_snapshots``.
    """
    if n_snapshots and n_steps % n_snapshots:
        raise ValueError("n_steps must be a multiple of n_snapshots")
    w = angular_frequencies(t)
    dz = fiber.length / n_steps
    # With numpy's FFT convention d/dT -> i*w, so the linear operator in the frequency domain is:
    linear = -fiber.alpha / 2.0 + 1j * fiber.beta2 / 2.0 * w**2 - 1j * fiber.beta3 / 6.0 * w**3
    half_linear = np.exp(linear * dz / 2.0)

    a = np.asarray(field, dtype=np.complex128).copy()
    snapshots = [a.copy()] if n_snapshots else None
    every = n_steps // n_snapshots if n_snapshots else 0
    for step in range(1, n_steps + 1):
        a = np.fft.ifft(half_linear * np.fft.fft(a))
        a = a * np.exp(1j * fiber.gamma * np.abs(a) ** 2 * dz)
        a = np.fft.ifft(half_linear * np.fft.fft(a))
        if n_snapshots and step % every == 0:
            snapshots.append(a.copy())
    if n_snapshots:
        return np.linspace(0.0, fiber.length, n_snapshots + 1), np.array(snapshots)
    return a


def energy(field: np.ndarray, t: np.ndarray) -> float:
    """Pulse energy in pJ (W × ps)."""
    return float(np.sum(np.abs(field) ** 2) * (t[1] - t[0]))


def peak_power(field: np.ndarray) -> float:
    """Peak instantaneous power in W."""
    return float(np.max(np.abs(field) ** 2))


def rms_width(field: np.ndarray, t: np.ndarray) -> float:
    """Root-mean-square temporal width of the intensity profile in ps.

    For a Gaussian pulse of half-width ``t0`` this is ``t0 / sqrt(2)``.
    """
    intensity = np.abs(field) ** 2
    total = np.sum(intensity)
    mean = np.sum(t * intensity) / total
    return float(np.sqrt(np.sum((t - mean) ** 2 * intensity) / total))


def spectrum(field: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(w, power)``: angular frequency offset (rad/ps, sorted) and normalised power spectrum."""
    w = np.fft.fftshift(angular_frequencies(t))
    power = np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
    return w, power / power.max()


def intensity_fidelity(field: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity in [0, 1] between the intensity profiles of ``field`` and ``reference``.

    Insensitive to overall power and phase, so it measures how well the pulse *shape*
    (what a direct-detection receiver sees) survived transmission.
    """
    a = np.abs(field) ** 2
    b = np.abs(reference) ** 2
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
