"""Offline FIR filtering utilities for uniformly sampled signals."""

import math
from collections.abc import Sequence


def design_lowpass_fir(
    sample_rate_hz: float,
    cutoff_hz: float,
    num_taps: int = 51,
) -> tuple[float, ...]:
    """Design a normalized, symmetric Hamming-windowed sinc FIR."""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Invalid sample rate")

    if not math.isfinite(cutoff_hz):
        raise ValueError("Invalid cutoff")

    if not 0 < cutoff_hz < sample_rate_hz / 2:
        raise ValueError("Cutoff must be below Nyquist")

    if isinstance(num_taps, bool) or not isinstance(num_taps, int):
        raise ValueError("num_taps must be an integer")

    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps must be odd and >= 3")

    normalized_cutoff = cutoff_hz / sample_rate_hz
    midpoint = (num_taps - 1) // 2
    taps = []

    for index in range(num_taps):
        distance = index - midpoint

        if distance == 0:
            ideal = 2 * normalized_cutoff
        else:
            ideal = (
                math.sin(2 * math.pi * normalized_cutoff * distance)
                / (math.pi * distance)
            )

        window = (
            0.54
            - 0.46 * math.cos(
                2 * math.pi * index / (num_taps - 1)
            )
        )

        taps.append(ideal * window)

    total = sum(taps)

    if not math.isfinite(total) or abs(total) < 1e-15:
        raise ValueError("FIR normalization failed")

    return tuple(value / total for value in taps)


def frequency_response_magnitude(
    taps: Sequence[float],
    sample_rate_hz: float,
    frequency_hz: float,
) -> float:
    """Return magnitude of a digital FIR response at one frequency."""

    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Invalid sample rate")

    if (
        not math.isfinite(frequency_hz)
        or not 0 <= frequency_hz <= sample_rate_hz / 2
    ):
        raise ValueError("Frequency outside Nyquist range")

    if not taps or not all(math.isfinite(v) for v in taps):
        raise ValueError("Invalid FIR coefficients")

    angle = 2 * math.pi * frequency_hz / sample_rate_hz

    real = sum(
        tap * math.cos(angle * index)
        for index, tap in enumerate(taps)
    )
    imag = sum(
        tap * math.sin(angle * index)
        for index, tap in enumerate(taps)
    )

    return math.hypot(real, imag)


def apply_fir(
    samples: Sequence[float],
    taps: Sequence[float],
) -> tuple[float, ...]:
    """Centered offline convolution with endpoint extension.

    Output length equals input length. Boundary samples are affected
    by the endpoint-extension policy; they require separate handling
    in downstream feature extraction.
    """

    values = tuple(samples)
    coefficients = tuple(taps)

    if not values:
        return ()

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite input sample")

    if (
        not coefficients
        or len(coefficients) % 2 == 0
        or not all(math.isfinite(v) for v in coefficients)
    ):
        raise ValueError("Invalid FIR coefficients")

    midpoint = len(coefficients) // 2
    last = len(values) - 1
    result = []

    for index in range(len(values)):
        total = 0.0

        for tap_index, coefficient in enumerate(coefficients):
            source = index + midpoint - tap_index
            source = max(0, min(last, source))
            total += coefficient * values[source]

        result.append(total)

    return tuple(result)
