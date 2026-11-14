"""Numerical tests for the offline low-pass FIR implementation."""

import math
import unittest

from adas.signal.filtering import (
    apply_fir,
    design_lowpass_fir,
    frequency_response_magnitude,
)


class TestFIRFiltering(unittest.TestCase):

    def setUp(self):
        self.fs = 20.0
        self.taps = design_lowpass_fir(self.fs, 6.0, 51)

    def test_tap_count_and_symmetry(self):
        self.assertEqual(len(self.taps), 51)
        for a, b in zip(self.taps, reversed(self.taps)):
            self.assertAlmostEqual(a, b, places=12)

    def test_dc_gain_is_unity(self):
        self.assertAlmostEqual(sum(self.taps), 1.0, places=12)
        self.assertAlmostEqual(
            frequency_response_magnitude(self.taps, self.fs, 0.0),
            1.0,
            places=12,
        )

    def test_passband_response(self):
        gain = frequency_response_magnitude(
            self.taps, self.fs, 2.0
        )
        self.assertGreater(gain, 0.95)
        self.assertLess(gain, 1.05)

    def test_stopband_response(self):
        gain = frequency_response_magnitude(
            self.taps, self.fs, 9.0
        )
        self.assertLess(gain, 0.05)

    def test_constant_signal_preserved(self):
        values = apply_fir([4.0] * 200, self.taps)
        self.assertEqual(len(values), 200)
        self.assertTrue(
            all(abs(value - 4.0) < 1e-9 for value in values)
        )

    def test_low_frequency_retained_high_frequency_suppressed(self):
        samples = [
            math.sin(2 * math.pi * 2 * i / self.fs)
            + math.sin(2 * math.pi * 9 * i / self.fs)
            for i in range(500)
        ]
        filtered = apply_fir(samples, self.taps)

        # Exclude endpoints affected by convolution padding.
        interior = filtered[50:-50]

        low_projection = abs(
            2 * sum(
                value * math.sin(2 * math.pi * 2 * (i + 50) / self.fs)
                for i, value in enumerate(interior)
            ) / len(interior)
        )

        high_projection = abs(
            2 * sum(
                value * math.sin(2 * math.pi * 9 * (i + 50) / self.fs)
                for i, value in enumerate(interior)
            ) / len(interior)
        )

        self.assertGreater(low_projection, 0.9)
        self.assertLess(high_projection, 0.05)

    def test_invalid_cutoff(self):
        for cutoff in [0, -1, 10, 11, float("nan")]:
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    design_lowpass_fir(20, cutoff)

    def test_invalid_tap_count(self):
        for count in [0, 2, 10, 2.5, True]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    design_lowpass_fir(20, 6, count)

    def test_rejects_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            apply_fir([1.0, float("nan")], self.taps)

    def test_empty_signal(self):
        self.assertEqual(apply_fir([], self.taps), ())


if __name__ == "__main__":
    unittest.main()
