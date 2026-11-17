#!/usr/bin/env python3
"""Small deterministic checks of the released calibration and metric functions."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asvspoof5_metrics import calibrate, ranking
from selfcal_common import holm, threshold


class CalibrationChecks(unittest.TestCase):
    def test_twenty_enrollment_scores_use_minimum(self):
        self.assertEqual(threshold(np.arange(20, 0, -1), ".05"), 1.0)

    def test_nineteen_is_the_first_nontrivial_enrollment_size(self):
        self.assertEqual(threshold(np.arange(19), ".05"), 0.0)
        self.assertTrue(np.isneginf(threshold(np.arange(18), ".05")))

    def test_strict_rejection_handles_ties(self):
        scores = np.array([0.0, 0.0, 1.0, 2.0])
        t = calibrate(scores, ["a"] * 4, alpha=.25)
        self.assertEqual(t, 0.0)
        self.assertEqual(int(np.sum(scores < t)), 0)

    def test_identity_weighting_differs_from_utterance_weighting(self):
        scores = [0.0, 1.0, 2.0, 3.0]
        weighted = calibrate(scores, ["a", "a", "a", "b"], alpha=.4)
        unweighted = calibrate(scores, ["one_group"] * 4, alpha=.4)
        self.assertEqual(weighted, 2.0)
        self.assertEqual(unweighted, 1.0)

    def test_perfect_genuine_high_ranking(self):
        result = ranking([3.0, 4.0], [0.0, 1.0])
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["step_pauc"], 1.0)
        self.assertEqual(result["oracle"]["0.0"], 1.0)

    def test_all_ties_have_half_auc_and_zero_zero_fa_detection(self):
        result = ranking([1.0, 1.0], [1.0, 1.0])
        self.assertEqual(result["auc"], .5)
        self.assertEqual(result["oracle"]["0.0"], 0.0)

    def test_holm_correction_preserves_input_order(self):
        np.testing.assert_allclose(holm([.9271072892710729, .0031996800319968005]), [.9271072892710729, .006399360063993601])


if __name__ == "__main__":
    unittest.main(verbosity=2)
