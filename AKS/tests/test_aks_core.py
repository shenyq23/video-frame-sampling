from __future__ import annotations

import heapq
import unittest

import numpy as np

from aks_core import normalize_scores, select_frames
from aks_keyframes_v2 import sample_candidate_indices, sample_uniform_indices


def legacy_reference(scores, frame_indices, n=64, t1=0.8, t2=-100.0, max_depth=5):
    """Executable form of the original frame_select.py rule for comparison."""

    values = np.asarray(scores, dtype=float)
    span = np.ptp(values)
    values = np.zeros_like(values) if span == 0 else (values - values.min()) / span

    def recurse(parts, frame_parts):
        split_parts = []
        split_frames = []
        terminal_parts = []
        terminal_frames = []
        for (part, depth), frames in zip(parts, frame_parts):
            # Empty descendants never contribute a selected frame.
            if len(part) == 0:
                continue
            top = heapq.nlargest(n, range(len(part)), part.__getitem__)
            difference = np.mean([part[index] for index in top]) - np.mean(part)
            if difference > t1 and np.std(part) > t2:
                terminal_parts.append((part, depth))
                terminal_frames.append(frames)
            elif depth < max_depth:
                middle = len(part) // 2
                split_parts.extend([(part[:middle], depth + 1), (part[middle:], depth + 1)])
                split_frames.extend([frames[:middle], frames[middle:]])
            else:
                terminal_parts.append((part, depth))
                terminal_frames.append(frames)
        if split_parts:
            descendants, descendant_frames = recurse(split_parts, split_frames)
        else:
            descendants, descendant_frames = [], []
        return terminal_parts + descendants, terminal_frames + descendant_frames

    if len(values) < n:
        return sorted(frame_indices)
    leaves, frame_parts = recurse([(values, 0)], [list(frame_indices)])
    selected = []
    for (leaf, depth), frames in zip(leaves, frame_parts):
        quota = int(n / 2**depth)
        top = heapq.nlargest(quota, range(len(leaf)), leaf.__getitem__)
        selected.extend(frames[index] for index in top)
    return sorted(selected)


class AKSCoreTests(unittest.TestCase):
    def test_original_matches_legacy_for_paper_budgets(self):
        rng = np.random.default_rng(2025)
        for budget in (16, 31, 32, 48, 64):
            for candidate_count in (budget, budget + 1, budget * 3 + 7):
                for _ in range(20):
                    scores = rng.normal(size=candidate_count).tolist()
                    indices = [index * 30 for index in range(candidate_count)]
                    expected = legacy_reference(scores, indices, n=budget)
                    actual = select_frames(
                        scores, indices, max_num_frames=budget, mode="original"
                    ).frame_indices
                    self.assertEqual(expected, actual)

    def test_robust_fills_non_divisible_budget(self):
        scores = np.linspace(0.0, 1.0, 480).tolist()
        indices = list(range(480))
        original = select_frames(scores, indices, max_num_frames=48, mode="original")
        robust = select_frames(scores, indices, max_num_frames=48, mode="robust")
        self.assertEqual(32, len(original.frame_indices))
        self.assertEqual(48, len(robust.frame_indices))
        self.assertEqual(48, sum(segment.quota for segment in robust.segments))

    def test_constant_scores_are_safe_and_deterministic(self):
        self.assertTrue(np.array_equal(normalize_scores([4.0, 4.0]), [0.0, 0.0]))
        first = select_frames([4.0] * 100, list(range(100)), mode="robust")
        second = select_frames([4.0] * 100, list(range(100)), mode="robust")
        self.assertEqual(64, len(first.frame_indices))
        self.assertEqual(first.frame_indices, second.frame_indices)

    def test_short_input_returns_all_candidates(self):
        result = select_frames([0.2, 0.1], [0, 30], max_num_frames=4, mode="robust")
        self.assertEqual([0, 30], result.frame_indices)

    def test_rejects_misaligned_or_unsorted_inputs(self):
        with self.assertRaises(ValueError):
            select_frames([0.1], [0, 1])
        with self.assertRaises(ValueError):
            select_frames([0.1, 0.2], [30, 0])

    def test_candidate_sampling_modes_are_explicit(self):
        self.assertEqual([0, 29, 58], sample_candidate_indices(100, 29.97, "original", 1.0))
        self.assertEqual(
            [0, 30, 60, 90], sample_candidate_indices(100, 29.97, "interval", 1.0)
        )

    def test_uniform_baseline_matches_aks_output_count(self):
        candidates = [0, 30, 60, 90, 120, 150, 180]
        self.assertEqual([0, 90, 180], sample_uniform_indices(candidates, 3))
        self.assertEqual([90], sample_uniform_indices(candidates, 1))
        self.assertEqual(candidates, sample_uniform_indices(candidates, 99))


if __name__ == "__main__":
    unittest.main()
