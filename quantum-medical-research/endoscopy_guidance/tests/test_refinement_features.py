"""Feature leakage regression checks without importing optional quantum backends.

Run: python -m unittest discover -s quantum-medical-research/endoscopy_guidance/tests
"""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]


def load_functions(filename, names, namespace):
    """Execute actual feature functions while avoiding heavyweight module imports."""
    tree = ast.parse((SOURCE / filename).read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)


class RefinementFeaturesTest(unittest.TestCase):
    def test_ground_truth_does_not_change_inputs(self):
        ns = {'np': np}
        load_functions('build_residual_patch_dataset.py',
            {'local_stats', 'image_patch_stats', 'mask_centroid', 'build_features'}, ns)
        # Select all pixels to isolate feature construction from candidate policy.
        ns['candidate_mask'] = lambda prob, pred, args: np.ones_like(pred, dtype=bool)
        load_functions('apply_residual_mask_refinement.py', {'frame_features'}, ns)
        for empty_prediction in (False, True):
            with self.subTest(empty_prediction=empty_prediction), TemporaryDirectory() as temp:
                base = Path(temp)
                for folder in ('images', 'gt_masks', 'prob_maps', 'pred_masks'):
                    (base / folder).mkdir()
                pred = np.zeros((4, 4), dtype=np.uint8)
                if not empty_prediction:
                    pred[0, 0] = 1
                np.save(base / 'images/frame.npy', np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
                np.save(base / 'prob_maps/frame.npy', np.full((4, 4), .4, dtype=np.float32))
                np.save(base / 'pred_masks/frame.npy', pred)
                args = SimpleNamespace(max_candidates_per_frame=16, patch_radius=1)
                row = SimpleNamespace(sample_id='frame')
                results = []
                for point in ((0, 0), (3, 3)):
                    gt = np.zeros((4, 4), dtype=np.uint8)
                    gt[point] = 1
                    np.save(base / 'gt_masks/frame.npy', gt)
                    results.append(ns['frame_features'](base, row, args))
                for before, after in zip(results[0][:4], results[1][:4]):
                    np.testing.assert_array_equal(before, after)
                self.assertFalse(np.array_equal(results[0][4], results[1][4]))
                features, ys, xs, _, _ = results[0]
                expected = np.ones(len(ys)) if empty_prediction else np.hypot(ys, xs) / 4
                np.testing.assert_allclose(features[:, 5], expected)


if __name__ == '__main__':
    unittest.main()
