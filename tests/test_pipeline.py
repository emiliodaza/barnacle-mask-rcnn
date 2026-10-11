"""Regression checks for counting errors, leakage guardrails, and tile seams."""

import unittest
from unittest.mock import patch
import numpy as np
import cv2
import torch
from src.instances import compact_mask, deduplicate, mask_iou
from src.dataset import make_patches, masks_to_target, BarnacleDataset
from src.regions import window_starts, keep_inside, validate_polygon
from src.masks import extract_blue_contours
from src.evaluate import evaluate_instances
from src.model import get_barnacle_maskrcnn
from src.predict import predict_instances


class CountingTests(unittest.TestCase):
    def test_all_edges_covered_with_awkward_dimensions(self):
        image = np.zeros((1000, 1000, 3), np.uint8)
        masks = []
        for y, x in [(100, 100), (100, 960), (960, 100), (960, 960)]:
            mask = np.zeros((1000, 1000), np.uint8)
            mask[y : y + 20, x : x + 20] = 1
            masks.append(mask)
        patches = make_patches(image, masks)
        self.assertEqual(len(patches), 9)
        coverage = np.zeros((1000, 1000), bool)
        for y in window_starts(1000, 512, 384):
            for x in window_starts(1000, 512, 384):
                coverage[y : y + 512, x : x + 512] = True
        self.assertTrue(coverage.all())

    def test_tiled_inference_maps_and_deduplicates_without_labels(self):
        class BrightPixelDetector:
            def __call__(self, images):
                patch = (images[0][0].numpy() > 0.5).astype(np.uint8)
                count, labels = cv2.connectedComponents(patch)
                masks = (
                    np.stack([labels == i for i in range(1, count)])
                    if count > 1
                    else np.zeros((0, *patch.shape))
                )
                return [
                    {
                        "scores": torch.full((count - 1,), 0.9),
                        "labels": torch.ones(count - 1, dtype=torch.int64),
                        "masks": torch.from_numpy(masks[:, None].astype(np.float32)),
                    }
                ]

        image = np.zeros((700, 700, 3), np.uint8)
        for x, y in [(100, 100), (350, 350), (550, 550), (650, 650)]:
            image[y : y + 20, x : x + 20] = 255
        polygon = [[30, 30], [600, 30], [600, 600], [30, 600]]
        predictions, timing = predict_instances(image, polygon, BrightPixelDetector())
        self.assertEqual(len(predictions), 3)
        self.assertGreater(timing["removed_duplicates"], 0)
        self.assertEqual(
            sorted(p.box[:2] for p in predictions), [(100, 100), (350, 350), (550, 550)]
        )

    def test_empty_patches_are_retained(self):
        self.assertEqual(len(make_patches(np.zeros((512, 512, 3), np.uint8), [])), 1)

    def test_separate_outlines_stay_separate(self):
        image = np.zeros((60, 60, 3), np.uint8)
        cv2.rectangle(image, (5, 5), (20, 20), (0, 0, 255), 1)
        cv2.rectangle(image, (22, 5), (37, 20), (0, 0, 255), 1)
        self.assertEqual(len(extract_blue_contours(image)), 2)

    def test_region_filter_and_boundary_rule(self):
        a = compact_mask(np.ones((3, 3), np.uint8), (3, 3))
        b = compact_mask(np.ones((3, 3), np.uint8), (20, 20))
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertEqual(keep_inside([a, b], polygon), [a])
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [10, 10], [0, 10], [10, 0]], (30, 30))

    def test_duplicate_instances_do_not_inflate_count(self):
        a = compact_mask(np.ones((5, 5), np.uint8), (10, 10), 0.9)
        b = compact_mask(np.ones((5, 5), np.uint8), (10, 10), 0.7)
        c = compact_mask(np.ones((5, 5), np.uint8), (20, 20), 0.8)
        self.assertEqual(len(deduplicate([a, b, c])), 2)
        self.assertEqual(mask_iou(a, b), 1)

    def test_equal_counts_can_hide_two_errors(self):
        ref = [
            compact_mask(np.ones((4, 4), np.uint8), (0, 0)),
            compact_mask(np.ones((4, 4), np.uint8), (10, 10)),
        ]
        pred = [ref[0], compact_mask(np.ones((4, 4), np.uint8), (20, 20))]
        metrics, matches = evaluate_instances(pred, ref)
        self.assertEqual(metrics["absolute_count_error"], 0)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertEqual(metrics["f1"], 0.5)

    def test_matching_maximizes_valid_pair_count(self):
        # A naive maximize-IoU assignment would choose 0.99+0.49 and lose a TP.
        instances = [compact_mask(np.ones((2, 2), np.uint8))] * 2
        with patch("src.evaluate.mask_iou", side_effect=[0.99, 0.5, 0.5, 0.49]):
            metrics, _ = evaluate_instances(instances, instances, 0.5)
        self.assertEqual(metrics["tp"], 2)

    def test_empty_metrics_and_targets(self):
        metrics, _ = evaluate_instances([], [])
        self.assertEqual(metrics["tp"], 0)
        self.assertIsNone(metrics["relative_count_error"])
        target = masks_to_target([], (512, 512, 3), 0)
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["masks"].shape), (0, 512, 512))

    def test_exclusive_box_ends(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[2:5, 3:7] = 1
        self.assertEqual(
            masks_to_target([mask], mask.shape, 0)["boxes"].tolist(), [[3, 2, 7, 5]]
        )

    def test_bad_inputs_fail_before_training(self):
        with self.assertRaises(FileNotFoundError):
            BarnacleDataset(["missing.png"], ["mask.png"])
        with self.assertRaises(ValueError):
            BarnacleDataset(["a"], [])
        with self.assertRaises(ValueError):
            window_starts(1000, 512, 600)

    def test_checkpoint_construction_does_not_download(self):
        with patch(
            "torch.hub.load_state_dict_from_url",
            side_effect=AssertionError("Unexpected download"),
        ):
            model = get_barnacle_maskrcnn(pretrained=False)
        self.assertFalse(
            any(isinstance(m, torch.nn.BatchNorm2d) for m in model.backbone.modules())
        )


if __name__ == "__main__":
    unittest.main()
