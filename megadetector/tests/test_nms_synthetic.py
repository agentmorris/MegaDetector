"""

Test script for validating NMS functionality with synthetic data.

This script creates synthetic detection scenarios where we know exactly which
boxes should be suppressed by NMS, allowing us to verify the correctness of
the NMS implementation.

This is an AI-generated test module.

"""


#%% Imports

import torch

from megadetector.detection.pytorch_detector import nms


#%% Support functions

def calculate_iou_boxes(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: torch.Tensor or list of [x1, y1, x2, y2]
        box2: torch.Tensor or list of [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
    """

    if isinstance(box1, list):
        box1 = torch.tensor(box1, dtype=torch.float)
    if isinstance(box2, list):
        box2 = torch.tensor(box2, dtype=torch.float)

    # Calculate intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return float(intersection / union) if union > 0 else 0.0


def create_synthetic_predictions():
    """
    Create synthetic model predictions for testing NMS.

    Returns:
        torch.Tensor: Synthetic predictions in the format expected by the NMS function
                     Shape: [batch_size=1, num_anchors, num_classes + 5]

    Test scenarios:
    1. Two highly overlapping boxes (IoU > 0.5) with different confidences - higher confidence should win
    2. Two boxes with low overlap (IoU < 0.5) - both should be kept
    3. Multiple boxes of different classes in same location - should be kept (class-independent NMS)
    4. Three overlapping boxes with cascading confidences - highest confidence should win
    """

    # We'll create predictions for a 640x640 image with 3 classes
    # Format: [x_center, y_center, width, height, objectness, class0_conf, class1_conf, class2_conf]

    synthetic_boxes = []

    # Scenario 1: Two highly overlapping boxes (IoU > 0.8)
    # Box A: center=(100, 100), size=80x80, high confidence for class 0
    # Box B: center=(105, 105), size=80x80, low confidence for class 0  (smaller offset = higher IoU)
    # Expected: Box A kept, Box B suppressed
    synthetic_boxes.append([100, 100, 80, 80, 0.9, 0.8, 0.1, 0.1])  # Box A - should be kept
    synthetic_boxes.append([105, 105, 80, 80, 0.9, 0.5, 0.1, 0.1])  # Box B - should be suppressed

    # Scenario 1b: Two nearly identical boxes (IoU ≈ 0.95)
    # Box A2: center=(200, 100), size=60x60, high confidence for class 0
    # Box B2: center=(202, 102), size=60x60, lower confidence for class 0
    # Expected: Box A2 kept, Box B2 suppressed
    synthetic_boxes.append([200, 100, 60, 60, 0.9, 0.9, 0.05, 0.05])  # Box A2 - should be kept
    synthetic_boxes.append([202, 102, 60, 60, 0.9, 0.7, 0.1, 0.1])    # Box B2 - should be suppressed

    # Scenario 2: Two boxes with low overlap (IoU ≈ 0.1)
    # Box C: center=(300, 100), size=60x60, class 0
    # Box D: center=(380, 100), size=60x60, class 0
    # Expected: Both kept
    synthetic_boxes.append([300, 100, 60, 60, 0.9, 0.7, 0.1, 0.1])  # Box C - should be kept
    synthetic_boxes.append([380, 100, 60, 60, 0.9, 0.6, 0.1, 0.1])  # Box D - should be kept

    # Scenario 3: Same location, different classes
    # Box E: center=(100, 300), size=70x70, class 0
    # Box F: center=(100, 300), size=70x70, class 1
    # Expected: Both kept (class-independent NMS)
    synthetic_boxes.append([100, 300, 70, 70, 0.9, 0.7, 0.1, 0.1])  # Box E - class 0, should be kept
    synthetic_boxes.append([100, 300, 70, 70, 0.9, 0.1, 0.7, 0.1])  # Box F - class 1, should be kept

    # Scenario 4: Three cascading overlapping boxes
    # Box G: center=(500, 300), size=80x80, highest confidence
    # Box H: center=(510, 310), size=80x80, medium confidence
    # Box I: center=(520, 320), size=80x80, lowest confidence
    # Expected: Only Box G kept
    synthetic_boxes.append([500, 300, 80, 80, 0.95, 0.9, 0.05, 0.05])  # Box G - highest conf, should be kept
    synthetic_boxes.append([510, 310, 80, 80, 0.9,  0.7, 0.1,  0.1])   # Box H - should be suppressed
    synthetic_boxes.append([520, 320, 80, 80, 0.85, 0.6, 0.15, 0.15])  # Box I - should be suppressed

    # Add some low-confidence boxes that should be filtered out before NMS
    synthetic_boxes.append([200, 500, 50, 50, 0.1, 0.05, 0.02, 0.03])  # Too low confidence

    # Convert to tensor format expected by NMS function
    # We need to pad to a reasonable number of anchors (let's use 20)
    num_anchors = 20
    num_classes = 3

    predictions = torch.zeros(1, num_anchors, num_classes + 5)  # batch_size=1

    # Fill in our synthetic boxes
    for i, box_data in enumerate(synthetic_boxes):
        if i < num_anchors:
            predictions[0, i, :] = torch.tensor(box_data)

    return predictions


#%% Main test function

def test_nms_functionality():
    """
    Test the NMS function with synthetic data to verify correct behavior.
    """

    print("Testing NMS functionality with synthetic data...")

    # Generate synthetic predictions
    predictions = create_synthetic_predictions()
    print(f"Created synthetic predictions with shape: {predictions.shape}")

    # Run NMS with IoU threshold = 0.5 and confidence threshold = 0.3
    results = nms(predictions, conf_thres=0.3, iou_thres=0.5, max_det=300)

    print(f"NMS returned {len(results)} batch results")
    detections = results[0]  # Get results for first (and only) image
    print(f"Number of detections after NMS: {detections.shape[0]}")

    assert detections.shape[0] != 0

    print("\nDetections after NMS:")
    print("Format: [x1, y1, x2, y2, confidence, class_id]")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        print(f"Detection {i}: center=({center_x:.1f}, {center_y:.1f}), "
              f"size={width:.1f}x{height:.1f}, conf={conf:.3f}, class={int(cls)}")

    # Verify expected results

    # Verify that high-confidence boxes are kept over low-confidence overlapping ones
    # Look for the scenario 1 boxes (around center 100,100 area)
    scenario1_boxes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if 80 <= center_x <= 130 and 80 <= center_y <= 130 and int(cls) == 0:
            scenario1_boxes.append((i, center_x, center_y, conf))

    # Check scenario 1b (around center 200,100 area)
    scenario1b_boxes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if 180 <= center_x <= 220 and 80 <= center_y <= 120 and int(cls) == 0:
            scenario1b_boxes.append((i, center_x, center_y, conf))

    # Both scenario 1 and 1b should have exactly 1 detection each
    total_high_overlap_boxes = len(scenario1_boxes) + len(scenario1b_boxes)
    if total_high_overlap_boxes != 2:
        print("Error: expected 2 detections in high-overlap scenarios (1 each), got {}".format(
            total_high_overlap_boxes
        ))
        print(f"  Scenario 1: {len(scenario1_boxes)} boxes")
        print(f"  Scenario 1b: {len(scenario1b_boxes)} boxes")
        raise AssertionError()
    # Should be the high-confidence box (0.8 * 0.9 = 0.72)
    elif len(scenario1_boxes) == 1 and scenario1_boxes[0][3] < 0.7:
        print("Error: wrong box kept in scenario 1. Expected conf > 0.7, got {}".format(
            scenario1_boxes[0][3]
        ))
        raise AssertionError()
    # Should be the high-confidence box (0.9 * 0.9 = 0.81)
    elif len(scenario1b_boxes) == 1 and scenario1b_boxes[0][3] < 0.8:
        print("Error: wrong box kept in scenario 1b. Expected conf > 0.8, got {}".format(
            scenario1b_boxes[0][3]
        ))
        raise AssertionError()
    else:
        print("Scenarios 1 & 1b passed: High-confidence boxes kept, low-confidence overlapping boxes suppressed")

        # Verify IoU calculations and ensure suppression actually works
        if len(scenario1_boxes) == 1 and len(scenario1b_boxes) == 1:
            # Calculate what the IoU would have been between the boxes that were supposed to overlap
            # Scenario 1: Box A (100,100,80x80) vs Box B (105,105,80x80)
            box_a = [100-40, 100-40, 100+40, 100+40]  # Convert center+size to corners
            box_b = [105-40, 105-40, 105+40, 105+40]
            iou_1 = calculate_iou_boxes(box_a, box_b)

            # Scenario 1b: Box A2 (200,100,60x60) vs Box B2 (202,102,60x60)
            box_a2 = [200-30, 100-30, 200+30, 100+30]
            box_b2 = [202-30, 102-30, 202+30, 102+30]
            iou_1b = calculate_iou_boxes(box_a2, box_b2)

            print(f"    Theoretical IoU for scenario 1 boxes: {iou_1:.3f}")
            print(f"    Theoretical IoU for scenario 1b boxes: {iou_1b:.3f}")

            # If IoU > threshold, suppression should have occurred
            if iou_1 <= 0.5:
                print(f"Error: scenario 1 IoU {iou_1:.3f} is too low - test setup is invalid!")
                raise AssertionError()
            elif iou_1b <= 0.5:
                print(f"Error: scenario 1b IoU {iou_1b:.3f} is too low - test setup is invalid!")
                raise AssertionError()
            else:
                print("    High IoU confirmed - suppression was correct")

    # Verify scenario 2 - both non-overlapping boxes should be kept
    scenario2_boxes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if 270 <= center_x <= 410 and 70 <= center_y <= 130 and int(cls) == 0:
            scenario2_boxes.append((i, center_x, center_y, conf))

    if len(scenario2_boxes) != 2:
        print(f"Error: expected 2 detections in scenario 2 area, got {len(scenario2_boxes)}")
        raise AssertionError()
    else:
        print("Scenario 2 passed: Both non-overlapping boxes kept")

    # Verify scenario 3 - different classes should both be kept
    scenario3_boxes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if 65 <= center_x <= 135 and 265 <= center_y <= 335:
            scenario3_boxes.append((i, center_x, center_y, conf, int(cls)))

    classes_found = set(box[4] for box in scenario3_boxes)
    if (len(scenario3_boxes) != 2) or (len(classes_found) != 2):
        print("Error: expected 2 detections of different classes , got {} detections of classes {}".format(
            len(scenario3_boxes),classes_found
        ))
        raise AssertionError()
    else:
        print("Scenario 3 passed: Both different-class boxes kept")

    # Verify scenario 4 - cascading overlapping boxes (only highest confidence should remain)
    scenario4_boxes = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if 460 <= center_x <= 560 and 260 <= center_y <= 360 and int(cls) == 0:
            scenario4_boxes.append((i, center_x, center_y, conf))

    print(f"\nScenario 4 analysis: Found {len(scenario4_boxes)} boxes in cascading area:")
    for i, (det_idx, cx, cy, conf) in enumerate(scenario4_boxes):
        print(f"  Box {i}: center=({cx:.1f}, {cy:.1f}), conf={conf:.3f}")

    # Check IoU between remaining boxes to ensure proper suppression
    if len(scenario4_boxes) >= 2:
        max_iou = 0
        for i in range(len(scenario4_boxes)):
            for j in range(i+1, len(scenario4_boxes)):
                det1 = detections[scenario4_boxes[i][0]]
                det2 = detections[scenario4_boxes[j][0]]
                iou = calculate_iou_boxes(det1[:4], det2[:4])
                print(f"  IoU between box {i} and box {j}: {iou:.3f}")
                max_iou = max(max_iou, iou)

        if len(scenario4_boxes) == 1:
            print("Scenario 4 passed: Only highest confidence box kept")
        else:
            # This is only OK if IoU < threshold
            if max_iou < 0.5:  # Our IoU threshold
                print("Scenario 4 passed: Multiple boxes kept due to low IoU (< 0.5)")
            else:
                print(f"ERROR: Scenario 4 failed - boxes with IoU {max_iou:.3f} > 0.5 were not suppressed!")
                raise AssertionError()

    # Create a scenario that requires IoU calculation
    print("\n=== COMPREHENSIVE IoU VALIDATION TEST ===")

    # Create two identical boxes that should definitely be suppressed
    identical_box_a = [100, 100, 50, 50, 0.9, 0.9, 0.05, 0.05]  # High confidence
    identical_box_b = [100, 100, 50, 50, 0.9, 0.7, 0.1, 0.1]    # Lower confidence

    test_predictions = torch.zeros(1, 5, 8)  # Small batch for focused test
    test_predictions[0, 0, :] = torch.tensor(identical_box_a)
    test_predictions[0, 1, :] = torch.tensor(identical_box_b)

    # Run NMS on this simple case
    test_results = nms(test_predictions, conf_thres=0.3, iou_thres=0.5, max_det=300)
    test_detections = test_results[0]

    print(f"Identical boxes test: Input 2 identical boxes, got {test_detections.shape[0]} detections")

    if test_detections.shape[0] != 1:
        print(f"Error Two identical boxes should result in 1 detection, got {test_detections.shape[0]}")
        raise AssertionError()
    else:
        # Verify it kept the higher confidence box
        kept_conf = test_detections[0, 4].item()
        expected_conf = 0.9 * 0.9  # objectness * class_conf
        if abs(kept_conf - expected_conf) > 0.01:
            print(f"ERROR: Wrong box kept. Expected conf ≈ {expected_conf:.3f}, got {kept_conf:.3f}")
            raise AssertionError()
        else:
            print("Identical boxes test passed: Higher confidence box kept")

    print("\nNMS tests passed")
