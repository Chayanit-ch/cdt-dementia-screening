import numpy as np


def get_vqa_feat(filename, vqa_dict):
    if filename in vqa_dict:
        return vqa_dict[filename]
    alt = filename.replace(".jpg", ".tif") \
          if ".jpg" in filename \
          else filename.replace(".tif", ".jpg")
    return vqa_dict.get(alt, {})


def features_to_vector(features):
    """VQA features v1 -> 12 dims"""
    if features is None or "error" in features:
        return np.zeros(12)
    vec = []
    vec.append(
        features.get("digit_count", 0) / 12.0)
    digits = features.get("digits_present", [])
    vec.append(len(digits) / 12.0)
    vec.append(1.0 if features.get(
        "digit_positions_correct") == "Yes" else 0.0)
    vec.append(
        features.get("quadrant_balance", 0) / 4.0)
    wq = features.get("worst_quadrant", "none")
    wq_map = {
        "top-left"    : [1,0,0,0,0],
        "top-right"   : [0,1,0,0,0],
        "bottom-left" : [0,0,1,0,0],
        "bottom-right": [0,0,0,1,0],
        "none"        : [0,0,0,0,1],
    }
    vec.extend(wq_map.get(wq, [0,0,0,0,1]))
    vec.append(1.0 if features.get(
        "has_hands") == "Yes" else 0.0)
    vec.append(
        features.get("hand_count", 0) / 2.0)
    vec.append(1.0 if features.get(
        "hands_at_correct_position") == "Yes"
        else 0.0)
    vec.append(
        features.get("spatial_regularity", 1) / 5.0)
    vec.append(
        features.get("overall_quality", 1) / 5.0)
    return np.array(vec, dtype=np.float32)
