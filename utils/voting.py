"""
 File name   : voting.py
 Description : description

 Date created : 24.09.2021
 Author:  Ihar Khakholka
"""

from typing import List, Union, Tuple

import numpy as np



def aggregate_detections(detections: Union[np.ndarray, List[list]], landmarks: Union[List[list], np.ndarray],
              thresh_iou: List[float], min_votes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """"
    Function to aggregate detections: supress extra ones and keep ones with <min_votes> and with preferably landmarks.

    """
    def _not_empty_landmarks(lndms: List[float]) -> bool:
        return any([p >= 0 for p in lndms])

    assert len(thresh_iou) == len(min_votes)

    detections = np.array(detections)
    landmarks = np.array(landmarks)

    if not len(thresh_iou):
        return detections, landmarks

    landmarks_mask = list(map(_not_empty_landmarks, landmarks))

    keep = nms_landmarks(detections, landmarks_mask, thresh_iou[0], min_votes[0])

    return aggregate_detections(detections[keep], landmarks[keep], thresh_iou[1:], min_votes[1:])



def nms_landmarks(detections: np.ndarray, landmarks_mask: List[bool], thresh_iou: float, min_votes: int = 1) -> List[int]:
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    And keep the ones with landmarks.
    Args:
        boxes: The location preds for the image
            along with the class predscores, Shape: [x1, y1, x2, y2, score].
        landmarks_mask: list of indicaters if landmarks are bresented for boxes.
        thresh_iou: The overlap thresh for suppressing unnecessary boxes.
        min_votes: min votes needed for detection to be not supressed
    Returns:
        A list boxes idxs to be selected
    """
    landmarks_mask = np.array(landmarks_mask)

    # we extract coordinates for every
    # prediction box present in P
    start_x = detections[:, 0]
    start_y = detections[:, 1]
    end_x = detections[:, 2]
    end_y = detections[:, 3]

    # we extract the confidence scores as well
    scores = detections[:, 4]

    # calculate area of every block in P
    areas = (end_x - start_x) * (end_y - start_y)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:

        # The index of largest confidence score
        index = order[-1]

        # Compute coordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order])
        x2 = np.minimum(end_x[index], end_x[order])
        y1 = np.maximum(start_y[index], start_y[order])
        y2 = np.minimum(end_y[index], end_y[order])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order] - intersection)

        # split by ious
        candidates = np.where(ratio >= thresh_iou)
        left = np.where(ratio < thresh_iou)

        # Choose the most confident with landmarks

        candidate_indexes = order[candidates]

        # sort first by confidence then by landmarks. The sorted() alg is stable.
        candidate_indexes = sorted(candidate_indexes, key=lambda i: scores[i], reverse=True)
        candidate_indexes = sorted(candidate_indexes, key=lambda i: landmarks_mask[i], reverse=True)

        # update index with one with landmarks
        index = candidate_indexes[0]

        # save result
        if len(candidate_indexes) >= min_votes:
            keep.append(index)

        # remove supressed boxes
        order = order[left]

    return keep