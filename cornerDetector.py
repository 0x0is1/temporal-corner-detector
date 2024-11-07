import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional

class TemporalCornerDetector:
    def __init__(self, corner_threshold: float = 0.01, smoothing_window: int = 5):
        self.prev_gray = None
        self.corner_threshold = corner_threshold
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.tracked_points = []
        self.smoothing_window = smoothing_window

    def detect_corners_harris(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        adaptive_threshold = self.corner_threshold + 0.0005 * len(self.tracked_points)
        corners = np.argwhere(dst > adaptive_threshold * dst.max())
        return corners.reshape(-1, 1, 2).astype(np.float32)

    def detect_corners_fast(self, frame: np.ndarray) -> np.ndarray:
        fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        keypoints = fast.detect(frame, None)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    def adaptive_corner_detection(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Frame is None, ensure valid frame input")
        return self.detect_corners_fast(frame) if np.mean(frame) < 128 else self.detect_corners_harris(frame)

    def filter_corners_by_density(self, corners: np.ndarray, density_threshold: int = 10) -> np.ndarray:
        if len(corners) < density_threshold:
            return corners

        nbrs = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(corners.reshape(-1, 2))
        distances, _ = nbrs.kneighbors(corners.reshape(-1, 2))

        dense_corners = [corner for i, corner in enumerate(corners) if np.mean(distances[i]) < density_threshold]
        return np.array(dense_corners).reshape(-1, 1, 2) if dense_corners else corners

    def track_corners(self, frame: np.ndarray, corners: np.ndarray, frame_count: int) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None and len(self.tracked_points) > 0 and frame_count % 10 != 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.tracked_points, None, **self.lk_params)
            self.tracked_points = next_points[status == 1].reshape(-1, 1, 2)
        else:
            self.tracked_points = corners
        self.prev_gray = gray
        return self.tracked_points

    def temporal_smoothing(self, corners_list: List[np.ndarray]) -> np.ndarray:
        if not corners_list or all(len(corners) == 0 for corners in corners_list):
            return np.array([])

        valid_corners = [corners for corners in corners_list if corners.size > 0]
        if len(valid_corners) < self.smoothing_window:
            return np.vstack(valid_corners)

        max_corners = max(len(c) for c in valid_corners[-self.smoothing_window:])
        padded_corners = [
            np.pad(corners, ((0, max_corners - len(corners)), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
            for corners in valid_corners[-self.smoothing_window:]
        ]
        stacked_corners = np.stack(padded_corners)
        smoothed_corners = np.nanmean(stacked_corners, axis=0)
        smoothed_corners = smoothed_corners[~np.isnan(smoothed_corners[:, :, 0])]
        return smoothed_corners

    def process_frame(self, frame: np.ndarray, frame_count: int) -> np.ndarray:
        corners = self.adaptive_corner_detection(frame)
        corners = self.filter_corners_by_density(corners)
        tracked_corners = self.track_corners(frame, corners, frame_count)
        return tracked_corners

    def update(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        smoothed_corners = []
        corners_list = []
        for frame_count, frame in enumerate(frames):
            corners = self.process_frame(frame, frame_count)
            corners_list.append(corners)
            smoothed_corners.append(self.temporal_smoothing(corners_list))
        return smoothed_corners
