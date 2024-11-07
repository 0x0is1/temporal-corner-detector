import matplotlib.pyplot as plt
import numpy as np

class MetricsEvaluator:
    def __init__(self):
        self.temporal_consistency_ratios = {"conventional": [], "enhanced": []}
        self.detected_corners_counts = {"conventional": [], "enhanced": []}
        self.consistent_corners = {"conventional": 0, "enhanced": 0}
        self.total_corners = {"conventional": 0, "enhanced": 0}

    def update_metrics(self, smoothed_corners, previous_corners, approach):
        if previous_corners is not None:
            consistency_threshold = 3
            consistent_corners = np.sum([np.linalg.norm(c - p) < consistency_threshold for c in smoothed_corners for p in previous_corners])
            self.consistent_corners[approach] += consistent_corners
            self.total_corners[approach] += len(smoothed_corners)

        temporal_consistency_ratio = self.consistent_corners[approach] / self.total_corners[approach] if self.total_corners[approach] > 0 else 0
        self.temporal_consistency_ratios[approach].append(temporal_consistency_ratio)
        self.detected_corners_counts[approach].append(len(smoothed_corners))

    def show_metrics(self):
        plt.figure(figsize=(10, 8))
        plt.suptitle("Metrics Comparison: Conventional vs Enhanced")

        plt.subplot(2, 1, 1)
        plt.plot(self.temporal_consistency_ratios["conventional"], label="Conventional", linestyle="--", color="orange")
        plt.plot(self.temporal_consistency_ratios["enhanced"], label="Enhanced", color="blue")
        plt.title("Temporal Consistency Ratio")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.detected_corners_counts["conventional"], label="Conventional", linestyle="--", color="orange")
        plt.plot(self.detected_corners_counts["enhanced"], label="Enhanced", color="green")
        plt.title("Number of Detected Corners")
        plt.legend()

        plt.tight_layout()
        plt.show()
