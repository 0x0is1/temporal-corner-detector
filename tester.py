import numpy as np

class Tester:
    def __init__(self):
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []

    def evaluate_detection(self, conventional, enhanced, threshold=3.0):
        matched_pairs = 0

        for ec in enhanced:
            if any(np.linalg.norm(ec - c) < threshold for c in conventional):
                matched_pairs += 1

        tp = matched_pairs
        fp = len(enhanced) - tp
        fn = len(conventional) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.f1_scores.append(f1_score)

    def display_metrics(self):
        avg_precision = np.mean(self.precision_scores)
        avg_recall = np.mean(self.recall_scores)
        avg_f1_score = np.mean(self.f1_scores)

        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1_score:.4f}")
