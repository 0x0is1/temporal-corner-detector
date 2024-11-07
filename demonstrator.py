import cv2
from metricEvaluator import MetricsEvaluator
from cornerDetector import TemporalCornerDetector
from tester import Tester

def runDemonstration(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = TemporalCornerDetector()
    evaluator = MetricsEvaluator()
    tester = Tester()
    corners_history = {"conventional": [], "enhanced": []}
    frame_count = 0
    FRAME_RATIO_CONSTANT = 160
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_RATIO_CONSTANT*4, FRAME_RATIO_CONSTANT*3))

        conventional_corners = detector.adaptive_corner_detection(frame)
        filtered_corners = detector.filter_corners_by_density(conventional_corners)
        evaluator.update_metrics(filtered_corners, corners_history["conventional"][-1] if corners_history["conventional"] else None, "conventional")
        corners_history["conventional"].append(filtered_corners)

        enhanced_corners = detector.track_corners(frame, filtered_corners, frame_count)
        smoothed_corners = detector.temporal_smoothing(corners_history["enhanced"])
        evaluator.update_metrics(smoothed_corners, corners_history["enhanced"][-1] if corners_history["enhanced"] else None, "enhanced")
        corners_history["enhanced"].append(enhanced_corners)

        for corner in filtered_corners:
            cv2.circle(frame, tuple(corner.ravel().astype(int)), 3, (0, 0, 255), -1)  # Red for conventional corners
        for corner in smoothed_corners:
            cv2.circle(frame, tuple(corner.ravel().astype(int)), 3, (0, 255, 0), -1)  # Green for enhanced corners

        tester.evaluate_detection(filtered_corners, smoothed_corners)

        cv2.imshow("Enhanced Temporal Consistency Corner Detection", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    evaluator.show_metrics()
    tester.display_metrics()
