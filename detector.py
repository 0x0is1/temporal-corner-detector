import cv2
import numpy as np

class Detector:
    def __init__(self, video_path):
        self.video_path = video_path

    def load_video(self, gap=1):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = int(fps * gap)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            for _ in range(skip_frames):
                cap.grab()

        cap.release()
        return frames

    def run_harris_detector(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        harris_dest = cv2.cornerHarris(gray_frame, 2, 5, 0.07)
        harris_dest = cv2.dilate(harris_dest, None)
        harris_image = frame.copy()
        harris_image[harris_dest > 0.01 * harris_dest.max()] = [0, 0, 255]
        return harris_image

    def run_fast_detector(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fast_detector = cv2.FastFeatureDetector_create()
        keypoints = fast_detector.detect(gray_frame, None)
        fast_image = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return fast_image
