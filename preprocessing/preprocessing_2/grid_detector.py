import cv2
import numpy as np


class GridPointDetector:
    """Detekcja punktów przecięcia siatki na zdjęciu EKG."""

    def detect_lines(self, image: np.ndarray):
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 3. Morphological Operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detected_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        detected_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        return detected_h, detected_v

    def find_intersections(self, h_lines: np.ndarray, v_lines: np.ndarray) -> np.ndarray:
        # Przecięcie masek (logic AND)
        intersection_mask = cv2.bitwise_and(h_lines, v_lines)

        # Znalezienie środków ciężkości "plamek" przecięć
        cnts, _ = cv2.findContours(intersection_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])

        return np.array(points, dtype=np.float32)