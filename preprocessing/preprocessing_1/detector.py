import cv2
import numpy as np
import logging

logger = logging.getLogger("ECGUnwarper.Detector")


class GridDetector:
    """Odpowiada za detekcję punktów siatki na obrazie przy użyciu filtracji HSV."""

    def __init__(self, grid_density=(10, 10)):
        self.grid_density = grid_density

    def detect(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definicja masek dla koloru siatki (róż/czerwień)
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        grid_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

        # Detekcja skrzyżowań linii siatki
        points = cv2.goodFeaturesToTrack(
            grid_mask,
            maxCorners=self.grid_density[0] * self.grid_density[1],
            qualityLevel=0.01,
            minDistance=20
        )

        if points is None or len(points) < (self.grid_density[0] * self.grid_density[1]) * 0.25:
            logger.warning("Zbyt mało punktów siatki wykrytych na obrazie.")
            return None

        # Sortowanie punktów dla zachowania spójności (lewa-góra -> prawa-dół)
        points = points.reshape(-1, 2)
        return points[np.lexsort((points[:, 0], points[:, 1]))]