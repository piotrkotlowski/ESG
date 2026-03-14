import cv2
import numpy as np
from .grid_detector import GridPointDetector
from .tps_unwarper import TPSUnwarper


class ECGImageProcessor:
    """Publiczne API do odkształcania zdjęć EKG."""

    def __init__(self):
        self.detector = GridPointDetector()
        self.unwarper = TPSUnwarper()

    def preprocess_photo(self, photo: np.ndarray) -> np.ndarray:
        # 1. Wykrywanie linii siatki
        h_lines, v_lines = self.detector.detect_lines(photo)

        # 2. Wyznaczanie punktów kontrolnych (skrzyżowań)
        distorted_points = self.detector.find_intersections(h_lines, v_lines)

        if len(distorted_points) < 4:
            raise ValueError("Wykryto zbyt mało punktów siatki, aby przeprowadzić transformację.")

        # 3. Generowanie idealnej, płaskiej siatki
        # Uwaga: Aby TPS działał poprawnie, punkty muszą być posortowane w tej samej kolejności
        # W tej implementacji sortujemy je według współrzędnych y, a potem x
        distorted_points = distorted_points[np.lexsort((distorted_points[:, 0], distorted_points[:, 1]))]
        ideal_points = self.unwarper.generate_target_grid(distorted_points)

        # 4. Wykonanie transformacji
        straightened_photo = self.unwarper.apply_transform(
            photo,
            distorted_points,
            ideal_points,
            method='piecewise'  # 'tps' lub 'piecewise'
        )

        return straightened_photo