import logging
import numpy as np
from .detector import GridDetector
from .engine import WarpingEngine

logger = logging.getLogger("ECGUnwarper.Processor")


class ECGUnwarper:
    """Klasa fasady integrująca detekcję i transformację."""

    def __init__(self, method='tps', grid_density=(10, 10)):
        self.detector = GridDetector(grid_density)
        self.warper = WarpingEngine(method)
        self.grid_density = grid_density

    def process_photo(self, img: np.ndarray) -> np.ndarray:
        """Główny punkt wejścia do przetwarzania zdjęcia EKG."""
        points_src = self.detector.detect(img)

        if points_src is None:
            return img

        # Generujemy siatkę o wymiarach wejściowych jako cel
        points_dst = self.warper.generate_ideal_grid(img.shape[:2], self.grid_density)

        try:
            return self.warper.apply_warp(img, points_src, points_dst)
        except Exception as e:
            logger.error(f"Krytyczny błąd podczas unwarpingu: {e}")
            return img