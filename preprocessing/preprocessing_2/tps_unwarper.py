import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp


class TPSUnwarper:
    """Klasa odpowiedzialna za nieliniowe prostowanie obrazu."""

    def generate_target_grid(self, source_points: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """
        Zamiast zgadywać sqrt, przekaż strukturę siatki wykrytą przez detektor.
        """
        x_min, y_min = np.min(source_points, axis=0)
        x_max, y_max = np.max(source_points, axis=0)

        # Generujemy idealne współrzędne dla dokładnie takiej liczby punktów, jaką mamy
        x_coords = np.linspace(x_min, x_max, cols)
        y_coords = np.linspace(y_min, y_max, rows)

        xv, yv = np.meshgrid(x_coords, y_coords)
        return np.vstack([xv.ravel(), yv.ravel()]).T.astype(np.float32)

    def apply_transform(self, image: np.ndarray, source: np.ndarray, target: np.ndarray,
                        method='piecewise') -> np.ndarray:
        """
        Domyślnie używa PiecewiseAffineTransform dla stabilności przy lokalnych zagięciach.
        """
        if method == 'piecewise':
            tform = PiecewiseAffineTransform()
            tform.estimate(target, source)  # Mapujemy docelowy (płaski) na źródłowy (zgięty)
            out = warp(image, tform, output_shape=image.shape[:2])
            return (out * 255).astype(np.uint8)

        elif method == 'tps':
            # Implementacja przy użyciu OpenCV TPS
            tps = cv2.createThinPlateSplineShapeTransformer()

            # TPS w OpenCV wymaga specyficznego formatu dopasowania (1, N, 2)
            source_reshaped = source.reshape(1, -1, 2)
            target_reshaped = target.reshape(1, -1, 2)

            # Tworzymy listę dopasowań (DMatch)
            matches = [cv2.DMatch(i, i, 0) for i in range(len(source))]

            tps.estimateTransformation(target_reshaped, source_reshaped, matches)
            return tps.warpImage(image)