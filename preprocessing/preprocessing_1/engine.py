import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp


class WarpingEngine:
    def __init__(self, method='tps'):
        self.method = method

    def generate_ideal_grid(self, target_size, grid_density):
        h, w = target_size
        rows, cols = grid_density
        y = np.linspace(0, h - 1, rows)
        x = np.linspace(0, w - 1, cols)
        xv, yv = np.meshgrid(x, y)
        return np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)

    def apply_warp(self, img, pts_src, pts_dst):
        # 1. Konwersja typów (TPS wymaga float32)
        pts_src = np.array(pts_src, dtype=np.float32).reshape(1, -1, 2)
        pts_dst = np.array(pts_dst, dtype=np.float32).reshape(1, -1, 2)

        # 2. Synchronizacja liczby punktów
        n = min(pts_src.shape[1], pts_dst.shape[1])
        pts_src = pts_src[:, :n, :]
        pts_dst = pts_dst[:, :n, :]

        h, w = img.shape[:2]

        if self.method == 'tps':
            # Tworzymy listę dopasowań 1:1
            matches = [cv2.DMatch(i, i, 0) for i in range(n)]

            tps = cv2.createThinPlateSplineShapeTransformer()

            try:
                # Mapowanie: z siatki docelowej (dst) do źródłowej (src)
                tps.estimateTransformation(pts_dst, pts_src, matches)
                return tps.warpImage(img)
            except cv2.error as e:
                print(f"Błąd TPS: {e} - prawdopodobnie punkty są współliniowe.")
                return img

        elif self.method == 'homography':
            # Jeśli zniekształcenie jest tylko perspektywiczne (np. zdjęcie pod kątem)
            # Homografia jest znacznie szybsza niż TPS.
            M, mask = cv2.findHomography(pts_src[0], pts_dst[0], cv2.RANSAC, 5.0)
            return cv2.warpPerspective(img, M, (w, h))

        return img