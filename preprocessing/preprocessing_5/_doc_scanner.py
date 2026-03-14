import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from typing import Tuple, Optional

class DocScanner:
    def __init__(self, model_name: str = "resnet34"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_name)
        self.input_size = (512, 512)

    def _load_model(self, backbone: str):
        model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet", classes=1)
        model.to(self.device).eval()
        return model

    def _get_mask(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Konwersja BGR -> RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, self.input_size) / 255.0

        # Prosta normalizacja (lub użyj smp.encoders.get_preprocessing_fn)
        img_tensor = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()

        mask = (mask > 0.5).astype(np.uint8)
        return mask  # Zwracamy małą maskę dla wydajności

    def _find_corners(self, mask: np.ndarray, orig_size: Tuple[int, int]) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        points = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(cnt))

        # Skalowanie punktów z 512x512 do oryginału
        scale_x = orig_size[0] / self.input_size[0]
        scale_y = orig_size[1] / self.input_size[1]
        points = points.astype(np.float32)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y

        return points

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Sortuje punkty: [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Suma min = top-left
        rect[2] = pts[np.argmax(s)] # Suma max = bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Różnica min = top-right
        rect[3] = pts[np.argmax(diff)] # Różnica max = bottom-left
        return rect

    def _warp_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Wykonuje transformację macierzową, aby wyprostować obraz."""
        rect = self._order_points(corners)
        (tl, tr, br, bl) = rect

        # Obliczanie szerokości nowego obrazu
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Obliczanie wysokości
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        # Macierz transformacji $M$
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def preprocess_photo(self, image: np.ndarray) -> np.ndarray:
        """
        Główny interfejs: 
        Prostuje geometrię obrazu, usuwa perspektywę i zwraca 'płaski' dokument.
        
        :param image: Obraz wejściowy BGR (np.ndarray)
        :return: Wyprostowany dokument (widok z góry)
        """
        # 1. Uzyskaj maskę obszaru dokumentu
        mask = self._get_mask(image)
        
        # 2. Znajdź punkty kontrolne (narożniki)
        corners = self._find_corners(mask)
        
        if corners is not None:
            # 3. Wyprostuj obraz na podstawie narożników
            rectified = self._warp_perspective(image, corners)
            return rectified
        
        return image # Fallback jeśli nie znaleziono dokumentu