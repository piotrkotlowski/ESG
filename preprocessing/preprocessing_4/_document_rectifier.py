import cv2
import numpy as np
from typing import Tuple, Optional

class DocumentRectifier:
    """
    Klasa realizująca geometryczne prostowanie dokumentów (Perspective Warp).
    """

    def __init__(self):
        self.image = None
        self.working_copy = None

    def load_image(self, path: str) -> np.ndarray:
        """Krok 1: Wczytanie obrazu z dysku."""
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Nie można wczytać pliku z lokalizacji: {path}")
        self.working_copy = self.image.copy()
        return self.image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Krok 2: Redukcja szumów i przygotowanie do detekcji krawędzi."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Rozmycie Gaussa usuwa drobne detale (np. kratkę EKG), zostawiając obrys kartki
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Krok 3: Wyznaczenie granic dokumentu algorytmem Canny."""
        # Parametry 75, 200 są standardem dla dokumentów o dobrym kontraście
        edges = cv2.Canny(image, 75, 200)
        return edges

    def find_document_contour(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """Krok 4: Identyfikacja największego zamkniętego obszaru."""
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Sortowanie po powierzchni (malejąco) i wybór największego
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[0]

    def approximate_polygon(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """Krok 5: Aproksymacja konturu do 4 punktów (prostokąt)."""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
        # Jeśli nie znaleziono 4 punktów, próba z wypukłą otoczką
        hull = cv2.convexHull(contour)
        peri_h = cv2.arcLength(hull, True)
        approx_h = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
        
        if len(approx_h) == 4:
            return approx_h.reshape(4, 2)
        
        return None

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Krok 6: Sortowanie punktów: [TL, TR, BR, BL]."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Suma x+y: TL ma najmniejszą, BR największą
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Różnica y-x: TR ma najmniejszą (lub x-y), BL największą
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    def compute_perspective_transform(self, rect: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Krok 7: Obliczenie macierzy transformacji i wymiarów docelowych."""
        (tl, tr, br, bl) = rect

        # Obliczanie maksymalnej szerokości (dół vs góra)
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Obliczanie maksymalnej wysokości (lewo vs prawo)
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return M, max_width, max_height

    def warp_image(self, image: np.ndarray, M: np.ndarray, width: int, height: int) -> np.ndarray:
        """Krok 8: Wykonanie właściwego prostowania obrazu."""
        return cv2.warpPerspective(image, M, (width, height))

    def process_photo(self, image: np.ndarray) -> np.ndarray:
        """Integracja wszystkich kroków w jeden proces."""
        processed = self.preprocess_image(image)
        edges = self.detect_edges(processed)
        contour = self.find_document_contour(edges)
        
        if contour is not None:
            pts = self.approximate_polygon(contour)
            if pts is not None:
                rect = self.order_points(pts)
                M, w, h = self.compute_perspective_transform(rect)
                return self.warp_image(image, M, w, h)
        
        print("Błąd: Nie udało się wykryć 4 narożników dokumentu.")
        return image