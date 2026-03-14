import cv2
import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import griddata

# --- 3.1 Interface ---
class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess_photo(self, photo: np.ndarray) -> np.ndarray:
        pass

# --- 3.2 Class: DepthEstimator ---
class DepthEstimator:
    def __init__(self, model_type: str = "MiDaS_small"):
        # Ładowanie modelu MiDaS z TorchHub
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.model.to(self.device).eval()

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        # Konwersja BGR do RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            # Resize do oryginalnego rozmiaru zdjęcia
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        # Normalizacja i wygładzanie (Step 1)
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
        return depth_map

# --- 3.3 Class: SurfaceReconstructor ---
class SurfaceReconstructor:
    def build_mesh(self, depth_map: np.ndarray):
        h, w = depth_map.shape
        # Tworzenie siatki punktów (x, y)
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xv, yv = np.meshgrid(x, y)
        
        # Współrzędne 3D (x, y, z)
        # Z jest przeskalowaną wartością głębi
        mesh = np.stack([xv, yv, depth_map], axis=-1)
        return mesh

# --- 3.4 Class: GeometricFlattener ---
class GeometricFlattener:
    def unroll_mesh(self, mesh: np.ndarray, original_shape: tuple):
        h, w = original_shape[:2]

        dz_dx = np.diff(mesh[:, :, 2], axis=1, prepend=0)
        dz_dy = np.diff(mesh[:, :, 2], axis=0, prepend=0)

        dist_x = np.sqrt(1 + dz_dx ** 2)
        dist_y = np.sqrt(1 + dz_dy ** 2)

        fwd_target_x = np.cumsum(dist_x, axis=1)
        fwd_target_y = np.cumsum(dist_y, axis=0)

        out_x, out_y = np.meshgrid(np.arange(w), np.arange(h))

        points = np.stack([fwd_target_x.ravel(), fwd_target_y.ravel()], axis=-1)

        map_x_inv = griddata(points, out_x.ravel(), (fwd_target_x, fwd_target_y), method='linear')
        map_y_inv = griddata(points, out_y.ravel(), (fwd_target_x, fwd_target_y), method='linear')

        return map_x_inv.astype(np.float32), map_y_inv.astype(np.float32)

    def remap_pixels(self, original_photo, map_x, map_y):
        # Teraz map_x i map_y są w formacie "skąd wziąć", więc remap zadziała
        return cv2.remap(original_photo, map_x, map_y,
                         interpolation=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REPLICATE)

# --- 3.5 Class: DepthProcessor (Main Implementation) ---
class DepthProcessor(BasePreprocessor):
    def __init__(self):
        self.estimator = DepthEstimator()
        self.reconstructor = SurfaceReconstructor()
        self.flattener = GeometricFlattener()

    def preprocess_photo(self, photo: np.ndarray) -> np.ndarray:
        # Krok 1: Predykcja głębi (MDE)
        depth_map = self.estimator.predict_depth(photo)
        
        # Krok 2: Generowanie topografii 3D
        mesh = self.reconstructor.build_mesh(depth_map)
        
        # Krok 3: Wyliczenie mapowania (Flattening)
        map_x, map_y = self.flattener.unroll_mesh(mesh, photo.shape)
        
        # Krok 4: Synteza obrazu końcowego
        unwarped = self.flattener.remap_pixels(photo, map_x, map_y)
        return unwarped

# --- Przykład użycia ---
if __name__ == "__main__":
    processor = DepthProcessor()
    # input_img = cv2.imread("pogniecione_ekg.jpg")
    # result = processor.preprocess_photo(input_img)