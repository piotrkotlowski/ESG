import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_long_lines(img, min_lenght_ratio=0.25):
    """
    Wydobywa najdłuższe proste linie z obrazu.

    :param image_path: ścieżka do pliku z rysunkiem
    :param min_line_length: minimalna długość linii w pikselach
    :param max_line_gap: maksymalna przerwa między segmentami linii
    :return: obraz z naniesionymi liniami, lista linii [(x1,y1,x2,y2),...]
    """

    # 1. Wczytanie obrazu i konwersja do odcieni szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    max_dim = max(gray.shape)
    min_line_length = min_lenght_ratio * max_dim
    max_line_gap = 0.01 * max_dim
    # 2. Wygładzenie szumu
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 3. Wykrywanie krawędzi
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # 4. Wydobycie linii prostych (Probabilistyczna Hougha)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # 5. Tworzenie obrazu do rysowania linii
    output_image = img.copy()
    extracted_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0,255,0), 2)
            extracted_lines.append((x1, y1, x2, y2))

    return output_image, extracted_lines

# Przykład użycia:
def filter_horizontal_lines(lines):
    """
    Filtruje linie, zachowując tylko te, dla których delta x > delta y.

    :param lines: lista linii [(x1,y1,x2,y2), ...]
    :return: lista przefiltrowanych linii
    """
    filtered = []
    for x1, y1, x2, y2 in lines:
        delta_x = abs(x2 - x1)
        delta_y = abs(y2 - y1)
        if delta_x > delta_y:
            filtered.append((x1, y1, x2, y2))
    return filtered


def plot_lines_on_image_rgb(image_rgb, lines, line_color=(0,255,0), line_thickness=2):
    """
    Rysuje linie na obrazie RGB i wyświetla wynik.

    :param image_rgb: obraz RGB (numpy array)
    :param lines: lista linii [(x1,y1,x2,y2), ...]
    :param line_color: kolor linii w BGR
    :param line_thickness: grubość linii
    :return: obraz z naniesionymi liniami
    """
    # Kopia obrazu, żeby nie nadpisać oryginału
    output_img = image_rgb.copy()

    # Rysowanie linii
    for x1, y1, x2, y2 in lines:
        cv2.line(output_img, (x1, y1), (x2, y2), line_color, line_thickness)

    # Wyświetlenie obrazu przy użyciu Matplotlib (zamiana BGR → RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Filtered Lines")
    plt.show()
# Przykład użycia:
# plotted_img = plot_lines_on_image_rgb(img, filtered_lines)

import cv2
import numpy as np
import math

def rotate_image_full_alpha(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Obraca obraz RGB o zadany kąt przeciwnie do ruchu wskazówek zegara,
    zwracając obraz czterokanałowy RGBA. Obszary, które nie mają danych,
    są przezroczyste (alpha = 0).

    :param image: obraz RGB (numpy array, 3 kanały)
    :param angle: kąt w stopniach, przeciwnie do ruchu wskazówek zegara
    :return: obrócony obraz RGBA
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Macierz obrotu
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Obliczamy nowe wymiary
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Korekta translacji
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Dodajemy kanał alfa wypełniony 255 (pełna widoczność)
    if image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        rgba = image.copy()

    # Obrót z przezroczystym tłem
    rotated = cv2.warpAffine(rgba, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))  # czarny + alpha=0

    return rotated

def rotate_image_full(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Obraca obraz o zadany kąt przeciwnie do ruchu wskazówek zegara,
    zachowując cały obraz (bez przycinania).
    
    :param image: obraz RGB lub BGR (numpy array, 3 kanały)
    :param angle: kąt w stopniach, przeciwnie do ruchu wskazówek zegara
    :return: obrócony obraz (trzy kanałowy)
    """
    (h, w) = image.shape[:2]
    # Środek obrotu
    center = (w / 2, h / 2)

    # Macierz obrotu
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Obliczamy nowe wymiary obrazu, żeby nic nie zostało obcięte
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Korekta translacji w macierzy obrotu
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Obrót obrazu
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def rad_to_deg(angle_rad):
    """
    Konwertuje kąt z radianów na stopnie.
    
    :param angle_rad: kąt w radianach
    :return: kąt w stopniach
    """
    return angle_rad * (180.0 / math.pi)

import numpy as np
from PIL import Image

def zapisz_rgba_bez_brzegow(macierz, sciezka_output):
    """
    Zapisuje macierz numpy (H, W, 4) jako plik PNG bez marginesów.
    
    Parametry:
        macierz (np.ndarray): Macierz w formacie RGBA (0-255 lub 0.0-1.0)
        sciezka_output (str): Ścieżka zapisu (powinna kończyć się na .png)
    """
    # 1. Sprawdzenie/Konwersja na uint8 (wymagane przez PIL)
    if macierz.dtype != np.uint8:
        # Jeśli dane są w formacie 0.0 - 1.0, przemnóż przez 255
        if np.max(macierz) <= 1.01:
            macierz = (macierz * 255).astype(np.uint8)
        else:
            macierz = macierz.astype(np.uint8)

    # 2. Utworzenie obiektu obrazu z macierzy
    # PIL automatycznie rozpoznaje format 'RGBA' dla 4 kanałów
    img = Image.fromarray(macierz, 'RGBA')

    # 3. Zapis bez żadnych dodatkowych metadanych czy marginesów
    img.save(sciezka_output, format='PNG')

import numpy as np
import cv2
from scipy.signal import find_peaks

def generate_mm_grid_bw(image_rgba, threshold_ratio=0.3, small_lines_per_segment=4, intensity_small=128, min_segment_for_small=9):
    """
    Tworzy czarno-biały obraz siatki papieru milimetrowego.
    Grube linie = 0, cienkie linie = intensity_small (np. 128), tło = 255.
    Cienkie linie nie są dodawane, jeśli segment między grubymi liniami jest mniejszy niż min_segment_for_small.

    :param image_rgba: numpy array, RGBA lub RGB
    :param threshold_ratio: czułość wykrywania grubych linii
    :param small_lines_per_segment: liczba cienkich linii między grubymi
    :param intensity_small: jasność cienkich linii (0-255)
    :param min_segment_for_small: minimalna odległość segmentu, aby dodać cienkie linie
    :return: czarno-biały obraz uint8
    """
    if image_rgba.shape[2] == 3:
        image_rgba = np.dstack([image_rgba, np.full(image_rgba.shape[:2], 255, dtype=np.uint8)])

    # Konwersja do szarości i wzmocnienie kontrastu
    gray = cv2.cvtColor(image_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Projekcje
    hor_proj = np.sum(255 - enhanced, axis=1)
    ver_proj = np.sum(255 - enhanced, axis=0)

    # Wykrycie grubych linii
    hor_peaks, _ = find_peaks(hor_proj, height=threshold_ratio*np.max(hor_proj))
    ver_peaks, _ = find_peaks(ver_proj, height=threshold_ratio*np.max(ver_proj))

    typical_h = int(np.median(np.diff(hor_peaks))) if len(hor_peaks) > 1 else 20
    typical_v = int(np.median(np.diff(ver_peaks))) if len(ver_peaks) > 1 else 20

    h_lines = np.arange(hor_peaks[0], hor_peaks[-1]+typical_h, typical_h)
    v_lines = np.arange(ver_peaks[0], ver_peaks[-1]+typical_v, typical_v)

    # Dodanie cienkich linii między grubymi (tylko jeśli segment >= min_segment_for_small)
    def add_small_lines(lines, n_small):
        full_lines = [lines[0]]
        for i in range(1, len(lines)):
            start, end = lines[i-1], lines[i]
            if end - start < min_segment_for_small:
                full_lines.append(end)  # pomijamy cienkie linie
                continue
            step = (end - start) / (n_small + 1)
            for j in range(1, n_small+1):
                full_lines.append(int(round(start + j*step)))
            full_lines.append(end)
        return np.unique(full_lines)

    h_full = add_small_lines(h_lines, small_lines_per_segment)
    v_full = add_small_lines(v_lines, small_lines_per_segment)

    # Tworzymy czarno-biały obraz 1px linie
    grid_bw = np.full(image_rgba.shape[:2], 255, dtype=np.uint8)  # tło białe

    # Cienkie linie najpierw
    for y in h_full:
        if 0 <= y < grid_bw.shape[0] and y not in h_lines:
            grid_bw[y, :] = intensity_small
    for x in v_full:
        if 0 <= x < grid_bw.shape[1] and x not in v_lines:
            grid_bw[:, x] = intensity_small

    # Grube linie nad cienkimi
    for y in h_lines:
        if 0 <= y < grid_bw.shape[0]:
            grid_bw[y, :] = 0
    for x in v_lines:
        if 0 <= x < grid_bw.shape[1]:
            grid_bw[:, x] = 0

    return grid_bw

def overlay_grid_on_image(image_rgb, grid_bw, thin_alpha=0.3):
    """
    Nakłada siatkę czarno-białą na obraz RGB zachowując kolory.
    
    :param image_rgb: numpy array, RGB
    :param grid_bw: numpy array, czarno-biały obraz siatki (0 = grube linie, 128 = cienkie, 255 = tło)
    :param thin_alpha: przezroczystość cienkich linii (0-1)
    :return: obraz RGB z nałożoną siatką
    """
    # Upewniamy się, że image_rgb ma 3 kanały
    if image_rgb.shape[2] != 3:
        raise ValueError("image_rgb musi być obrazem RGB (3 kanały)")

    overlay = image_rgb.copy().astype(np.float32)

    # Grube linie (czarne) – pełne pokrycie
    mask_thick = grid_bw == 0
    overlay[mask_thick] = 0

    # Cienkie linie (np. szare) – częściowe pokrycie
    mask_thin = (grid_bw > 0) & (grid_bw < 255)
    overlay[mask_thin] = (1 - thin_alpha) * overlay[mask_thin] + thin_alpha * grid_bw[mask_thin, None]

    return overlay.astype(np.uint8)

import cv2
import numpy as np

def boost_dark_colors_v2(image, alpha=1.5, beta=-40):
    """
    Funkcja podkręcająca ciemne kolory na jasnym tle.
    Zwiększa kontrast i obniża jasność, czyniąc ciemne linie wyraźniejszymi.

    Parametry:
    :param image: Obraz wejściowy w formacie RGB (Numpy array)
    :param alpha: Współczynnik kontrastu. Wartości > 1 zwiększają kontrast.
                  Dla EKG zakres 1.2 - 2.0 jest zazwyczaj dobry.
    :param beta: Współczynnik jasności. Wartości < 0 zmniejszają jasność.
                 Zazwyczaj zakres od -20 do -60 działa dobrze.
    :return: Przetworzony obraz (Numpy array)
    """
    # 1. Konwersja do float, aby zapobiec nasyceniu podczas obliczeń
    image_float = image.astype(np.float32) / 255.0

    # 2. Zastosowanie transformacji liniowej: new_pixel = alpha * old_pixel + beta/255
    # Beta jest dzielona przez 255, ponieważ pracujemy na zakresie [0, 1]
    new_image_float = alpha * image_float + (beta / 255.0)

    # 3. Nasycenie (clip): ograniczenie wartości do zakresu [0, 1]
    # Wartości < 0 staną się 0 (czarny), a > 1 staną się 1 (biały)
    new_image_float = np.clip(new_image_float, 0, 1)

    # 4. Konwersja z powrotem do uint8 (0-255)
    processed_image = (new_image_float * 255).astype(np.uint8)

    return processed_image

def calculate_alpha(lines):
    mean_tanges = 0
    for line in lines:
        mean_tanges += (line[3] - line[1]) / (line[2] - line[0])

    return np.arctan(mean_tanges / len(lines))

import cv2
import numpy as np

def preprocess(img):
    """
    Pełny pipeline: Rotacja -> Generowanie siatki -> Nakładanie -> Podkręcanie detali.
    """
    # --- KROK 1: Wyznaczenie kąta i rotacja ---
    # Używamy Twoich funkcji do detekcji linii na podstawie ścieżki
    output_img, lines = extract_long_lines(img)
    filtered_lines = filter_horizontal_lines(lines)

    # Rezerwowy mechanizm, jeśli nie wykryto linii za pierwszym razem
    if len(filtered_lines) == 0:
        output_img, lines = extract_long_lines(image_path, min_lenght_ratio=0.1)
        filtered_lines = filter_horizontal_lines(lines)
    
    # Obliczenie kąta (zakładamy, że zwraca radiany, które zamieniamy na stopnie)
    angle_rad = calculate_alpha(filtered_lines)
    angle_deg = rad_to_deg(angle_rad)

    # Wczytanie obrazu do właściwej obróbki (już jako numpy array)
    # Zakładam, że wolisz pracować na RGB
    
    # Wykonanie rotacji
    rotated_img = rotate_image_full_alpha(img, angle=angle_deg)

    # Jeśli obraz po rotacji ma kanał Alpha (RGBA), konwertujemy go do RGB 
    # (lub ignorujemy alfę), by funkcje siatki działały poprawnie
    if rotated_img.shape[2] == 4:
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGBA2RGB)

    # --- KROK 2: Generacja i nakładanie siatki ---
    # Generujemy siatkę na już wyprostowanym obrazie
    grid_bw = generate_mm_grid_bw(rotated_img, threshold_ratio=0.3, small_lines_per_segment=4)

    # Nakładamy siatkę na obraz
    image_with_grid = overlay_grid_on_image(rotated_img, grid_bw, thin_alpha=0.3)

    # --- KROK 3: Finalny szlif (podkręcanie ciemnych kolorów) ---
    # To sprawi, że zarówno krzywa EKG, jak i nowa siatka będą wyraźniejsze
    final_result = boost_dark_colors_v2(image_with_grid, alpha=1.5, beta=-40)

    return final_result
