import torch
import torch.nn as nn
import cv2
import numpy as np
from skimage.feature import peak_local_max

from scipy.interpolate import griddata

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.act(out)
        return out

class DotterUNet(nn.Module):
    """
    Phase 2: Stage 1 - Image Flattening (Dotter U-Net)
    Initialize a standard U-Net in PyTorch. The goal is to correct spatial distortions by
    detecting millimeter grid intersections.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(DotterUNet, self).__init__()
        # Skeleton U-Net backbone
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        
        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = self.conv_block(256, 128)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_c, out_c):
        return ResBlock(in_c, out_c)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e2))
        
        # Decoder
        d1 = self.up_conv1(b)
        # Assuming sizes match (in a real scenario, pad or crop)
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up_conv2(d1)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.dec2(d2)
        
        # Output Heatmap (using MSE or Keypoint focal loss later)
        out = self.final_conv(d2)
        return out


def extract_grid_intersections(heatmap, min_distance=10, threshold_abs=0.5):
    """
    Apply a peak-finding algorithm to the output heatmap to get exact (X, Y)
    coordinates of every grid intersection.
    """
    # Assuming heatmap is a 2D numpy array probability map
    coordinates = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold_abs)
    # peak_local_max returns (row, col) which equates to (Y, X)
    # Convert to (X, Y) for OpenCV
    coordinates_xy = coordinates[:, ::-1]
    return coordinates_xy


def build_grid_matrix(points, tolerance=15):
    """
    Step 1: The Gridder Algorithm
    Sorts unstructured [X,Y] peaks into a structured 2D grid matrix and interpolates missing gaps.
    """
    if len(points) == 0:
        return np.array([])
        
    points = sorted(points.tolist(), key=lambda p: p[1])
    rows = []
    current_row = [points[0]]
    
    for p in points[1:]:
        if abs(p[1] - current_row[0][1]) <= tolerance:
            current_row.append(p)
        else:
            current_row = sorted(current_row, key=lambda p: p[0])
            rows.append(current_row)
            current_row = [p]
    
    if current_row:
        current_row = sorted(current_row, key=lambda p: p[0])
        rows.append(current_row)
        
    dxs, dys = [], []
    for r in rows:
        for i in range(len(r) - 1):
            dxs.append(r[i+1][0] - r[i][0])
    for i in range(len(rows) - 1):
        for p1 in rows[i]:
            closest = min(rows[i+1], key=lambda p2: abs(p1[0] - p2[0]))
            if abs(closest[0] - p1[0]) <= tolerance:
                dys.append(closest[1] - p1[1])
                
    if not dxs or not dys:
        return np.array([])

    median_dx = np.median(dxs)
    median_dy = np.median(dys)
    
    min_x = min(p[0] for r in rows for p in r)
    max_x = max(p[0] for r in rows for p in r)
    min_y = min(p[1] for r in rows for p in r)
    max_y = max(p[1] for r in rows for p in r)
    
    num_cols = int(round((max_x - min_x) / median_dx)) + 1
    num_rows = int(round((max_y - min_y) / median_dy)) + 1
    
    grid_matrix = np.zeros((num_rows, num_cols, 2))
    mask = np.zeros((num_rows, num_cols), dtype=bool)
    
    for r in rows:
        for p in r:
            col_idx = int(round((p[0] - min_x) / median_dx))
            row_idx = int(round((p[1] - min_y) / median_dy))
            if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
                grid_matrix[row_idx, col_idx] = p
                mask[row_idx, col_idx] = True
                
    known_coords = np.argwhere(mask)
    known_values = grid_matrix[mask]
    all_coords = np.indices((num_rows, num_cols)).reshape(2, -1).T
    
    try:
        grid_matrix_x = griddata(known_coords, known_values[:, 0], all_coords, method='linear')
        grid_matrix_y = griddata(known_coords, known_values[:, 1], all_coords, method='linear')
        
        nan_mask = np.isnan(grid_matrix_x)
        if np.any(nan_mask):
            grid_x_nearest = griddata(known_coords, known_values[:, 0], all_coords, method='nearest')
            grid_y_nearest = griddata(known_coords, known_values[:, 1], all_coords, method='nearest')
            grid_matrix_x[nan_mask] = grid_x_nearest[nan_mask]
            grid_matrix_y[nan_mask] = grid_y_nearest[nan_mask]
            
        grid_matrix[:, :, 0] = grid_matrix_x.reshape((num_rows, num_cols))
        grid_matrix[:, :, 1] = grid_matrix_y.reshape((num_rows, num_cols))
    except Exception as e:
        print("Interpolation failed:", e)
        
    return grid_matrix

def undistort_image(image, predicted_points):
    """
    Step 2: Undistortion Algorithm
    Iterate through the matrix to warp each grid cell individually into 20x20 patches.
    """
    if len(predicted_points) < 4:
        return image
        
    grid_matrix = build_grid_matrix(predicted_points)
    if len(grid_matrix) == 0:
        return image
        
    num_rows, num_cols, _ = grid_matrix.shape
    if num_rows < 2 or num_cols < 2:
        return image
        
    target_size = 20
    out_h = (num_rows - 1) * target_size
    out_w = (num_cols - 1) * target_size
    
    canvas = np.zeros((out_h, out_w, image.shape[2] if len(image.shape) > 2 else 1), dtype=np.uint8)
    
    dst_pts = np.array([
        [0, 0],
        [target_size, 0],
        [target_size, target_size],
        [0, target_size]
    ], dtype=np.float32)
    
    for r in range(num_rows - 1):
        for c in range(num_cols - 1):
            tl = grid_matrix[r, c]
            tr = grid_matrix[r, c+1]
            br = grid_matrix[r+1, c+1]
            bl = grid_matrix[r+1, c]
            
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            
            try:
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(image, M, (target_size, target_size))
                canvas[r*target_size : (r+1)*target_size, c*target_size : (c+1)*target_size] = warped
            except cv2.error as e:
                pass
                
    return canvas
