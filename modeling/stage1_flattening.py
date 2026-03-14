import torch
import torch.nn as nn
import cv2
import numpy as np
from skimage.feature import peak_local_max

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
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

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


def undistort_image(image, predicted_points):
    """
    Iterate through sets of four adjacent intersection points.
    Calculate warping matrix and cv2.warpPerspective to morph the warped square
    into a mathematically flat square. Stitch these together into normalized image.
    """
    # Skeleton logic: We would logically find the 4 corners of the grid
    # or a piece-wise affine transformation on the 4-point sets.
    
    # As a mock setup for the skeleton, assuming we have top-left, top-right,
    # bottom-right, bottom-left of the entire grid to undistort the bounding box.
    if len(predicted_points) < 4:
        # Robust error handling for degraded images
        return image
        
    # Assume we ordered the points for a single big warp for simplicity in the skeleton
    src_pts = np.array(predicted_points[:4], dtype=np.float32)
    
    # Define ideal mathematically flat square (e.g. 500x500 pixels)
    dst_pts = np.array([
        [0, 0],
        [500 - 1, 0],
        [500 - 1, 500 - 1],
        [0, 500 - 1]
    ], dtype=np.float32)
    
    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (500, 500))
        return warped
    except cv2.error as e:
        # Fallback in case of singular matrix or OpenCV failure
        print(f"Failed to warp perspective: {e}")
        return image
