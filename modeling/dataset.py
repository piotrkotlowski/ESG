import os
import glob
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

try:
    import wfdb
except ImportError:
    print("WARNING: wfdb not installed. Please install: pip install wfdb")
    wfdb = None

class ECGDigitizationDataset(Dataset):
    """
    Dataset for loading ECG images and target 1D signals.
    Phase 1: Data Preparation & Ground Truth Generation.
    
    Structure your custom dataset to mirror the competition, separating it into a training folder
    (containing images and 1D digital signal files).
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Directory containing the images and signal data.
            split (str): 'train', 'val', or 'test'. 
            transform: Optional transforms to be applied on an image.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 1. File Parsing & Record Pairing
        # Scan for file groups sharing the same prefix (e.g., ecg_train_0001)
        self.records = []
        if os.path.exists(data_dir):
            # Find all images
            image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
            for img_path in image_paths:
                # Extract the base prefix to find corresponding files
                base_name = os.path.splitext(img_path)[0]
                
                record = {
                    "image": img_path,
                    "base_name": base_name,
                    "json": f"{base_name}.json",
                    "hea": f"{base_name}.hea",
                    "dat": f"{base_name}.dat"
                }
                self.records.append(record)

    def __len__(self):
        return max(len(self.records), 1)  # Skeleton fallback

    def __getitem__(self, idx):
        # Skeleton fallback for missing data
        if not self.records:
            # Create a mock image and masks for skeleton return
            mock_img = np.zeros((800, 1200, 3), dtype=np.uint8)
            grid_mask = self.generate_grid_mask(mock_img)
            target_mask = self.generate_segmentation_mask(mock_img, leads=[])
            return {"image": mock_img, "grid_mask": grid_mask, "target_mask": target_mask}

        record = self.records[idx]
        image = cv2.imread(record["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Default empty metadata dictionary
        metadata = {
            "sampling_frequency": 500.0,
            "grid_line_color_major": [255, 0, 0],
            "grid_line_color_minor": [255, 128, 128],
            "leads": []
        }
        
        # Parse JSON Meta-data if it exists
        if os.path.exists(record["json"]):
            with open(record["json"], 'r') as f:
                json_data = json.load(f)
                metadata.update(json_data)
                
        # Parse WFDB Signal (.hea/.dat pair)
        # Assuming you may want to extract ground truth millivolt signals to return for Phase 4 evaluation:
        wfdb_signal = None
        if wfdb is not None and os.path.exists(record["hea"]) and os.path.exists(record["dat"]):
            # Note: wfdb reads the prefix without extension
            sig, fields = wfdb.rdsamp(record["base_name"])
            wfdb_signal = {"signal": sig, "fields": fields}

        sample = {
            "image": image,
            "metadata": metadata, # Expose for Stage 4/5 calibration logic
            "wfdb": wfdb_signal
        }

        if self.split in ['train', 'val']:
            # Load your 1D ground truth and generate corresponding 2D masks
            grid_mask = self.generate_grid_mask(image, metadata)
            
            # Note: Due to overlap complexity, instead of concatenating all traces onto one image,
            # we isolate leads into separate channels or a combined dictionary.
            # In Stage 2 training, you will pick a lead and use its specific mask to train on.
            # `generate_segmentation_mask` now handles the 'plotted_pixels' from the JSON.
            target_mask_dict = self.generate_segmentation_mask(image, leads=metadata.get("leads", []))
            
            sample["grid_mask"] = grid_mask
            
            # You can concatenate them [Num_Leads, H, W] for the model, or just return the dict.
            # We'll return the dict of lead_name -> (Height, Width) mask tensor to let the DataLoaders organize it.
            sample["target_mask"] = target_mask_dict
            
        if self.transform:
            sample = self.transform(**sample)
            
        return sample

    def generate_grid_mask(self, image, metadata=None):
        """
        Stage 1: The "Dotter" Grid Mask
        Uses the JSON provided colors (grid_line_color_major / minor) and cv2.inRange
        to find grid line intersections, placing a 2D Gaussian dot (radius ~3-5 pixels).
        """
        h, w = image.shape[:2]
        grid_mask = np.zeros((h, w), dtype=np.float32)
        
        if not metadata or "grid_line_color_major" not in metadata:
            return grid_mask

        # Extract color (assuming JSON is [R, G, B])
        # cv2 uses RGB here because we converted earlier
        major_color = np.array(metadata["grid_line_color_major"])
        
        # Set a small range around the target color to find grid lines
        lower_bound = np.clip(major_color - 30, 0, 255)
        upper_bound = np.clip(major_color + 30, 0, 255)
        
        grid_lines = cv2.inRange(image, lower_bound, upper_bound)
        
        # To find exact intersections geometrically, we extract horizontal and vertical lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal = cv2.morphologyEx(grid_lines, cv2.MORPH_OPEN, kernel_h)
        
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical = cv2.morphologyEx(grid_lines, cv2.MORPH_OPEN, kernel_v)
        
        # Intersections are where horizontal and vertical lines overlap
        intersections = cv2.bitwise_and(horizontal, vertical)
        
        # Find points and plot them
        # (Where intersections > 0)
        y_coords, x_coords = np.where(intersections > 0)
        for x, y in zip(x_coords, y_coords):
             grid_mask[y, x] = 1.0

        # Apply a Gaussian blur to create the 3-5 pixel "Gaussian dots" target
        # Kernel size and sigma controls dot spread
        grid_mask = cv2.GaussianBlur(grid_mask, (7, 7), sigmaX=2, sigmaY=2)

        # Normalize the mask back to the [0.0, 1.0] range
        max_val = np.max(grid_mask)
        if max_val > 0:
            grid_mask = grid_mask / max_val

        return grid_mask

    def generate_segmentation_mask(self, image, leads=[]):
        """
        Stage 2: Signal Segmentation Mask
        Logic: Use the plotted_pixels from the JSON to create the ground truth.
        Iterate through the leads list, drawing `cv2.polylines` to map a continuous
        white line (value 1.0) on a black mask.
        Returns a dictionary mapping lead_name -> lead_mask resolving overlap.
        """
        h, w = image.shape[:2]
        masks = {}
        
        if not leads:
            return masks

        for lead_obj in leads:
            lead_name = lead_obj.get("lead_name", "UNKNOWN")
            plotted_pixels = lead_obj.get("plotted_pixels", [])
            
            # Initialize a pure black [h, w] mask for this specific lead
            lead_mask = np.zeros((h, w), dtype=np.float32)
            
            if not plotted_pixels:
                masks[lead_name] = lead_mask
                continue

            # Convert to numpy format for polylines: shape (number_of_points, 1, 2)
            # The JSON plotted_pixels are assumed to be a list of [x, y] coordinates
            pts_array = np.array(plotted_pixels, np.int32).reshape((-1, 1, 2))
            
            # Draw a continuous white line (1.0) on the black mask
            cv2.polylines(lead_mask, [pts_array], isClosed=False, color=1.0, thickness=3)
            
            masks[lead_name] = lead_mask

        return masks
