import numpy as np
import cv2

def binarize_probability_map(prob_map):
    """
    Phase 4: Stage 3 - 1D Time-Series Extraction
    Apply adaptive Otsu thresholding to the raw probability output 
    to create a crisp, binary black-and-white mask.
    """
    # Scale probabilities to uint8 for Otsu
    img_uint8 = (prob_map * 255).astype(np.uint8)
    
    # Simple Otsu threshold
    ret, thresh = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 255 denotes trace, 0 denotes background
    return thresh

def setup_cost_matrix(binary_mask):
    """
    Treat the binary mask as a grid. Assign low costs to white pixels
    (predicted trace) and extremely high costs to black pixels.
    """
    # Low cost (e.g. 1) for white pixel, High Cost (100) for black
    cost_matrix = np.where(binary_mask > 127, 1.0, 100.0)
    return cost_matrix

def extract_1d_signal_viterbi(cost_matrix):
    """
    Implement the Viterbi algorithm (dynamic programming) to find the path of least resistance
    from the leftmost column to the rightmost column. Output exactly one Y-coordinate
    for every X-coordinate (time step).
    """
    h, w = cost_matrix.shape
    
    # DP table to store minimum costs
    dp = np.zeros_like(cost_matrix, dtype=np.float32)
    # Pointer table for backtracking
    ptr = np.zeros_like(cost_matrix, dtype=np.int32)
    
    dp[:, 0] = cost_matrix[:, 0]
    
    # Forward pass: left to right
    for x in range(1, w):
        for y in range(h):
            # Window for y: allowing jumps 
            # In skeleton, a reasonable neighbor window: [-1, 0, +1]
            # to enforce continuity. For 1D extraction, the trace could jump more sharply
            # so window might be larger depending on sampling rate. Let's assume w=[-5, 5]
            
            y_min = max(0, y - 5)
            y_max = min(h, y + 6)
            
            prev_costs = dp[y_min:y_max, x-1]
            min_idx = np.argmin(prev_costs)
            min_cost = prev_costs[min_idx]
            
            dp[y, x] = min_cost + cost_matrix[y, x]
            ptr[y, x] = y_min + min_idx  # Keep track of actual Y coordinate in prev column
            
    # Backtracking: right to left
    extracted_signal = np.zeros(w, dtype=np.int32)
    
    # Find argmin in the last column
    best_y = np.argmin(dp[:, w-1])
    extracted_signal[w-1] = best_y
    
    curr_y = best_y
    for x in range(w-1, 0, -1):
        curr_y = ptr[curr_y, x]
        extracted_signal[x-1] = curr_y
        
    return extracted_signal

