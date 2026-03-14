import numpy as np
import scipy.interpolate as interp

def calibrate_amplitude(y_pixel_array, pixel_height_per_mm, baseline_y=0):
    """
    Amplitude Calibration (Max 20 pts)
    Scale the Y-axis to millivolts.
    Gain is 10 mm/mV. Divide your Y-values by the pixel equivalent of 10 mm.
    """
    # Inverse Y-axis if needed (image Y goes down, signal Y goes up)
    y_shifted = baseline_y - y_pixel_array
    
    pixels_per_10mm = pixel_height_per_mm * 10.0
    
    # Scale to millivolts
    mz_signal = y_shifted / pixels_per_10mm
    return mz_signal


def calibrate_time_and_resample(signal_1d, pixel_width_per_mm, signal_start_x=0):
    """
    Time Calibration (Max 20 pts) and Resampling
    Paper speed is 25 mm/s. 1 mm = 40 ms = 20 samples (for exactly 500 Hz).
    Interpolate array horizontally so final signal is sampled exactly at 500 Hz.
    Cast the final array to np.float16.
    """
    # For a given array of size W, assume W = total pixels in time.
    width_pixels = len(signal_1d)
    
    total_mm = width_pixels / pixel_width_per_mm
    total_seconds = total_mm / 25.0
    
    # We want exactly 500 samples per second
    target_samples = int(np.round(total_seconds * 500))
    
    if target_samples <= 1:
        return np.array(signal_1d, dtype=np.float16)

    # Create original time stamps and new time stamps
    old_x = np.linspace(0, total_seconds, width_pixels)
    new_x = np.linspace(0, total_seconds, target_samples)
    
    # Interpolate using Scipy
    interpolator = interp.interp1d(old_x, signal_1d, kind='linear', fill_value="extrapolate")
    resampled_signal = interpolator(new_x)
    
    # Cast to float16 to reduce file size as per requirements
    return resampled_signal.astype(np.float16)

def create_submission(predictions_dict, filename="submission.npz"):
    """
    Packaging: Save predictions as a flat, dictionary-like mapping inside a single submission.npz file.
    Keys must strictly follow the {record_name}_{lead_name} convention.
    """
    # ensure float16
    for key, val in predictions_dict.items():
        if val.dtype != np.float16:
            predictions_dict[key] = val.astype(np.float16)
            
    np.savez_compressed(filename, **predictions_dict)
    print(f"Successfully saved {len(predictions_dict)} leads to {filename}")
