import os
import cv2
import numpy as np
from tqdm import tqdm

# Configuration Parameters
IMAGE_DIR = r"Datasets/RS-AOD-YOLO/images/train"
LABEL_DIR = r"RS-AOD-YOLO/labels/train"

OUTPUT_DIR = r"../apsm_amplitude"  # Output directory for saving amplitude spectrum
AMPLITUDE_NAME = "plane_priori_amplitude_normalized.npy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SIZE = (640, 640)
GAUSSIAN_NOISE_MEAN = 0
GAUSSIAN_NOISE_SIGMA = 15


def get_pure_target_from_yolo(img_path, label_path, target_size):
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None: return None, None
    img_h, img_w = original_img.shape
    pure_target_img = np.zeros_like(original_img)
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line: return None, None  # Skip empty label files
            _, x_center_norm, y_center_norm, w_norm, h_norm = map(float, line.split())
            w, h = int(w_norm * img_w), int(h_norm * img_h)
            x, y = int((x_center_norm * img_w) - w / 2), int((y_center_norm * img_h) - h / 2)
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(img_w, x + w), min(img_h, y + h)
            pure_target_img[y_min:y_max, x_min:x_max] = original_img[y_min:y_max, x_min:x_max]
    except (IOError, ValueError) as e:
        print(f"Failed to read label {label_path}: {e}")
        return None, None
    original_img_resized = cv2.resize(original_img, target_size)
    pure_target_img_resized = cv2.resize(pure_target_img, target_size)
    return original_img_resized, pure_target_img_resized

def reconstruct_image(amplitude_spectrum, phase_spectrum):
    complex_spectrum = amplitude_spectrum * np.exp(1j * phase_spectrum)
    reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(complex_spectrum)))
    return reconstructed


def main():
    print("Generating generic amplitude spectrum from clean data")

    cwa_sum = np.zeros(TARGET_SIZE, dtype=np.float64)
    processed_count = 0

    image_files = sorted(os.listdir(IMAGE_DIR))
    test_sample = None

    for img_filename in tqdm(image_files, desc="Processing images"):
        img_name, _ = os.path.splitext(img_filename)
        img_path = os.path.join(IMAGE_DIR, img_filename)
        label_path = os.path.join(LABEL_DIR, img_name + '.txt')
        if not os.path.exists(label_path):
            continue

        g, t = get_pure_target_from_yolo(img_path, label_path, TARGET_SIZE)
        if g is None:
            continue

        G = np.fft.fftshift(np.fft.fft2(g))
        T = np.fft.fftshift(np.fft.fft2(t))

        mag_T, phase_T = np.abs(T), np.angle(T)
        phase_G = np.angle(G)

        C_i = mag_T * np.cos(phase_T - phase_G)

        cwa_sum += C_i
        processed_count += 1

        if test_sample is None:
            test_sample = {'original_g': g, 'pure_target_t': t}

    if processed_count == 0:
        print("Error: No images were successfully processed. Please check image and label paths.")
        return

    print(f"\nSuccessfully processed {processed_count} images.")

    print("\nCalculating final amplitude spectrum...")
    cwa_mean = cwa_sum / processed_count

    A_coherent = cwa_mean
    print("Amplitude spectrum generated!")

    if test_sample is None:
        print("Error: Failed to generate test sample, cannot proceed with reconstruction and evaluation.")
        return

    print("\nPerforming reconstruction")
    g_clean = test_sample['original_g']
    phase_G_clean = np.angle(np.fft.fftshift(np.fft.fft2(g_clean)))

    g_recon_coherent_clean = reconstruct_image(A_coherent, phase_G_clean)

    # Save normalized coherent amplitude spectrum
    print("\n" + "=" * 50)
    print("Saving normalized amplitude spectrum")
    print("=" * 50)

    max_pixel_value = np.max(g_recon_coherent_clean)
    print(f"Maximum pixel value of reconstruction for calibration: {max_pixel_value:.4f}")

    if max_pixel_value > 1e-6:
        scaling_factor = 1.0 / max_pixel_value
        A_coherent_normalized = A_coherent * scaling_factor
        print(f"Calculated scaling factor: {scaling_factor:.6f}")

        save_path = os.path.join(OUTPUT_DIR, AMPLITUDE_NAME)
        np.save(save_path, A_coherent_normalized)
        print(f"Normalized coherent amplitude spectrum saved to: {save_path}")

        print("\nLoading and Verification")
        loaded_spectrum = np.load(save_path)
        g_recon_from_loaded = reconstruct_image(loaded_spectrum, phase_G_clean)
        min_val, max_val = np.min(g_recon_from_loaded), np.max(g_recon_from_loaded)
        print("Pixel value range of reconstructed image using loaded spectrum:")
        print(f"  Minimum: {min_val:.6f}")
        print(f"  Maximum: {max_val:.6f}")
        if 0.95 < max_val < 1.05:
            print("Verification successful! Maximum value of reconstructed image is very close to 1.0.")
        else:
            print("Verification warning: Maximum value of reconstructed image deviates significantly from 1.0.")
    else:
        print("Calibration failed: Reconstructed image pixel values are too low to calculate valid scaling factor.")
    print("=" * 50)


if __name__ == '__main__':
    main()