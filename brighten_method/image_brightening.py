import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def brighten_exposure(image_path, exposure_compensation=0.5, save_path=None):
    """
    Brighten image by simulating exposure compensation
    exposure_compensation: positive values brighten, negative darken
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Apply exposure compensation (similar to camera exposure)
    brightened = img_float * (2 ** exposure_compensation)
    
    # Clip values to prevent overflow
    brightened = np.clip(brightened, 0, 1)
    
    # Convert back to uint8
    result = (brightened * 255).astype(np.uint8)

    if save_path is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    else:
        # Save with method name if save_path not provided
        method_save_path = f"{image_path.rsplit('.',1)[0]}_exposure.jpg"
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(method_save_path, result_bgr)

    return result

def brighten_gamma_correction(image_path, gamma=0.7, save_path=None):
    """
    Brighten using gamma correction (natural looking)
    gamma < 1 brightens, gamma > 1 darkens
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    result = cv2.LUT(img, table)
    if save_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    return result

def brighten_shadow_highlight(image_path, shadow_lift=30, highlight_reduction=10, save_path=None):
    """
    Selectively brighten shadows while protecting highlights
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Create masks for shadows and highlights
    luminance = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
    
    # Shadow mask (dark areas)
    shadow_mask = 1 - luminance
    shadow_mask = np.power(shadow_mask, 2)  # Stronger effect on darker areas
    
    # Highlight mask (bright areas)
    highlight_mask = luminance
    highlight_mask = np.power(highlight_mask, 2)  # Stronger effect on brighter areas
    
    # Apply adjustments
    for i in range(3):  # RGB channels
        # Lift shadows
        img_float[:,:,i] += (shadow_mask * shadow_lift / 255.0)
        # Reduce highlights
        img_float[:,:,i] -= (highlight_mask * highlight_reduction / 255.0)
    
    # Clip and convert back
    result = np.clip(img_float, 0, 1)
    result = (result * 255).astype(np.uint8)
    if save_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    return result

def brighten_curves(image_path, midpoint_lift=0.15, save_path=None):
    """
    Brighten using curve adjustment (lifts midtones)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create curve lookup table
    def create_curve(midpoint_lift):
        x = np.arange(256)
        # S-curve that lifts midtones
        curve = x + midpoint_lift * 255 * np.sin(np.pi * x / 255)
        curve = np.clip(curve, 0, 255)
        return curve.astype(np.uint8)
    
    curve_table = create_curve(midpoint_lift)
    result = cv2.LUT(img, curve_table)
    if save_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    return result

def brighten_pil_natural(image_path, brightness_factor=1.3, contrast_factor=1.1, save_path=None):
    """
    Natural brightening using PIL with brightness and slight contrast boost
    """
    img = Image.open(image_path)
    
    # Enhance brightness
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(brightness_factor)
    
    # Slight contrast boost to maintain natural look
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(contrast_factor)
    
    result = np.array(img)
    if save_path:
        # Convert to BGR for saving with OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    return result

def adaptive_brighten(image_path, clip_limit=2.0, save_path=None):
    """
    Adaptive histogram equalization for natural local brightening
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if save_path:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
    return result

# Example usage and comparison
def compare_brightening_methods(image_path):
    """
    Compare different brightening methods and save results
    """
    # Load original
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Prepare save paths
    base = image_path.rsplit('.', 1)[0]
    save_paths = {
        'Exposure': f"{base}_exposure.jpg",
        'Gamma': f"{base}_gamma.jpg",
        'Shadow/Highlight': f"{base}_shadow_highlight.jpg",
        'Curves': f"{base}_curves.jpg",
        'PIL Natural': f"{base}_pil_natural.jpg",
        'Adaptive': f"{base}_adaptive.jpg"
    }
    
    # Apply different methods and save results
    methods = {
        'Original': original,
        'Exposure': brighten_exposure(image_path, 0.6, save_paths['Exposure']),
        'Gamma': brighten_gamma_correction(image_path, 0.8, save_paths['Gamma']),
        'Shadow/Highlight': brighten_shadow_highlight(image_path, 40, 5, save_paths['Shadow/Highlight']),
        'Curves': brighten_curves(image_path, 0.12, save_paths['Curves']),
        'PIL Natural': brighten_pil_natural(image_path, 1.4, 1.05, save_paths['PIL Natural']),
        'Adaptive': adaptive_brighten(image_path, 2.5, save_paths['Adaptive'])
    }
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, (name, img) in enumerate(methods.items()):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(name)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Simple function for quick brightening
def quick_brighten(image_path, method='gamma', save_path=None):
    """
    Quick brightening with best default settings
    """
    if method == 'gamma':
        result = brighten_gamma_correction(image_path, 0.8, save_path)
    elif method == 'exposure':
        result = brighten_exposure(image_path, 0.5, save_path)
    elif method == 'shadow_highlight':
        result = brighten_shadow_highlight(image_path, 35, 8, save_path)
    else:
        result = brighten_gamma_correction(image_path, 0.8, save_path)
    return result

# Usage example:
if __name__ == "__main__":
    # Replace with your image path
    image_path = "000000_scene6.png"
    
    # Quick brighten (recommended)
    # brightened = quick_brighten(image_path, method='gamma')
    
    # Or compare all methods
    compare_brightening_methods(image_path)
    
    # Save result
    # quick_brighten(image_path, method='shadow_highlight', save_path='brightened_image.jpg')
    # exposure
    brighten_exposure(image_path, exposure_compensation=0.5)
    # gamma
    brighten_gamma_correction(image_path, gamma=0.7)    
    # shadow/highlight
    brighten_shadow_highlight(image_path, shadow_lift=30, highlight_reduction=10)
    # curves
    brighten_curves(image_path, midpoint_lift=0.15)
    # PIL natural
    brighten_pil_natural(image_path, brightness_factor=1.3, contrast_factor=1.1)
    # adaptive
    adaptive_brighten(image_path, clip_limit=2.0)
    