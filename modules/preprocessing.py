import cv2
import numpy as np
from PIL import Image


# =========================================================
# IMAGE CONVERSION
# =========================================================

def pil_to_rgb(image):
    return np.array(image.convert("RGB"))


def pil_to_gray(image):
    rgb = pil_to_rgb(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def array_to_pil(array):
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


# =========================================================
# FILTERING
# =========================================================

def mean_filter(image, kernel_size=5):

    rgb = pil_to_rgb(image)

    result = cv2.blur(
        rgb,
        (kernel_size, kernel_size)
    )

    return array_to_pil(result)


def median_filter(image, kernel_size=5):

    if kernel_size % 2 == 0:
        kernel_size += 1

    rgb = pil_to_rgb(image)

    result = cv2.medianBlur(
        rgb,
        kernel_size
    )

    return array_to_pil(result)


def gaussian_filter(image, kernel_size=5, sigma=0):

    if kernel_size % 2 == 0:
        kernel_size += 1

    rgb = pil_to_rgb(image)

    result = cv2.GaussianBlur(
        rgb,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )

    return array_to_pil(result)


# =========================================================
# OTSU MASKING
# =========================================================

def otsu_mask(image):

    rgb = pil_to_rgb(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Smooth image
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu threshold
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Border intensity menentukan foreground
    border = np.concatenate([
        binary[0, :],
        binary[-1, :],
        binary[:, 0],
        binary[:, -1]
    ])

    # Background biasanya dominan pada border
    border_mean = np.mean(border)

    if border_mean > 127:
        mask = cv2.bitwise_not(binary)
    else:
        mask = binary.copy()

    # =====================================================
    # MORPHOLOGY
    # =====================================================

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    # Menghilangkan noise kecil
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # Menyambungkan area yang terputus
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2
    )

    # =====================================================
    # LARGEST CONTOUR
    # =====================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(mask)

    if contours:

        # Kontur terbesar
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            final_mask,
            [largest_contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED
        )

    return Image.fromarray(final_mask)


# =========================================================
# SEGMENTATION
# =========================================================

def apply_segmentation(image, mask=None):

    rgb = pil_to_rgb(image)

    if mask is None:
        mask = otsu_mask(image)

    mask_array = np.array(
        mask.convert("L")
    )

    # Pastikan mask binary
    _, mask_array = cv2.threshold(
        mask_array,
        127,
        255,
        cv2.THRESH_BINARY
    )

    segmented = cv2.bitwise_and(
        rgb,
        rgb,
        mask=mask_array
    )

    return array_to_pil(segmented)


# =========================================================
# ENHANCEMENT
# =========================================================

def enhance_image(image, clip_limit=2.0, tile_grid_size=8):

    rgb = pil_to_rgb(image)

    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit if False else clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    return array_to_pil(enhanced_rgb)


# =========================================================
# HAIR REMOVAL
# =========================================================

def hair_removal(image):

    rgb = pil_to_rgb(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Deteksi rambut gelap
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Dilatasi mask rambut
    hair_mask = cv2.dilate(
        hair_mask,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Inpainting
    result = cv2.inpaint(
        rgb,
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )

    return array_to_pil(result)


# =========================================================
# GRAYSCALE
# =========================================================

def grayscale_image(image):

    gray = pil_to_gray(image)

    gray_rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )

    return array_to_pil(gray_rgb)


# =========================================================
# PROCESSING ROUTER
# =========================================================

def process_image(image, method):

    method = method.lower()

    if method == "mean":
        return mean_filter(image)

    elif method == "median":
        return median_filter(image)

    elif method == "gaussian":
        return gaussian_filter(image)

    elif method == "enhancement":
        return enhance_image(image)

    elif method == "hair removal":
        return hair_removal(image)

    elif method == "grayscale":
        return grayscale_image(image)

    else:
        return image
