# src/cv/skeletonize.py
import cv2
import numpy as np

def zhang_suen_skeleton(binary_img: np.ndarray) -> np.ndarray:
    """Fast thinning using OpenCV ximgproc guided morph (fallback: erode)."""
    if binary_img.dtype != np.uint8:
        binary_img = (binary_img>0).astype(np.uint8)*255
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    img = binary_img.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        done = (cv2.countNonZero(img) == 0)
    return skel
