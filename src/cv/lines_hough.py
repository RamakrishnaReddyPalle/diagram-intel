# src/cv/lines_hough.py
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def detect_lines(image_path: str,
                 canny1: int = 50, canny2: int = 150,
                 rho: float = 1.0, theta: float = np.pi/180,
                 threshold: int = 120, min_line_len: int = 60,
                 max_line_gap: int = 10) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Basic probabilistic Hough lines. Returns list of ((x1,y1),(x2,y2)).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    edges = cv2.Canny(img, canny1, canny2)
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    out = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            out.append(((int(x1),int(y1)),(int(x2),int(y2))))
    return out
