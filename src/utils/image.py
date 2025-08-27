# raster ops (deskew/binarize)
# src/utils/image.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps, ImageFilter

def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def open_image(path: str | Path) -> Image.Image:
    return ensure_rgb(Image.open(path))

def to_grayscale(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)

def binarize(img: Image.Image, threshold: int = 200) -> Image.Image:
    gray = to_grayscale(img)
    return gray.point(lambda p: 255 if p > threshold else 0).convert("1")

def deskew_hint(img: Image.Image) -> float:
    # quick-and-dirty; return 0 for now (placeholder for future Hough-based skew)
    return 0.0

def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1: return img
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
