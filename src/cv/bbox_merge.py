# src/cv/bbox_merge.py
from __future__ import annotations
from typing import List, Tuple

def iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter == 0: return 0.0
    aarea = (ax2-ax1)*(ay2-ay1)
    barea = (bx2-bx1)*(by2-by1)
    return inter / float(aarea + barea - inter)

def merge_overlaps(bboxes: List[Tuple[float,float,float,float]], iou_thr=0.5):
    """
    Greedy merge boxes with IoU ≥ iou_thr.
    """
    boxes = list(bboxes)
    out = []
    while boxes:
        b = boxes.pop()
        merged = True
        while merged:
            merged = False
            for i, c in list(enumerate(boxes)):
                if iou(b,c) >= iou_thr:
                    b = (min(b[0],c[0]), min(b[1],c[1]), max(b[2],c[2]), max(b[3],c[3]))
                    boxes.pop(i); merged=True; break
        out.append(b)
    return out
