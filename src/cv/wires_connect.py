# src/cv/wires_connect.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

def connect_segments(segments: List[Tuple[Tuple[int,int],Tuple[int,int]]],
                     join_tol: int = 6) -> List[List[Tuple[int,int]]]:
    """
    Merge collinear/nearby segments into polylines (very simple).
    """
    # naive chaining
    segs = [list(s) for s in segments]
    polylines = []
    while segs:
        a = segs.pop()
        poly = [tuple(a[0]), tuple(a[1])]
        changed = True
        while changed:
            changed = False
            for i, s in list(enumerate(segs)):
                x1,y1 = s[0]; x2,y2 = s[1]
                if np.hypot(poly[-1][0]-x1, poly[-1][1]-y1) <= join_tol:
                    poly.append((x2,y2)); segs.pop(i); changed=True; break
                if np.hypot(poly[-1][0]-x2, poly[-1][1]-y2) <= join_tol:
                    poly.append((x1,y1)); segs.pop(i); changed=True; break
        polylines.append(poly)
    return polylines
