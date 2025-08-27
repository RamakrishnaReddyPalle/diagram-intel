# src/cv/snap_ports.py
from __future__ import annotations
from typing import List, Tuple

def snap_points_to_bbox_edges(points: List[Tuple[float,float]],
                              bbox: Tuple[float,float,float,float],
                              max_dist: float = 6.0) -> List[Tuple[float,float]]:
    """
    Move points onto the nearest edge of bbox if within max_dist.
    """
    x1,y1,x2,y2 = bbox
    out=[]
    for (x,y) in points:
        candidates = [(x1,y),(x2,y),(x,y1),(x,y2)]
        dists = [abs(x-x1),abs(x-x2),abs(y-y1),abs(y-y2)]
        mi = min(range(4), key=lambda i:dists[i])
        if dists[mi] <= max_dist:
            out.append(candidates[mi])
        else:
            out.append((x,y))
    return out
