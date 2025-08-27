from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2, numpy as np
from skimage.morphology import skeletonize as skel
from src.utils.io import ensure_dir, write_json, read_json
from src.utils.logging import setup_logging
import math

BBox = Tuple[int,int,int,int]

def _binarize(img: np.ndarray, cfg) -> np.ndarray:
    bs = int(cfg.geometry.binarize.blocksize)
    C  = int(cfg.geometry.binarize.C)
    if bs % 2 == 0:
        bs += 1
    thr = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bs, C
    )
    thr = cv2.medianBlur(thr, 3)
    return thr

def _skeletonize(thr: np.ndarray, do_skel: bool) -> np.ndarray:
    if not do_skel:
        return thr
    bw = (thr > 0).astype(np.uint8)
    sk = skel(bw > 0)
    return (sk * 255).astype(np.uint8)

def _hough_segments(sk: np.ndarray, cfg) -> List[Tuple[int,int,int,int]]:
    lines = cv2.HoughLinesP(
        sk,
        rho=1,
        theta=np.pi/180,
        threshold=int(cfg.geometry.hough.threshold),
        minLineLength=int(cfg.geometry.hough.min_line_length),
        maxLineGap=int(cfg.geometry.hough.max_line_gap),
    )
    segs=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            segs.append((int(x1),int(y1),int(x2),int(y2)))
    return segs

def _merge_colinear(segs: List[Tuple[int,int,int,int]], cfg):
    if not segs:
        return []
    angle_eps = math.radians(float(cfg.geometry.merge_lines.angle_deg_eps))
    dist_eps  = float(cfg.geometry.merge_lines.endpoint_px_eps)

    def angle(s):
        x1,y1,x2,y2 = s
        return math.atan2(y2-y1, x2-x1)

    used = [False]*len(segs)
    polys = []
    for i,s in enumerate(segs):
        if used[i]: continue
        ax = angle(s)
        x1,y1,x2,y2 = s
        pts=[(x1,y1),(x2,y2)]
        used[i]=True
        changed=True
        while changed:
            changed=False
            for j,t in enumerate(segs):
                if used[j]: continue
                bx = angle(t)
                # angle close?
                da = abs(math.atan2(math.sin(ax-bx), math.cos(ax-bx)))
                if da < angle_eps:
                    # share an endpoint (within dist_eps)?
                    for a in (pts[0], pts[-1]):
                        for b in ((t[0],t[1]), (t[2],t[3])):
                            if max(abs(a[0]-b[0]), abs(a[1]-b[1])) <= dist_eps:
                                pts.append(b); used[j]=True; changed=True
                                break
                        if used[j]: break
        # compress to longest pair
        xs = np.array(pts, dtype=np.int32)
        d = ((xs[:,None,:]-xs[None,:,:])**2).sum(-1)
        i1,i2 = np.unravel_index(np.argmax(d), d.shape)
        polys.append({"polyline":[(int(xs[i1,0]),int(xs[i1,1])), (int(xs[i2,0]),int(xs[i2,1]))]})
    return polys

def _endpoints_from_polys(polys):
    pts=[]
    for p in polys:
        (x1,y1),(x2,y2) = p["polyline"]
        pts.append((x1,y1)); pts.append((x2,y2))
    # unique
    return list({(int(x),int(y)) for (x,y) in pts})

def extract_wires_for_pdf(cfg, pdf_stem: str):
    log = setup_logging(cfg.logging.level)
    mani_path = Path(cfg.paths.raw)/"manifests"/f"{pdf_stem}.json"
    assert mani_path.exists(), f"manifest not found: {mani_path}"
    mani = read_json(mani_path)

    out_root = Path(cfg.paths.processed)/"wires"/pdf_stem
    ensure_dir(out_root)

    for pg in mani["pages"]:
        png = Path(pg["png"])
        assert png.exists(), f"png not found: {png}"
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        thr = _binarize(img, cfg)
        sk  = _skeletonize(thr, bool(cfg.geometry.skeletonize))
        segs = _hough_segments(sk, cfg)
        polys = _merge_colinear(segs, cfg)
        endpoints = _endpoints_from_polys(polys)

        data = {
            "png": str(png),
            "page": int(pg["page"]),
            "n_segments_raw": len(segs),
            "n_polylines": len(polys),
            "polylines": polys,          # [{polyline:[(x1,y1),(x2,y2)]}]
            "endpoints": endpoints       # [(x,y)]
        }
        out_path = out_root / f"page-{pg['page']}.json"
        write_json(data, out_path)
        log.info(f"[wires] {pdf_stem} page-{pg['page']}: segs={len(segs)} polys={len(polys)} → {out_path}")
