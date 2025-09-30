import os
import cv2
import math
import numpy as np
import exifread
from pyproj import Transformer

# ------------------------
# Tunables
# ------------------------
GSD_M_PER_PX = 0.03          # meters per pixel (assumed constant per your original)
MIN_GOOD_MATCHES = 20        # need at least this many descriptor matches after Lowe ratio
RANSAC_REPROJ_THRESH = 3.0   # px; robustness of affine fit
MAX_NEIGHBOR_DIST_M = 100.0   # only try keypoint refine if GPS-predicted centers within this many meters
ORB_NFEATURES = 4000         # ORB feature cap
LOWE_RATIO = 0.75            # Lowe ratio test threshold

# ------------------------
# EXIF helpers
# ------------------------
def _to_deg(val):
    parts = val.values
    def f(r): return r.num / r.den
    return f(parts[0]) + f(parts[1]) / 60.0 + f(parts[2]) / 3600.0

def read_latlon(path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        lat = _to_deg(tags['GPS GPSLatitude'])
        lon = _to_deg(tags['GPS GPSLongitude'])
        lat_ref = str(tags['GPS GPSLatitudeRef'])
        lon_ref = str(tags['GPS GPSLongitudeRef'])
        # Correct sign logic: N/E positive, S/W negative
        if lat_ref == 'N': lat = -lat
        if lon_ref == 'E': lon = -lon
        return lat, lon
    except KeyError:
        return None, None

# Optionally use image heading if present (not required)
def read_img_direction(path):
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        val = tags.get('GPS ImgDirection')
        if val is not None:
            num = val.values[0].num
            den = val.values[0].den
            return (num / den) * math.pi / 180.0  # radians
    except Exception:
        pass
    return None

# ------------------------
# Geodesy helpers
# ------------------------
def make_transformer(lat0, lon0):
    # Local Azimuthal Equidistant projection centered at the anchor
    return Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs",
        always_xy=True
    )

def ll_to_xy(T, lat, lon):
    x, y = T.transform(lon, lat)  # (lon, lat) because always_xy=True
    return np.array([x, y], dtype=np.float32)

# ------------------------
# Warps
# ------------------------
def gps_affine_to_canvas(center_xy_m, ref_xy_m, gsd_m_per_px, img_w, img_h):
    """
    Build a 2x3 affine that places an image so that its CENTER is at the gps-projected
    (relative) location. Only translation (no rotation/scale).
    Canvas Y grows downward; meters +y (north) maps to +pixels Y.
    """
    delta_m = center_xy_m - ref_xy_m
    tx = -delta_m[0] / gsd_m_per_px
    ty =  delta_m[1] / gsd_m_per_px
    # move image center to (tx, ty)
    return np.array([[1, 0, tx - img_w / 2.0],
                     [0, 1, ty - img_h / 2.0]], dtype=np.float32)

def compose_affine(A, B):
    """
    Compose two 2x3 affines: result = A @ B
    (treat as 3x3 with last row [0 0 1]).
    """
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    C3 = A3 @ B3
    return C3[:2, :]

def corners_after_affine(w, h, A):
    pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1,1,2)
    warped = cv2.transform(pts, A).reshape(-1, 2)
    return warped

# ------------------------
# Keypoint refine (ORB + RANSAC affine)
# ------------------------
def estimate_affine_from_matches(img_src, img_dst):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    k1, d1 = orb.detectAndCompute(img_src, None)
    k2, d2 = orb.detectAndCompute(img_dst, None)
    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in knn:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None, len(good)

    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    A, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
        maxIters=2000,
        refineIters=50,
        confidence=0.99
    )
    inliers_count = int(inliers.sum()) if inliers is not None else 0
    if A is None or inliers_count < MIN_GOOD_MATCHES:
        return None, len(good)
    return A.astype(np.float32), inliers_count

# ------------------------
# Main mosaic
# ------------------------
def mosaic_gps_keypoints(folder="images/", gsd_m_per_px=GSD_M_PER_PX, output="mosaic_gps_kp.jpg"):
    # 1) Load images & EXIF
    items = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        path = os.path.join(folder, fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: 
            continue
        lat, lon = read_latlon(path)
        items.append({
            "name": fn,
            "path": path,
            "img": img,
            "lat": lat,
            "lon": lon
        })

    if not items:
        raise RuntimeError("No images found.")

    # 2) Anchor & transformer
    anchor = next(((it["lat"], it["lon"]) for it in items if it["lat"] is not None), None)
    if anchor is None:
        raise RuntimeError("No EXIF GPS found; cannot seed GPS placement.")
    lat0, lon0 = anchor
    T = make_transformer(lat0, lon0)

    # 3) Project GPS to local XY (meters)
    for it in items:
        it["xy_m"] = None if it["lat"] is None else ll_to_xy(T, it["lat"], it["lon"])

    # Reference (first with GPS)
    ref_xy = next((it["xy_m"] for it in items if it["xy_m"] is not None), np.array([0,0], np.float32))

    # 4) Predict GPS placement (affine to a pre-shift canvas frame)
    for it in items:
        h, w = it["img"].shape[:2]
        if it["xy_m"] is None:
            it["A_gps"] = None
        else:
            it["A_gps"] = gps_affine_to_canvas(it["xy_m"], ref_xy, gsd_m_per_px, w, h)

    # 5) Determine canvas bounds using GPS-only placement
    all_corners = []
    for it in items:
        if it["A_gps"] is None:
            continue
        h, w = it["img"].shape[:2]
        cw = corners_after_affine(w, h, it["A_gps"])
        all_corners.append(cw)
    if not all_corners:
        raise RuntimeError("No GPS-capable images to bound the canvas.")
    corners = np.vstack(all_corners)
    min_x, min_y = np.floor(corners.min(0)).astype(int)
    max_x, max_y = np.ceil(corners.max(0)).astype(int)

    # Shift so everything is >= 0
    shift = np.array([[1, 0, -min_x],
                      [0, 1, -min_y]], dtype=np.float32)
    out_w, out_h = int(max_x - min_x), int(max_y - min_y)

    # 6) Prepare blending accumulators
    pano = np.zeros((out_h, out_w, 3), np.float32)
    acc  = np.zeros((out_h, out_w), np.float32)

    # Track placed images with their final canvas warp
    placed = []

    # Utility to blend an image with a 2x3 affine into the pano
    def blend_into(img, A_canvas):
        h, w = img.shape[:2]
        warped = cv2.warpAffine(img, A_canvas, (out_w, out_h))
        m = (warped.sum(axis=2) > 0).astype(np.float32)
        # incremental feathered averaging
        nonlocal pano, acc
        pano = (pano * acc[..., None] + warped.astype(np.float32) * m[..., None]) / np.clip(acc[..., None] + m[..., None], 1e-6, None)
        acc += m

    # 7) Place images
    # Strategy:
    #   - seed with the first GPS image (GPS-only)
    #   - for each next image (prefer those near placed ones), try keypoint refine against nearest neighbor
    #   - if refine fails, fall back to GPS-only affine

    # Seed
    seed_idx = next(i for i, it in enumerate(items) if it["A_gps"] is not None)
    it0 = items[seed_idx]
    A0_canvas = compose_affine(shift, it0["A_gps"])
    blend_into(it0["img"], A0_canvas)
    it0["A_canvas"] = A0_canvas
    placed.append(seed_idx)

    # Remaining indices
    remaining = [i for i in range(len(items)) if i != seed_idx]

    while remaining:
        # Choose next: the one whose GPS center is nearest to any placed image
        best_i, best_dist = None, float('inf')
        for i in remaining:
            if items[i]["xy_m"] is None:
                continue
            for j in placed:
                if items[j]["xy_m"] is None:
                    continue
                d = np.linalg.norm(items[i]["xy_m"] - items[j]["xy_m"])
                if d < best_dist:
                    best_dist, best_i = d, i

        # If no GPS among remaining, just take the next by file order and GPS-skip
        if best_i is None:
            best_i = remaining[0]

        it = items[best_i]
        h, w = it["img"].shape[:2]

        # Default warp is GPS-only if available
        A_final_canvas = None
        used_refine = False

        if it.get("A_gps") is not None:
            A_gps_canvas = compose_affine(shift, it["A_gps"])

            # Try keypoint refinement if a close neighbor exists
            neighbor_j = None
            if best_dist <= MAX_NEIGHBOR_DIST_M and items[best_i]["xy_m"] is not None:
                neighbor_j = min(
                    placed,
                    key=lambda j: np.linalg.norm(items[best_i]["xy_m"] - items[j]["xy_m"]) if items[j]["xy_m"] is not None else float('inf')
                )
                # Make sure neighbor has a valid warp
                if items[neighbor_j].get("A_canvas") is None:
                    neighbor_j = None

            if neighbor_j is not None:
                # Estimate affine between raw images: A_ij maps current -> neighbor
                A_ij, inl = estimate_affine_from_matches(it["img"], items[neighbor_j]["img"])
                if A_ij is not None:
                    # Compose to canvas: W_i = W_j @ A_ij
                    A_neighbor_canvas = items[neighbor_j]["A_canvas"]
                    A_candidate_canvas = compose_affine(A_neighbor_canvas, A_ij)

                    # Simple sanity: check the candidate center isn't crazy far from GPS center
                    # Measure predicted center from both warps
                    center = np.array([[w/2.0, h/2.0]], np.float32).reshape(-1,1,2)
                    c_gps = cv2.transform(center, A_gps_canvas).reshape(-1,2)[0]
                    c_kp  = cv2.transform(center, A_candidate_canvas).reshape(-1,2)[0]
                    if np.linalg.norm(c_kp - c_gps) < (100.0 / gsd_m_per_px):  # e.g., within 100 m
                        A_final_canvas = A_candidate_canvas
                        used_refine = True

            if A_final_canvas is None:
                A_final_canvas = A_gps_canvas
        else:
            # No GPS → if any neighbor exists, we can still try pure keypoints to a placed neighbor
            neighbor_j = placed[-1]  # last placed as a fallback
            A_ij, inl = estimate_affine_from_matches(it["img"], items[neighbor_j]["img"])
            if A_ij is not None:
                A_neighbor_canvas = items[neighbor_j]["A_canvas"]
                A_final_canvas = compose_affine(A_neighbor_canvas, A_ij)
                used_refine = True
            else:
                # Give up on this image; skip
                remaining.remove(best_i)
                continue

        # Blend
        blend_into(it["img"], A_final_canvas)
        it["A_canvas"] = A_final_canvas
        placed.append(best_i)
        remaining.remove(best_i)

        # (Optional) print progress
        print(f"Placed {it['name']}  | refine={'yes' if used_refine else 'no'}  | canvas set.")

    # 8) Save
    out = np.clip(pano, 0, 255).astype(np.uint8)
    cv2.imwrite(output, out)
    print(f"Saved {output} ({out_w}x{out_h}), GSD={gsd_m_per_px} m/px")

if __name__ == "__main__":
    mosaic_gps_keypoints(folder="images/", gsd_m_per_px=GSD_M_PER_PX, output="mosaic_gps_kp.jpg")

