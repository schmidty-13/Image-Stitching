import cv2
import numpy as np
import exifread
from PIL import Image
from pyproj import Transformer


def extract_gps(image_path):
    """Extract (lat, lon) from EXIF metadata."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
    try:
        lat_ref = str(tags["GPS GPSLatitudeRef"])
        lon_ref = str(tags["GPS GPSLongitudeRef"])
        lat = tags["GPS GPSLatitude"].values
        lon = tags["GPS GPSLongitude"].values

        # Convert from degrees/minutes/seconds to decimal
        def dms_to_dd(dms, ref):
            d = float(dms[0].num) / float(dms[0].den)
            m = float(dms[1].num) / float(dms[1].den)
            s = float(dms[2].num) / float(dms[2].den)
            dd = d + m / 60.0 + s / 3600.0
            if ref in ["S", "W"]:
                dd *= -1
            return dd

        return dms_to_dd(lat, lat_ref), dms_to_dd(lon, lon_ref)
    except Exception:
        return None, None


def gps_to_xy(lat, lon, epsg_out="EPSG:32617"):  # UTM Zone 17N for Columbus, OH
    """Convert lat/lon to projected (x, y) coordinates (e.g. UTM)."""
    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def gps_assisted_stitch(image_paths, canvas_scale=0.1):
    """
    GPS-assisted image stitching.
    - image_paths: list of file paths
    - canvas_scale: meters per pixel (rough scale)
    """
    gps_coords = []
    imgs = []

    # Load images + GPS
    for path in image_paths:
        img = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2BGR)
        lat, lon = extract_gps(path)
        if lat is None or lon is None:
            raise ValueError(f"No GPS data found in {path}")
        x, y = gps_to_xy(lat, lon)
        gps_coords.append((x, y))
        imgs.append(img)

    # Normalize GPS coords to start at (0,0)
    min_x = min([c[0] for c in gps_coords])
    min_y = min([c[1] for c in gps_coords])
    gps_coords = [(x - min_x, y - min_y) for (x, y) in gps_coords]

    # Estimate canvas size
    max_x = max([c[0] for c in gps_coords])
    max_y = max([c[1] for c in gps_coords])
    canvas_w = int(max_x / canvas_scale) + 2000
    canvas_h = int(max_y / canvas_scale) + 2000
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images on canvas based on GPS
    for (img, (x, y)) in zip(imgs, gps_coords):
        h, w = img.shape[:2]
        cx = int(x / canvas_scale)
        cy = int(y / canvas_scale)

        # Rough placement (centered)
        x0 = cx - w // 2
        y0 = cy - h // 2
        x1, y1 = x0 + w, y0 + h

        # Place on canvas (simple overwrite)
        canvas[y0:y1, x0:x1] = img

    # Refine stitching with OpenCV (optional)
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        return canvas  # fallback to GPS placement


# --- Example usage ---
if __name__ == "__main__":
    images = ["drone1.jpg", "drone2.jpg", "drone3.jpg"]
    result = gps_assisted_stitch(images)
    cv2.imwrite("stitched_output.jpg", result)
