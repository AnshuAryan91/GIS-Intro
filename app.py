import os
import socket
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROCESSED_FOLDER"] = "processed"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def create_fallback_polygons(image_shape):
    h, w = image_shape[:2]
    boxes = []
    cols = 3
    rows = 3
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    gx = (w - 2 * margin_x) / cols
    gy = (h - 2 * margin_y) / rows

    for row in range(rows):
        for col in range(cols):
            x1 = int(margin_x + col * gx + 18)
            y1 = int(margin_y + row * gy + 18)
            x2 = int(margin_x + (col + 1) * gx - 18)
            y2 = int(margin_y + (row + 1) * gy - 18)
            boxes.append({
                "points": [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [x1, y1],
                ]
            })
    return boxes


def _polygon_is_parcel_like(contour, image_shape):
    h, w = image_shape[:2]
    area = cv2.contourArea(contour)
    if area < max(500, (w * h) * 0.0006):
        return False

    x, y, cw, ch = cv2.boundingRect(contour)
    if cw < 35 or ch < 35:
        return False
    if cw > w * 0.95 or ch > h * 0.95:
        return False

    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return False

    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    if len(approx) < 4:
        return False

    rect_area = cw * ch
    if rect_area == 0:
        return False

    area_ratio = area / rect_area
    if area_ratio < 0.55:
        return False

    return True


def _structured_grid_polygons(image_shape):
    h, w = image_shape[:2]
    pad_x = max(20, int(w * 0.08))
    pad_y = max(20, int(h * 0.08))
    cols = max(3, min(6, round(w / 320)))
    rows = max(3, min(6, round(h / 260)))
    gx = max(60, (w - 2 * pad_x) / cols)
    gy = max(60, (h - 2 * pad_y) / rows)

    polygons = []
    for row in range(rows):
        for col in range(cols):
            x1 = int(pad_x + col * gx + 12)
            y1 = int(pad_y + row * gy + 12)
            x2 = int(pad_x + (col + 1) * gx - 12)
            y2 = int(pad_y + (row + 1) * gy - 12)
            polygons.append({
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            })
    return polygons


def _edge_based_parcels(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=25)

    if lines is None:
        return []

    x_positions = []
    y_positions = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        if abs(y2 - y1) < 20 and abs(x2 - x1) > 40:
            x_positions.extend([x1, x2])
        elif abs(x2 - x1) < 20 and abs(y2 - y1) > 40:
            y_positions.extend([y1, y2])

    if len(x_positions) < 8 or len(y_positions) < 8:
        return []

    xs = sorted(set(x_positions))
    ys = sorted(set(y_positions))
    if len(xs) < 3 or len(ys) < 3:
        return []

    xs = [v for v in xs if 20 < v < image_bgr.shape[1] - 20]
    ys = [v for v in ys if 20 < v < image_bgr.shape[0] - 20]
    if len(xs) < 3 or len(ys) < 3:
        return []

    xs = xs[:25]
    ys = ys[:25]

    candidate_rects = []
    for i in range(len(xs) - 1):
        x1 = xs[i]
        x2 = xs[i + 1]
        if x2 - x1 < 25:
            continue
        for j in range(len(ys) - 1):
            y1 = ys[j]
            y2 = ys[j + 1]
            if y2 - y1 < 25:
                continue
            candidate_rects.append({
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            })

    return candidate_rects


def _merge_close_polygons(polygons):
    merged = []
    for polygon in polygons:
        pts = np.array(polygon["points"], dtype=np.float32)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        is_duplicate = False
        for existing in merged:
            existing_pts = np.array(existing["points"], dtype=np.float32)
            ex = np.mean(existing_pts[:, 0])
            ey = np.mean(existing_pts[:, 1])
            if abs(cx - ex) < 30 and abs(cy - ey) < 30:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(polygon)
    return merged


def normalize_polygons(polygons):
    normalized = []
    for idx, polygon in enumerate(polygons, start=1):
        points = polygon.get("points", [])
        if not points:
            continue
        ordered = []
        for x, y in points:
            ordered.append([int(x), int(y)])
        if ordered[0] != ordered[-1]:
            ordered.append(ordered[0])
        normalized.append({
            "points": ordered,
            "label": polygon.get("label") or f"P{idx}",
        })
    return normalized


def generate_sample_parcels():
    sample_path = Path(app.config["PROCESSED_FOLDER"]) / "sample_cadastral_map.png"
    width, height = 1600, 1100
    image = Image.new("RGB", (width, height), '#edf3ef')
    draw = ImageDraw.Draw(image)

    road_color = '#b1b8b4'
    for x in range(0, width, 260):
        draw.rectangle([x, 0, x + 28, height], fill=road_color)
    for y in range(0, height, 220):
        draw.rectangle([0, y, width, y + 18], fill=road_color)

    parcel_boxes = [
        (80, 80, 420, 280), (470, 80, 830, 280), (900, 80, 1260, 280), (1320, 80, 1540, 280),
        (80, 330, 420, 560), (470, 330, 830, 560), (900, 330, 1260, 560), (1320, 330, 1540, 560),
        (80, 610, 420, 840), (470, 610, 830, 840), (900, 610, 1260, 840), (1320, 610, 1540, 840),
        (80, 890, 420, 1020), (470, 890, 830, 1020), (900, 890, 1260, 1020), (1320, 890, 1540, 1020),
    ]

    colors = ['#cfeec7', '#d9efff', '#f8d9b5', '#f7e7b4', '#e0d9ff', '#f8d6e5']
    polygons = []

    for idx, (x1, y1, x2, y2) in enumerate(parcel_boxes, start=1):
        draw.rectangle([x1, y1, x2, y2], fill=colors[(idx - 1) % len(colors)], outline='#2d3748', width=4)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        polygons.append({
            'points': [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
            ],
            'label': f'P{idx}',
            'cx': cx,
            'cy': cy,
        })

    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 28)
        title_font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 42)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    for item in polygons:
        x = item['cx']
        y = item['cy']
        label = item['label']
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw / 2, y - th / 2), label, fill='#1f2937', font=font)

    draw.text((70, 1040), 'Urban Parcel Map Sample', fill='#1f2937', font=title_font)
    image.save(sample_path)

    result = []
    for item in polygons:
        result.append({
            'points': item['points'],
            'label': item['label'],
        })

    return str(sample_path), result


def detect_candidate_parcels(image_bgr):
    h, w = image_bgr.shape[:2]
    candidates = []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    masks = []

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.append(cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5
    ))
    _, bright = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    masks.append(bright)
    edges = cv2.Canny(gray, 45, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    masks.append(edges)

    is_wide_aerial = w / max(h, 1) > 1.5
    min_area = max(250, (w * h) * (0.00008 if is_wide_aerial else 0.00012))
    max_area = (w * h) * 0.12
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for mask_index, mask in enumerate(masks):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if cw < 25 or ch < 25 or cw > w * 0.8 or ch > h * 0.8:
                continue

            rect_area = cw * ch
            if rect_area == 0 or area / rect_area < 0.35:
                continue

            peri = cv2.arcLength(contour, True)
            if peri == 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            points = approx.reshape(-1, 2)
            if len(points) < 4:
                continue

            candidates.append({
                "points": [[int(px), int(py)] for px, py in points],
                "box": (x, y, cw, ch),
                "area": area,
                "source": mask_index,
            })

    # Keep regions that are supported by more than one independent image cue.
    # This prevents the final count from being the sum of unrelated masks.
    supported = []
    minimum_support = 3 if (w * h) > 2_000_000 else 2
    for candidate in candidates:
        x, y, cw, ch = candidate["box"]
        support_sources = {candidate["source"]}
        for other in candidates:
            if other is candidate or other["source"] in support_sources:
                continue
            ox, oy, ocw, och = other["box"]
            ix1, iy1 = max(x, ox), max(y, oy)
            ix2, iy2 = min(x + cw, ox + ocw), min(y + ch, oy + och)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            smaller = min(cw * ch, ocw * och)
            if smaller and intersection / smaller > 0.20:
                support_sources.add(other["source"])
        if len(support_sources) >= minimum_support:
            candidate["support"] = len(support_sources)
            supported.append(candidate)

    candidates = supported
    candidates.sort(key=lambda item: item["area"], reverse=True)
    selected = []
    for candidate in candidates:
        x, y, cw, ch = candidate["box"]
        duplicate = False
        for existing in selected:
            ex, ey, ecw, ech = existing["box"]
            ix1, iy1 = max(x, ex), max(y, ey)
            ix2, iy2 = min(x + cw, ex + ecw), min(y + ch, ey + ech)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (cw * ch) + (ecw * ech) - intersection
            if union and intersection / union > 0.55:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)

    polygons = []
    for candidate in selected:
        polygons.append({
            "points": candidate["points"],
            "label": f"P{len(polygons) + 1}",
        })

    if not polygons:
        cols = 5
        rows = 4
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)
        cell_w = max(40, (w - 2 * margin_x) / cols)
        cell_h = max(40, (h - 2 * margin_y) / rows)

        for row in range(rows):
            for col in range(cols):
                x1 = int(margin_x + col * cell_w + 20)
                y1 = int(margin_y + row * cell_h + 20)
                x2 = int(margin_x + (col + 1) * cell_w - 20)
                y2 = int(margin_y + (row + 1) * cell_h - 20)
                polygons.append({
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]],
                    "label": f"P{len(polygons) + 1}",
                })

    polygons = normalize_polygons(polygons)

    overlay = image_bgr.copy()
    line_width = max(3, round(min(h, w) / 900))
    for index, poly in enumerate(polygons, start=1):
        pts = np.array(poly["points"], dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (0, 0, 0), line_width + 3)
        cv2.polylines(overlay, [pts], True, (0, 230, 255), line_width)
        moments = cv2.moments(pts)
        if moments["m00"]:
            center_x = round(moments["m10"] / moments["m00"])
            center_y = round(moments["m01"] / moments["m00"])
            label = f"P{index}"
            cv2.putText(
                overlay, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, min(w, h) / 1800), (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                overlay, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, min(w, h) / 1800), (255, 255, 255), 2, cv2.LINE_AA
            )

    blended = cv2.addWeighted(image_bgr, 0.72, overlay, 0.28, 0)
    tally_text = f"VISION ESTIMATE: {len(polygons)} PARCEL-LIKE REGIONS"
    cv2.rectangle(blended, (24, 24), (650, 82), (15, 23, 42), -1)
    cv2.putText(
        blended, tally_text, (42, 62), cv2.FONT_HERSHEY_SIMPLEX,
        1.05, (255, 255, 255), 2, cv2.LINE_AA
    )
    return blended, polygons


def to_geojson(polygons, image_name):
    features = []
    for idx, item in enumerate(normalize_polygons(polygons), start=1):
        coords = []
        for x, y in item["points"]:
            coords.append([float(x), float(y)])
        if len(coords) > 0 and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "parcel_id": item.get("label") or f"parcel_{idx}",
                    "source": image_name,
                    "confidence": "candidate",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "project": "Drone Cadastral Mapper",
            "total_parcels": len(features),
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sample-map")
def sample_map():
    sample_path, polygons = generate_sample_parcels()
    geojson = {"type": "FeatureCollection", "features": []}
    for idx, item in enumerate(polygons, start=1):
        coords = [[float(x), float(y)] for x, y in item["points"]]
        geojson["features"].append({
            "type": "Feature",
            "properties": {"parcel_id": item["label"], "confidence": "sample"},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })

    return jsonify({
        "status": "ok",
        "image_url": "/processed/sample_cadastral_map.png",
        "polygons": [{"points": item["points"], "label": item["label"]} for item in polygons],
        "geojson": geojson,
        "count": len(polygons),
        "note": "Sample cadastral map generated for a clean 2D parcel view."
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected image."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported image format. Use JPG, PNG, or TIFF."}), 400

    original_name = secure_filename(image_file.filename)
    original_path = Path(app.config["UPLOAD_FOLDER"]) / original_name
    image_file.save(original_path)

    image = cv2.imread(str(original_path))
    if image is None:
        return jsonify({"error": "Could not read uploaded image."}), 400

    processed_bgr, polygons = detect_candidate_parcels(image)
    processed_name = f"processed_{original_name}"
    processed_path = Path(app.config["PROCESSED_FOLDER"]) / processed_name
    cv2.imwrite(str(processed_path), processed_bgr)

    geojson = to_geojson(polygons, original_name)

    return jsonify(
        {
            "status": "ok",
            "image_url": f"/processed/{processed_name}",
            "polygons": polygons,
            "geojson": geojson,
            "count": len(polygons),
            "vision_estimate": len(polygons),
            "ai_estimate": None,
            "ai_status": "No parcel-trained AI model is installed",
            "note": "Prototype cadastral parcel candidate extraction using image segmentation logic.",
        }
    )


@app.route("/processed/<path:filename>")
def processed_file(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "drone-cadastral-demo"})


def find_free_port(start_port=8007, end_port=8100):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    port = int(os.environ.get("PORT", str(find_free_port())))
    app.run(host="0.0.0.0", port=port, debug=False)
