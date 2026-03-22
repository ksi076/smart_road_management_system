from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import serial
import pyrealsense2 as rs
import sqlite3

# =========================
# DB / 저장 폴더 설정
# =========================
DB_PATH = "/home/rapi20/workspace/jaywalk_monitor.db"

BASE_SAVE_DIR = "/home/rapi20/workspace/captures"
JAYWALK_DIR = os.path.join(BASE_SAVE_DIR, "jaywalk")
ILLEGAL_PARK_DIR = os.path.join(BASE_SAVE_DIR, "illegal_park")
ILLEGAL_UTURN_DIR = os.path.join(BASE_SAVE_DIR, "illegal_uturn")

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(JAYWALK_DIR, exist_ok=True)
os.makedirs(ILLEGAL_PARK_DIR, exist_ok=True)
os.makedirs(ILLEGAL_UTURN_DIR, exist_ok=True)

# =========================
# 설정
# =========================
MODEL_PATH = "/home/rapi20/workspace/8stest/best.onnx"
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 9600

# D435
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 추가 USB 웹캠
AUX_CAM_INDEX = 0
AUX_CAM_WIDTH = 640
AUX_CAM_HEIGHT = 480

DASHBOARD_WIDTH = 1280
DASHBOARD_HEIGHT = 720

YOLO_IMGSZ = 320
YOLO_CONF = 0.5
EMERGENCY_BLINK_INTERVAL = 0.5

# =========================
# 시간 설정
# =========================
JAYWALK_COUNT_COOLDOWN = 10
PARKING_CAPTURE_COOLDOWN = 10
UTURN_CAPTURE_COOLDOWN = 10
CAPTURE_DISPLAY_DURATION = 5
AMBULANCE_EMERGENCY_DURATION = 15
WARNING_HOLD_SECONDS = 3.0

# 불법주차 기준
ILLEGAL_PARK_SEC = 3.0

# 판정 비율 기준
ROAD_RATIO_THRESHOLD = 0.30
ILLEGAL_ZONE_RATIO_THRESHOLD = 0.55
FALL_VALID_RATIO_THRESHOLD = 1 / 3   # fall이 유효 ROI에 이 비율 이상 겹쳐야 인정

# =========================
# D435 고정 ROI
# =========================
LEFT_ROAD_ROIS = [
    (229, 136, 348, 286),
    (206, 170, 231, 290),
    (177, 209, 203, 288),
    (152, 240, 176, 290),
    (1, 429, 635, 477),
]

RIGHT_ROAD_ROIS = [
    (355, 132, 500, 280),
    (499, 154, 514, 280),
    (514, 182, 534, 279),
    (532, 211, 551, 277),
]

ROAD_ROIS = LEFT_ROAD_ROIS + RIGHT_ROAD_ROIS

CROSSWALK_ROIS = [
    (63, 294, 635, 427),
    (17, 353, 61, 426),
]

SIGNAL_ROI = (568, 111, 628, 133)

# 클래스 이름
WARNING_CLASSES = {"ON", "OFF"}
PERSON_CLASSES = {"person", "personnight"}
VEHICLE_CLASSES = {"vehicle", "carnight"}
FALL_CLASSES = {"fall", "fallnight"}

# 거리 측정 설정
PATCH_SIZE = 5
LINE_SAMPLE_COUNT = 80

# =========================
# DB 함수
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            image_path TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            stat_date TEXT PRIMARY KEY,
            jaywalk_count INTEGER NOT NULL DEFAULT 0,
            last_detect_time TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_event(event_type, image_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO event_logs (event_type, detected_at, image_path)
        VALUES (?, ?, ?)
    """, (event_type, now_str, image_path))

    conn.commit()
    conn.close()


def update_jaywalk_daily_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        INSERT INTO daily_stats (stat_date, jaywalk_count, last_detect_time)
        VALUES (?, 1, ?)
        ON CONFLICT(stat_date)
        DO UPDATE SET
            jaywalk_count = jaywalk_count + 1,
            last_detect_time = excluded.last_detect_time
    """, (today_str, now_str))

    conn.commit()
    conn.close()


def load_today_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today_str = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT jaywalk_count, last_detect_time
        FROM daily_stats
        WHERE stat_date = ?
    """, (today_str,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return 0, "없음"

    jaywalk_count, last_detect_time = row
    return jaywalk_count, (last_detect_time if last_detect_time else "없음")


def load_latest_event_image(event_type):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_path
        FROM event_logs
        WHERE event_type = ?
        ORDER BY id DESC
        LIMIT 1
    """, (event_type,))

    row = cursor.fetchone()
    conn.close()

    if row is None or row[0] is None:
        return None

    image_path = row[0]
    if not os.path.exists(image_path):
        return None

    return cv2.imread(image_path)


# =========================
# 폰트 로드
# =========================
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(
        f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}\n"
        f"먼저 sudo apt install fonts-nanum 으로 설치하세요."
    )

font_title = ImageFont.truetype(FONT_PATH, 38)
font_panel_title = ImageFont.truetype(FONT_PATH, 24)
font_big = ImageFont.truetype(FONT_PATH, 72)
font_medium = ImageFont.truetype(FONT_PATH, 34)
font_small = ImageFont.truetype(FONT_PATH, 22)
font_tiny = ImageFont.truetype(FONT_PATH, 18)
font_status = ImageFont.truetype(FONT_PATH, 28)
font_emergency_big = ImageFont.truetype(FONT_PATH, 96)
font_emergency_mid = ImageFont.truetype(FONT_PATH, 42)

# =========================
# 유틸 함수
# =========================
def draw_korean_text(img, text, pos, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)


def put_center_text_pil(img, text, area_x1, area_y1, area_x2, area_y2, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = area_x1 + (area_x2 - area_x1 - text_w) // 2
    y = area_y1 + (area_y2 - area_y1 - text_h) // 2
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)


def draw_panel(img, x1, y1, x2, y2, title, title_color=(255, 255, 255)):
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 2)
    cv2.rectangle(img, (x1, y1), (x2, y1 + 48), (75, 75, 75), -1)
    img = draw_korean_text(img, title, (x1 + 14, y1 + 10), font_panel_title, title_color)
    return img


def resize_with_padding(src, target_w, target_h, bg_color=(0, 0, 0)):
    if src is None or src.size == 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        return canvas

    h, w = src.shape[:2]
    if h == 0 or w == 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        return canvas

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(src, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = bg_color

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def resize_with_padding_info(src, target_w, target_h, bg_color=(0, 0, 0)):
    if src is None or src.size == 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        return canvas, 0, 0, 0, 0

    h, w = src.shape[:2]
    if h == 0 or w == 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        return canvas, 0, 0, 0, 0

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(src, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = bg_color

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, new_w, new_h, x_offset, y_offset


def open_aux_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    return cv2.VideoCapture(index)


def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_ratio(det_box, roi_box):
    det_area = box_area(det_box)
    if det_area == 0:
        return 0.0
    inter = intersection_area(det_box, roi_box)
    return inter / det_area


def build_rect_mask(width, height, rects):
    mask = np.zeros((height, width), dtype=np.uint8)
    for (x1, y1, x2, y2) in rects:
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def overlap_ratio_with_mask(det_box, mask):
    x1, y1, x2, y2 = det_box
    h, w = mask.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = mask[y1:y2, x1:x2]
    white = cv2.countNonZero(roi)
    total = (x2 - x1) * (y2 - y1)
    return white / total if total > 0 else 0.0


def crop_box_with_margin(frame, box, margin=20):
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def judge_warning_light(roi):
    if roi is None or roi.size == 0:
        return "OFF", 0.0, 0.0, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0

    v_channel = hsv[:, :, 2]
    mean_v = cv2.mean(v_channel, mask=red_mask)[0] if red_pixels > 0 else 0.0

    RED_RATIO_THRESHOLD = 0.03
    BRIGHTNESS_THRESHOLD = 180

    state = "ON" if red_ratio >= RED_RATIO_THRESHOLD and mean_v >= BRIGHTNESS_THRESHOLD else "OFF"
    return state, red_ratio, mean_v, red_mask


def judge_traffic_signal(roi):
    if roi is None or roi.size == 0:
        return "UNKNOWN", 0.0, 0.0, None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([90, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)

    total_pixels = roi.shape[0] * roi.shape[1]
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

    RED_THRESHOLD = 0.03
    GREEN_THRESHOLD = 0.03

    if red_ratio >= RED_THRESHOLD and red_ratio > green_ratio:
        signal_state = "RED"
    elif green_ratio >= GREEN_THRESHOLD and green_ratio > red_ratio:
        signal_state = "GREEN"
    else:
        signal_state = "UNKNOWN"

    return signal_state, red_ratio, green_ratio, red_mask, green_mask


def send_serial_command(ser, cmd, last_cmd):
    if ser is None:
        return last_cmd

    if cmd != last_cmd:
        try:
            ser.write((cmd + "\n").encode())
            print(f"[SERIAL] sent: {cmd}")
            return cmd
        except Exception as e:
            print(f"[SERIAL] 전송 실패: {e}")
            return last_cmd

    return last_cmd


def send_led_command(ser, cmd, last_cmd):
    return send_serial_command(ser, cmd, last_cmd)


# =========================
# USB 웹캠 분석
# =========================
def crop_best_aux_person(aux_frame, aux_person_boxes, margin=20):
    if aux_frame is None or len(aux_person_boxes) == 0:
        return None

    best_box = None
    best_area = -1

    for (x1, y1, x2, y2, conf, label_name) in aux_person_boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    return crop_box_with_margin(aux_frame, best_box, margin=margin)


def analyze_aux_frame(aux_frame, results, model_names, hold_state):
    out = aux_frame.copy()
    aux_red_masks = []
    aux_extra_masks = []
    aux_person_boxes = []
    aux_vehicle_boxes = []

    now = time.time()
    h, w = aux_frame.shape[:2]
    emergency_detected_now = False

    if results is not None and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names if hasattr(result, "names") else model_names

            for idx, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, classes)):
                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                label_name = names[int(cls_id)]
                label_lower = label_name.lower()

                if label_lower in {c.lower() for c in WARNING_CLASSES}:
                    roi = aux_frame[y1:y2, x1:x2]
                    final_state, red_ratio, mean_v, red_mask = judge_warning_light(roi)

                    if red_mask is not None:
                        aux_red_masks.append((idx, red_mask))

                    color = (0, 0, 255) if final_state == "ON" else (255, 0, 0)

                    if final_state == "ON":
                        emergency_detected_now = True
                        hold_state["last_warning_on_time"] = now

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{final_state}", (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(out, f"model:{label_name} {conf:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(out, f"R={red_ratio:.3f} V={mean_v:.1f}",
                                (x1, min(y2 + 34, h - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                elif label_lower in {c.lower() for c in PERSON_CLASSES}:
                    aux_person_boxes.append((x1, y1, x2, y2, float(conf), label_name))
                    color = (0, 255, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                elif label_lower in {c.lower() for c in VEHICLE_CLASSES}:
                    aux_vehicle_boxes.append((x1, y1, x2, y2, float(conf), label_name))
                    color = (0, 255, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                elif label_lower in {c.lower() for c in FALL_CLASSES}:
                    color = (255, 0, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    warning_override_active = (now - hold_state["last_warning_on_time"]) <= WARNING_HOLD_SECONDS
    hold_state["np_cmd"] = "NP_N"
    hold_state["signal_cmd"] = "TL_ALL" if warning_override_active else "TL_NORMAL"

    status_text = "USB EMERGENCY LIGHT ON" if emergency_detected_now else "USB NORMAL"
    status_color = (0, 0, 255) if emergency_detected_now else (0, 255, 0)
    cv2.putText(out, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if warning_override_active:
        remain = WARNING_HOLD_SECONDS - (now - hold_state["last_warning_on_time"])
        cv2.putText(out, f"TRAFFIC ALL ON: {remain:.1f}s", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return out, aux_red_masks, aux_extra_masks, hold_state, aux_person_boxes, aux_vehicle_boxes


# =========================
# 대시보드
# =========================
def make_dashboard(jaywalk_capture, illegal_park_capture, uturn_capture, aux_frame, d435_frame):
    dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    dashboard[:] = (18, 18, 22)

    cv2.rectangle(dashboard, (0, 0), (DASHBOARD_WIDTH, 78), (25, 52, 96), -1)
    dashboard = draw_korean_text(dashboard, "AI기반 스마트도로 시스템", (28, 18), font_title, (255, 255, 255))

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dashboard = draw_korean_text(dashboard, current_time_str, (860, 20), font_medium, (255, 255, 255))

    left_x1 = 30
    left_x2 = 640

    panel1_y1, panel1_y2 = 110, 270
    panel2_y1, panel2_y2 = 290, 450
    panel3_y1, panel3_y2 = 470, 630

    dashboard = draw_panel(dashboard, left_x1, panel1_y1, left_x2, panel1_y2, "최근 무단횡단자 사진", (0, 220, 255))
    dashboard = draw_panel(dashboard, left_x1, panel2_y1, left_x2, panel2_y2, "최근 불법주차 사진", (255, 100, 255))
    dashboard = draw_panel(dashboard, left_x1, panel3_y1, left_x2, panel3_y2, "최근 불법 유턴 사진", (0, 200, 255))

    img_x = 85
    img_w = 500
    img_h = 90

    if jaywalk_capture is not None and jaywalk_capture.size > 0:
        capture_view = resize_with_padding(jaywalk_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[170:170 + img_h, img_x:img_x + img_w] = capture_view
    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[170:170 + img_h, img_x:img_x + img_w] = empty_view

    if illegal_park_capture is not None and illegal_park_capture.size > 0:
        park_view = resize_with_padding(illegal_park_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[350:350 + img_h, img_x:img_x + img_w] = park_view
    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[350:350 + img_h, img_x:img_x + img_w] = empty_view

    if uturn_capture is not None and uturn_capture.size > 0:
        uturn_view = resize_with_padding(uturn_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[530:530 + img_h, img_x:img_x + img_w] = uturn_view
    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[530:530 + img_h, img_x:img_x + img_w] = empty_view

    dashboard = draw_panel(dashboard, 690, 110, 1240, 360, "D435 실시간", (255, 255, 255))

    if d435_frame is not None and d435_frame.size > 0:
        d435_view = resize_with_padding(d435_frame, 470, 180, bg_color=(0, 0, 0))
        dashboard[160:160 + 180, 730:730 + 470] = d435_view
    else:
        d435_empty = np.zeros((180, 470, 3), dtype=np.uint8)
        d435_empty[:] = (25, 25, 25)
        d435_empty = put_center_text_pil(d435_empty, "D435 연결 안됨", 0, 0, 470, 180, font_medium, (180, 180, 180))
        dashboard[160:160 + 180, 730:730 + 470] = d435_empty

    dashboard = draw_panel(dashboard, 690, 380, 1240, 650, "추가 카메라 탐지 화면", (255, 210, 80))

    if aux_frame is not None and aux_frame.size > 0:
        aux_view = resize_with_padding(aux_frame, 470, 200, bg_color=(0, 0, 0))
        dashboard[430:430 + 200, 730:730 + 470] = aux_view
    else:
        aux_empty = np.zeros((200, 470, 3), dtype=np.uint8)
        aux_empty[:] = (25, 25, 25)
        aux_empty = put_center_text_pil(aux_empty, "추가 카메라 연결 안됨", 0, 0, 470, 200, font_medium, (180, 180, 180))
        dashboard[430:430 + 200, 730:730 + 470] = aux_empty

    dashboard = draw_korean_text(
        dashboard,
        "자동 감지: 무단횡단/불법주정차/불법유턴 캡처 표시 / k=캡처 숨김 / ESC=종료",
        (110, 675),
        font_tiny,
        (210, 210, 210)
    )

    return dashboard


def make_capture_fullscreen(capture_img, title_text="캡처 화면", distance_cm=None):
    screen = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)
    screen[:] = (10, 10, 10)

    cv2.rectangle(screen, (0, 0), (DASHBOARD_WIDTH, 80), (120, 30, 30), -1)
    screen = draw_korean_text(screen, title_text, (30, 20), font_title, (255, 255, 255))

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    screen = draw_korean_text(screen, current_time_str, (860, 22), font_small, (255, 255, 255))

    panel_x1, panel_y1 = 60, 110
    panel_x2, panel_y2 = 1220, 660
    cv2.rectangle(screen, (panel_x1, panel_y1), (panel_x2, panel_y2), (35, 35, 35), -1)
    cv2.rectangle(screen, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 2)

    img_x = 100
    img_y = 135
    img_w = 1080
    img_h = 500

    full_capture, new_w, new_h, x_offset, y_offset = resize_with_padding_info(capture_img, img_w, img_h, bg_color=(0, 0, 0))
    screen[img_y:img_y + img_h, img_x:img_x + img_w] = full_capture

    if distance_cm is not None:
        right_padding = img_w - (x_offset + new_w)

        if right_padding >= 220:
            box_x1 = img_x + x_offset + new_w + 20
            box_y1 = img_y + 120
            box_x2 = img_x + img_w - 20
            box_y2 = img_y + 320
        else:
            box_x1 = 910
            box_y1 = 170
            box_x2 = 1180
            box_y2 = 360

        cv2.rectangle(screen, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
        cv2.rectangle(screen, (box_x1, box_y1), (box_x2, box_y2), (0, 200, 255), 2)

        screen = draw_korean_text(screen, "거리 정보", (box_x1 + 20, box_y1 + 18), font_small, (0, 220, 255))
        screen = draw_korean_text(screen, f"{distance_cm:.1f} cm", (box_x1 + 22, box_y1 + 78), font_medium, (255, 255, 255))
        screen = draw_korean_text(screen, "무단횡단자와", (box_x1 + 22, box_y1 + 138), font_tiny, (210, 210, 210))
        screen = draw_korean_text(screen, "횡단보도 중앙 거리", (box_x1 + 22, box_y1 + 165), font_tiny, (210, 210, 210))

    screen = draw_korean_text(screen, "k 키를 누르면 캡처 화면이 사라집니다.", (410, 675), font_small, (255, 255, 0))
    return screen


def make_emergency_fullscreen():
    screen = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)

    now = time.time()
    blink_on = int(now / EMERGENCY_BLINK_INTERVAL) % 2 == 0

    if blink_on:
        screen[:] = (20, 20, 60)
        banner_color = (0, 0, 255)
        title_color = (255, 255, 0)
        sub_color = (255, 255, 255)
        red_color = (0, 0, 255)
        orange_color = (0, 165, 255)
        green_color = (0, 255, 0)
        border_color = (255, 255, 255)
    else:
        screen[:] = (10, 10, 20)
        banner_color = (0, 0, 120)
        title_color = (120, 120, 255)
        sub_color = (180, 180, 180)
        red_color = (0, 0, 80)
        orange_color = (0, 90, 140)
        green_color = (0, 100, 0)
        border_color = (140, 140, 140)

    cv2.rectangle(screen, (0, 0), (DASHBOARD_WIDTH, 90), banner_color, -1)
    screen = draw_korean_text(screen, "비상사태 경고", (35, 18), font_title, (255, 255, 255))

    cv2.rectangle(screen, (80, 120), (1200, 650), (35, 35, 35), -1)
    cv2.rectangle(screen, (80, 120), (1200, 650), border_color, 5)

    screen = put_center_text_pil(screen, "비상사태 !!", 150, 150, 1130, 300, font_emergency_big, title_color)
    screen = put_center_text_pil(screen, "즉시 주변 상황을 확인하세요", 180, 280, 1100, 360, font_emergency_mid, sub_color)

    cv2.rectangle(screen, (420, 380), (860, 620), (55, 55, 55), -1)
    cv2.rectangle(screen, (420, 380), (860, 620), border_color, 4)

    cv2.circle(screen, (520, 500), 55, red_color, -1)
    cv2.circle(screen, (640, 500), 55, orange_color, -1)
    cv2.circle(screen, (760, 500), 55, green_color, -1)

    cv2.circle(screen, (520, 500), 55, border_color, 3)
    cv2.circle(screen, (640, 500), 55, border_color, 3)
    cv2.circle(screen, (760, 500), 55, border_color, 3)

    screen = draw_korean_text(screen, "빨강", (485, 580), font_small, (255, 255, 255))
    screen = draw_korean_text(screen, "주황", (605, 580), font_small, (255, 255, 255))
    screen = draw_korean_text(screen, "초록", (725, 580), font_small, (255, 255, 255))

    guide_color = (255, 255, 0) if blink_on else (180, 180, 120)
    screen = put_center_text_pil(screen, "비상상태가 유지되고 있습니다.", 250, 640, 1030, 700, font_small, guide_color)

    return screen


# =========================
# D435 / 거리 계산
# =========================
def get_valid_depth_median(depth_frame, cx, cy, patch_size=5):
    half = patch_size // 2
    values = []

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for y in range(max(0, cy - half), min(height, cy + half + 1)):
        for x in range(max(0, cx - half), min(width, cx + half + 1)):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                values.append(d)

    if not values:
        return 0.0

    return float(np.median(values))


def deproject_to_3d(intrinsics, x, y, depth_m):
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
    return np.array(point_3d, dtype=np.float32)


def sample_line_points(pt1, pt2, n=80):
    x1, y1 = pt1
    x2, y2 = pt2
    pts = []

    for i in range(n + 1):
        t = i / n
        x = int(round(x1 * (1 - t) + x2 * t))
        y = int(round(y1 * (1 - t) + y2 * t))
        pts.append((x, y))

    return pts


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


# =========================
# 메인
# =========================
def main():
    init_db()

    print("1. 모델 로드 시작")
    model = YOLO(MODEL_PATH)
    print("2. 모델 로드 완료")
    print("클래스 목록:", model.names)

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        print(f"시리얼 연결 성공: {SERIAL_PORT}")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        print("LED 제어 없이 계속 진행합니다.")

    aux_cap = open_aux_camera(AUX_CAM_INDEX)
    if aux_cap is not None and aux_cap.isOpened():
        aux_cap.set(cv2.CAP_PROP_FRAME_WIDTH, AUX_CAM_WIDTH)
        aux_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AUX_CAM_HEIGHT)
        aux_cap.set(cv2.CAP_PROP_FPS, 10)
        aux_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"추가 카메라 연결 성공: index={AUX_CAM_INDEX}")
    else:
        print(f"추가 카메라 연결 실패: index={AUX_CAM_INDEX}")
        aux_cap = None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, CAM_WIDTH, CAM_HEIGHT, rs.format.z16, 30)

    align = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"D435 시작 실패: {e}")
        if ser:
            ser.close()
        if aux_cap:
            aux_cap.release()
        return

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    occupied_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, CROSSWALK_ROIS + [SIGNAL_ROI] + ROAD_ROIS)
    illegal_zone_mask = cv2.bitwise_not(occupied_mask)
    left_road_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, LEFT_ROAD_ROIS)
    right_road_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, RIGHT_ROAD_ROIS)

    today_count, last_detect_time = load_today_stats()
    emergency_status = "미감지"

    last_jaywalk_capture = load_latest_event_image("jaywalk")
    last_illegal_park_capture = load_latest_event_image("illegal_park")
    last_uturn_capture = load_latest_event_image("illegal_uturn")

    last_capture = None
    capture_mode = None  # "jaywalk", "parking", "uturn"

    show_capture_fullscreen = False
    show_capture_until = 0
    display_distance_cm = None

    show_emergency_fullscreen = False

    last_jaywalk_count_time = -9999
    last_parking_capture_time = -9999
    last_uturn_capture_time = -9999
    ambulance_emergency_until = 0
    prev_raw_on_detected = False

    frame_count = 0
    last_boxes = []
    last_labels = []
    last_scores = []

    last_d435_led_cmd = None
    last_aux_np_cmd = None
    last_aux_signal_cmd = None

    aux_display_frame = None
    aux_red_masks = []
    aux_extra_masks = []
    aux_person_boxes = []
    aux_vehicle_boxes = []
    last_aux_person_boxes = []

    aux_hold_state = {
        "last_warning_on_time": 0.0,
        "np_cmd": "NP_N",
        "signal_cmd": "TL_NORMAL",
    }

    illegal_park_state = {}

    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Dashboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("통합 시스템 시작. ESC 종료 / k 캡처화면 숨김 / j 비상해제(표시안함)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()

            aux_frame = None

            if aux_cap is not None:
                ret_aux, aux_tmp = aux_cap.read()
                if ret_aux:
                    aux_frame = aux_tmp.copy()

            frame_count += 1
            now = time.time()
            h, w = frame.shape[:2]

            crosswalk_rois = CROSSWALK_ROIS
            road_rois = ROAD_ROIS
            signal_roi = SIGNAL_ROI
            fall_valid_rois = road_rois + crosswalk_rois

            for cw in crosswalk_rois:
                cv2.rectangle(display_frame, (cw[0], cw[1]), (cw[2], cw[3]), (0, 255, 0), 2)

            rx1, ry1, rx2, ry2 = CROSSWALK_ROIS[0]
            line_y = (ry1 + ry2) // 2
            line_pt1 = (rx1, line_y)
            line_pt2 = (rx2, line_y)
            cv2.line(display_frame, line_pt1, line_pt2, (255, 255, 0), 2)

            for rr in LEFT_ROAD_ROIS:
                cv2.rectangle(display_frame, (rr[0], rr[1]), (rr[2], rr[3]), (255, 255, 0), 1)

            for rr in RIGHT_ROAD_ROIS:
                cv2.rectangle(display_frame, (rr[0], rr[1]), (rr[2], rr[3]), (0, 200, 255), 1)

            sx1, sy1, sx2, sy2 = signal_roi
            signal_crop = frame[sy1:sy2, sx1:sx2]
            signal_state, signal_red_ratio, signal_green_ratio, signal_red_mask, signal_green_mask = judge_traffic_signal(signal_crop)

            signal_color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0) if signal_state == "GREEN" else (200, 200, 200)
            cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), signal_color, 2)
            cv2.putText(display_frame, f"SIGNAL: {signal_state}", (sx1, max(sy1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
            cv2.putText(display_frame, f"R={signal_red_ratio:.2f} G={signal_green_ratio:.2f}",
                        (sx1, min(sy2 + 15, h - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, signal_color, 1)

            if frame_count % 3 == 0:
                results = model.predict(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    verbose=False
                )

                last_boxes = []
                last_labels = []
                last_scores = []

                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    names = result.names if hasattr(result, "names") else model.names

                    for box, conf, cls_id in zip(boxes_xyxy, confs, classes):
                        x1, y1, x2, y2 = map(int, box)

                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        label_name = names[int(cls_id)]
                        last_boxes.append((x1, y1, x2, y2))
                        last_labels.append(label_name)
                        last_scores.append(float(conf))

                if aux_frame is not None:
                    aux_results = model.predict(
                        aux_frame,
                        imgsz=YOLO_IMGSZ,
                        conf=YOLO_CONF,
                        verbose=False
                    )
                    aux_display_frame, aux_red_masks, aux_extra_masks, aux_hold_state, aux_person_boxes, aux_vehicle_boxes = analyze_aux_frame(
                        aux_frame, aux_results, model.names, aux_hold_state
                    )

                    if len(aux_person_boxes) > 0:
                        last_aux_person_boxes = aux_person_boxes.copy()
                else:
                    aux_display_frame = None
                    aux_red_masks = []
                    aux_extra_masks = []
                    aux_person_boxes = []
                    aux_vehicle_boxes = []

            if aux_display_frame is None and aux_frame is not None:
                aux_display_frame = aux_frame.copy()

            emergency_detected = False
            jaywalking_detected = False
            vehicle_intrusion_detected = False
            illegal_uturn_detected = False
            illegal_parking_detected = False
            fall_detected = False
            current_min_dist_cm = None

            best_illegal_parking_box = None
            best_illegal_parking_ratio = 0.0
            best_uturn_box = None
            best_uturn_score = 0.0

            line_pixels_cache = None
            line_points_3d_cache = None

            def prepare_line_points():
                nonlocal line_pixels_cache, line_points_3d_cache
                if line_pixels_cache is not None and line_points_3d_cache is not None:
                    return

                line_pixels = sample_line_points(line_pt1, line_pt2, LINE_SAMPLE_COUNT)
                valid_pixels = []
                line_points_3d = []

                for lx, ly in line_pixels:
                    d = get_valid_depth_median(depth_frame, lx, ly, patch_size=PATCH_SIZE)
                    if d > 0:
                        p3d = deproject_to_3d(intr, lx, ly, d)
                        valid_pixels.append((lx, ly))
                        line_points_3d.append(p3d)

                line_pixels_cache = valid_pixels
                line_points_3d_cache = line_points_3d

            current_vehicle_keys = set()

            for i, det_box in enumerate(last_boxes):
                x1, y1, x2, y2 = det_box
                label_name = last_labels[i]
                score = last_scores[i]
                label_name_lower = label_name.lower()

                if label_name.upper() in WARNING_CLASSES:
                    roi = frame[y1:y2, x1:x2]
                    final_state, red_ratio, mean_v, _ = judge_warning_light(roi)
                    color = (0, 0, 255) if final_state == "ON" else (255, 0, 0)

                    if final_state == "ON":
                        emergency_detected = True

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{final_state}", (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(display_frame, f"model:{label_name} {score:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(display_frame, f"R={red_ratio:.3f} V={mean_v:.1f}",
                                (x1, min(y2 + 34, h - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                elif label_name_lower in {c.lower() for c in PERSON_CLASSES}:
                    person_road_ratio_max = 0.0
                    for road_roi in road_rois:
                        ratio = overlap_ratio(det_box, road_roi)
                        person_road_ratio_max = max(person_road_ratio_max, ratio)

                    person_cross_ratio_max = 0.0
                    for cw in crosswalk_rois:
                        person_cross_ratio_max = max(person_cross_ratio_max, overlap_ratio(det_box, cw))

                    is_jaywalking = (person_road_ratio_max >= (1 / 3)) or ((signal_state == "GREEN") and (person_cross_ratio_max >= (1 / 3)))

                    if is_jaywalking:
                        jaywalking_detected = True
                        color = (0, 0, 255)
                        text = f"{label_name} JAYWALK"
                    else:
                        color = (0, 255, 255)
                        text = f"{label_name}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    if is_jaywalking:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)

                        person_depth = get_valid_depth_median(depth_frame, cx, cy, patch_size=PATCH_SIZE)

                        if person_depth > 0:
                            prepare_line_points()

                            if line_points_3d_cache:
                                person_3d = deproject_to_3d(intr, cx, cy, person_depth)

                                min_dist_m = float("inf")
                                closest_line_pixel = None

                                for lpix, l3d in zip(line_pixels_cache, line_points_3d_cache):
                                    dist = euclidean_distance(person_3d, l3d)
                                    if dist < min_dist_m:
                                        min_dist_m = dist
                                        closest_line_pixel = lpix

                                dist_cm = min_dist_m * 100.0
                                current_min_dist_cm = dist_cm

                                if closest_line_pixel is not None:
                                    cv2.line(display_frame, (cx, cy), closest_line_pixel, (255, 0, 0), 2)
                                    cv2.circle(display_frame, closest_line_pixel, 5, (255, 0, 255), -1)

                                cv2.putText(display_frame, f"{text} {score:.2f} | dist={dist_cm:.1f}cm",
                                            (x1, max(y1 - 10, 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            else:
                                cv2.putText(display_frame, f"{text} {score:.2f} | line depth N/A",
                                            (x1, max(y1 - 10, 20)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        else:
                            cv2.putText(display_frame, f"{text} {score:.2f} | depth N/A",
                                        (x1, max(y1 - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    else:
                        cv2.putText(display_frame, f"{text} {score:.2f}",
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(display_frame, f"road={person_road_ratio_max:.2f} cross={person_cross_ratio_max:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                elif label_name_lower in {c.lower() for c in VEHICLE_CLASSES}:
                    left_ratio = overlap_ratio_with_mask(det_box, left_road_mask)
                    right_ratio = overlap_ratio_with_mask(det_box, right_road_mask)
                    illegal_zone_ratio = overlap_ratio_with_mask(det_box, illegal_zone_mask)

                    cross_ratio_max = 0.0
                    for cw in crosswalk_rois:
                        cross_ratio_max = max(cross_ratio_max, overlap_ratio(det_box, cw))

                    intrusion_cond = (signal_state == "RED") and (cross_ratio_max >= (1 / 3))
                    if intrusion_cond:
                        vehicle_intrusion_detected = True

                    illegal_uturn_cond = (left_ratio >= ROAD_RATIO_THRESHOLD) and (right_ratio >= ROAD_RATIO_THRESHOLD)
                    if illegal_uturn_cond:
                        illegal_uturn_detected = True
                        if score > best_uturn_score:
                            best_uturn_score = score
                            best_uturn_box = det_box

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    vehicle_key = (cx // 20, cy // 20)
                    current_vehicle_keys.add(vehicle_key)

                    if illegal_zone_ratio >= ILLEGAL_ZONE_RATIO_THRESHOLD:
                        if vehicle_key not in illegal_park_state:
                            illegal_park_state[vehicle_key] = {
                                "start_time": now,
                                "reported": False
                            }
                        else:
                            parked_time = now - illegal_park_state[vehicle_key]["start_time"]
                            if parked_time >= ILLEGAL_PARK_SEC:
                                illegal_park_state[vehicle_key]["reported"] = True
                                illegal_parking_detected = True
                                if illegal_zone_ratio > best_illegal_parking_ratio:
                                    best_illegal_parking_ratio = illegal_zone_ratio
                                    best_illegal_parking_box = det_box
                    else:
                        if vehicle_key in illegal_park_state:
                            del illegal_park_state[vehicle_key]

                    color = (0, 255, 255)
                    text = f"{label_name}"

                    if intrusion_cond:
                        color = (255, 0, 0)
                        text = f"{label_name} INTRUSION"
                    elif illegal_uturn_cond:
                        color = (0, 165, 255)
                        text = f"{label_name} U-TURN"
                    elif illegal_zone_ratio >= ILLEGAL_ZONE_RATIO_THRESHOLD and illegal_park_state.get(vehicle_key, {}).get("reported", False):
                        color = (180, 0, 255)
                        text = f"{label_name} ILLEGAL PARK"
                    elif illegal_zone_ratio >= ILLEGAL_ZONE_RATIO_THRESHOLD:
                        parked_time = now - illegal_park_state[vehicle_key]["start_time"] if vehicle_key in illegal_park_state else 0.0
                        text = f"{label_name} ZONE1 {parked_time:.1f}s"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{text} {score:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display_frame, f"L={left_ratio:.2f} R={right_ratio:.2f} Z1={illegal_zone_ratio:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                elif label_name_lower in {c.lower() for c in FALL_CLASSES}:
                    signal_overlap = overlap_ratio(det_box, signal_roi)

                    fall_roi_overlap_max = 0.0
                    for roi in fall_valid_rois:
                        fall_roi_overlap_max = max(fall_roi_overlap_max, overlap_ratio(det_box, roi))

                    is_valid_fall = (signal_overlap < 0.30) and (fall_roi_overlap_max >= FALL_VALID_RATIO_THRESHOLD)

                    if is_valid_fall:
                        fall_detected = True
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(display_frame, f"{label_name} {score:.2f}",
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                        cv2.putText(display_frame, f"fall_roi={fall_roi_overlap_max:.2f}",
                                    (x1, min(y2 + 18, h - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    else:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                        cv2.putText(display_frame, f"IGNORE {label_name} {score:.2f}",
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                        cv2.putText(display_frame, f"fall_roi={fall_roi_overlap_max:.2f}",
                                    (x1, min(y2 + 18, h - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                else:
                    color = (0, 255, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{label_name} {score:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            keys_to_delete = []
            for k, v in illegal_park_state.items():
                if k not in current_vehicle_keys and now - v["start_time"] > 1.0:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del illegal_park_state[k]

            temp_emergency_for_status = fall_detected or (time.time() < ambulance_emergency_until)

            if temp_emergency_for_status:
                status_text = "EMERGENCY"
                status_color = (0, 0, 255)
                d435_led_cmd = "N"
            elif jaywalking_detected:
                status_text = "JAYWALKING"
                status_color = (0, 0, 255)
                d435_led_cmd = "R"
            elif vehicle_intrusion_detected:
                status_text = "VEHICLE INTRUSION"
                status_color = (255, 0, 0)
                d435_led_cmd = "B"
            elif illegal_uturn_detected:
                status_text = "ILLEGAL U-TURN"
                status_color = (0, 165, 255)
                d435_led_cmd = "B"
            elif illegal_parking_detected:
                status_text = "ILLEGAL PARKING"
                status_color = (180, 0, 255)
                d435_led_cmd = "B"
            else:
                status_text = "NORMAL"
                status_color = (0, 255, 0)
                d435_led_cmd = "N"

            cv2.putText(display_frame, status_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

            last_d435_led_cmd = send_led_command(ser, d435_led_cmd, last_d435_led_cmd)
            last_aux_np_cmd = send_serial_command(ser, aux_hold_state["np_cmd"], last_aux_np_cmd)
            last_aux_signal_cmd = send_serial_command(ser, aux_hold_state["signal_cmd"], last_aux_signal_cmd)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('k'):
                show_capture_fullscreen = False
                show_capture_until = 0

            if key == ord('j'):
                show_emergency_fullscreen = False
                emergency_status = "미감지"
                ambulance_emergency_until = 0

            now = time.time()

            raw_on_detected = emergency_detected
            raw_fall_detected = fall_detected

            if raw_on_detected and (not prev_raw_on_detected):
                ambulance_emergency_until = now + AMBULANCE_EMERGENCY_DURATION
                last_detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            fall_emergency_active = raw_fall_detected
            ambulance_emergency_active = (now < ambulance_emergency_until)
            emergency_now = fall_emergency_active or ambulance_emergency_active

            if emergency_now:
                emergency_status = "감지"
                show_emergency_fullscreen = True
                show_capture_fullscreen = False
            else:
                emergency_status = "미감지"
                show_emergency_fullscreen = False

            if show_capture_fullscreen and now >= show_capture_until:
                show_capture_fullscreen = False

            # 무단횡단: USB 웹캠 person 크롭
            if (not emergency_now) and jaywalking_detected and aux_frame is not None:
                if now - last_jaywalk_count_time >= JAYWALK_COUNT_COOLDOWN:
                    boxes_for_crop = aux_person_boxes if len(aux_person_boxes) > 0 else last_aux_person_boxes
                    aux_crop = crop_best_aux_person(aux_frame, boxes_for_crop, margin=20)

                    if aux_crop is not None:
                        capture_img = aux_crop
                        print("[추가 웹캠] person crop 저장")
                    else:
                        capture_img = aux_frame.copy()
                        print("[추가 웹캠] person crop 실패 -> 전체 화면 저장")

                    last_jaywalk_count_time = now
                    last_detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    last_jaywalk_capture = capture_img
                    last_capture = capture_img
                    capture_mode = "jaywalk"
                    display_distance_cm = current_min_dist_cm

                    show_capture_fullscreen = True
                    show_capture_until = now + CAPTURE_DISPLAY_DURATION

                    filename = os.path.join(JAYWALK_DIR, f"jaywalk_{int(now)}.jpg")
                    cv2.imwrite(filename, capture_img)
                    print(f"[무단횡단자 캡처 저장 - 추가 웹캠] {filename}")

                    save_event("jaywalk", filename)
                    update_jaywalk_daily_stats()
                    today_count, last_detect_time = load_today_stats()

            # 불법주차: D435 크롭
            if (not emergency_now) and illegal_parking_detected and best_illegal_parking_box is not None:
                if now - last_parking_capture_time >= PARKING_CAPTURE_COOLDOWN:
                    capture_img = crop_box_with_margin(frame, best_illegal_parking_box, margin=25)
                    if capture_img is None:
                        capture_img = frame.copy()

                    last_parking_capture_time = now
                    last_detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    last_illegal_park_capture = capture_img
                    last_capture = capture_img
                    capture_mode = "parking"
                    display_distance_cm = None

                    show_capture_fullscreen = True
                    show_capture_until = now + CAPTURE_DISPLAY_DURATION

                    filename = os.path.join(ILLEGAL_PARK_DIR, f"illegal_park_{int(now)}.jpg")
                    cv2.imwrite(filename, capture_img)
                    print(f"[불법주차 차량 캡처 저장 - D435] {filename}")

                    save_event("illegal_park", filename)

            # 불법유턴: D435 크롭
            if (not emergency_now) and illegal_uturn_detected and best_uturn_box is not None:
                if now - last_uturn_capture_time >= UTURN_CAPTURE_COOLDOWN:
                    capture_img = crop_box_with_margin(frame, best_uturn_box, margin=25)
                    if capture_img is None:
                        capture_img = frame.copy()

                    last_uturn_capture_time = now
                    last_detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    last_uturn_capture = capture_img
                    last_capture = capture_img
                    capture_mode = "uturn"
                    display_distance_cm = None

                    show_capture_fullscreen = True
                    show_capture_until = now + CAPTURE_DISPLAY_DURATION

                    filename = os.path.join(ILLEGAL_UTURN_DIR, f"illegal_uturn_{int(now)}.jpg")
                    cv2.imwrite(filename, capture_img)
                    print(f"[불법유턴 캡처 저장 - D435] {filename}")

                    save_event("illegal_uturn", filename)

            if jaywalking_detected and current_min_dist_cm is not None:
                display_distance_cm = current_min_dist_cm

            prev_raw_on_detected = raw_on_detected

            if show_emergency_fullscreen:
                screen = make_emergency_fullscreen()
            elif show_capture_fullscreen and last_capture is not None:
                if capture_mode == "jaywalk":
                    screen = make_capture_fullscreen(
                        last_capture,
                        title_text="무단횡단자 캡처 화면",
                        distance_cm=display_distance_cm
                    )
                elif capture_mode == "parking":
                    screen = make_capture_fullscreen(
                        last_capture,
                        title_text="불법주차 차량 캡처 화면",
                        distance_cm=None
                    )
                elif capture_mode == "uturn":
                    screen = make_capture_fullscreen(
                        last_capture,
                        title_text="불법유턴 차량 캡처 화면",
                        distance_cm=None
                    )
                else:
                    screen = make_capture_fullscreen(
                        last_capture,
                        title_text="캡처 화면",
                        distance_cm=None
                    )
            else:
                screen = make_dashboard(
                    last_jaywalk_capture,
                    last_illegal_park_capture,
                    last_uturn_capture,
                    aux_display_frame,
                    display_frame,
                )

            cv2.imshow("Dashboard", screen)

            if signal_red_mask is not None:
                cv2.imshow("D435 Signal Red Mask", signal_red_mask)
            if signal_green_mask is not None:
                cv2.imshow("D435 Signal Green Mask", signal_green_mask)

            cv2.imshow("D435 Illegal Zone Mask", illegal_zone_mask)

            for idx, mask in aux_red_masks:
                cv2.imshow(f"AUX Red Mask {idx}", mask)

            for name, mask in aux_extra_masks:
                cv2.imshow(name, mask)

            if key == 27:
                break

    finally:
        if ser:
            try:
                ser.write(b"N\n")
                ser.write(b"NP_N\n")
                ser.write(b"TL_NORMAL\n")
                ser.close()
            except Exception:
                pass

        if aux_cap:
            aux_cap.release()

        pipeline.stop()
        cv2.destroyAllWindows()
        print("종료 완료")


if __name__ == "__main__":
    main()
