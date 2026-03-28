from ultralytics import YOLO  # YOLO모델을 불러오기 위한 라이브러리
import cv2                    # 카메라프레임읽기, 박스그리기,마스크연산,대시보드출력
import os                     # 파일/폴더 경로처리
import time                   # 시간로직
from datetime import datetime  # DB감지시간, 화면에 현재시간
import numpy as np             #  마스크처리, 좌표계산
from PIL import ImageFont, ImageDraw, Image  # 한글텍스트출력용
import serial                   # 라즈베리가 아두이노에게 문자열(명령) 전송
import pyrealsense2 as rs     # D435라이브러리
import sqlite3                # 사건발생기록을 DB에 저장할때 필요

# =========================
# DB / 저장 폴더 설정
# =========================
DB_PATH = "/home/rapi20/workspace/jaywalk_monitor.db"   # SQLite DB 파일 경로

BASE_SAVE_DIR = "/home/rapi20/workspace/captures"                   # 모든 캡처 이미지가 저장되는 상위 폴더 
JAYWALK_DIR = os.path.join(BASE_SAVE_DIR, "jaywalk")                # 무단횡단 이미지 저장 폴더
ILLEGAL_PARK_DIR = os.path.join(BASE_SAVE_DIR, "illegal_park")      # 불법주차 이미지 저장 폴더
ILLEGAL_UTURN_DIR = os.path.join(BASE_SAVE_DIR, "illegal_uturn")    # 불법유턴 이미지 저장 폴더

# 폴더가 없으면 생성, 있으면 무시
os.makedirs(BASE_SAVE_DIR, exist_ok=True)                      
os.makedirs(JAYWALK_DIR, exist_ok=True)                     
os.makedirs(ILLEGAL_PARK_DIR, exist_ok=True)
os.makedirs(ILLEGAL_UTURN_DIR, exist_ok=True)               

# =========================
# 설정
# =========================
MODEL_PATH = "/home/rapi20/workspace/8stest/best.onnx"              # 학습 완료된 YOLO ONNX 모델
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"       # 한글 폰트 경로

SERIAL_PORT = "/dev/ttyACM0"            # 아두이노가 연결된 포트
BAUDRATE = 9600                         # 시리얼 통신 속도

# D435
CAM_WIDTH = 640      # D435  프레임 너비
CAM_HEIGHT = 480     # D435  프레임 높이

# 추가 USB 웹캠
AUX_CAM_INDEX = 0       # 보조 웹캠 인덱스번호
AUX_CAM_WIDTH = 640     # 보조 웹캠 너비
AUX_CAM_HEIGHT = 480    # 보조 웹캠 높이

DASHBOARD_WIDTH = 1280  # 대시보드(디스플레이) 너비
DASHBOARD_HEIGHT = 720  # 대시보드(디스플레이) 높이

YOLO_IMGSZ = 320                # YOLO 추론용 입력 크기
YOLO_CONF = 0.5                 # confidence 50% 이상만 믿겠다
EMERGENCY_BLINK_INTERVAL = 0.5  # 비상화면 대시보드 깜빡이는 속도 0.5초 간격

# =========================
# 시간 설정
# =========================
JAYWALK_COUNT_COOLDOWN = 10         # 무단횡단 중복 카운트 방지 시간(10초)
PARKING_CAPTURE_COOLDOWN = 10       # 불법주차 중복 캡처 방지 시간(10초)
UTURN_CAPTURE_COOLDOWN = 10         # 불법유턴 중복 캡처 방지 시간(10초)
CAPTURE_DISPLAY_DURATION = 5        # 캡처화면 전체 표시 시간(5초)
AMBULANCE_EMERGENCY_DURATION = 15   # 경광등 ON 강지 후 비상 상태 유지시간(15초)
WARNING_HOLD_SECONDS = 3.0          # 경광등 ON 상태 유지 판정시간(3초)

# 네오픽셀 유지 시간
LED_HOLD_SECONDS = 1.5

# 불법주차 기준
ILLEGAL_PARK_SEC = 3.0         # 차량이 불법구역내에 3초 이상 머물면 불법주차로 판단

# 판정 비율 기준
ROAD_RATIO_THRESHOLD = 0.30                 # 사람이 도로 ROI에 30%이상 겹치면 무단횡단 
ILLEGAL_ZONE_RATIO_THRESHOLD = 0.55         # 차량이 불법구역과 55% 이상 겹치면 불법주차 후보
FALL_VALID_RATIO_THRESHOLD = 1 / 3          # fall이 유효 ROI에 이 비율 이상 겹쳐야 인정 (엉뚱한위치에서 검출된 오탐을 줄이기위한 기준)

# =========================
# D435 고정 ROI
# =========================

# 화면 왼쪽 차로에 해당하는 영역
LEFT_ROAD_ROIS = [
    (229, 136, 348, 286),
    (206, 170, 231, 290),
    (177, 209, 203, 288),
    (152, 240, 176, 290),
    (1, 429, 635, 477),
]

# 오른쪽 차로 영역
RIGHT_ROAD_ROIS = [
    (355, 132, 500, 280),
    (499, 154, 514, 280),
    (514, 182, 534, 279),
    (532, 211, 551, 277),
]

#전체 도로 영역
ROAD_ROIS = LEFT_ROAD_ROIS + RIGHT_ROAD_ROIS

# 횡단보도 영역
CROSSWALK_ROIS = [
    (63, 294, 635, 427),
    (17, 353, 61, 426),
]

# 신호등 고정위치 좌표
SIGNAL_ROI = (568, 111, 628, 133)


# 클래스 이름
WARNING_CLASSES = {"ON", "OFF"}                 # 경광등 ON/OFF 관련 클래스
PERSON_CLASSES = {"person", "personnight"}      # 사람 클래스
VEHICLE_CLASSES = {"vehicle", "carnight"}       # 차량 클래스
FALL_CLASSES = {"fall", "fallnight"}            # 낙상 클래스

# 거리 측정 설정
PATCH_SIZE = 5              # depth 카메라 거리 계산 시 카메라의 한 점만 믿지 않고 주변 5x5 영역으로 계산
LINE_SAMPLE_COUNT = 80      # 횡단보도 기준선 위에 80갸위 샘플 점을 뿌려서 사람과의 최소거리 계산

# =========================
# DB 함수
# =========================

# DB 준비 작업 함수(프로그램 시작할 때 자동으로 테이블 생성)
def init_db():
    conn = sqlite3.connect(DB_PATH)    # DB파일을 연다(없으면 새로 생성)
    cursor = conn.cursor()             # SQL문을 실제로 실행할 객채 생성(커서)


    # 사건 하나하나를 저장할 로그 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,       -- 자동 증가 ID
            event_type TEXT NOT NULL,                   -- 사건 종류
            detected_at TEXT NOT NULL,                  -- 발생 시각
            image_path TEXT                             -- 저장 이미지 경로
        )
    """)

    # 날짜 별 집계 통계 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            stat_date TEXT PRIMARY KEY,                  -- 날짜
            jaywalk_count INTEGER NOT NULL DEFAULT 0,    -- 그 날 무단횡단 횟수 
            last_detect_time TEXT                        -- 마지막 감지 시각
        )
    """)

    conn.commit()       # 테이블 생성 반영
    conn.close()        # 연결 닫기


# 사건 발생 로그 저장하는 함수
def save_event(event_type, image_path=None):
    conn = sqlite3.connect(DB_PATH)     
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  #현재 시각 문자열 생성

    cursor.execute("""
        INSERT INTO event_logs (event_type, detected_at, image_path)    -- 사건이 발생할 때마다 DB에 한 줄 추가
        VALUES (?, ?, ?)
    """, (event_type, now_str, image_path))     # 사건 정보 저장 ex) event_type : "jaywalk", "illegal_park", "illegal_uturn" 같은 값이 들어감
                                                #               ex) image_path : 사진경로저장
                                                #               ex) now_str    : 사람이 보기 좋게 시간을 문자열로 저장
                                                
                                                            
    conn.commit()
    conn.close()


def update_jaywalk_daily_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시각
    today_str = datetime.now().strftime("%Y-%m-%d")         # 오늘 날짜만 추출

    cursor.execute("""
        INSERT INTO daily_stats (stat_date, jaywalk_count, last_detect_time) 
        VALUES (?, 1, ?)
        ON CONFLICT(stat_date)
        DO UPDATE SET
            jaywalk_count = jaywalk_count + 1,
            last_detect_time = excluded.last_detect_time
    """, (today_str, now_str))

    # 오늘 날짜가 DB에 없으면 새로 만들고 jaywalk_count = 1

    conn.commit()
    conn.close()


# 역할 -> 대시보드나 통계 표시용으로 오늘 상황을 불러오는 함수다
def load_today_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today_str = datetime.now().strftime("%Y-%m-%d")   # 오늘 날짜 문자열

    cursor.execute("""
        SELECT jaywalk_count, last_detect_time   
        FROM daily_stats
        WHERE stat_date = ?
    """, (today_str,))
    
    # SELECT ~ : 오늘 날짜에 해당하는 통계만 가져온다
    # fetchone() : 결과가 한 줄일 거라서 하나만 가져옴
    # if row is None : 아직 오늘 무단횡단이 한 번도 기록되지 않았으면 기본값 반환
    

    row = cursor.fetchone()     # 한 줄만 가져옴
    conn.close()

    if row is None:             # 오늘 데이터가 없으면 기본값 반환
        return 0, "없음"        

    jaywalk_count, last_detect_time = row   # 무단횡단자수와 최근감지시간을 가져옴
    return jaywalk_count, (last_detect_time if last_detect_time else "없음")




# 이미지를 불러오는 함수 (프로그램을 다시 켰을때도 가장 최근의 이미지를 대시보드에 다시 보여주기 위함)



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
    # DB에는 경로만 저장
    # 실제이미지는 파일로 저장
    # 필요할 떄 경로를 통해 다시 불러옴

    row = cursor.fetchone()
    conn.close()

    if row is None or row[0] is None:       # 해당 사건의 이미지 기록이 없으면 None
        return None

    image_path = row[0]
    if not os.path.exists(image_path):      # 경로는 있는데 파일이 없으면 None
        return None

    return cv2.imread(image_path)           # 실제 이미지 파일을 읽어서 반환


# =========================
# 폰트 로드
# =========================

# 폰트가 없으면 한글 출력이 안되기 떄문에 시작할 때 바로 검사
if not os.path.exists(FONT_PATH):

    # 폰트가 없으면 정확한 에러를 보여주고 설치 방법까지 알려줌
    raise FileNotFoundError(
        f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}\n"
        f"먼저 sudo apt install fonts-nanum 으로 설치하세요."
    )

font_title = ImageFont.truetype(FONT_PATH, 38)          # 상단 큰 제목용
font_panel_title = ImageFont.truetype(FONT_PATH, 24)    # 패널 제목용
font_big = ImageFont.truetype(FONT_PATH, 72)            # 큰 강조 문구용
font_medium = ImageFont.truetype(FONT_PATH, 34)         # 중간 글씨
font_small = ImageFont.truetype(FONT_PATH, 22)          # 일반 안내 문구
font_tiny = ImageFont.truetype(FONT_PATH, 18)           # 작은 안내 문구
font_status = ImageFont.truetype(FONT_PATH, 28)         # 상태 표시용
font_emergency_big = ImageFont.truetype(FONT_PATH, 96)  # 비상화면 큰 경고 문구
font_emergency_mid = ImageFont.truetype(FONT_PATH, 42)  # 비상화면 부제목
# 같은 폰트라도 크기를 여러개 만들어 놓고 상황에 맞게 쓰기 위한 구조





# =========================
# 유틸 함수
# =========================

# 이 함수가 없으면 대시보드에 한글 제목이나 문구를 예쁘게 넣기 어렵다
def draw_korean_text(img, text, pos, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)                  # OpenCV 이미지(numpy 배열)를 PIL 이미지로 변환
    draw = ImageDraw.Draw(img_pil)                  # PIL 이미지에 글씨나 도형을 그릴 수 있게 준비
    draw.text(pos, text, font=font, fill=color)     # 지정 위치에 텍스트 출력
    return np.array(img_pil)                        # 다시 numpy(OpenCV스타일)배열로 되돌려서 이후 처리 가능하게 함





# 지정한 영역 안에 자동 중앙정렬을 해주는 함수이다.
def put_center_text_pil(img, text, area_x1, area_y1, area_x2, area_y2, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)       # 텍스트 크기 계산용 박스
    text_w = bbox[2] - bbox[0]                          # 글자 폭
    text_h = bbox[3] - bbox[1]                          # 글자 높이
    x = area_x1 + (area_x2 - area_x1 - text_w) // 2     # 사각형 중앙 X 계산
    y = area_y1 + (area_y2 - area_y1 - text_h) // 2     # 사각형 중앙 y 계산
    draw.text((x, y), text, font=font, fill=color)      
    return np.array(img_pil)


# 대시보드에서 "최근 무단횡단자 사진" 같은 박스 툴을 만드는 함수
def draw_panel(img, x1, y1, x2, y2, title, title_color=(255, 255, 255)):
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)                # 패널 내부 배경색
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 2)                 # 패널 테두리
    cv2.rectangle(img, (x1, y1), (x2, y1 + 48), (75, 75, 75), -1)           # 상단 제목 바
    img = draw_korean_text(img, title, (x1 + 14, y1 + 10), font_panel_title, title_color)   # 패널 제목 표시
    return img


# 비율 유지, 중앙 정렬, 남는 공간 배경 채우기 역할을 해주는 함수
def resize_with_padding(src, target_w, target_h, bg_color=(0, 0, 0)):
    if src is None or src.size == 0:                                # 입력 이미지가 비어 있으면
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # 빈 캔버스 생성
        canvas[:] = bg_color
        return canvas

    h, w = src.shape[:2]            #원본 이미지 높이/너비
    if h == 0 or w == 0:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:] = bg_color
        return canvas

    scale = min(target_w / w, target_h / h)     # 비율 유지용 배율
    new_w = max(1, int(w * scale))              # 새 너비
    new_h = max(1, int(h * scale))              # 새 높이

    resized = cv2.resize(src, (new_w, new_h))                   # 비율 유지 resize
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # 목표 크기 빈 캔버스
    canvas[:] = bg_color

    x_offset = (target_w - new_w) // 2          # 가로 중앙 정렬
    y_offset = (target_h - new_h) // 2          # 세로 중앙 정렬
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


# 위의 함수는 이미지 하나만 반환하지만, 이 함수는 추가로 새 너비, 새 높이, 좌측 여백, 상단 여백 까지 반환한다.
#  캡처 전체화면에 거리 정보 박스를 이미지 옆에 띄울 때, 이미지가 실제 어디에 놓여있는지 알기 위한 함수
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
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)         # 리눅스 환경에서 카메라를 여는 일반적인 방식
    if cap.isOpened():              
        return cap                                      # 성공하면 바로 반환
    return cv2.VideoCapture(index)                      # 살패하면 일반 방식 재시도




# 사람 박스가 횡단보도 ROI와 얼마나 겹치는지, 차량 박스가 차도 ROI와 얼마나 겹치는지를 계산하기 위한 함수이다.
def intersection_area(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)     # 겹치는 시작 X
    iy1 = max(ay1, by1)     # 겹치는 시작 Y
    ix2 = min(ax2, bx2)     # 겹치는 끝 X
    iy2 = min(ay2, by2)     # 겹치는 끝 Y

    if ix2 <= ix1 or iy2 <= iy1:    # 안 겹치면 면적 0
        return 0

    return (ix2 - ix1) * (iy2 - iy1)        #겹치는 사각형 면적




# 박스자체 면적 계산
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)



def overlap_ratio(det_box, roi_box):
    det_area = box_area(det_box)    # 검출 박스 면적
    if det_area == 0:
        return 0.0
    inter = intersection_area(det_box, roi_box)     # 겹침 면적
    return inter / det_area                         # 검출 박스 기준 겹침 비율



# ROI들을 사각형으로 그린 마스크를 만드는 함수이다.
def build_rect_mask(width, height, rects):      
    mask = np.zeros((height, width), dtype=np.uint8)    # 전체 검은 배경 마스크
    for (x1, y1, x2, y2) in rects:
        x1 = max(0, min(x1, width - 1))             # 좌표 보정
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255        # ROI 부분만 흰색으로 채움
    return mask


# 사각형 ROI 하나하나 비교하는 대신, 미리 만든 흰색 마스크와 겹치는 비율을 빠르게 계산하기 위한 함수
def overlap_ratio_with_mask(det_box, mask):
    x1, y1, x2, y2 = det_box
    h, w = mask.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = mask[y1:y2, x1:x2]        # 박스에 해당하는 마스크 부분만 잘라냄
    white = cv2.countNonZero(roi)
    total = (x2 - x1) * (y2 - y1)
    return white / total if total > 0 else 0.0


# 탐지 박스에 딱 맞게 자르면 보기좋지 않기 때문에 주변 여유를 살짝 포함해서 저장하기 위한 함수이다.
def crop_box_with_margin(frame, box, margin=20):
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # 여유 공간 확보
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def judge_warning_light(roi):
    if roi is None or roi.size == 0:
        return "OFF", 0.0, 0.0, None        # ROI가 비어 있으면 OFF 처리

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # RGB -> HSV 변환 (색 범위 분리가 쉬워서 빨강/초록 판정에 유리함)

    lower_red_1 = np.array([0, 80, 80])         # 빨강 범위 1
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 80, 80])       # 빨강 범위 2
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)      # 빨강 마스크 1
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)      # 빨강 마스크 2
    red_mask = cv2.bitwise_or(mask1, mask2)                 # 둘 합치기

    kernel = np.ones((3, 3), np.uint8)          #노이즈 제거용 커널
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   # 작은 잡음 제거
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel) # 빨간 영역 확장

    red_pixels = cv2.countNonZero(red_mask)         # 빨간 픽셀 수
    total_pixels = roi.shape[0] * roi.shape[1]      # ROI 전체 픽셀 수
    red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0.0  # 빨간 비율

    v_channel = hsv[:, :, 2]    # 밝기 채널(V)
    mean_v = cv2.mean(v_channel, mask=red_mask)[0] if red_pixels > 0 else 0.0   # 빨간 부분 평균 밝기

    RED_RATIO_THRESHOLD = 0.03  # 빨간색 비율 3% 이상
    BRIGHTNESS_THRESHOLD = 180  # 밝기까지 고려 (단순 빨간 물체가 아니라 실제로 켜진 경광등인지 확인(오탐 방지 로직))

    state = "ON" if red_ratio >= RED_RATIO_THRESHOLD and mean_v >= BRIGHTNESS_THRESHOLD else "OFF"
    return state, red_ratio, mean_v, red_mask


# 신호등 색깔 판정 함수 (위치가 고정이기 떄문에 좌표 고정으로 판단)
def judge_traffic_signal(roi):
    if roi is None or roi.size == 0:
        return "UNKNOWN", 0.0, 0.0, None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    lower_green = np.array([40, 80, 80])        # 초록 범위
    upper_green = np.array([90, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green) # 초록 마스크

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

    if red_ratio >= RED_THRESHOLD and red_ratio > green_ratio:          # 빨강 비율이 충분히 크고 초록보다 크면 RED
        signal_state = "RED"
    elif green_ratio >= GREEN_THRESHOLD and green_ratio > red_ratio:    # 초록 비율이 충분히 크고 빨강보다 크면 GREEN
        signal_state = "GREEN"
    else:                                                               # 둘 다 애매하면 UNKNOWN
        signal_state = "UNKNOWN"

    return signal_state, red_ratio, green_ratio, red_mask, green_mask


def send_serial_command(ser, cmd, last_cmd):
    if ser is None or cmd is None:              # 시리얼 객체가 없으면
        return last_cmd                         # 아무것도 하지 않고 이전 명령 유지

    if cmd != last_cmd:                         # 이번 명령이 이전 명령과 다를 때만 전송 ( 같은 명령을 매 프레임 보내면 비효율적이라서)
        try:
            ser.write((cmd + "\n").encode())    # 문자열 + 줄바꿈 전송
            print(f"[SERIAL] sent: {cmd}")      # 디버그 출력
            return cmd                          # 방금 보낸 명령을 새로운 Last_cmd로 저장 
        except Exception as e:
            print(f"[SERIAL] 전송 실패: {e}")
            return last_cmd

    return last_cmd                             # 같은 명령이면 중복 전송 안함


def drain_serial_feedback(ser):
    if ser is None:
        return
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"[ARDUINO] {line}")
    except Exception:
        pass


# =========================
# USB 웹캠 분석
# =========================


# 보조 웹캠에서 가장 크게 잡힌 사람을 저장
def crop_best_aux_person(aux_frame, aux_person_boxes, margin=20):
    if aux_frame is None or len(aux_person_boxes) == 0:
        return None

    best_box = None
    best_area = -1

    for (x1, y1, x2, y2, conf, label_name) in aux_person_boxes:
        area = (x2 - x1) * (y2 - y1)            # 사람 박스 면적 계산
        if area > best_area:                    # 가장 큰 사람 박스를 선택
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    return crop_box_with_margin(aux_frame, best_box, margin=margin)



# 경광등 ON/OFF 확인, 사람/차량 박스 저장, 화면에 시각화, hold_state 업데이트 수행 함수
def analyze_aux_frame(aux_frame, results, model_names, hold_state):
    out = aux_frame.copy()      # 원본을 보존하고 출력용 프레임 생성
    aux_red_masks = []          
    aux_extra_masks = []
    aux_person_boxes = []       # 보조 카메라 사람 박스 목록
    aux_vehicle_boxes = []      # 보조 카메라 차량 박스 목록

    now = time.time()
    h, w = aux_frame.shape[:2]
    emergency_detected_now = False

    if results is not None and len(results) > 0:
        result = results[0]                                     
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()            # 박스 좌표
            confs = result.boxes.conf.cpu().numpy()                 # confidence
            classes = result.boxes.cls.cpu().numpy()                # 클래스 ID
            names = result.names if hasattr(result, "names") else model_names

            for idx, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confs, classes)):    # 탐지된 객체를 하나씩 순회 
                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                # 좌표를 정수로 바꾸고, 이미지 범위 넘지 않게 보정, 폭이나 높이가 0인 이상한 박스 건너뜀

                label_name = names[int(cls_id)]
                label_lower = label_name.lower()


                # YOLO가 경광등 후보를 잡으면 그 ROI만 잘라서 judge_warning_light()로 진짜 켜졌는지 색 분석을 함
                if label_lower in {c.lower() for c in WARNING_CLASSES}:
                    roi = aux_frame[y1:y2, x1:x2]
                    final_state, red_ratio, mean_v, red_mask = judge_warning_light(roi)

                    if red_mask is not None:
                        aux_red_masks.append((idx, red_mask))

                    # 실제 ON이면 빨강 박스, OFF면 파랑 박스료 표시
                    color = (0, 0, 255) if final_state == "ON" else (255, 0, 0)


                    if final_state == "ON":  # ON이 감지되면
                        emergency_detected_now = True  # 비상상황 플래그를 킴
                        hold_state["last_warning_on_time"] = now  # 마지막 감지 시각 갱신

                    # 최종상태 ON/OFF 출력, YOLO모델이 무슨 클래스를 잡았는지, 빨강비율과 밝기 표시 (왜 ON/OFF가 나왔는지 확인가능)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{final_state}", (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(out, f"model:{label_name} {conf:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(out, f"R={red_ratio:.3f} V={mean_v:.1f}",
                                (x1, min(y2 + 34, h - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                 # 사람 클래스 처리            
                elif label_lower in {c.lower() for c in PERSON_CLASSES}:
                    aux_person_boxes.append((x1, y1, x2, y2, float(conf), label_name)) # 사람박스를 리스트에 저장
                    color = (0, 255, 255)   #화면에는 노란색 박스로 표시
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 차량 클래스 처리 (사람과 비슷하게 리스트에 저장하고 시각화한다.)
                elif label_lower in {c.lower() for c in VEHICLE_CLASSES}:
                    aux_vehicle_boxes.append((x1, y1, x2, y2, float(conf), label_name))
                    color = (0, 255, 255)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                # fall 클래스 처리 
                elif label_lower in {c.lower() for c in FALL_CLASSES}:
                    color = (255, 0, 255) # fall클래스는 보라색으로 표시
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(out, f"{label_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    #경광등 유지 로직 (USB웹캠은 신호등전체켜기 (비상제어) 판단에 큰 역할을 함)
    warning_override_active = (now - hold_state["last_warning_on_time"]) <= WARNING_HOLD_SECONDS # 방금 ON이 아니더라도 마지막 ON시간으로부터 3초 이내면 비상상태 유지
    hold_state["signal_cmd"] = "TL_ALL" if warning_override_active else "TL_NORMAL" #경광등 유지 중이면 TL_ALL, 아니면 TL_NORMAL

    status_text = "USB EMERGENCY LIGHT ON" if emergency_detected_now else "USB NORMAL"
    status_color = (0, 0, 255) if emergency_detected_now else (0, 255, 0)
    cv2.putText(out, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if warning_override_active:  #비상유지시간이 아직 남아있으면 화면에 몇초남았는지 같이 표시
        remain = WARNING_HOLD_SECONDS - (now - hold_state["last_warning_on_time"])
        cv2.putText(out, f"TRAFFIC ALL ON: {remain:.1f}s", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return out, aux_red_masks, aux_extra_masks, hold_state, aux_person_boxes, aux_vehicle_boxes
    # out : 시각화 완료된 보조 카메라 화면, aux_red_masks : 경광등 빨간 마스크들, aux_extra_masks : 추가마스크, hold_state: 유지상태정보, aux_person_boxes : 보조카메라 사람목록, aux_vehucle_boxes : 보조카메라 차량목록
    # 이 함수는 USB웹캠 쪽 전체 판단 결과를 한번에 정리해서 돌려주는 함수임





# =========================
# 대시보드
# =========================

def make_dashboard(jaywalk_capture, illegal_park_capture, uturn_capture, aux_frame, d435_frame):
    dashboard = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8) #전체 배경 캔버스 생성
    dashboard[:] = (18, 18, 22)     # 어두운 회식 배경으로 채움

    cv2.rectangle(dashboard, (0, 0), (DASHBOARD_WIDTH, 78), (25, 52, 96), -1)   #상단 타이틀 바
    dashboard = draw_korean_text(dashboard, "AI기반 스마트도로 시스템", (28, 18), font_title, (255, 255, 255))

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") #현재 시각 문자열
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

    # 최근 무단횡단자 사진 넣기
    if jaywalk_capture is not None and jaywalk_capture.size > 0:
        capture_view = resize_with_padding(jaywalk_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[170:170 + img_h, img_x:img_x + img_w] = capture_view

        # 최근 무단횡단자 사진이 있으면 패널 크기에 맞게 resize해서 넣음

    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[170:170 + img_h, img_x:img_x + img_w] = empty_view

        # 없다면 빈박스를 생성하고, 가운데 "캡처 이미지 없음"으로 표시

        # 최근 불법주차 사진도 위와 같은 방식으로 진행
    if illegal_park_capture is not None and illegal_park_capture.size > 0:
        park_view = resize_with_padding(illegal_park_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[350:350 + img_h, img_x:img_x + img_w] = park_view
    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[350:350 + img_h, img_x:img_x + img_w] = empty_view

        # 최근 불법유턴 사진도 위와 같은 방식으로 진행
    if uturn_capture is not None and uturn_capture.size > 0:
        uturn_view = resize_with_padding(uturn_capture, img_w, img_h, bg_color=(0, 0, 0))
        dashboard[530:530 + img_h, img_x:img_x + img_w] = uturn_view
    else:
        empty_view = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        empty_view[:] = (25, 25, 25)
        empty_view = put_center_text_pil(empty_view, "캡처 이미지 없음", 0, 0, img_w, img_h, font_small, (180, 180, 180))
        dashboard[530:530 + img_h, img_x:img_x + img_w] = empty_view

    dashboard = draw_panel(dashboard, 690, 110, 1240, 360, "D435 실시간", (255, 255, 255))

    # D435웹캠 화면을 실시간으로 출력
    if d435_frame is not None and d435_frame.size > 0:
        d435_view = resize_with_padding(d435_frame, 470, 180, bg_color=(0, 0, 0))
        dashboard[160:160 + 180, 730:730 + 470] = d435_view
    else:
        d435_empty = np.zeros((180, 470, 3), dtype=np.uint8)
        d435_empty[:] = (25, 25, 25)
        d435_empty = put_center_text_pil(d435_empty, "D435 연결 안됨", 0, 0, 470, 180, font_medium, (180, 180, 180))
        dashboard[160:160 + 180, 730:730 + 470] = d435_empty

    dashboard = draw_panel(dashboard, 690, 380, 1240, 650, "추가 카메라 탐지 화면", (255, 210, 80))

    # USB웹캠화면도 실시간으로 출력
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


# 캡처 전체화면 함수
def make_capture_fullscreen(capture_img, title_text="캡처 화면", distance_cm=None):
    screen = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8) # 사건이 발생했을때 대시보드 대신 크게보여줄 전체화면용 캔버스를 만든다
    screen[:] = (10, 10, 10)

    cv2.rectangle(screen, (0, 0), (DASHBOARD_WIDTH, 80), (120, 30, 30), -1)
    screen = draw_korean_text(screen, title_text, (30, 20), font_title, (255, 255, 255))

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    screen = draw_korean_text(screen, current_time_str, (860, 22), font_small, (255, 255, 255))

    # 캡처이미지를 넣을 메인 패널을 중앙에 크게 하나 만든다.
    panel_x1, panel_y1 = 60, 110
    panel_x2, panel_y2 = 1220, 660
    cv2.rectangle(screen, (panel_x1, panel_y1), (panel_x2, panel_y2), (35, 35, 35), -1)
    cv2.rectangle(screen, (panel_x1, panel_y1), (panel_x2, panel_y2), (100, 100, 100), 2)

    # 실제 캡처 이미지를 큰 영역에 배치하는 부분
    img_x = 100
    img_y = 135
    img_w = 1080
    img_h = 500

    full_capture, new_w, new_h, x_offset, y_offset = resize_with_padding_info(capture_img, img_w, img_h, bg_color=(0, 0, 0))
    screen[img_y:img_y + img_h, img_x:img_x + img_w] = full_capture

    # 거리 정보 박스
    if distance_cm is not None:
        right_padding = img_w - (x_offset + new_w)

        if right_padding >= 220:    # 이미지 오른쪽에 여백이 충분하면 그 여백안에 거리 정보 박스를 붙임
            box_x1 = img_x + x_offset + new_w + 20
            box_y1 = img_y + 120
            box_x2 = img_x + img_w - 20
            box_y2 = img_y + 320
        else:                       # 여백이 부족하면 미리 정한 고정위치에 박스 배치
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


# =========================
# 비상화면 + D435 실시간 화면 포함
# =========================

#비상 전체화면 함수
def make_emergency_fullscreen(live_frame=None):
    screen = np.zeros((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8)

    now = time.time()
    blink_on = int(now / EMERGENCY_BLINK_INTERVAL) % 2 == 0   # 비상화면 깜박임 코드 (0.5초마다 True/False가바뀐다)

    if blink_on:        #비상화면 전체가 번쩍이면서 비상상태 시각적 강조
        screen[:] = (20, 20, 60)
        banner_color = (0, 0, 255)
        title_color = (255, 255, 0)
        sub_color = (255, 255, 255)
        red_color = (0, 0, 255)
        orange_color = (0, 165, 255)
        green_color = (0, 255, 0)
        border_color = (255, 255, 255)
        live_box_border = (0, 200, 255)
    else:
        screen[:] = (10, 10, 20)
        banner_color = (0, 0, 120)
        title_color = (120, 120, 255)
        sub_color = (180, 180, 180)
        red_color = (0, 0, 80)
        orange_color = (0, 90, 140)
        green_color = (0, 100, 0)
        border_color = (140, 140, 140)
        live_box_border = (0, 120, 180)

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




    # =========================
    # 우측 상단 D435 실시간 화면 박스
    # =========================
    live_box_x1, live_box_y1 = 900, 120
    live_box_x2, live_box_y2 = 1190, 250
    live_box_w = live_box_x2 - live_box_x1
    live_box_h = live_box_y2 - live_box_y1

    cv2.rectangle(screen, (live_box_x1, live_box_y1), (live_box_x2, live_box_y2), (30, 30, 30), -1)
    cv2.rectangle(screen, (live_box_x1, live_box_y1), (live_box_x2, live_box_y2), live_box_border, 3)

    cv2.rectangle(screen, (live_box_x1, live_box_y1), (live_box_x2, live_box_y1 + 28), (55, 55, 55), -1)
    screen = draw_korean_text(screen, "D435 실시간", (live_box_x1 + 10, live_box_y1 + 4), font_tiny, (255, 255, 255))

    inner_x1 = live_box_x1 + 6
    inner_y1 = live_box_y1 + 34
    inner_x2 = live_box_x2 - 6
    inner_y2 = live_box_y2 - 6
    inner_w = inner_x2 - inner_x1
    inner_h = inner_y2 - inner_y1

    if live_frame is not None and live_frame.size > 0:
        live_view = resize_with_padding(live_frame, inner_w, inner_h, bg_color=(0, 0, 0))
        screen[inner_y1:inner_y2, inner_x1:inner_x2] = live_view
    else:
        empty_live = np.zeros((inner_h, inner_w, 3), dtype=np.uint8)
        empty_live[:] = (20, 20, 20)
        empty_live = put_center_text_pil(empty_live, "영상 없음", 0, 0, inner_w, inner_h, font_tiny, (180, 180, 180))
        screen[inner_y1:inner_y2, inner_x1:inner_x2] = empty_live

    return screen


# =========================
# D435 / 거리 계산
# =========================

# D435 깊이 프레임에서 한 점만 보는게 아니라 주변 5x5영역을 돌면서 유효한 깊이 값들을 모은다.(안정적임)
# 평균보다 중앙값이 노이즈나 이상치에 강함 (중앙값 사용)
def get_valid_depth_median(depth_frame, cx, cy, patch_size=5):
    half = patch_size // 2
    values = []

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for y in range(max(0, cy - half), min(height, cy + half + 1)):
        for x in range(max(0, cx - half), min(width, cx + half + 1)):
            d = depth_frame.get_distance(x, y)    # 해당 픽셀 거리
            if d > 0:
                values.append(d)

    if not values:
        return 0.0

    return float(np.median(values))


# 2D 화면 픽셀 (x, y)와 깊이값 depth_m를 실제 3차원 좌표 (X,Y,Z)로 바꿔준다.
def deproject_to_3d(intrinsics, x, y, depth_m):
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_m)
    return np.array(point_3d, dtype=np.float32)


# 횡단보도 기준선을 한번에 처리하지않고 선 위에 샘플 점 80개를 뿌리는 함수
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


# 3차원 점 p1, p2사이의 실제 거리 계산 (사람과 횡단보도 기준선 거리 계산의 마지막 단계)
def euclidean_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


# =========================
# 메인
# =========================
def main():
    init_db()       # DB테이블 준비

    print("1. 모델 로드 시작")
    model = YOLO(MODEL_PATH)    #학습완료된 YOLO 모델 로드
    print("2. 모델 로드 완료")
    print("클래스 목록:", model.names)

    ser = None
    try:        # 아두이노 연결 성공하면 ser 객체 생성, 연결 실패 시 프로그램 계속 진행
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        print(f"시리얼 연결 성공: {SERIAL_PORT}")
    except Exception as e:
        print(f"시리얼 연결 실패: {e}")
        print("LED 제어 없이 계속 진행합니다.")

    # USB 웹캠 연결 및 해상도/FPS/buffer설정
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

    # D435 컬러/깊이 스트림 둘다 킴
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, CAM_WIDTH, CAM_HEIGHT, rs.format.z16, 30)

    align = rs.align(rs.stream.color) # 깊이 프레임을 컬러 프레임 기준으로 맞춰주기 위한 객체(안 맞추면 화면의 사람위치와 깊이값 위치가 어긋날 수 있다.)

    try:    # D435는 메인카메라 이기 때문에 시작 실패하면 프로그램 종료
        profile = pipeline.start(config)
    except Exception as e:
        print(f"D435 시작 실패: {e}")
        if ser:
            ser.close()
        if aux_cap:
            aux_cap.release()
        return

    # 컬러 카메라 내부 파라미터를 가져온다(이 값이 있어야 픽셀 좌표를 3D 좌표로 바꿀 수 있다.)
    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    #불법주차 / 불법유턴 판단에 사용하는 마스크설정
    occupied_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, CROSSWALK_ROIS + [SIGNAL_ROI] + ROAD_ROIS) #횡단보도,신호등,도로 등 이미 의미있는 영역 마스크
    illegal_zone_mask = cv2.bitwise_not(occupied_mask)  #비어있는 구역
    left_road_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, LEFT_ROAD_ROIS) #왼쪽 차로 전용마스크
    right_road_mask = build_rect_mask(CAM_WIDTH, CAM_HEIGHT, RIGHT_ROAD_ROIS) #오른쪽 차로 전용마스크

    # 오늘 통계 불러오기 (이전에 저장된 최근 사건 이미지 불러오기) -> 프로그램을 껏다켜도 최근 사건정보가 대시보드에 남는다.
    today_count, last_detect_time = load_today_stats()
    emergency_status = "미감지"

    last_jaywalk_capture = load_latest_event_image("jaywalk")
    last_illegal_park_capture = load_latest_event_image("illegal_park")
    last_uturn_capture = load_latest_event_image("illegal_uturn")

    last_capture = None # 최근 캡처 이미지
    capture_mode = None # 현재 캡처가 무단횡단이지 주차인지 유턴인지

    show_capture_fullscreen = False      #지금 캡처 전체화면 띄울지
    show_capture_until = 0               # 캡처화면을 언제까지 보여줄지
    display_distance_cm = None           # 거리정보 표시용

    show_emergency_fullscreen = False   # 비상 전체화면 띄울지

    # 중복 저장 쿨다운 시간 관리
    last_jaywalk_count_time = -9999     
    last_parking_capture_time = -9999
    last_uturn_capture_time = -9999

    ambulance_emergency_until = 0      # 비상차량 감지 후 언제까지 비상 유지할지
    prev_raw_on_detected = False

    frame_count = 0       #몇 프레임 쨰인지

    #추론결과 캐시
    last_boxes = []        
    last_labels = []
    last_scores = []

    # 직전에 보낸 시리얼 명령 기억
    last_d435_led_cmd = None
    last_aux_signal_cmd = None

    # 네오픽셀 유지용 상태
    led_hold_until = 0.0
    led_hold_cmd = "NP_N"


    #USB 웹캠 관련 상태들
    aux_display_frame = None
    aux_red_masks = []
    aux_extra_masks = []
    aux_person_boxes = []
    aux_vehicle_boxes = []
    last_aux_person_boxes = []

    # 경광등 최근 감지 시각, 보조 네오픽셀 명령, 보조 신호등 명령
    aux_hold_state = {
        "last_warning_on_time": 0.0,
        "signal_cmd": "TL_NORMAL",
    }

    illegal_park_state = {} # 차랑별 불법주차 추적 상태 저장용 dict (몇 초 이상 머물렀는지 추척하는데 사용)

    cv2.namedWindow("Dashboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Dashboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("통합 시스템 시작. ESC 종료 / k 캡처화면 숨김 / j 비상해제(표시안함)")

    try:
        while True:

            # D435에서 프레임 획득, 깊이 프레임을 컬러 프레임 기준으로 정렬 (D435는 매 프레임마다 컬러 + 깊이가 함께 필요)
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()

            # USB 웹캠도 프레임 읽기 (D435가 메인이고 USB웹캠은 보조)
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

            # 횡단보도 ROI를 초록색 박스로 화면에 표시
            for cw in crosswalk_rois:
                cv2.rectangle(display_frame, (cw[0], cw[1]), (cw[2], cw[3]), (0, 255, 0), 2)

            rx1, ry1, rx2, ry2 = CROSSWALK_ROIS[0]
            line_y = (ry1 + ry2) // 2
            line_pt1 = (rx1, line_y)
            line_pt2 = (rx2, line_y)
            cv2.line(display_frame, line_pt1, line_pt2, (255, 255, 0), 2)

            # 좌우 차로 ROI표시 (불법유턴 판정 영역)
            for rr in LEFT_ROAD_ROIS:
                cv2.rectangle(display_frame, (rr[0], rr[1]), (rr[2], rr[3]), (255, 255, 0), 1)

            for rr in RIGHT_ROAD_ROIS:
                cv2.rectangle(display_frame, (rr[0], rr[1]), (rr[2], rr[3]), (0, 200, 255), 1)

            # 신호등 ROI를 흰색 박스로 그림
            sx1, sy1, sx2, sy2 = signal_roi
            # 신호등 영역만 잘라서 빨강/초록 여부 판정 (신호등 색 판정)
            signal_crop = frame[sy1:sy2, sx1:sx2]
            signal_state, signal_red_ratio, signal_green_ratio, signal_red_mask, signal_green_mask = judge_traffic_signal(signal_crop)
            signal_color = (0, 0, 255) if signal_state == "RED" else (0, 255, 0) if signal_state == "GREEN" else (200, 200, 200)

            # 현재 신호등 상태와 빨강/초록 비율표시
            cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), signal_color, 2)
            cv2.putText(display_frame, f"SIGNAL: {signal_state}", (sx1, max(sy1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 2)
            cv2.putText(display_frame, f"R={signal_red_ratio:.2f} G={signal_green_ratio:.2f}",
                        (sx1, min(sy2 + 15, h - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, signal_color, 1)

            # YOLO추론 최적화 (모든프레임에서 추론하지 않고 3프레임마다 한번만 추론)
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
                        # 현재 프레임의 탐지 결과를 last_*캐시에 저장하여 추론안하는 프레임에서는 이전 결과를 재사용하는 구조
                        label_name = names[int(cls_id)]
                        last_boxes.append((x1, y1, x2, y2))
                        last_labels.append(label_name)
                        last_scores.append(float(conf))

                # USB웹캠 추론
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

            # 사건 판정용 변수 초기화
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

            # 내부 보조 함수 횡단보도 기준선 샘플점을 프레임마다 한번만 계산, 이미 계산되어 있으면 계산안함
            def prepare_line_points():
                nonlocal line_pixels_cache, line_points_3d_cache
                if line_pixels_cache is not None and line_points_3d_cache is not None:
                    return

                line_pixels = sample_line_points(line_pt1, line_pt2, LINE_SAMPLE_COUNT)
                valid_pixels = []
                line_points_3d = []

                # 각 기준선 샘플점에 대해 depth 측정, 가능하면 3D 좌표로 변환
                for lx, ly in line_pixels:
                    d = get_valid_depth_median(depth_frame, lx, ly, patch_size=PATCH_SIZE)
                    if d > 0:
                        p3d = deproject_to_3d(intr, lx, ly, d)
                        valid_pixels.append((lx, ly))
                        line_points_3d.append(p3d)

                # 계산 결과를 캐시에 저장하고 반환
                line_pixels_cache = valid_pixels
                line_points_3d_cache = line_points_3d

            current_vehicle_keys = set()

            # D435화면에서 탐지된 객채들을 하나씩 처리
            for i, det_box in enumerate(last_boxes):
                x1, y1, x2, y2 = det_box
                label_name = last_labels[i]
                score = last_scores[i]
                label_name_lower = label_name.lower()

                # D435화면에서 경광등 후보가 나오면 실제로 빨간불이 켜졌는지 색 분석으로 다시 확인
                if label_name.upper() in WARNING_CLASSES:
                    roi = frame[y1:y2, x1:x2]
                    final_state, red_ratio, mean_v, _ = judge_warning_light(roi)
                    color = (0, 0, 255) if final_state == "ON" else (255, 0, 0)

                    #ON이면 빨강 박스 + 비상상황 TRUE
                    if final_state == "ON":
                        emergency_detected = True
                    
                    # YOLO 클래스 명과 색 분석 결과를 같이 표시
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"{final_state}", (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(display_frame, f"model:{label_name} {score:.2f}",
                                (x1, min(y2 + 18, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(display_frame, f"R={red_ratio:.3f} V={mean_v:.1f}",
                                (x1, min(y2 + 34, h - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # 무단횡단 핵심
                elif label_name_lower in {c.lower() for c in PERSON_CLASSES}:
                    person_road_ratio_max = 0.0
                    # 사람 박스가 도로 ROI와 얼마나 겹치는지 최대값 계산
                    for road_roi in road_rois:
                        ratio = overlap_ratio(det_box, road_roi)
                        person_road_ratio_max = max(person_road_ratio_max, ratio)

                    person_cross_ratio_max = 0.0
                    # 사람 박스가 횡단보도 ROI와 얼마나 겹치는지 최대값 계산
                    for cw in crosswalk_rois:
                        person_cross_ratio_max = max(person_cross_ratio_max, overlap_ratio(det_box, cw))

                    # 사람이 차량신호 초록불에 횡단보도영역에 1/3 이상 침입, 차량신호 상관없이 횡단보도로 건너지 않고 차도로 건널시 차도 영역에 1/3 침입시 무단횡단
                    is_jaywalking = (person_road_ratio_max >= (1 / 3)) or ((signal_state == "GREEN") and (person_cross_ratio_max >= (1 / 3)))

                    #무단횡단으로 인식되면 빨간 박스
                    if is_jaywalking:
                        jaywalking_detected = True
                        color = (0, 0, 255)
                        text = f"{label_name} JAYWALK"
                    else:
                        color = (0, 255, 255)
                        text = f"{label_name}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    # 거리 계산
                    if is_jaywalking:
                        #사람 중심점 계산
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1) # 그 중심점에 점 표시

                        person_depth = get_valid_depth_median(depth_frame, cx, cy, patch_size=PATCH_SIZE) # 사람중심점 깊이 값 구하기

                        
                        if person_depth > 0:
                            prepare_line_points()
                            
                            # 사람 중심 점을 3D 좌표로 변환
                            if line_points_3d_cache:
                                person_3d = deproject_to_3d(intr, cx, cy, person_depth)

                                min_dist_m = float("inf")
                                closest_line_pixel = None

                                for lpix, l3d in zip(line_pixels_cache, line_points_3d_cache):
                                    dist = euclidean_distance(person_3d, l3d)

                                    # 사람과 기준선 위 샘플 점들 사이 거리중 가장 작은 값 찾기
                                    if dist < min_dist_m:
                                        min_dist_m = dist
                                        closest_line_pixel = lpix

                                dist_cm = min_dist_m * 100.0    # D435 거리값은 meter 단위라서 cm단위로 변환
                                current_min_dist_cm = dist_cm

                                if closest_line_pixel is not None:
                                    cv2.line(display_frame, (cx, cy), closest_line_pixel, (255, 0, 0), 2)
                                    cv2.circle(display_frame, closest_line_pixel, 5, (255, 0, 255), -1)

                                cv2.putText(display_frame, f"{text} {score:.2f} | dist={dist_cm:.1f}cm",    # 거리값 디스플레이에 출력
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

                # 차량 클래스
                elif label_name_lower in {c.lower() for c in VEHICLE_CLASSES}:
                    left_ratio = overlap_ratio_with_mask(det_box, left_road_mask)       #차량이 왼쪽 차로와 얼마나 겹치는지
                    right_ratio = overlap_ratio_with_mask(det_box, right_road_mask)     #차량이 오른쪽 차로와 얼마나 겹치는지
                    illegal_zone_ratio = overlap_ratio_with_mask(det_box, illegal_zone_mask)  # 불법 구역과 얼마나 겹치는지

                    cross_ratio_max = 0.0

                     #횡단보도 침범 정도 계산
                    for cw in crosswalk_rois:
                        cross_ratio_max = max(cross_ratio_max, overlap_ratio(det_box, cw))

                    # 차량 신호등 빨간불에 횡단보도 ROI 1/3 이상 침범 시 차량 침범
                    intrusion_cond = (signal_state == "RED") and (cross_ratio_max >= (1 / 3))
                    if intrusion_cond:
                        vehicle_intrusion_detected = True

                    # 차량이 왼쪽차로와 오른쪽 차로를 동시에 1/3 영역 침범시 불법 유턴 감지
                    illegal_uturn_cond = (left_ratio >= ROAD_RATIO_THRESHOLD) and (right_ratio >= ROAD_RATIO_THRESHOLD)
                    if illegal_uturn_cond:
                        illegal_uturn_detected = True
                        if score > best_uturn_score:
                            best_uturn_score = score
                            best_uturn_box = det_box

                    # 차량 중심점을 20 픽셀 단위격자로 묶어서 간단 추적 key생성, 비슷한 위치에 있는 차를 같은 차량으로 보기 위해서
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    vehicle_key = (cx // 20, cy // 20)
                    current_vehicle_keys.add(vehicle_key)


                    #불법 주정차 계산
                    if illegal_zone_ratio >= ILLEGAL_ZONE_RATIO_THRESHOLD:
                        if vehicle_key not in illegal_park_state:   #처음 보는 차량이라면 start_time 기록
                            illegal_park_state[vehicle_key] = {
                                "start_time": now,
                                "reported": False
                            }   
                        else:   # 이미 보던 차량이면 last_seen 갱신
                            parked_time = now - illegal_park_state[vehicle_key]["start_time"]
                            if parked_time >= ILLEGAL_PARK_SEC: # 불법 주정차 구역에 3초이상 머물렀으면 불법 주차 확정
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


                # fall 클래스
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

            
            # 1초 이상 안보인 차량은 상태 dict에서 제거, 이미 사라진 차가 계속 불법주정차로 남아있을수 있기에 제거
            keys_to_delete = []
            for k, v in illegal_park_state.items():
                if k not in current_vehicle_keys and now - v["start_time"] > 1.0:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                del illegal_park_state[k]

            temp_emergency_for_status = fall_detected or (time.time() < ambulance_emergency_until)

            # 최종 상태 우선순위 결정 1등: 비상/낙상/비상유지시간, 2등 : 무단횡단, 3등:차량침범, 4등:불법유턴, 5등: 불법주차
            # 이에 따라 LED명령도 전달
            if temp_emergency_for_status:
                status_text = "EMERGENCY"
                status_color = (0, 0, 255)
                requested_led_cmd = "NP_N"
                led_hold_cmd = "NP_N"
                led_hold_until = 0.0
            elif jaywalking_detected:
                status_text = "JAYWALKING"
                status_color = (0, 0, 255)
                requested_led_cmd = "NP_R"
                led_hold_cmd = requested_led_cmd
                led_hold_until = now + LED_HOLD_SECONDS
            elif vehicle_intrusion_detected:
                status_text = "VEHICLE INTRUSION"
                status_color = (255, 0, 0)
                requested_led_cmd = "NP_B"
                led_hold_cmd = requested_led_cmd
                led_hold_until = now + LED_HOLD_SECONDS
            elif illegal_uturn_detected:
                status_text = "ILLEGAL U-TURN"
                status_color = (0, 165, 255)
                requested_led_cmd = "NP_B"
                led_hold_cmd = requested_led_cmd
                led_hold_until = now + LED_HOLD_SECONDS
            elif illegal_parking_detected:
                status_text = "ILLEGAL PARKING"
                status_color = (180, 0, 255)
                requested_led_cmd = "NP_B"
                led_hold_cmd = requested_led_cmd
                led_hold_until = now + LED_HOLD_SECONDS
            else:
                status_text = "NORMAL"
                status_color = (0, 255, 0)
                requested_led_cmd = "NP_N"

            if now < led_hold_until:
                d435_led_cmd = led_hold_cmd
            else:
                d435_led_cmd = "NP_N"

            cv2.putText(display_frame, status_text, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
            cv2.putText(display_frame, f"LED={d435_led_cmd}", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # 시리얼 명령 전송
            last_d435_led_cmd = send_serial_command(ser, d435_led_cmd, last_d435_led_cmd) # 메인 상태를 D435용 LED명령으로 전송
            last_aux_signal_cmd = send_serial_command(ser, aux_hold_state["signal_cmd"], last_aux_signal_cmd)   # USB웹캠 쪽 네오픽셀 명령 전송, 신호등전체점등/정상복귀 명령 전송
            drain_serial_feedback(ser)  

            key = cv2.waitKey(1) & 0xFF

            if key == ord('k'):
                show_capture_fullscreen = False
                show_capture_until = 0

            if key == ord('j'):
                show_emergency_fullscreen = False
                emergency_status = "미감지"
                ambulance_emergency_until = 0

            now = time.time()

            # 비상 유지 로직
            raw_on_detected = emergency_detected
            raw_fall_detected = fall_detected
            
            # 경광등이 새로 켜진(ON처음 감지)에만 15초타이머 시작 (계속 켜져있는동안 무한으로 늘어나지 않도록)
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

            # 무단횡단자 캡처 (지금 비상상태가 아니면서, 무단횡단이 감지되었을때, 마지막 무단횡단 감지 이후 10초가 지났을것->중복저장 막기)
            if (not emergency_now) and jaywalking_detected and aux_frame is not None:
                if now - last_jaywalk_count_time >= JAYWALK_COUNT_COOLDOWN:
                    boxes_for_crop = aux_person_boxes if len(aux_person_boxes) > 0 else last_aux_person_boxes
                    aux_crop = crop_best_aux_person(aux_frame, boxes_for_crop, margin=20)

                    # 가능하면 USB웹캠에서 사람위주 크롭저장, 실패하면 D435전체화면 저장
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

                    # 사건 기반 파일명 생성 (무단횡단 폴더에 이미지 저장)
                    filename = os.path.join(JAYWALK_DIR, f"jaywalk_{int(now)}.jpg")
                    cv2.imwrite(filename, capture_img)
                    print(f"[무단횡단자 캡처 저장 - 추가 웹캠] {filename}")

                    #DB 로그에 저장, 오늘통계증가
                    save_event("jaywalk", filename)
                    update_jaywalk_daily_stats()
                    today_count, last_detect_time = load_today_stats()

            # 불법 주정차 저장 (D435화면에서 크롭 저장), 사건로그 저장
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

            #불법 유턴 저장 (D435화면에서 크롭 저장), 사건로그 저장
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

            #화면 모드 비상이면 무조건 비상전체화면 우선
            if show_emergency_fullscreen:
                screen = make_emergency_fullscreen(display_frame)

            # 비상이 아니고 캡처화면 표시 중이면 시간이 다됐으면 다시 대시보드 복귀, 아직 시간이 남았으면 사건별 제목으로 전체화면 표시
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
            #아무 이벤트 발생이 아니라면 기본 대시보드 표시
            else:
                screen = make_dashboard(
                    last_jaywalk_capture,
                    last_illegal_park_capture,
                    last_uturn_capture,
                    aux_display_frame,
                    display_frame,
                )

            cv2.imshow("Dashboard", screen) #최종 선택된 화면 출력

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
        if ser:         #종료할 때 아두이노를 원상태로 복귀
            try:
                ser.write(b"NP_N\n")
                ser.write(b"TL_NORMAL\n")
                time.sleep(0.1)
                ser.close()
            except Exception:
                pass

        if aux_cap: # USB 웹캠 해제
            aux_cap.release()

        pipeline.stop() #D435 파이프라인 종료
        cv2.destroyAllWindows()
        print("종료 완료")


if __name__ == "__main__":
    main()
