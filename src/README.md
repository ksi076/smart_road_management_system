## 🧠 핵심 기능 설명

---

### 1️⃣ 시스템 초기화 및 통합 제어 구조

- **핵심 코드**

 ```python
 # =========================
# 시스템 초기화
# =========================

# DB 초기화
init_db()

# YOLO 모델 로드
model = YOLO(MODEL_PATH)

# 아두이노 시리얼 연결
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

# 보조 카메라 초기화 (USB 웹캠)
aux_cap = open_aux_camera(AUX_CAM_INDEX)

# RealSense D435 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
```
- **설명**
  - YOLO ONNX 모델, Intel RealSense D435, USB 웹캠, SQLite DB, Arduino 시리얼 통신을 초기화한다.


---

### 2️⃣ ROI 기반 영역 설정

  
- **핵심 코드**

 ```python
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
```
- **설명**
  - 도로, 횡단보도, 신호등 영역을 ROI(Region of Interest) 좌표로 직접 설정하였다.
  - 이후 YOLO로 검출된 사람 및 차량 객체와 ROI의 겹침 비율을 계산하여 무단횡단, 차량 침범, 불법 유턴 등의 이벤트를 위치 기반으로 판정하였다.

---

### 3️⃣ SQLite 기반 이벤트 저장 구조

- **핵심 코드**
```python
# =========================
# SQLite 기반 이벤트 저장 구조
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 이벤트 로그 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            image_path TEXT
        )
    """)

    # 일별 무단횡단 통계 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            stat_date TEXT PRIMARY KEY,
            jaywalk_count INTEGER NOT NULL DEFAULT 0,
            last_detect_time TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_event(event_type, detected_at, image_path=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO event_logs (event_type, detected_at, image_path)
        VALUES (?, ?, ?)
    """, (event_type, detected_at, image_path))

    conn.commit()
    conn.close()


def update_jaywalk_daily_stats(detected_at):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 날짜 추출 (YYYY-MM-DD)
    date = detected_at[:10]

    cursor.execute("""
        SELECT jaywalk_count FROM daily_stats WHERE stat_date = ?
    """, (date,))
    row = cursor.fetchone()

    if row:
        # 기존 데이터 있으면 count 증가
        cursor.execute("""
            UPDATE daily_stats
            SET jaywalk_count = jaywalk_count + 1,
                last_detect_time = ?
            WHERE stat_date = ?
        """, (detected_at, date))
    else:
        # 없으면 새로 생성
        cursor.execute("""
            INSERT INTO daily_stats (stat_date, jaywalk_count, last_detect_time)
            VALUES (?, 1, ?)
        """, (date, detected_at))

    conn.commit()
    conn.close()
```
- **설명**
  - SQLite DB를 활용하여 이벤트 로그와 일별 통계를 저장하는 구조를 설계하였다.
  - event_logs 테이블에는 이벤트 종류, 발생 시각, 이미지 경로를 저장하고,
  - daily_stats 테이블에는 날짜별 무단횡단 횟수와 마지막 감지 시간을 관리하였다.
  - 이를 통해 실시간 감지뿐 아니라 사후 로그 조회 및 통계 분석이 가능하도록 구현하였다.
  - 이벤트 발생 시 DB 저장과 동시에 이미지 파일 경로를 함께 기록하여, 추후 이벤트 이미지 조회 기능과 연동되도록 구성하였다.
---

### 4️⃣ OpenCV 기반 신호등/경광등 상태 판별

- **핵심 코드**
 ```python
# =========================
# OpenCV 기반 신호등 / 경광등 상태 판별
# =========================

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
```

- **설명**
  - OpenCV의 HSV 색 공간 변환과 마스크 연산을 활용하여 신호등과 경광등의 실제 점등 상태를 판별하였다.
  - YOLO 객체 검출 결과만 사용하는 것이 아니라, ROI 내부의 색상 비율을 추가로 분석하여 빨간불, 초록불, 경광등 ON 상태를 구분하도록 구현하였다.
  - 이를 통해 단순 객체 검출을 넘어 상태 기반 이벤트 판단이 가능하도록 설계하였다.

---

### 5️⃣ 무단횡단 판정 로직

- **무시 영역 설정:**
  ○ ignore_zone 함수로 특정 영역 사람 감지 제외  

- **중복 방지:**
  ○ duplicate_limit으로 일정 시간 내 중복 카운트 방지  

- **비상 모드:**
  ○ emergency_mode 실행 시 15초 유지  
  ○ 'J' 키 입력 시 해제  

- **오류 처리:**
  ○ 카메라 오류 및 파일 저장 오류 대응


---

### 6️⃣ D435 기반 거리 측정

- **화면 출력:**
  ○ draw_box 함수로 bbox 및 라벨 표시  

- **이미지 저장:**
  ○ save_image 함수로 이벤트 이미지 저장  

- **DB 저장:**
  ○ save_db 함수로 로그 기록  
    ■ 시간 (detected_at)  
    ■ 이벤트 종류 (event_type)  
    ■ 이미지 경로 (image_path)  

---


### 7️⃣ 차량 이벤트 판정 로직

- **화면 출력:**
  ○ draw_box 함수로 bbox 및 라벨 표시  

- **이미지 저장:**
  ○ save_image 함수로 이벤트 이미지 저장  

- **DB 저장:**
  ○ save_db 함수로 로그 기록  
    ■ 시간 (detected_at)  
    ■ 이벤트 종류 (event_type)  
    ■ 이미지 경로 (image_path)  

---


### 8️⃣ fall 감지 및 비상상황 판정 로직

- **화면 출력:**
  ○ draw_box 함수로 bbox 및 라벨 표시  

- **이미지 저장:**
  ○ save_image 함수로 이벤트 이미지 저장  

- **DB 저장:**
  ○ save_db 함수로 로그 기록  
    ■ 시간 (detected_at)  
    ■ 이벤트 종류 (event_type)  
    ■ 이미지 경로 (image_path)  

---


### 9️⃣ 대시보드 및 Arduino 연동

- **화면 출력:**
  ○ draw_box 함수로 bbox 및 라벨 표시  

- **이미지 저장:**
  ○ save_image 함수로 이벤트 이미지 저장  

- **DB 저장:**
  ○ save_db 함수로 로그 기록  
    ■ 시간 (detected_at)  
    ■ 이벤트 종류 (event_type)  
    ■ 이미지 경로 (image_path)  

---
  
