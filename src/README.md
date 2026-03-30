## 📄 best.onnx파일

---

### [best.onnx](https://drive.google.com/file/d/1uawiLiPnFgE_XmMgUQVAdI_l8eMrahd5/view?usp=drive_link)


### 📦 Model Info

- Format: ONNX
- Model: YOLOv8s
- Input Size: 320 x 320
- Channels: 3 (RGB)
- Batch Size: 1
- Parameters: 11M
- Task: Object Detection

### 🎯 Classes
- vehicle
- person
- fall
- ON
- OFF
- carnight
- personnight
- fallnight
  
### ⚡ 특징
- Raspberry Pi 환경에 최적화된 경량 모델
- 실시간 객체 탐지 가능
- YOLOv8s 기반 커스텀 학습 모델


---


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

- **핵심 코드**

```python
# =========================
# 무단횡단 판정 로직
# =========================

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
```
- **설명**
  - 사람 객체가 치량 신호 초록불에 횡단보도에 검출되면 횡단보도 ROI와의 겹침 비율을  계산하였고 신호와 상관없이 횡단보도가 아닌 도로에 침범시 도로 ROI와의 겹침비율을 계산하였다.
  - 이후 겹침 비율이 기준값 이상인 경우를 무단횡단 상황으로 판정하도록 구현하였다.
  - 이를 통해 단순히 사람을 검출하는 것이 아니라, 객체의 위치 정보를 바탕으로 실제 도로 위반 상황을 해석하도록 설계하였다.
---

### 6️⃣ D435 기반 거리 측정

- **핵심 코드**

```python
# =========================
# D435 기반 거리 측정
# =========================

 # 거리 계산
                    if is_jaywalking:
                        #사람 중심점 계산
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1) # 그 중심점에 점 표시

                        person_depth = get_valid_depth_median(depth_frame, cx, cy, patch_size=PATCH_SIZE) # 사람중심점 깊이 값 구하기

                        
                        if person_depth > 0: #depth값 유효성 체크
                            prepare_line_points() # 기준선 준비
                            
                            # 사람 중심 점을 3D 좌표로 변환
                            if line_points_3d_cache:
                                person_3d = deproject_to_3d(intr, cx, cy, person_depth)

                                min_dist_m = float("inf")
                                closest_line_pixel = None

                                for lpix, l3d in zip(line_pixels_cache, line_points_3d_cache): # 최소 거리 찾기 (사람 vs 기준선 각 점 거리 계산)
                                    dist = euclidean_distance(person_3d, l3d)

                                    # 사람과 기준선 위 샘플 점들 사이 거리중 가장 작은 값 찾기
                                    if dist < min_dist_m:
                                        min_dist_m = dist
                                        closest_line_pixel = lpix

                                dist_cm = min_dist_m * 100.0    # D435 거리값은 meter 단위라서 cm단위로 변환
                                current_min_dist_cm = dist_cm
```
- **설명**
 - **설명**
  - Intel RealSense D435의 깊이 정보를 활용하여 사람 객체 중심점의 실제 거리를 계산하였다.
  - 중심점의 depth 값을 구한 뒤, 2D 픽셀 좌표를 3D 공간 좌표로 변환하고, 횡단보도 기준선 샘플 점들과의 유클리드 거리를 비교하여 최소 거리를 산출하였다.
  - 이를 통해 무단횡단 상황을 단순 검출이 아닌 실제 거리(cm) 정보와 함께 표시할 수 있도록 구현하였다.
---


### 7️⃣ 차량 이벤트 판정 로직

- **핵심 코드**

```python
# =========================
# 차량 침범 이벤트 판정
# =========================

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
    illegal_zone_ratio = 0.0
    for roi in CROSSWALK_ROIS:
        illegal_zone_ratio = max(illegal_zone_ratio, overlap_ratio(vehicle_box, roi))

# =========================
# 차량 불법 유턴 이벤트 판정
# =========================
   # 차량이 왼쪽차로와 오른쪽 차로를 동시에 1/3 영역 침범시 불법 유턴 감지
                    illegal_uturn_cond = (left_ratio >= ROAD_RATIO_THRESHOLD) and (right_ratio >= ROAD_RATIO_THRESHOLD)
                    if illegal_uturn_cond:
                        illegal_uturn_detected = True
                        if score > best_uturn_score:
                            best_uturn_score = score
                            best_uturn_box = det_box
# =========================
# 차량 불법 주정차 이벤트 판정
# =========================
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
```
- **설명**
  - 차량 객체가 차량신호 빨간불일 경우에 횡단보도 ROI와의 겹침비율을 계산하였다.
  - 차량 객체가 검출되면 좌측 차로, 우측 차로 ROI와의 겹침 비율을 각각 계산하였다.
  - 이를 기반으로 차량 침범, 불법 유턴, 불법주차를 구분하여 판정하도록 구현하였다.
  - 특히 불법주차는 특정 구역에 일정 시간 이상 머무는 경우로 정의하여 단순 위치 검출을 넘어 시간 기반 이벤트 판단이 가능하도록 설계하였다.


---


### 8️⃣ fall 감지 및 비상상황 유지 로직

- **핵심 코드**

```python
# =========================
# 낙상 감지
# =========================

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

# =========================
# 비상상황 유지
# =========================
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
# =========================
# 경광등 ON 판정시 차량신호 3개 모두 점등
# =========================
warning_override_active = (now - hold_state["last_warning_on_time"]) <= WARNING_HOLD_SECONDS # 방금 ON이 아니더라도 마지막 ON시간으로부터 3초 이내면 비상상태 유지
    hold_state["signal_cmd"] = "TL_ALL" if warning_override_active else "TL_NORMAL" #경광등 유지 중이면 TL_ALL, 아니면 TL_NORMAL

    status_text = "USB EMERGENCY LIGHT ON" if emergency_detected_now else "USB NORMAL"
    status_color = (0, 0, 255) if emergency_detected_now else (0, 255, 0)
    cv2.putText(out, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

```
- **설명**
  - fall 클래스가 검출될 경우와 경광등 ON을 감지할 경우 비상상황으로 판정한다.
  - 경광등 ON 감지 시 비상상황 전환 및 차량신호 3개를 모두 점등하여 비상상황임을 인지시킬 수 있게 한다.

---



  
