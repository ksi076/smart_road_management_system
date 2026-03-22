## 🧠 기능 세분화

---

### 1️⃣ 시스템 초기화

- **load_model, init_camera, init_db, create_directory**
  ○ YOLO 모델, 카메라, DB, 저장 폴더 초기화.

- **초기 상태:**
  ○ 카메라 ON 상태  
  ○ 시스템 대기 상태  

---

### 2️⃣ 입력 처리

- **카메라 프레임 입력:**
  ○ read_frame 함수로 실시간 영상 획득  

- **전처리:**
  ○ resize_frame으로 YOLO 입력 크기 조정  

---

### 3️⃣ 객체 및 위반 감지

- **객체 탐지:**
  ○ predict 함수로 YOLO 추론 수행  
  ○ 사람(person), 차량(car, bus, truck) 인식  

- **무단 횡단 감지:**
  ○ check_jaywalk 함수로 제한 구역 내 사람 확인  

- **불법 주정차 감지:**
  ○ check_illegal_parking 함수로 일정 시간 정지 차량 판단  

- **불법 유턴 감지:**
  ○ check_illegal_uturn 함수로 이동 방향 변화 분석  

---

### 4️⃣ 결과 처리

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

### 5️⃣ 예외 처리 및 추가 기능

- **무시 영역 설정:**
  ○ ignore_zone 함수로 특정 영역 사람 감지 제외  

- **중복 방지:**
  ○ duplicate_limit으로 일정 시간 내 중복 카운트 방지  

- **비상 모드:**
  ○ emergency_mode 실행 시 15초 유지  
  ○ 'J' 키 입력 시 해제  

- **오류 처리:**
  ○ 카메라 오류 및 파일 저장 오류 대응  
