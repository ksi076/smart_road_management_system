# 🚦 스마트 교통 관리 시스템 (Smart Traffic Management System)

스마트 교통 관리 시스템 프로젝트는 라즈베리파이5 환경에서 YOLO 기반 객체 인식과 OpenCV 영상처리를 활용하여  
도로 상황을 실시간으로 분석하고, 무단횡단 / 불법 주정차 / 불법 유턴 / 낙상사고 / 비상상황 등의 교통 위반 이벤트를 감지 및 저장하는 통합 시스템입니다.

---

## 📌 프로젝트 개요

- **수행 기간:** 2025.03.06 ~ 2025.03.21
- **사용 기술:**
  - Python
  - YOLOv8
  - OpenCV
  - Raspberry Pi5
  - SQLite
  - Arduino IDE
- **주요 기능:**
  ![구성도](img/구성도1.png)
  - 무단횡단 감지
  - 불법 주정차 감지
  - 불법 유턴 감지
  - 낙상 감지
  - 긴급 차량 처리
  - 이벤트 이미지 저장
  - 로그 데이터 기록 및 일별 통계 관리
 
  


---

## 🛠 기술 스택

| 기술 | 설명 |
|------|------|
| Python | 전체 시스템 로직 및 영상 처리 구현 |
| YOLOv8 | 객체 탐지 모델 기반 보행자/차량 인식 |
| OpenCV | 영상 프레임 처리, 시각화, 이미지 저장 |
| Raspberry Pi5 | 임베디드 환경에서 실시간 시스템 구동 |
| XPT2046 Touch Controller | 임베디드 환경에서 실시간 시스템 구동 |
| SQLite | 이벤트 로그 및 일별 통계 저장 |
| Serial Communication | 외부 장치와의 상태 제어 및 연동 |
| Arduino IDE| 외부 장치와의 상태 제어 및 연동 |

---

## 📋 기능 설명

### 1. 무단횡단 감지
- 보행자와 신호 상황을 기반으로 무단횡단 여부 판단
- 이벤트 발생 시 이미지 저장 및 DB 기록
- 동일 인물 반복 인식에 따른 중복 카운트를 방지하기 위해 쿨타임 적용

### 2. 불법 주정차 감지
- 차량이 지정 구역 내 일정 시간 이상 정차할 경우 이벤트 처리
- 위반 차량 이미지 저장 및 로그 기록

### 3. 불법 유턴 감지
- 차량 이동 방향과 관심 구역을 기준으로 불법 유턴 여부 판단
- 감지 결과를 이미지와 함께 저장

### 4. 긴급 차량 처리
- 긴급 차량 인식 시 비상 상태를 일정 시간 유지
- 일반 이벤트 처리와 구분하여 예외적으로 동작

### 5. 데이터 저장 및 통계 관리
- 이벤트 발생 시 발생 시간, 이미지 경로, 이벤트 종류를 저장
- 날짜별 누적 통계를 관리하여 추후 분석 가능



---
##  실물 사진

![실물사진](img/실물사진.png)

---




## 🧠 시스템 구성도

![시스템 구성도](img/구성도2.png)


---


## 📂 소스 코드

### [소스코드 바로가기](https://github.com/ksi076/smart_road_management_system/tree/main/src)

---

## 🎥 시연 영상

### [인도 무단횡단 감지 시연](https://drive.google.com/file/d/1JJZ4wy2REE9QvrCth4uMI0Oh-UzQre7v/view?usp=sharing)
[![인도무단횡단](gif/display/횡단보도무단횡단.gif)]

### [차도 무단횡단 감지 시연](https://drive.google.com/file/d/10VPleeBBzlbaidgrZ4XxjRO3DYnDbJa4/view?usp=sharing)
![차도무단횡단](gif/display/차도무단횡단.gif)

### [불법 주정차 감지 시연](https://drive.google.com/file/d/1wICn6sA5SGs-cMUMmPEFmAYt1xEubBA2/view?usp=sharing)
![불법주정차](gif/display/불법주정차2.gif)

### [불법 유턴 감지 시연](https://drive.google.com/file/d/1-yff9gF1twIYAe5XEUdBGuQiPEu5qhGJ/view?usp=sharing)
![불법유턴](gif/display/불법유턴1.gif)

### [차량 횡단보도 침범](https://drive.google.com/file/d/1e-4tieU3bb9hKjmdHmfrGHj2JFM-pdN3/view?usp=sharing)
![차량침범](gif/display/차량횡단보도침범.gif)

### [긴급상황 사고](https://drive.google.com/file/d/11_sgPJO63pYdR7drzoCO-xOwAlElfMGV/view?usp=sharing)
![긴급상황사고](gif/display/긴급상황사고최종.gif)

### [긴급 차 비켜주기](https://drive.google.com/file/d/1XEe5XvLOEKhPmtaGWWo1Pxdk5H6INKlp/view?usp=sharing)
![긴급차비킴](gif/display/긴급차비켜주기.gif)

---

## ⚠️ 문제 해결 과정 (Trouble Shooting)

### 🚦 신호등을 사람으로 잘못 인식하는 문제

<p>
<img src="img/신호등트러블슈팅.png" width="300"/>
<img src="img/신호등트러블슈팅해결.png" width="300"/>
</p>

- **문제:** 빨간 사람 학습 후 신호등의 빨간 신호를 실제 사람으로 잘못 탐지  
- **해결:** 특정 ROI 영역 안의 person 감지를 continue하여 오탐 방지  

### car클래스를 밤에 인식하지 못하는 문제

<p>
<img src="img/car트러블슈팅.png" width="300"/>
<img src="img/car트러블슈팅해결.png" width="300"/>
</p>

- **문제:** 낮과 밤을 묶어 car 클래스를 학습시킨 결과 밤에 car를 인식하지 못함  
- **해결:** 낮과 밤을 클래스로 나눠 학습하여 해결 → vehicle, carnigh 클래스로 분류  

### 사람을 인식하지 못하는 문제

<p>
<img src="img/욜로모델변경, 파일변경.png" width="300"/>
</p>

- **문제:** 카메라 2대를 사용하기 때문에 카메라 속도 유지를 위해 YOLO5n을 사용하자 인식하지 못함  
- **해결:** 학습 완료된 best.pt를 best.onnx파일로 교체 후 YOLO8s로 학습함  
---

## 📈 향후 개선 방향

- 번호판 인식 기능 추가
- 웹 대시보드 구축
- 다중 카메라 연동
- 클라우드 기반 통합 모니터링 시스템 확장
