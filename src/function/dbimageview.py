import sqlite3  # sqlite DB 연결 및 SQL 실행
import cv2      # 이미지 읽기 및 화면 출력(OpenCV)
import os       # 파일 경로 존재 여부 확인

DB_PATH = "/home/rapi20/workspace/jaywalk_monitor.db"   # SQLite DB 파일의 실체 위치 저장

# =========================
# 1. 이벤트 종류 입력
# =========================
event_type = input("이벤트 종류 입력 (jaywalk / illegal_park / illegal_uturn): ").strip()
# 사용자한테 콘솔에서 문자열을 입력받는다.
# 조회하고 싶은 이벤트 종류 직접 입력가능


if event_type not in ["jaywalk", "illegal_park", "illegal_uturn"]:
    print("잘못된 이벤트 종류입니다.")
    exit()

# =========================
# 2. 날짜 입력
# 엔터만 누르면 오늘 날짜
# =========================
input_date = input("조회할 날짜 입력 (YYYY-MM-DD, 엔터=오늘): ").strip()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

if input_date == "":
    cursor.execute("SELECT date('now', 'localtime')")
    selected_date = cursor.fetchone()[0]
else:
    selected_date = input_date

print(f"\n조회 이벤트: {event_type}")
print(f"조회 날짜: {selected_date}")
print("=" * 80)

# =========================
# 3. event_date 컬럼 있는지 확인
# 없으면 detected_at 기준으로 날짜 필터
# =========================
cursor.execute("PRAGMA table_info(event_logs)")   # SQLite에서 테이블 구조를 조회하는 명령어(event_logs 테이블의 컬럼 목록을 가져온다.)
columns = [row[1] for row in cursor.fetchall()]   # fetchall()로 모든 컬럼 정보를 가져오고 그 중 row[1]만 뽑아 컬럼 이름 리스트 생성
# DB구조 변경에도 대응 가능함

if "event_date" in columns:
    cursor.execute("""
        SELECT detected_at, image_path
        FROM event_logs
        WHERE event_type = ? AND event_date = ?
        ORDER BY id DESC
    """, (event_type, selected_date))

    #  event_logs 테이블에 event_data라는 컬럼이 있으면
    #  감지시각과 이미지 경로 가져옴
    #  event_logs 테이블에서 조회
    #  사건종류와 날짜가 모두 일치하는 데이터만 가져옴
    #  최신 데이터가 먼저 오도록 내림차순 정렬


    # event_data가 없을 경우
else:
    cursor.execute("""
        SELECT detected_at, image_path
        FROM event_logs
        WHERE event_type = ? AND substr(detected_at, 1, 10) = ?
        ORDER BY id DESC
    """, (event_type, selected_date))

    # detected_at 문자열에서 날짜 부분만 잘라서 비교 (evnet_data라는 컬럼을 따로 만들지 않았을 경우에도 날짜조회가 가능하도록 처리)

rows = cursor.fetchall()    #조회된 결과를 전부 리스트로 가져옴
conn.close()

if not rows:
    print("조회된 이미지가 없습니다.")
    exit()

# =========================
# 4. 이미지 출력
# =========================
for i, (detected_at, image_path) in enumerate(rows, start=1):   #조회된 결과를 하나씩 반복, enumrate : 번호를 1부터 붙여서 출력하기 위해 사용
    print(f"{i}. [{detected_at}] {image_path}")  #콘솔에서 몇번째 이미지인지, 언제감지됐는지, 경로가 무엇인지 출력

    #이미지 파일 존재 여부 확인
    if image_path is None or not os.path.exists(image_path):
        print("이미지 파일 없음:", image_path)
        continue

    #파일이 있어도 손상됐거나 읽을수 없는 경우 건너뛰고 다음으로 이동
    img = cv2.imread(image_path)    
    if img is None:
        print("이미지 읽기 실패:", image_path)
        continue

    title = f"{event_type} | {selected_date} | {i}/{len(rows)}" #창 제목 만들기
    cv2.imshow(title, img) #실제 이미지 화면 출력

    # 사용자가 키를 누르기 전까지다음 이미지로 넘어가지 않음
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    # ESC 누르면 종료
    if key == 27:
        break
