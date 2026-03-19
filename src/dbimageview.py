import sqlite3
import cv2
import os

DB_PATH = "/home/rapi20/workspace/jaywalk_monitor.db"

# =========================
# 1. 이벤트 종류 입력
# =========================
event_type = input("이벤트 종류 입력 (jaywalk / illegal_park / illegal_uturn): ").strip()

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
cursor.execute("PRAGMA table_info(event_logs)")
columns = [row[1] for row in cursor.fetchall()]

if "event_date" in columns:
    cursor.execute("""
        SELECT detected_at, image_path
        FROM event_logs
        WHERE event_type = ? AND event_date = ?
        ORDER BY id DESC
    """, (event_type, selected_date))
else:
    cursor.execute("""
        SELECT detected_at, image_path
        FROM event_logs
        WHERE event_type = ? AND substr(detected_at, 1, 10) = ?
        ORDER BY id DESC
    """, (event_type, selected_date))

rows = cursor.fetchall()
conn.close()

if not rows:
    print("조회된 이미지가 없습니다.")
    exit()

# =========================
# 4. 이미지 출력
# =========================
for i, (detected_at, image_path) in enumerate(rows, start=1):
    print(f"{i}. [{detected_at}] {image_path}")

    if image_path is None or not os.path.exists(image_path):
        print("이미지 파일 없음:", image_path)
        continue

    img = cv2.imread(image_path)
    if img is None:
        print("이미지 읽기 실패:", image_path)
        continue

    title = f"{event_type} | {selected_date} | {i}/{len(rows)}"
    cv2.imshow(title, img)

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    # ESC 누르면 종료
    if key == 27:
        break