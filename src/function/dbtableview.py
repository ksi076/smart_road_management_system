import sqlite3

DB_PATH = "/home/rapi20/workspace/jaywalk_monitor.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# =========================
# 1. event_logs 테이블 없으면 생성
# =========================
cursor.execute("""
    CREATE TABLE IF NOT EXISTS event_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        detected_at TEXT NOT NULL,
        image_path TEXT
    )
""")

# =========================
# 2. daily_stats 테이블 없으면 생성
# =========================
cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_stats (
        stat_date TEXT PRIMARY KEY,
        jaywalk_count INTEGER NOT NULL DEFAULT 0,
        last_detect_time TEXT
    )
""")

# =========================
# 3. event_date 컬럼 존재 여부 확인
# =========================
cursor.execute("PRAGMA table_info(event_logs)")
columns = [row[1] for row in cursor.fetchall()]

if "event_date" not in columns:
    cursor.execute("ALTER TABLE event_logs ADD COLUMN event_date TEXT")
    print("event_date 컬럼 추가 완료")
else:
    print("event_date 컬럼이 이미 존재함")

# =========================
# 4. 기존 데이터의 event_date 채우기
# =========================
cursor.execute("""
    UPDATE event_logs
    SET event_date = substr(detected_at, 1, 10)
    WHERE event_date IS NULL OR event_date = ''
""")

conn.commit()

# =========================
# 5. 조회 날짜 입력
# 엔터면 오늘 날짜
# =========================
input_date = input("조회할 날짜를 입력하세요 (YYYY-MM-DD, 엔터=오늘): ").strip()

if input_date == "":
    cursor.execute("SELECT date('now', 'localtime')")
    selected_date = cursor.fetchone()[0]
else:
    selected_date = input_date

# =========================
# 6. 조회 이벤트 종류 입력
# =========================
print("\n조회할 이벤트를 선택하세요:")
print("1. jaywalk        (무단횡단)")
print("2. illegal_park   (불법주차)")
print("3. illegal_uturn  (불법유턴)")
print("4. all            (전체)")

event_input = input("번호 또는 이름 입력: ").strip().lower()

event_map = {
    "1": "jaywalk",
    "2": "illegal_park",
    "3": "illegal_uturn",
    "4": "all",
    "jaywalk": "jaywalk",
    "illegal_park": "illegal_park",
    "illegal_uturn": "illegal_uturn",
    "all": "all"
}

selected_event = event_map.get(event_input, "all")

print("\n" + "=" * 80)
print(f"📅 조회 날짜: {selected_date}")
print(f"📌 조회 이벤트: {selected_event}")
print("=" * 80)

# =========================
# 7. 무단횡단 통계 출력
# daily_stats는 jaywalk만 있으므로 날짜 기준으로만 표시
# =========================
cursor.execute("""
    SELECT jaywalk_count
    FROM daily_stats
    WHERE stat_date = ?
""", (selected_date,))
row = cursor.fetchone()

selected_count = row[0] if row else 0
print(f"🔥 {selected_date} 무단횡단자 수(daily_stats 기준): {selected_count}")
print("=" * 80)

# =========================
# 8. 조건에 맞는 로그 조회
# =========================
if selected_event == "all":
    cursor.execute("""
        SELECT id, event_type, detected_at, event_date, image_path
        FROM event_logs
        WHERE event_date = ?
        ORDER BY id DESC
    """, (selected_date,))
else:
    cursor.execute("""
        SELECT id, event_type, detected_at, event_date, image_path
        FROM event_logs
        WHERE event_date = ? AND event_type = ?
        ORDER BY id DESC
    """, (selected_date, selected_event))

rows = cursor.fetchall()

print(f"📋 {selected_date} / {selected_event} 로그 목록")
print("-" * 80)

if not rows:
    print("해당 조건의 저장된 로그가 없습니다.")
else:
    for row in rows:
        print(row)

print("=" * 80)

# =========================
# 9. event_logs 기준 개수 출력
# =========================
if selected_event == "all":
    cursor.execute("""
        SELECT event_type, COUNT(*)
        FROM event_logs
        WHERE event_date = ?
        GROUP BY event_type
        ORDER BY event_type
    """, (selected_date,))

    counts = cursor.fetchall()

    print("📊 이벤트별 개수")
    print("-" * 80)

    if not counts:
        print("해당 날짜의 로그가 없습니다.")
    else:
        for event_type, count in counts:
            print(f"{event_type}: {count}건")
else:
    cursor.execute("""
        SELECT COUNT(*)
        FROM event_logs
        WHERE event_date = ? AND event_type = ?
    """, (selected_date, selected_event))

    count = cursor.fetchone()[0]
    print(f"📊 {selected_event} 개수: {count}건")

print("=" * 80)

conn.close()
