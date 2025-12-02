import sqlite3
import datetime

# GANTI NAMA DB AGAR STRUKTUR BARU TER-LOAD OTOMATIS
DB_NAME = "sahamwajar_social.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            device_id TEXT PRIMARY KEY,
            email TEXT,
            username TEXT,
            avatar TEXT,
            tokens INTEGER DEFAULT 3,
            last_claim_date TEXT,
            is_premium INTEGER DEFAULT 0,
            join_date TEXT
        )
    ''')
    
    # UPDATE: Tambah kolom parent_id untuk fitur Reply
    c.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            user_id TEXT,
            username TEXT,
            avatar TEXT,
            content TEXT,
            parent_id INTEGER, 
            timestamp DATETIME,
            likes INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# --- USER MANAGEMENT ---
def get_user(device_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE device_id=?", (device_id,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(device_id, email=None, username="Guest"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.date.today().isoformat()
    try:
        c.execute("INSERT INTO users (device_id, email, username, tokens, last_claim_date, join_date) VALUES (?, ?, ?, 3, ?, ?)",
                  (device_id, email, username, now, now))
        conn.commit()
    except: pass
    conn.close()

def update_token(device_id, amount):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT tokens FROM users WHERE device_id=?", (device_id,))
    cur = c.fetchone()
    if cur:
        new_bal = cur[0] + amount
        c.execute("UPDATE users SET tokens=? WHERE device_id=?", (new_bal, device_id))
        conn.commit()
    conn.close()

def login_user_google(device_id, email, username, avatar):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("UPDATE users SET email=?, username=?, avatar=?, is_premium=1, tokens=999 WHERE device_id=?", 
              (email, username, avatar, device_id))
    conn.commit()
    conn.close()

def check_monthly_reset(device_id):
    pass 

# --- COMMENT SYSTEM PRO (NESTED) ---
def add_comment(ticker, device_id, username, avatar, content, parent_id=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") # Format jam lebih pendek
    c.execute("INSERT INTO comments (ticker, user_id, username, avatar, content, parent_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (ticker, device_id, username, avatar, content, parent_id, ts))
    conn.commit()
    conn.close()

def get_comments(ticker):
    """
    Mengambil semua komentar dan mengurutkannya:
    Parent di atas, Child (Reply) di bawahnya.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Ambil semua komentar untuk ticker ini
    c.execute("SELECT id, username, avatar, content, timestamp, parent_id FROM comments WHERE ticker=? ORDER BY timestamp DESC", (ticker,))
    rows = c.fetchall()
    conn.close()
    
    # Konversi ke List of Dicts agar mudah diproses di UI
    comments = []
    for r in rows:
        comments.append({
            "id": r[0], "username": r[1], "avatar": r[2], 
            "content": r[3], "timestamp": r[4], "parent_id": r[5]
        })
    return comments