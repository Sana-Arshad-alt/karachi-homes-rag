import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import closing
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import rag

BASE = Path(__file__).parent
DB_PATH = BASE / "users.db"

SESSION_DAYS = 7
PBKDF2_ITERATIONS = 600_000
COOKIE = "session"
MAX_MESSAGE_CHARS = 1000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = FastAPI(title="Karachi Homes Realty Assistant")
lock = threading.Lock()   
histories = {}          
attempts = defaultdict(deque) 

# database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with closing(db()) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            cards TEXT,
            info TEXT,
            created_at INTEGER NOT NULL
        );
        """)
        conn.commit()


# passwords
def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        _, iterations, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
    except Exception:
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


DUMMY_HASH = hash_password("dummy-password")  # keeps login timing similar for unknown emails

# sessions
def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(conn, user_id, response):
    token = secrets.token_urlsafe(32)
    expires = int(time.time()) + SESSION_DAYS * 86400
    # only a hash of the token is stored, so a leaked database cannot be used to log in
    conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                 (token_hash(token), user_id, expires))
    conn.commit()
    response.set_cookie(COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax")


def current_user(request: Request):
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "Please log in.")
    with closing(db()) as conn:
        row = conn.execute(
            "SELECT u.id, u.name, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash(token), int(time.time())),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Your session has expired. Please log in again.")
    return dict(row)


# rate limiting
def is_blocked(key, limit, window):
    now = time.time()
    q = attempts[key]
    while q and now - q[0] > window:
        q.popleft()
    return len(q) >= limit


def record(key):
    attempts[key].append(time.time())


def client_ip(request):
    return request.client.host if request.client else "unknown"


# chats
def latest_chat_id(conn, user_id):
    row = conn.execute("SELECT id FROM chats WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO chats (user_id, created_at) VALUES (?, ?)", (user_id, int(time.time())))
    conn.commit()
    return cur.lastrowid


def rag_history(conn, chat_id):
    """Rebuild (user, bot) pairs from the database so follow-up questions keep working after a restart."""
    rows = conn.execute("SELECT role, text FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
    pairs, pending = [], None
    for r in rows:
        if r["role"] == "user":
            pending = r["text"]
        elif pending is not None:
            pairs.append((pending, r["text"]))
            pending = None
    return pairs[-6:]


# errors and headers
@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse({"error": "Invalid request."}, status_code=400)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


# ---------------------------------------------------------------- request models
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------- routes
@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.post("/signup")
def signup(req: SignupRequest, request: Request, response: Response):
    ip_key = f"signup:{client_ip(request)}"
    if is_blocked(ip_key, 10, 3600):
        raise HTTPException(429, "Too many sign-ups from this device. Please try again later.")
    record(ip_key)

    name = " ".join(req.name.split())
    email = req.email.strip().lower()
    if not 1 <= len(name) <= 60:
        raise HTTPException(400, "Please enter your name (up to 60 characters).")
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email address.")
    if not 8 <= len(req.password) <= 128:
        raise HTTPException(400, "Password must be 8 to 128 characters long.")

    pw_hash = hash_password(req.password)
    with closing(db()) as conn:
        try:
            cur = conn.execute("INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                               (name, email, pw_hash, int(time.time())))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "This email is already registered. Please log in.")
        conn.commit()
        start_session(conn, cur.lastrowid, response)
    return {"user": {"name": name, "email": email}}


@app.post("/login")
def login(req: LoginRequest, request: Request, response: Response):
    email = req.email.strip().lower()[:254]
    key = f"login:{email}:{client_ip(request)}"
    if is_blocked(key, 5, 600):
        raise HTTPException(429, "Too many failed attempts. Please try again in 10 minutes.")

    with closing(db()) as conn:
        row = conn.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        ok = verify_password(req.password[:128], row["password_hash"] if row else DUMMY_HASH)
        if not row or not ok:
            record(key)
            raise HTTPException(401, "Invalid email or password.")
        attempts.pop(key, None)
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
        start_session(conn, row["id"], response)
    return {"user": {"name": row["name"], "email": row["email"]}}


@app.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE)
    if token:
        with closing(db()) as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))
            conn.commit()
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/me")
def me(user=Depends(current_user)):
    return {"user": {"name": user["name"], "email": user["email"]}}


@app.get("/history")
def get_history(user=Depends(current_user)):
    with closing(db()) as conn:
        chat_id = latest_chat_id(conn, user["id"])
        rows = conn.execute("SELECT role, text, cards, info FROM messages WHERE chat_id = ? ORDER BY id",
                            (chat_id,)).fetchall()
    return {"messages": [
        {"role": r["role"], "text": r["text"],
         "cards": json.loads(r["cards"]) if r["cards"] else [],
         "info": json.loads(r["info"]) if r["info"] else None}
        for r in rows
    ]}


@app.post("/new_chat")
def new_chat(user=Depends(current_user)):
    with closing(db()) as conn:
        chat_id = latest_chat_id(conn, user["id"])
        count = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()[0]
        if count:  # an empty chat is reused instead of creating another one
            conn.execute("INSERT INTO chats (user_id, created_at) VALUES (?, ?)", (user["id"], int(time.time())))
            conn.commit()
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest, user=Depends(current_user)):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Empty message.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(400, f"Message too long (maximum {MAX_MESSAGE_CHARS} characters).")

    rate_key = f"chat:{user['id']}"
    if is_blocked(rate_key, 15, 60):
        raise HTTPException(429, "You are sending messages too fast. Please wait a moment.")
    record(rate_key)

    with closing(db()) as conn:
        chat_id = latest_chat_id(conn, user["id"])
        hist = histories.get(chat_id)
        if hist is None:
            hist = histories[chat_id] = rag_history(conn, chat_id)

    try:
        with lock:
            reply = rag.ask(message, hist)
            info = dict(rag.last_stats)
    except Exception as e:
        traceback.print_exc()  # the full error stays in the server terminal
        low = str(e).lower()
        if "429" in low or "rate" in low or "limit" in low:
            raise HTTPException(503, "The AI service is busy or its usage limit was reached. Please try again later.")
        raise HTTPException(500, "Sorry, the assistant could not answer right now. Please try again.")

    cards = info.pop("cards", [])
    del hist[:-20]
    now = int(time.time())
    with closing(db()) as conn:
        conn.execute("INSERT INTO messages (chat_id, role, text, cards, info, created_at) VALUES (?, 'user', ?, NULL, NULL, ?)",
                     (chat_id, message, now))
        conn.execute("INSERT INTO messages (chat_id, role, text, cards, info, created_at) VALUES (?, 'bot', ?, ?, ?, ?)",
                     (chat_id, reply, json.dumps(cards), json.dumps(info), now))
        conn.commit()
    return {"reply": reply, "info": info, "cards": cards}


init_db()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
