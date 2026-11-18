"""
ARIA backend (Flask + SQLite).
Run:  pip install -r requirements.txt   ->   python app.py
"""
import os, json, base64, sqlite3, datetime, functools
import bcrypt, jwt
from flask import Flask, request, jsonify, g
from flask_cors import CORS
try:
    from dotenv import load_dotenv; load_dotenv()      # reads the .env file
except ImportError:
    pass

# ---------- 1. Settings (from .env or environment) ----------
SECRET      = os.getenv("SECRET_KEY", "change-me-in-production")
FACULTY_CODE = os.getenv("FACULTY_CODE", "ARIA-FACULTY")      # only people with this code can register
ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # later: https://ariaLab.com
DB_PATH     = os.getenv("DB_PATH", "aria.db")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")   # use the same model name as your robot code
YOLO_MODEL  = os.getenv("YOLO_MODEL", "best.pt")           # your trained tool detector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024        # 16 MB (photos come as base64)
CORS(app, origins=ORIGINS)

# ---------- 2. Database ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB_PATH)
    c.executescript("""
      CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw_hash BLOB);
      CREATE TABLE IF NOT EXISTS items(id INTEGER PRIMARY KEY, data TEXT);
      CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, name TEXT, text TEXT);
    """)
    c.commit(); c.close()

# ---------- 3. Login system ----------
def make_token(user):
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return jwt.encode({"uid": user["id"], "name": user["name"], "exp": exp}, SECRET, algorithm="HS256")

def faculty_only(f):
    @functools.wraps(f)
    def wrap(*a, **k):
        tok = request.headers.get("Authorization", "").replace("Bearer ", "")
        try:
            g.user = jwt.decode(tok, SECRET, algorithms=["HS256"])
        except Exception:
            return jsonify(error="Please sign in again."), 401
        return f(*a, **k)
    return wrap

@app.post("/api/auth/register")
def register():
    d = request.json or {}
    name, email, pw = d.get("name", "").strip(), d.get("email", "").strip().lower(), d.get("password", "")
    if d.get("code") != FACULTY_CODE:
        return jsonify(error="Wrong faculty code."), 403
    if not name or "@" not in email or len(pw) < 6:
        return jsonify(error="Enter name, valid email and a password of 6+ characters."), 400
    h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt())
    try:
        db().execute("INSERT INTO users(name,email,pw_hash) VALUES(?,?,?)", (name, email, h))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="This email is already registered."), 409
    u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    return jsonify(token=make_token(u), name=u["name"])

@app.post("/api/auth/login")
def login():
    d = request.json or {}
    u = db().execute("SELECT * FROM users WHERE email=?", (d.get("email", "").strip().lower(),)).fetchone()
    if not u or not bcrypt.checkpw(d.get("password", "").encode(), u["pw_hash"]):
        return jsonify(error="Email or password is incorrect."), 401   # same message for both = safer
    return jsonify(token=make_token(u), name=u["name"])

# ---------- 4. Inventory (tools, photos, rack/shelf/box) ----------
@app.get("/api/inventory")
@faculty_only
def inv_list():
    rows = db().execute("SELECT id,data FROM items ORDER BY id DESC").fetchall()
    return jsonify(items=[{**json.loads(r["data"]), "id": r["id"]} for r in rows])

@app.post("/api/inventory")
@faculty_only
def inv_add():
    cur = db().execute("INSERT INTO items(data) VALUES(?)", (json.dumps(request.json or {}),))
    db().commit()
    return jsonify(id=cur.lastrowid)

@app.put("/api/inventory/<int:i>")
@faculty_only
def inv_edit(i):
    db().execute("UPDATE items SET data=? WHERE id=?", (json.dumps(request.json or {}), i)); db().commit()
    return jsonify(ok=True)

@app.delete("/api/inventory/<int:i>")
@faculty_only
def inv_del(i):
    db().execute("DELETE FROM items WHERE id=?", (i,)); db().commit()
    return jsonify(ok=True)

# "Robot feed" button: replace the whole inventory with the list sent by the website
@app.post("/api/inventory/sync")
@faculty_only
def inv_sync():
    items = (request.json or {}).get("items", [])
    db().execute("DELETE FROM items")
    db().executemany("INSERT INTO items(data) VALUES(?)", [(json.dumps(x),) for x in items])
    db().commit()
    return jsonify(ok=True, count=len(items))

# ---------- 5. Lab notes (used for answers) ----------
@app.post("/api/notes")
@faculty_only
def notes_add():
    d = request.json or {}
    db().execute("INSERT INTO notes(name,text) VALUES(?,?)", (d.get("name", "note"), d.get("text", "")))
    db().commit()
    return jsonify(ok=True)

def search_notes(q, n=2):
    words = {w for w in q.lower().split() if len(w) > 3}
    best = []
    for r in db().execute("SELECT name,text FROM notes"):
        for para in r["text"].split("\n\n"):
            score = sum(w in para.lower() for w in words)
            if score: best.append((score, para.strip()))
    return [p for _, p in sorted(best, reverse=True)[:n]]

# ---------- 6. Vision (YOLO) ----------
_model = None
def yolo():
    global _model
    if _model is None:
        from ultralytics import YOLO          # pip install ultralytics
        _model = YOLO(YOLO_MODEL)
    return _model

def decode_image(data_url):
    import numpy as np, cv2
    raw = base64.b64decode(data_url.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def detect_objects(data_url):
    r = yolo()(decode_image(data_url), verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxyn[0].tolist()
        out.append({"label": r.names[int(b.cls)], "conf": float(b.conf), "box": [x1, y1, x2 - x1, y2 - y1]})
    return out

@app.post("/api/detect")
def detect():
    try:
        return jsonify(detections=detect_objects(request.json["image"]))
    except Exception:
        return jsonify(detections=[], error="Live detect needs the YOLO backend (laptop or Pi). Use Capture & ask instead."), 200

# ---------- 7. Chat: Gemini (text + vision) with lab notes and inventory ----------
_client = None
def gemini(prompt, image_data_url=None):
    global _client
    from google import genai                      # pip install google-genai
    from google.genai import types
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    parts = [prompt]
    if image_data_url:
        head, b64 = image_data_url.split(",", 1)
        mime = head.split(";")[0].replace("data:", "") or "image/jpeg"
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
    return (_client.models.generate_content(model=GEMINI_MODEL, contents=parts).text or "").strip()

def inventory_text(limit=60):
    out = []
    for r in db().execute("SELECT data FROM items LIMIT ?", (limit,)):
        it = json.loads(r["data"]); L = it.get("location", {})
        out.append(f"- {it.get('name','?')}: Rack {L.get('rack','?')}, Shelf {L.get('shelf','?')}, Box {L.get('box','?')}")
    return "\n".join(out) or "(no items added yet)"

def where_is(label):
    for r in db().execute("SELECT data FROM items"):
        it = json.loads(r["data"])
        if label.lower() in it.get("name", "").lower():
            L = it.get("location", {})
            return f"Rack {L.get('rack','?')}, Shelf {L.get('shelf','?')}, Box {L.get('box','?')}"
    return None

def text_reply(text):
    notes = search_notes(text, 3)
    if os.getenv("GEMINI_API_KEY"):
        prompt = ("You are ARIA, a voice assistant in a robotics and technology lab. Answer the student in simple English, "
                  "in at most 4 short sentences (your answer will be spoken aloud). Prefer the LAB NOTES. If the answer is not in "
                  "the notes, answer briefly from general knowledge and say it is not from the lab notes. If the student asks where "
                  "a tool is, use the TOOLS list.\n\nLAB NOTES:\n" + ("\n---\n".join(notes) or "(none found)") +
                  "\n\nTOOLS AND WHERE THEY ARE KEPT:\n" + inventory_text() + "\n\nSTUDENT QUESTION: " + text)
        return gemini(prompt)
    try:                                           # fallback 1: your own assistant.py
        import assistant
        return assistant.process_command(text)
    except ImportError:
        pass
    return notes[0] if notes else "I do not have that in the lab notes yet."   # fallback 2

def photo_reply(image, question):
    if os.path.exists(YOLO_MODEL):                 # 1) your trained YOLO model (fast)
        try:
            found = detect_objects(image)
            if found:
                top = max(found, key=lambda x: x["conf"])
                if top["conf"] >= 0.5:
                    loc = where_is(top["label"])
                    return f"This looks like a {top['label']}. " + (f"It is kept at {loc}." if loc else "I do not have its storage location yet.")
        except Exception:
            app.logger.exception("YOLO failed")
    if os.getenv("GEMINI_API_KEY"):                # 2) Gemini vision
        prompt = ("You are ARIA, a lab assistant. Identify the main lab tool or component in this photo. LAB TOOLS AND WHERE THEY ARE KEPT:\n"
                  + inventory_text() + "\nIf it matches a tool in the list, say its name and where it is kept. If not, name it and say its "
                  "location is not recorded. Use at most 3 short sentences." + (f"\nStudent also asked: {question}" if question else ""))
        return gemini(prompt, image)
    return "I cannot recognise tools yet. Add best.pt or a Gemini key on the backend."

@app.post("/api/chat")
def chat():
    d = request.json or {}
    text, image = (d.get("text") or "").strip(), d.get("image")
    try:
        return jsonify(reply=photo_reply(image, text) if image else text_reply(text))
    except Exception:
        app.logger.exception("chat failed")
        return jsonify(reply="Sorry, I could not answer right now. Please try again.")

@app.get("/api/diag")
def diag():
    """Open in browser: /api/diag?code=YOUR_FACULTY_CODE  -> shows why Gemini fails (never prints the key)."""
    if request.args.get("code") != FACULTY_CODE:
        return jsonify(error="code missing or wrong"), 403
    key = os.getenv("GEMINI_API_KEY", "")
    out = {"key_set": bool(key), "key_length": len(key), "key_first3": key[:3],
           "key_has_space_or_quote": any(c in key for c in " \"'\n\r"), "model": GEMINI_MODEL}
    try:
        out["gemini_reply"] = gemini("Reply with the single word OK")
    except Exception as e:
        out["gemini_error"] = f"{type(e).__name__}: {str(e)[:400]}"
    return jsonify(out)

@app.get("/api/health")
def health():
    return jsonify(ok=True)

init_db()   # also runs under gunicorn

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
