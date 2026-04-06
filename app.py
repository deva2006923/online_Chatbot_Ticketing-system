from flask import Flask, render_template, request, jsonify
import sqlite3, os, uuid, re, json, smtplib, csv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv
import pytesseract
from datetime import datetime, date
import cv2
import numpy as np
from io import StringIO

# ---------------- CONFIG ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY missing in .env")

client = Groq(api_key=api_key)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EMAIL_SENDER   = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_INPUT_LENGTH = 2000
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect("tickets.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def create_table():
    with get_db() as conn:
        # Create tickets table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets(
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id     TEXT UNIQUE,
            name          TEXT NOT NULL,
            issue         TEXT,
            category      TEXT,
            priority      TEXT,
            sentiment     TEXT,
            status        TEXT DEFAULT 'Open',
            language      TEXT DEFAULT 'English',
            message_count INTEGER DEFAULT 1,
            date          TEXT,
            resolved_date TEXT
        )
        """)
        # Create chat history table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history(
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            role      TEXT,
            message   TEXT,
            timestamp TEXT
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_name_status ON tickets(name, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticket_history ON chat_history(ticket_id)")
        conn.commit()

def migrate_db():
    """Auto-add any missing columns to existing DB — safe to run every startup."""
    new_columns = [
        ("language",      "TEXT DEFAULT 'English'"),
        ("message_count", "INTEGER DEFAULT 1"),
        ("resolved_date", "TEXT"),
    ]
    with get_db() as conn:
        # Get existing columns
        existing = {row[1] for row in conn.execute("PRAGMA table_info(tickets)")}
        for col_name, col_def in new_columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {col_name} {col_def}")
                print(f"✅ Migrated: added column '{col_name}'")
        conn.commit()

create_table()
migrate_db()  # ✅ Always run on startup — safe for existing DBs

# ---------------- EMAIL ----------------
def send_email(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    except Exception as e:
        app.logger.error(f"Email error: {e}")

def email_ticket_created(ticket_id, name, issue, priority, category, sentiment):
    priority_color = {"High": "#FF4D6D", "Medium": "#FFB800", "Low": "#00E5A0"}.get(priority, "#ccc")
    body = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:auto;background:#0E1521;color:#C9D8E8;padding:30px;border-radius:12px;">
        <h2 style="color:#00D4FF;">🎫 New Support Ticket Created</h2>
        <table style="width:100%;border-collapse:collapse;margin-top:15px;">
            <tr><td style="padding:8px;color:#4A6080;">Ticket ID</td><td style="padding:8px;color:#00D4FF;font-family:monospace;">{ticket_id}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Customer</td><td style="padding:8px;">{name}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Issue</td><td style="padding:8px;"><strong>{issue}</strong></td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Priority</td><td style="padding:8px;color:{priority_color};font-weight:bold;">{priority}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Category</td><td style="padding:8px;">{category}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Sentiment</td><td style="padding:8px;">{sentiment}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Date</td><td style="padding:8px;">{date.today()}</td></tr>
        </table>
        <p style="margin-top:20px;color:#4A6080;font-size:12px;">Login to your admin dashboard to manage this ticket.</p>
    </div>
    """
    send_email(f"[{priority}] New Ticket: {issue} — {ticket_id}", body)

def email_ticket_resolved(ticket_id, name, issue):
    body = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:auto;background:#0E1521;color:#C9D8E8;padding:30px;border-radius:12px;">
        <h2 style="color:#00E5A0;">✅ Ticket Resolved</h2>
        <table style="width:100%;border-collapse:collapse;margin-top:15px;">
            <tr><td style="padding:8px;color:#4A6080;">Ticket ID</td><td style="padding:8px;color:#00D4FF;font-family:monospace;">{ticket_id}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Customer</td><td style="padding:8px;">{name}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Issue</td><td style="padding:8px;">{issue}</td></tr>
            <tr><td style="padding:8px;color:#4A6080;">Resolved On</td><td style="padding:8px;">{date.today()}</td></tr>
        </table>
    </div>
    """
    send_email(f"✅ Resolved: {ticket_id} — {name}", body)

# ---------------- HELPERS ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_input(text):
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"[<>\"']", "", text)
    return text[:MAX_INPUT_LENGTH].strip()

def clean_json_string(raw):
    raw = re.sub(r"```json|```", "", raw).strip()
    raw = re.sub(r"[\x00-\x1f\x7f]", " ", raw)
    return raw

def quick_resolve_check(text):
    lower = text.lower().strip()
    resolve_phrases = [
        "issue is over", "issue is resolved", "problem is solved",
        "its fixed", "it's fixed", "fixed now", "resolved now",
        "my issue is over", "my problem is over", "issue over",
        "problem over", "solved", "all good", "working now",
        "it works", "it is working", "thank you", "thanks",
        "no more issues", "issue fixed", "problem fixed",
        "got it working", "got it fixed", "ok thanks",
        "its working", "its done", "done now"
    ]
    return any(phrase in lower for phrase in resolve_phrases)

def save_chat_history(ticket_id, role, message):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO chat_history (ticket_id, role, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (ticket_id, role, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

def get_chat_history(ticket_id, limit=6):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT role, message FROM chat_history
            WHERE ticket_id=? ORDER BY id DESC LIMIT ?
        """, (ticket_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["message"]} for r in reversed(rows)]

# ---------------- OCR ----------------
def extract_text(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return ""
        results = []

        def try_ocr(processed):
            try:
                t = pytesseract.image_to_string(processed, config="--psm 6").strip()
                if t: results.append(t)
            except Exception:
                pass

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try_ocr(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        try_ocr(cv2.bitwise_not(gray))
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try_ocr(cv2.fastNlMeansDenoising(upscaled, h=30))
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try_ocr(otsu)
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        try_ocr(otsu_inv)

        if not results:
            return ""
        best = max(results, key=lambda t: len([w for w in t.split() if len(w) > 2]))
        return re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]", " ", best)).strip()
    except Exception:
        return ""

# ---------------- AI ANALYSIS ----------------
def analyze_with_ai(full_text, has_image=False, history=None):
    full_text = re.sub(r"[\x00-\x1f\x7f]", " ", full_text).strip()
    image_hint = "Note: Customer uploaded a screenshot. OCR text may be noisy — infer the issue from context." if has_image else ""

    system_prompt = """You are a senior multilingual technical support engineer.
Always respond with plain valid JSON only. No markdown fences, no extra text.
NEVER say the message is unclear — always make your best effort to help.
If text is from OCR and garbled, infer the most likely issue.
Adapt your tone: be empathetic and gentle when the user is frustrated."""

    prompt = f"""Analyze this customer support message and respond ONLY in valid JSON:

{{
  "resolved": false,
  "summary": "2-4 word issue title",
  "category": "Technical",
  "priority": "High",
  "sentiment": "Frustrated",
  "language": "English",
  "solution": "step-by-step support response in markdown"
}}

Rules:
- resolved=true if user says issue is fixed, resolved, over, solved, working now, thanks etc.
- priority=High if error/fail/crash/urgent/down/not working appear
- priority=Low for simple questions
- category: exactly Technical, Billing, or General
- priority: exactly High, Medium, or Low
- sentiment: exactly Frustrated, Neutral, or Satisfied
- language: detect the language the customer wrote in
- solution: MUST be written in the same language as the customer message
- If sentiment=Frustrated, start solution with an empathetic acknowledgment
- solution must be specific numbered steps, never generic
{image_hint}

Customer message: {full_text}"""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.1
    )

    raw = clean_json_string(res.choices[0].message.content.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(clean_json_string(match.group())) if match else {}

    return {
        "resolved":  bool(data.get("resolved", False)),
        "summary":   str(data.get("summary", "Unknown issue"))[:100],
        "category":  data.get("category", "General") if data.get("category") in ["Technical","Billing","General"] else "General",
        "priority":  data.get("priority", "Medium") if data.get("priority") in ["High","Medium","Low"] else "Medium",
        "sentiment": data.get("sentiment", "Neutral") if data.get("sentiment") in ["Frustrated","Neutral","Satisfied"] else "Neutral",
        "language":  str(data.get("language", "English")),
        "solution":  str(data.get("solution", "Please describe your issue in more detail."))
    }

# ---------------- RESOLVE HANDLER ----------------
def handle_resolve(name):
    with get_db() as conn:
        ticket = conn.execute("""
            SELECT ticket_id, issue, name FROM tickets
            WHERE name=? AND status='Open'
            ORDER BY id DESC LIMIT 1
        """, (name,)).fetchone()

        if ticket:
            conn.execute("""
                UPDATE tickets SET status='Resolved', resolved_date=?
                WHERE ticket_id=?
            """, (str(date.today()), ticket["ticket_id"]))
            conn.commit()
            email_ticket_resolved(ticket["ticket_id"], ticket["name"], ticket["issue"])
            save_chat_history(ticket["ticket_id"], "assistant", "Ticket resolved.")
            return jsonify({"reply": f"## Ticket Resolved\n\nGreat to hear your issue is fixed! Your ticket has been closed.\n\n**Ticket ID:** `{ticket['ticket_id']}`"})
        else:
            return jsonify({"reply": "Glad your issue is resolved! Feel free to start a new conversation anytime."})

# ---------------- FAQ ----------------
FAQ = [
    {"keywords": ["password", "reset", "forgot"], "answer": "## Password Reset\n\n1. Go to the login page\n2. Click **Forgot Password**\n3. Enter your email and check your inbox\n4. Follow the reset link\n\nStill not working? I'll create a ticket for you."},
    {"keywords": ["refund", "money back", "charge"], "answer": "## Refund Request\n\n1. Refunds are processed within **5-7 business days**\n2. Go to **My Orders → Request Refund**\n3. Select the order and reason\n\nIf the option is not available, I'll escalate to billing."},
    {"keywords": ["slow", "loading", "speed"], "answer": "## Performance Issue\n\n1. Clear your browser cache (Ctrl+Shift+Delete)\n2. Try a different browser\n3. Check your internet speed at fast.com\n4. Disable browser extensions\n\nIf still slow, I'll create a ticket for deeper investigation."},
    {"keywords": ["login", "cant login", "cannot login", "sign in"], "answer": "## Login Issue\n\n1. Clear cookies and cache\n2. Try incognito/private mode\n3. Check if Caps Lock is on\n4. Try resetting your password\n\nIf none work, I'll escalate to our team."},
]

def check_faq(text):
    lower = text.lower()
    for faq in FAQ:
        if any(kw in lower for kw in faq["keywords"]):
            return faq["answer"]
    return None

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        name       = sanitize_input(request.form.get("name", "User"))
        user_input = sanitize_input(request.form.get("issue", ""))
        image      = request.files.get("image")

        if not name:
            return jsonify({"reply": "Please provide your name."}), 400

        extracted_text = ""
        has_image = False

        if image and image.filename != "":
            if not allowed_file(image.filename):
                return jsonify({"reply": "Invalid file type. Please upload an image."}), 400
            filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
            path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(path)
            extracted_text = extract_text(path)
            has_image = True

        full_text = f"{user_input} {extracted_text}".strip()
        if not full_text:
            return jsonify({"reply": "Please describe your issue or upload an image."}), 400

        # Fast resolve check
        if quick_resolve_check(full_text):
            return handle_resolve(name)

        # FAQ check
        faq_answer = check_faq(full_text)
        if faq_answer:
            return jsonify({"reply": faq_answer + "\n\n---\n💬 Did this help? Reply **yes** to close or describe more details to create a ticket."})

        # Get existing ticket + chat history
        with get_db() as conn:
            existing = conn.execute("""
                SELECT ticket_id, issue, message_count FROM tickets
                WHERE name=? AND status != 'Resolved'
                ORDER BY id DESC LIMIT 1
            """, (name,)).fetchone()

        history = []
        if existing:
            history = get_chat_history(existing["ticket_id"])

        # AI Analysis
        ai = analyze_with_ai(full_text, has_image=has_image, history=history)

        if ai["resolved"]:
            return handle_resolve(name)

        # Upsert ticket
        with get_db() as conn:
            existing = conn.execute("""
                SELECT ticket_id, issue, message_count FROM tickets
                WHERE name=? AND status != 'Resolved'
                ORDER BY id DESC LIMIT 1
            """, (name,)).fetchone()

            if not existing:
                ticket_id = "TKT-" + str(uuid.uuid4())[:8].upper()
                conn.execute("""
                    INSERT INTO tickets
                    (ticket_id, name, issue, category, priority, sentiment, status, language, message_count, date)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (ticket_id, name, ai["summary"], ai["category"],
                      ai["priority"], ai["sentiment"], "Open",
                      ai["language"], 1, str(date.today())))
                conn.commit()
                email_ticket_created(ticket_id, name, ai["summary"],
                                     ai["priority"], ai["category"], ai["sentiment"])
            else:
                ticket_id     = existing["ticket_id"]
                message_count = (existing["message_count"] or 1) + 1

                # Auto-escalate
                escalated_priority = ai["priority"]
                if message_count >= 3 and ai["priority"] == "Low":
                    escalated_priority = "Medium"
                if message_count >= 5:
                    escalated_priority = "High"

                bad_summaries = ["unknown issue", "issue is over", "issue over",
                                 "customer message is not understandable", "not understandable", ""]
                new_summary = ai["summary"] if ai["summary"].lower() not in bad_summaries else existing["issue"]

                conn.execute("""
                    UPDATE tickets
                    SET issue=?, category=?, priority=?, sentiment=?, language=?, message_count=?
                    WHERE ticket_id=?
                """, (new_summary, ai["category"], escalated_priority,
                      ai["sentiment"], ai["language"], message_count, ticket_id))
                conn.commit()
                ai["priority"] = escalated_priority

        # Save chat history
        save_chat_history(ticket_id, "user", full_text)
        save_chat_history(ticket_id, "assistant", ai["solution"])

        escalation_notice = ""
        if existing and (existing["message_count"] or 1) + 1 >= 3:
            escalation_notice = "\n\n> ⚡ **Note:** This ticket has been escalated due to multiple follow-ups."

        reply = f"""## {ai['summary']}

---

{ai['solution']}
{escalation_notice}

---
**Ticket ID:** `{ticket_id}` | **Priority:** {ai['priority']} | **Category:** {ai['category']} | **Lang:** {ai['language']}
"""
        return jsonify({"reply": reply})

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({"reply": f"Something went wrong. Please try again.\n\n`{str(e)}`"}), 500


@app.route("/admin")
def admin():
    with get_db() as conn:
        tickets = [dict(row) for row in conn.execute("SELECT * FROM tickets ORDER BY id DESC")]
    from collections import Counter
    categories = Counter(t["category"] for t in tickets)
    priorities = Counter(t["priority"] for t in tickets)
    sentiments = Counter(t["sentiment"] for t in tickets)
    statuses   = Counter(t["status"]   for t in tickets)
    return render_template("admin.html",
        tickets=tickets,
        total=len(tickets),
        open_t=statuses.get("Open", 0),
        resolved=statuses.get("Resolved", 0),
        high=priorities.get("High", 0),
        chart_categories=json.dumps(dict(categories)),
        chart_priorities=json.dumps(dict(priorities)),
        chart_sentiments=json.dumps(dict(sentiments)),
    )


@app.route("/api/tickets")
def api_tickets():
    with get_db() as conn:
        tickets = [dict(row) for row in conn.execute("SELECT * FROM tickets ORDER BY id DESC")]
    return jsonify(tickets)


@app.route("/api/history/<ticket_id>")
def api_history(ticket_id):
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT role, message, timestamp FROM chat_history
            WHERE ticket_id=? ORDER BY id ASC
        """, (ticket_id,))]
    return jsonify(rows)


@app.route("/export/csv")
def export_csv():
    from flask import Response
    with get_db() as conn:
        tickets = [dict(row) for row in conn.execute("SELECT * FROM tickets ORDER BY id DESC")]
    si = StringIO()
    writer = csv.DictWriter(si, fieldnames=["ticket_id","name","issue","category","priority",
                                             "sentiment","status","language","message_count","date","resolved_date"])
    writer.writeheader()
    writer.writerows(tickets)
    return Response(si.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=tickets.csv"})


@app.route("/reset", methods=["POST"])
def reset():
    with get_db() as conn:
        conn.execute("DELETE FROM tickets")
        conn.execute("DELETE FROM chat_history")
        conn.commit()
    return jsonify({"status": "success"})


if __name__ == "__main__":
    print("🚀 Next-Level AI Support System Running")
    app.run(debug=True)