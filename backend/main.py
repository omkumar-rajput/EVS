from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import os
import httpx
from datetime import datetime, timezone
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Vertical Garden API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "garden.db")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Database ──────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS plants (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT    NOT NULL,
                type         TEXT    NOT NULL DEFAULT 'herb',
                position     TEXT    NOT NULL DEFAULT 'Unassigned',
                freq         INTEGER NOT NULL DEFAULT 2,
                sunlight     TEXT    NOT NULL DEFAULT 'full',
                added        TEXT    NOT NULL,
                lastWatered  TEXT    NOT NULL,
                wateredCount INTEGER NOT NULL DEFAULT 0,
                growthDays   INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS watering_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id   INTEGER NOT NULL,
                watered_at TEXT    NOT NULL,
                FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ai_chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id   INTEGER,
                plant_name TEXT,
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        count = db.execute("SELECT COUNT(*) as c FROM plants").fetchone()["c"]
        if count == 0:
            now = datetime.now(timezone.utc).isoformat()
            sample_plants = [
                ("Mint",          "herb",      "Row 1, Bottle 1", 2, "partial", 8, 2, 4, 8),
                ("Basil",         "herb",      "Row 1, Bottle 2", 1, "full",    5, 1, 5, 5),
                ("Cherry Tomato", "vegetable", "Row 2, Bottle 1", 2, "full",   12, 3, 6, 12),
            ]
            for name, ptype, pos, freq, sun, days_ago, watered_ago, w_count, g_days in sample_plants:
                from datetime import timedelta
                added   = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
                watered = (datetime.now(timezone.utc) - timedelta(days=watered_ago)).isoformat()
                db.execute(
                    "INSERT INTO plants (name,type,position,freq,sunlight,added,lastWatered,wateredCount,growthDays) VALUES (?,?,?,?,?,?,?,?,?)",
                    (name, ptype, pos, freq, sun, added, watered, w_count, g_days)
                )


init_db()


# ── Pydantic Models ───────────────────────────────────────────────

class PlantCreate(BaseModel):
    name: str
    type: Optional[str] = "herb"
    position: Optional[str] = "Unassigned"
    freq: Optional[int] = 2
    sunlight: Optional[str] = "full"

class PlantUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    position: Optional[str] = None
    freq: Optional[int] = None
    sunlight: Optional[str] = None

class AIQuestion(BaseModel):
    question: str
    plant_id: Optional[str] = "general"


def row_to_dict(row):
    return dict(row) if row else None

def rows_to_list(rows):
    return [dict(r) for r in rows]


# ── Plant Routes ──────────────────────────────────────────────────

@app.get("/api/plants")
def get_plants():
    with get_db() as db:
        plants = db.execute("SELECT * FROM plants ORDER BY id ASC").fetchall()
    return rows_to_list(plants)


@app.get("/api/plants/{plant_id}")
def get_plant(plant_id: int):
    with get_db() as db:
        plant = db.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    return row_to_dict(plant)


@app.post("/api/plants", status_code=201)
def create_plant(data: PlantCreate):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        cursor = db.execute(
            "INSERT INTO plants (name,type,position,freq,sunlight,added,lastWatered,wateredCount,growthDays) VALUES (?,?,?,?,?,?,?,0,0)",
            (data.name, data.type, data.position, data.freq, data.sunlight, now, now)
        )
        plant = db.execute("SELECT * FROM plants WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return row_to_dict(plant)


@app.put("/api/plants/{plant_id}")
def update_plant(plant_id: int, data: PlantUpdate):
    with get_db() as db:
        existing = db.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Plant not found")
        db.execute(
            "UPDATE plants SET name=?, type=?, position=?, freq=?, sunlight=? WHERE id=?",
            (
                data.name     or existing["name"],
                data.type     or existing["type"],
                data.position or existing["position"],
                data.freq     or existing["freq"],
                data.sunlight or existing["sunlight"],
                plant_id
            )
        )
        plant = db.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
    return row_to_dict(plant)


@app.delete("/api/plants/{plant_id}")
def delete_plant(plant_id: int):
    with get_db() as db:
        result = db.execute("DELETE FROM plants WHERE id = ?", (plant_id,))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Plant not found")
    return {"success": True}


@app.post("/api/plants/{plant_id}/water")
def water_plant(plant_id: int):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        plant = db.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
        if not plant:
            raise HTTPException(status_code=404, detail="Plant not found")
        db.execute(
            "UPDATE plants SET lastWatered = ?, wateredCount = wateredCount + 1 WHERE id = ?",
            (now, plant_id)
        )
        db.execute(
            "INSERT INTO watering_logs (plant_id, watered_at) VALUES (?, ?)",
            (plant_id, now)
        )
        updated = db.execute("SELECT * FROM plants WHERE id = ?", (plant_id,)).fetchone()
    return row_to_dict(updated)


@app.get("/api/plants/{plant_id}/logs")
def get_watering_logs(plant_id: int):
    with get_db() as db:
        logs = db.execute(
            "SELECT * FROM watering_logs WHERE plant_id = ? ORDER BY watered_at DESC LIMIT 20",
            (plant_id,)
        ).fetchall()
    return rows_to_list(logs)


# ── Stats ─────────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    with get_db() as db:
        plants = db.execute("SELECT * FROM plants").fetchall()
        total_waterings = db.execute("SELECT COUNT(*) as c FROM watering_logs").fetchone()["c"]

    now_ts = datetime.now(timezone.utc).timestamp()
    due = overdue = healthy = 0

    for p in plants:
        last = datetime.fromisoformat(p["lastWatered"]).timestamp()
        days = (now_ts - last) / 86400
        if days >= p["freq"] + 1:
            overdue += 1
        elif days >= p["freq"]:
            due += 1
        else:
            healthy += 1

    return {
        "total": len(plants),
        "due": due,
        "overdue": overdue,
        "healthy": healthy,
        "totalWaterings": total_waterings
    }


# ── AI Advisor ────────────────────────────────────────────────────

@app.post("/api/ai/ask")
async def ask_ai(data: AIQuestion):
    if not data.question:
        raise HTTPException(status_code=400, detail="Question is required")

    plant = None
    plant_context = ""

    if data.plant_id and data.plant_id != "general":
        with get_db() as db:
            plant = db.execute(
                "SELECT * FROM plants WHERE id = ?",
                (data.plant_id,)
            ).fetchone()

        if plant:
            plant = dict(plant)

            now_ts = datetime.now(timezone.utc).timestamp()
            last_ts = datetime.fromisoformat(
                plant["lastWatered"]
            ).timestamp()

            days_since = int((now_ts - last_ts) / 86400)

            plant_context = (
                f"Plant info — Name: {plant['name']}, "
                f"Type: {plant['type']}, "
                f"Position: {plant['position']}, "
                f"Sunlight: {plant['sunlight']}, "
                f"Watered every {plant['freq']} day(s), "
                f"Last watered {days_since} day(s) ago, "
                f"Growing for {plant['growthDays']} days, "
                f"Total waterings: {plant['wateredCount']}."
            )

        with get_db() as db:
        all_plants = db.execute(
            "SELECT name, freq, lastWatered FROM plants"
        ).fetchall()

    garden_summary = ""

    for p in all_plants:
        garden_summary += (
            f"Plant: {p['name']}, "
            f"Water every {p['freq']} day(s). "
        )

    system_prompt = (
        "You are a friendly expert plant care advisor for a Smart Vertical Garden project. "
        "The garden uses reused plastic bottles mounted vertically — an eco-friendly sustainable approach. "
        "Give practical, concise advice. Use simple language suitable for students. "
        "Keep responses under 130 words. Use short paragraphs. "
        "Always connect advice to eco-friendly or sustainable gardening where relevant. "
        + plant_context
    )

    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not set in .env file"
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openrouter/auto",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": data.question
                    }
                ]
            }
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"OpenRouter API error: {response.text}"
        )

    result = response.json()

    answer = result["choices"][0]["message"]["content"]

    plant_name = plant["name"] if plant else "General"

    now = datetime.now(timezone.utc).isoformat()

    with get_db() as db:
        db.execute(
            "INSERT INTO ai_chat_history "
            "(plant_id, plant_name, question, answer, created_at) "
            "VALUES (?,?,?,?,?)",
            (
                data.plant_id if data.plant_id != "general" else None,
                plant_name,
                data.question,
                answer,
                now
            )
        )

    return {
        "answer": answer,
        "plant": plant_name
    }


@app.get("/api/ai/history")
def get_ai_history():
    with get_db() as db:
        history = db.execute(
            "SELECT * FROM ai_chat_history ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return rows_to_list(history)


# ── Health Check ──────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Smart Garden API is running"}


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
