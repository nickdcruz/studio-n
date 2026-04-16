import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn

import httpx
import requests
from apify_client import ApifyClient
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from typing import List
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Persistent storage (SQLite) ───────────────────────────────────
# Set DATA_DIR=/data in Railway env vars + mount a Railway Volume at /data.
# Without a volume, data persists in ./data (lost on redeploy — set the volume).
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "studio_n.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL DEFAULT '',
            client          TEXT NOT NULL DEFAULT 'other',
            project_name    TEXT DEFAULT '',
            brief           TEXT NOT NULL DEFAULT '',
            timestamp       TEXT NOT NULL,
            marcus_analysis TEXT DEFAULT '',
            stage1_outputs  TEXT DEFAULT '{}',
            marcus_review   TEXT DEFAULT '',
            cascade_outputs TEXT DEFAULT '{}',
            marcus_cascade_review TEXT DEFAULT '',
            assembled_files TEXT DEFAULT '{}',
            assembled_content TEXT DEFAULT '{}',
            status          TEXT DEFAULT 'complete',
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN archived INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    conn.close()
    logging.info("DB ready at %s", DB_PATH)

def db_save_job(job_id: str, job: dict, assembled_content: dict = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO jobs
        (id, title, client, project_name, brief, timestamp,
         marcus_analysis, stage1_outputs, marcus_review,
         cascade_outputs, marcus_cascade_review,
         assembled_files, assembled_content, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        job_id,
        job.get("title", "Untitled"),
        job.get("client", "other"),
        job.get("project_name", ""),
        job.get("brief", ""),
        job["timestamp"].isoformat() if hasattr(job.get("timestamp"), "isoformat") else str(job.get("timestamp", "")),
        job.get("marcus_analysis", ""),
        json.dumps(job.get("stage1_outputs") or {}),
        job.get("marcus_review", "") or "",
        json.dumps(job.get("cascade_outputs") or {}),
        job.get("marcus_cascade_review", "") or "",
        json.dumps(job.get("assembled_files") or {}),
        json.dumps(assembled_content or {}),
        "complete",
    ))
    conn.commit()
    conn.close()

def db_load_all_jobs(show_archived: bool = False) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if show_archived:
        rows = conn.execute("SELECT * FROM jobs WHERE archived=1 ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM jobs WHERE archived=0 OR archived IS NULL ORDER BY created_at DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        job = dict(r)
        for k in ("stage1_outputs", "cascade_outputs", "assembled_files", "assembled_content"):
            try:
                job[k] = json.loads(job.get(k) or "{}")
            except Exception:
                job[k] = {}
        result.append(job)
    return result

def db_recover_missing_files():
    """Recreate output files from DB if they are missing (e.g. after redeploy)."""
    try:
        jobs = db_load_all_jobs()
    except Exception as e:
        logging.error("DB recovery failed: %s", e)
        return
    recovered = 0
    for job in jobs:
        for key, content in (job.get("assembled_content") or {}).items():
            fname = (job.get("assembled_files") or {}).get(key)
            if not fname or not content:
                continue
            path = OUTPUTS_DIR / fname
            if not path.exists():
                try:
                    path.write_text(content, encoding="utf-8")
                    recovered += 1
                except Exception as e:
                    logging.error("Recovery failed for %s: %s", fname, e)
    if recovered:
        logging.info("Recovered %d output files from DB", recovered)

init_db()
db_recover_missing_files()

app = FastAPI(title="Studio N")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

GITHUB_REPO         = os.getenv("GITHUB_REPO", "nickdcruz/nicklaus-marketing-agents")
GITHUB_TOKEN        = os.environ.get("GITHUB_TOKEN")
ANTHROPIC_API_KEY    = os.environ["ANTHROPIC_API_KEY"]
ARCADS_CLIENT_ID     = os.getenv("ARCADS_CLIENT_ID", os.getenv("ARCADS_API_KEY", ""))
ARCADS_CLIENT_SECRET = os.getenv("ARCADS_CLIENT_SECRET", "")
ARCADS_BASE_URL      = os.getenv("ARCADS_BASE_URL", "https://external-api.arcads.ai")
ARCADS_CREDIT_BUDGET = float(os.getenv("ARCADS_CREDIT_BUDGET", "200"))
HTTP_USER            = os.getenv("HTTP_USER", "admin")
HTTP_PASS           = os.getenv("HTTP_PASS", "changeme")
SESSION_SECRET      = os.getenv("SESSION_SECRET", secrets.token_hex(32))
PORT                = int(os.environ.get("PORT", "5050"))
APIFY_API_TOKEN     = os.environ.get("APIFY_API_TOKEN")
BUFFER_ACCESS_TOKEN = os.environ.get("BUFFER_ACCESS_TOKEN")
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── Arcads API helpers ────────────────────────────────────────────

def _arcads_headers():
    import base64
    # HTTP Basic auth: Client ID as username, Client Secret as password
    creds = base64.b64encode(f"{ARCADS_CLIENT_ID}:{ARCADS_CLIENT_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}

async def arcads_get_products() -> list:
    if not ARCADS_CLIENT_ID:
        return []
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ARCADS_BASE_URL}/v1/products", headers=_arcads_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("items") or data.get("data") or []

async def arcads_generate_video(payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        logging.info("Arcads generate payload: %s", json.dumps(payload))
        r = await client.post(f"{ARCADS_BASE_URL}/v2/videos/generate",
                              json=payload, headers=_arcads_headers(), timeout=30)
        logging.info("Arcads generate response %s: %s", r.status_code, r.text[:500])
        r.raise_for_status()
        return r.json()

async def arcads_generate_image(payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ARCADS_BASE_URL}/v2/images/generate",
                              json=payload, headers=_arcads_headers(), timeout=30)
        r.raise_for_status()
        return r.json()

async def arcads_poll_video(video_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ARCADS_BASE_URL}/v1/videos/{video_id}",
                             headers=_arcads_headers(), timeout=15)
        r.raise_for_status()
        return r.json()

async def arcads_poll_asset(asset_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ARCADS_BASE_URL}/v1/assets/{asset_id}",
                             headers=_arcads_headers(), timeout=15)
        r.raise_for_status()
        return r.json()

async def arcads_upload_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Upload file to Arcads presigned storage, return filePath for use in generation requests."""
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ARCADS_BASE_URL}/v1/uploads/presigned",
                              json={"filename": filename, "contentType": content_type},
                              headers=_arcads_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        upload_url = data.get("uploadUrl") or data.get("url", "")
        file_path   = data.get("filePath") or data.get("path", "")
        await client.put(upload_url, content=file_bytes,
                         headers={"Content-Type": content_type}, timeout=120)
        return file_path

# SQLite table for video studio jobs
def init_video_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_jobs (
            id          TEXT PRIMARY KEY,
            client      TEXT DEFAULT 'other',
            job_type    TEXT DEFAULT 'video',
            model       TEXT,
            prompt      TEXT,
            status      TEXT DEFAULT 'pending',
            arcads_id   TEXT,
            result_url  TEXT,
            formats     TEXT DEFAULT '{}',
            created_at  TEXT DEFAULT (datetime('now')),
            error       TEXT
        )
    """)
    conn.commit()
    conn.close()

init_video_db()

_agent_cache: dict = {}
_jobs:        dict = {}
_completed:   dict = {}

VALID_AGENTS = {"callum", "priya", "dante", "suki", "felix", "nadia", "zara", "reeva", "kiara", "rex", "nova"}

THIRD_PARTY_MESSAGES = {
    "video_production": (
        "Dante's video concept is saved and ready. Production requires Canva, CapCut, or Adobe Premiere — "
        "use real screen recordings of the software, not AI-generated imagery. Brief saved to outputs."
    ),
    "social_images": (
        "Suki's design brief is saved. Static images require real screen recordings or photography. "
        "Use the Canva JSON template in your outputs folder to build designs in Canva."
    ),
    "publishing": (
        "Content is approved. Connect Buffer, Hootsuite, or a platform API to enable auto-scheduling."
    ),
}

# ── Auth middleware ───────────────────────────────────────────────

PUBLIC_PATHS = {"/health", "/login"}

class SessionAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith("/api/stream/"):
            return await call_next(request)
        if not request.session.get("authenticated"):
            if path.startswith("/api/"):
                return JSONResponse({"error": "Not authenticated"}, status_code=401)
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)

app.add_middleware(SessionAuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="studio_session",
                   max_age=86400 * 7, https_only=False, same_site="lax")

# ── Orchestration prompts ─────────────────────────────────────────

ORCHESTRATION_PREFIX = """SYSTEM INSTRUCTION — READ THIS BEFORE ANYTHING ELSE:
You are in orchestration mode. Your job is to analyse the brief and decide which specialist agents are needed.
After your analysis, end your response with a JSON block in this exact format — this is required:

```json
{"agents_needed": ["agent1", "agent2"], "briefs": {"agent1": "full brief text", "agent2": "full brief text"}}
```

Valid agent names: callum, priya, dante, suki, felix, nadia, zara, reeva, kiara, rex, nova

Agent specialisms:
- callum: LinkedIn and long-form content
- priya: social and short-form content
- dante: video concepts and reels
- suki: static visuals and design briefs
- felix: decks and presentations
- nadia: web copy and landing pages
- zara: research and competitive intelligence
- reeva: brand identity and strategy
- kiara: AI video generation (Arcads/Seedance — use when actual video output is needed)
- rex: Remotion programmatic video code
- nova: Nano Banana brand visuals and stills

---
"""

REVIEW_CASCADE_SUFFIX = """

---
SYSTEM — REVIEW + CASCADE: After your verdicts, append ONE cascade block and nothing after it:

```cascade
{
  "verdicts": {"agent_name": "approved"},
  "cascade": {
    "next_agents": [],
    "assembly": "none",
    "flags": []
  }
}
```

ASSEMBLY GUIDE:
• Brand foundation delivered        → next_agents:[felix,nadia,suki]  assembly:html_website
• Website or landing page           → next_agents:[nadia,suki]        assembly:html_website
• One-pager or sales sheet          → next_agents:[felix,suki]        assembly:html_onepager
• Pitch deck or presentation        → next_agents:[felix]             assembly:html_deck
• Full social/content campaign      → next_agents:[priya,dante,suki]  assembly:social_pack  flags:[video_production,social_images]
• LinkedIn or long-form only        → next_agents:[]  assembly:none
• Research, strategy, or analysis   → next_agents:[]  assembly:none

flags options: video_production, social_images, publishing
"""

# ── Brand data extraction ─────────────────────────────────────────

def extract_brand_data(outputs: dict) -> dict:
    """Extract hex colors, font names, and brand name from any agent output."""
    combined = "\n".join(outputs.values())

    hex_colors = list(dict.fromkeys(re.findall(r"#[0-9A-Fa-f]{6}\b", combined)))[:6]

    google_fonts = [
        "Inter", "Plus Jakarta Sans", "DM Sans", "Sora", "Poppins",
        "Montserrat", "Raleway", "Lato", "Nunito", "Work Sans",
    ]
    found_fonts = [f for f in google_fonts if f.lower() in combined.lower()]
    chosen_fonts = found_fonts[:2] if found_fonts else ["Inter"]

    name_match = re.search(r"^#\s+(.+)$", combined, re.MULTILINE)
    brand_name = name_match.group(1).strip() if name_match else ""

    photo_match = re.search(
        r"(?:photography[:\s]+|photo\s+direction[:\s]+|visual\s+style[:\s]+|image\s+direction[:\s]+)"
        r"([^\n]{10,120})",
        combined, re.IGNORECASE)
    photography = photo_match.group(1).strip() if photo_match else \
        "Clean, professional product photography with natural lighting"

    return {
        "hex_colors":  hex_colors,
        "fonts":       chosen_fonts,
        "brand_name":  brand_name,
        "primary":     hex_colors[0] if hex_colors else "#1e293b",
        "secondary":   hex_colors[1] if len(hex_colors) > 1 else "#64748b",
        "accent":      hex_colors[2] if len(hex_colors) > 2 else "#3b82f6",
        "photography": photography,
    }

def _font_link(fonts: list[str]) -> str:
    families = "|".join(f.replace(" ", "+") + ":wght@300;400;600;700" for f in fonts)
    return f'<link href="https://fonts.googleapis.com/css2?family={families}&display=swap" rel="stylesheet">'

def _font_stack(fonts: list[str]) -> str:
    return ", ".join(f'"{f}"' for f in fonts) + ", -apple-system, sans-serif"

# ── Assembly prompts ──────────────────────────────────────────────

def _build_website_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    p  = bd["primary"]
    s  = bd["secondary"]
    a  = bd["accent"]
    font_link  = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)

    return f"""You are a senior frontend engineer at a world-class design agency. Build a visually stunning, \
premium B2B marketing website as a single self-contained HTML file. \
Use only HTML, CSS, and inline SVG — zero external images, zero placeholder divs, \
zero dashed-border boxes. Every pixel must look intentionally designed.

BRAND PALETTE — use these exact hex values, no substitutions:
  Primary:   {p}
  Secondary: {s}
  Accent:    {a}
  Tint A:    {p}14   (primary at ~8% opacity — subtle section backgrounds)
  Tint B:    {p}28   (primary at ~16% opacity — cards, hover states)

TYPOGRAPHY:
  Include in <head>: {font_link}
  CSS font-family: {font_stack}

CONTENT INPUTS (all copy, features, and brand details come from here):
{inputs}

BRIEF: {brief}

━━━ SECTION-BY-SECTION VISUAL SPEC ━━━

Each section below has a mandatory visual treatment. Follow it exactly.

── NAV (id="top") ──
  Full-width bar. background: {p}; height: 68px.
  Max-width 1200px centred. Logo text: white, 700 weight, 17px.
  Nav links: color rgba(255,255,255,0.65); hover color white; font-size 14px.
  CTA button: background {a}; color white; border-radius 999px; padding 9px 22px; font-size 13px; font-weight 700; no border.
  Smooth underline animation on nav link hover using CSS ::after pseudo-element.

── HERO (id="hero") ──
  min-height: 100vh; position: relative; overflow: hidden.
  Background: radial-gradient(ellipse 130% 90% at 65% 50%, {p} 0%, {p}ee 45%, {s} 100%).
  Floating decorative shapes (position: absolute, pointer-events: none, z-index 0):
    — Large circle: width 700px, height 700px, border-radius 50%, background rgba(255,255,255,0.04), top: -15%, right: -10%
    — Medium circle: width 320px, height 320px, border-radius 50%, background {a}28, bottom: -8%, left: -5%
    — Rotated square: width 180px, height 180px, background rgba(255,255,255,0.03), transform rotate(45deg), top: 20%, right: 20%
  Two-column flex layout (align-items: center, gap: 60px, padding: 120px 8% 80px):
    LEFT COLUMN (flex 1.1):
      Eyebrow: text from inputs, uppercase, {a}, font-size 11px, letter-spacing 3px, font-weight 700, margin-bottom 18px.
      H1: main headline from copy inputs, white, font-size clamp(42px,5.5vw,72px), font-weight 800, line-height 1.08, margin-bottom 22px.
      Subtitle: white at 75% opacity, font-size 18px, line-height 1.65, max-width 480px, margin-bottom 36px.
      Two CTA buttons side-by-side (flex, gap 14px):
        Primary: background {a}, color white, padding 14px 28px, border-radius 10px, font-size 15px, font-weight 700.
        Secondary: background rgba(255,255,255,0.1), color white, border: 1px solid rgba(255,255,255,0.25), same padding.
    RIGHT COLUMN (flex 1): inline SVG product screen mockup, viewBox="0 0 520 360":
      Outer drop shadow: filter drop-shadow(0 32px 64px rgba(0,0,0,0.4)).
      Frame rect: x=0 y=0 w=520 h=360 rx=14, fill=#0f172a.
      Title bar: rect x=0 y=0 w=520 h=40 rx=14, fill={p}.
        Traffic-light circles: cx=20,32,44 cy=20 r=6, fills #ff5f57, #febc2e, #28c840.
        Title text: x=260 y=25, text-anchor=middle, fill=rgba(255,255,255,0.5), font-size=12.
      Content area (below y=40):
        — Sidebar: rect x=0 y=40 w=130 h=320, fill=rgba(0,0,0,0.2).
          3 sidebar nav items as rounded rects: y=60,96,132, x=12, w=108, h=24, rx=6;
          first one filled {a}40, others rgba(255,255,255,0.05). Small square icons (8×8) at x=20.
        — Main content (x>130):
          Metric cards row (y=56): 3 rects, each 90×56, rx=8, fill=rgba(255,255,255,0.06), x=148,256,364.
            Inside each: big number text in {a} (font-size=18, y=80), small label in rgba(255,255,255,0.4) (font-size=9, y=96).
          Bar chart (y=128 to 260): 6 vertical bars at x=152,178,204,230,256,282, width=18, rx=3.
            Heights vary: 80,110,60,130,90,120 (bars grow downward from y=260, so y = 260-height).
            Bars filled {a}; last bar slightly lighter ({a}99).
          Table rows (x=310, y=136): 4 rows, height=28 each, width=200, rx=4;
            alternating fill rgba(255,255,255,0.04) and transparent.
            Short text lines: rect w=60 h=6 rx=3 fill=rgba(255,255,255,0.2) and w=40 h=6 rx=3 fill={a}60.

── FEATURES (id="features") — value proposition ──
  background: white; padding: 100px 8%.
  Section header centred (margin-bottom 56px):
    Overline: "WHY CHOOSE US" (or relevant), {a}, 11px, letter-spacing 3px, font-weight 700.
    H2: brand proposition headline from inputs, {p}, font-size clamp(28px,3.5vw,42px), font-weight 800.
    Subtitle: 16px, color #64748b, max-width 560px, centred.
  3-column CSS grid (grid-template-columns: repeat(3,1fr); gap: 28px). Each card:
    background: white; border-radius: 20px; padding: 36px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04);
    border-top: 3px solid {a};
    transition: transform 0.2s, box-shadow 0.2s;
    On hover: transform translateY(-6px); box-shadow: 0 16px 48px rgba(0,0,0,0.12).
    Icon: inline SVG 52×52px — a circle fill={p}18, inside it a relevant geometric path/shape in {p} (e.g. checkmark, graph bars, gear spokes, lightning bolt — use actual SVG path data, not emoji).
    H3: {p}, font-size 17px, font-weight 700, margin: 18px 0 10px.
    Body: #64748b, font-size 14px, line-height 1.7.

── HOW IT WORKS (id="how-it-works") — features in depth ──
  background: {p}08; padding: 100px 8%.
  Section header centred (same style as above).
  2–3 alternating rows (flex, align-items centre, gap 72px, margin-bottom 80px).
  Odd rows: text left, visual right. Even rows: visual left, text right.
  TEXT SIDE (flex 1):
    Step badge: inline block, {a}, font-size 48px, font-weight 900, opacity 0.12, position absolute, top -10px, left -6px. (The number sits as a giant ghost behind the heading.)
    H3: {p}, 22px, 700.
    Body: #475569, 15px, line-height 1.7, margin-bottom 16px.
    Bullet list: each item has a 4px wide left border in {a} and padding-left 14px.
  VISUAL SIDE (flex 1): inline SVG "feature screen", viewBox="0 0 440 300":
    Container rect: rx=12, fill=#1e293b, w=440, h=300.
    Top bar: h=36, fill={p}, rx=12 (top corners).
      Three circles traffic-light style. Tab labels: 2–3 small rounded rects fill=rgba(255,255,255,0.15).
    Content: design a unique abstract UI for each feature step using rects, circles, polylines.
      Use {a} for highlighted elements, rgba(255,255,255,0.08) for row/card backgrounds.
      Make each SVG visually different — vary the layout between a table, a chart, a form, a map grid, etc.

── TESTIMONIALS (id="testimonials") ──
  background: {p}; padding: 90px 8%.
  Section header centred in white.
  2–3 testimonial cards in flex (gap 24px). Each card:
    background: rgba(255,255,255,0.07); border-radius: 20px; padding: 36px 32px; flex: 1;
    border: 1px solid rgba(255,255,255,0.1);
    Quote mark SVG (position: absolute, top: 20px, left: 24px): a large " mark, fill={a}, opacity 0.35, font-size 80px (use SVG text element).
    Quote text: white at 90% opacity, 15px, italic, line-height 1.7, padding-top 32px.
    Separator: 1px solid rgba(255,255,255,0.15), margin 20px 0.
    Avatar row: inline SVG circle (r=22, fill={a}33, stroke={a}, stroke-width=1.5) with initials text, plus name/title.
    Name: white, 13px, 700. Title: rgba(255,255,255,0.55), 12px.

── CTA (id="contact") ──
  position: relative; overflow: hidden; padding: 110px 8%; text-align: centre.
  Background: linear-gradient(135deg, {a} 0%, {p} 100%).
  Decorative rings (position: absolute, pointer-events none):
    — Circle 600px, border: 2px solid rgba(255,255,255,0.08), border-radius 50%, centred top-left -200px,-200px.
    — Circle 400px, same style, bottom-right -150px,-150px.
  H2: white, clamp(32px,4vw,52px), 800, margin-bottom 18px.
  Subtitle: rgba(255,255,255,0.8), 17px, margin-bottom 40px.
  Two buttons: Primary white bg + {p} text; Secondary transparent + white border + white text.

── FOOTER ──
  background: #0f172a; padding: 60px 8% 32px; color: rgba(255,255,255,0.45).
  3-column grid: brand+tagline+social icons | nav links | contact info.
  Brand name white 700. Links hover white. Divider line then copyright bar, font-size 12px.

━━━ TECHNICAL REQUIREMENTS ━━━

MANDATORY IDs — these are non-negotiable. Every section must carry exactly these id attributes.
Do NOT invent custom ids based on brand names or content. Use exactly these strings:
  <nav>  or  <header>        →  id="top"
  Hero section               →  id="hero"
  Value proposition section  →  id="features"
  How-it-works section       →  id="how-it-works"
  Testimonials section       →  id="testimonials"
  CTA / contact section      →  id="contact"

MANDATORY NAV HREFS — every nav link must use exactly these href values:
  href="#hero"  href="#features"  href="#how-it-works"  href="#testimonials"  href="#contact"

MANDATORY CSS — include in the <style> block:
  html {{ scroll-behavior: smooth; }}

- Single self-contained HTML5 file. Google Fonts CDN <link> in <head>. All CSS in one <style> block.
- Responsive: columns stack below 768px; nav links collapse on mobile.
- All SVG elements must be self-contained inline SVG with explicit viewBox. No <image> tags inside SVG.
- No external images. No placeholder divs. No dashed-border boxes. No emoji as icons.
- All content (headlines, copy, feature names, testimonials) must come from the marketing inputs above.
- You MUST generate ALL six sections plus footer before closing </html>. Do not stop early.

Output ONLY valid HTML. Start with <!DOCTYPE html>. End with </html>. Zero preamble, zero explanation."""

def _build_onepager_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    p  = bd["primary"]
    s  = bd["secondary"]
    a  = bd["accent"]
    font_link  = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    return f"""You are a senior designer producing a premium B2B sales one-pager as a print-ready HTML file. \
This document goes in front of enterprise buyers and investors. \
It must look like a top-agency production piece. \
Use only HTML, CSS, and inline SVG — zero external images, zero placeholder divs, zero dashed boxes.

BRAND PALETTE:
  Primary:   {p}
  Secondary: {s}
  Accent:    {a}

TYPOGRAPHY:
  Include in <head>: {font_link}
  CSS font-family: {font_stack}

CONTENT INPUTS:
{inputs}

BRIEF: {brief}

━━━ PRINT LAYOUT SPEC — ONE A4 PAGE ━━━

The entire document must fit on a single A4 sheet (210mm × 297mm) when printed.
Use a fixed-width wrapper: width: 794px; margin: 0 auto; (794px ≈ A4 at 96 dpi).
No section should overflow. Tight spacing throughout.

── PRINT TOOLBAR (.no-print) ──
  Visible on screen, hidden on print. White bar above the page.
  "Save as PDF" button: background {p}; color white; border-radius 7px; padding 8px 20px; font-size 13px; font-weight 700.
  "Close" button: border 1px solid #e2e8f0; color #475569; same padding.

── HEADER BAND (full 794px width, height ~80px) ──
  background: linear-gradient(90deg, {p} 0%, {s} 100%); padding: 0 36px.
  Brand name: white, 20px, 800 weight. Tagline beside it: rgba(255,255,255,0.65), 12px.
  Right side: inline SVG logomark — a 40×40 geometric shape using {a} and white (abstract mark, not text).

── HERO STRIP (background {p}14, padding 28px 36px) ──
  Two columns (flex, align-items centre):
    Left (~58%): Value proposition headline from inputs, {p}, 26px, 800, line-height 1.2, max 2 lines.
    Right (~42%): One sentence supporting statement, #475569, 13px, line-height 1.6.

── STATS BAR (background {p}, padding 18px 36px) ──
  Flex row, justify-content space-evenly.
  Extract 3 concrete metrics from inputs (e.g. speed improvement, ROI, time-to-value).
  Each stat block: centred, divider lines between.
    Number: {a}, 30px, 800 weight.
    Label: rgba(255,255,255,0.65), 10px, uppercase, letter-spacing 1.5px.
  Thin vertical dividers: 1px solid rgba(255,255,255,0.18).

── MAIN BODY (padding 28px 36px, flex row, gap 28px) ──
  LEFT COLUMN (~52%, flex-direction column):
    "Why [BrandName]" heading: {p}, 14px, 700, letter-spacing 0.5px, margin-bottom 16px.
    3 feature blocks stacked (margin-bottom 18px each):
      Inline SVG icon (28×28): circle fill={p}18, geometric path inside in {p}.
      Feature title: {p}, 13px, 700, margin-bottom 5px.
      Feature body: #64748b, 12px, line-height 1.6.
      Left accent: border-left 3px solid {a}; padding-left 12px; on the title+body.
    Testimonial quote block (margin-top 20px):
      background {p}0d; border-radius 10px; padding 16px 18px;
      Large inline SVG open-quote mark, fill={a}, opacity 0.4, float left, width 20px.
      Quote text: #334155, 12px, italic, line-height 1.6.
      Attribution: {p}, 11px, 700, margin-top 8px.
  RIGHT COLUMN (~48%):
    Inline SVG product dashboard illustration, viewBox="0 0 340 280":
      Frame: rect w=340 h=280 rx=10 fill=#0f172a.
      Top bar: h=32 fill={p} rx=10 (top corners only — bottom is square).
        Traffic circles: cx=14,26,38 cy=16 r=5 fills #ff5f57, #febc2e, #28c840.
        Tab strip: 3 small rounded rects fill=rgba(255,255,255,0.1), w=48, h=16, y=8, x=60/116/172.
      Left sidebar: rect x=0 y=32 w=80 h=248 fill=rgba(0,0,0,0.18).
        4 nav items: rounded rects y=48,76,104,132 x=8 w=64 h=20 rx=5;
        first fill={a}50, rest rgba(255,255,255,0.05).
      Main panel (x>80):
        KPI cards row (y=42): 3 cards, each 70×44 rx=6 fill=rgba(255,255,255,0.06) x=90,172,254.
          Big number text {a} font-size=14 y=62; small label rgba(255,255,255,0.35) font-size=7 y=76.
        Chart area (y=96 to 200 x=88 to 330):
          Background rect fill=rgba(255,255,255,0.03) rx=6.
          5 vertical bars x=100,130,160,190,220 width=18 rx=3 fill={a}; heights 60,85,45,100,70 (from y=200 upward).
          Line overlay: polyline points connecting bar tops, stroke={a}80 stroke-width=1.5 fill=none.
        Table rows (y=212 to 272): 3 rows h=18 w=240 x=88 rx=3;
          alternating fill rgba(255,255,255,0.03)/transparent.
          Two text stubs per row: w=50 h=5 rx=2 and w=30 h=5 rx=2, fill=rgba(255,255,255,0.15) and {a}60.

── FOOTER BAR (background {p}, height 36px, padding 0 36px) ──
  Flex, align-items centre, justify-content space-between.
  Brand name: white, 11px, 700. Centre: website URL placeholder: rgba(255,255,255,0.6), 10px. Right: "Confidential": rgba(255,255,255,0.4), 10px.

━━━ TECHNICAL REQUIREMENTS ━━━
- Self-contained HTML5. Google Fonts CDN in <head>. All CSS in one <style> block.
- @page {{ size: A4; margin: 0; }}
- body {{ margin: 0; background: #f1f5f9; }}
- .page {{ width: 794px; margin: 20px auto; background: white; }}
- @media print {{ .no-print {{ display: none !important; }} body {{ background: white; margin: 0; }} .page {{ margin: 0; box-shadow: none; }} }}
- Zero external images. Zero placeholder divs. Zero dashed boxes. Every zone fully designed.
- All content (metrics, features, testimonial, brand name) must come from the inputs above.

Output ONLY valid HTML. Start with <!DOCTYPE html>. No explanation."""

def _build_deck_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    font_link = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    return f"""Generate a complete HTML presentation deck for a B2B product pitch.

BRAND COLORS: Primary: {bd['primary']} | Secondary: {bd['secondary']} | Accent: {bd['accent']}
FONTS CDN: {font_link}
Font stack: {font_stack}

INPUTS:
{inputs}

ORIGINAL BRIEF: {brief}

REQUIREMENTS:
- Self-contained HTML5, Google Fonts CDN in <head>, all CSS inline.
- Each slide is a full-viewport <section class="slide"> div.
- 10–12 slides: Title, Problem, Solution, Key Features (3), Demo slide (demo-placeholder: "📹 Live demo / screen recording"), Proof/Results, Pricing/Plans, Team placeholder, CTA/Next Steps.
- Keyboard navigation (ArrowLeft/ArrowRight) + on-screen prev/next buttons.
- Slide counter (e.g. "3 / 12") in corner.
- Brand color scheme: header/title slides use primary color background; content slides white with primary accents.
- demo-placeholder on demo slide: background:rgba(255,255,255,0.15); border:2px dashed rgba(255,255,255,0.5); border-radius:10px; padding:60px; text-align:center; color:rgba(255,255,255,0.8); font-size:16px;
- @media print: each slide prints as one page.

Output ONLY valid HTML. Start with <!DOCTYPE html>. No explanation."""

async def assemble_html(prompt: str, system_msg: str) -> str:
    html = await call_agent(system_msg, prompt, model="claude-opus-4-6", max_tokens=16000)
    m = re.search(r"<!DOCTYPE", html, re.IGNORECASE)
    return html[m.start():] if m else html

async def assemble_social_pack(outputs: dict, brief: str) -> str:
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    prompt = f"""Compile these marketing outputs into a structured Social Media Content Pack.

INPUTS:\n{inputs}\n\nORIGINAL BRIEF: {brief}

Format: clean markdown with sections: Cover (client, date, brief summary) | LinkedIn Posts (numbered, copy-paste ready) | Instagram Captions (numbered, with hashtag sets) | TikTok/Reels Scripts | Facebook Posts | Design Briefs for Static Posts | Video Production Brief (flagged: use real screen recordings, not AI images) | Hashtag Master List.

Output clean final markdown. No meta-commentary."""
    return await call_agent("You compile marketing content packs.", prompt,
                            model="claude-opus-4-6", max_tokens=4096)

ASSEMBLERS = {
    "html_website":        (".html",  "website"),
    "html_onepager":       (".html",  "one-pager"),
    "social_pack":         (".md",    "social-pack"),
    "canva_json":          (".json",  "canva"),
    "social_pack_cascade": (".md",    "social-pack"),
    "video_brief":         (".html",  "video-brief"),
}

# ── Canva JSON template ───────────────────────────────────────────

def _build_canva_json_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    palette_json = json.dumps(bd["hex_colors"])
    hfont = bd["fonts"][0]
    bfont = bd["fonts"][-1] if len(bd["fonts"]) > 1 else bd["fonts"][0]
    return f"""Extract all approved brand and copy data into a structured JSON template for Canva.

BRAND COLORS:
Primary: {bd['primary']}
Secondary: {bd['secondary']}
Accent: {bd['accent']}
Full palette: {', '.join(bd['hex_colors'])}

FONTS: {', '.join(bd['fonts'])}

APPROVED MARKETING OUTPUTS:
{inputs}

ORIGINAL BRIEF: {brief}

Generate a JSON object with this exact structure — fill ALL fields with real extracted content:

{{
  "brand": {{
    "name": "extracted brand name",
    "tagline": "extracted tagline",
    "description": "one paragraph brand description",
    "url": "website URL if mentioned, else empty string",
    "colors": {{
      "primary": "{bd['primary']}",
      "secondary": "{bd['secondary']}",
      "accent": "{bd['accent']}",
      "palette": {palette_json}
    }},
    "typography": {{
      "heading_font": "{hfont}",
      "body_font": "{bfont}"
    }}
  }},
  "copy": {{
    "headline": "main headline from approved copy",
    "subheadline": "subheadline",
    "value_proposition": "one sentence value prop",
    "cta_primary": "primary CTA button text",
    "cta_secondary": "secondary CTA text",
    "features": [
      {{"title": "feature 1 name", "description": "brief description"}},
      {{"title": "feature 2 name", "description": "brief description"}},
      {{"title": "feature 3 name", "description": "brief description"}}
    ],
    "testimonials": [
      {{"quote": "testimonial text", "author": "Name, Title, Company"}},
      {{"quote": "second testimonial", "author": "Name, Title, Company"}}
    ],
    "social_posts": [
      {{"platform": "linkedin", "caption": "copy-paste ready 150-word post"}},
      {{"platform": "instagram", "caption": "short caption with 10-15 hashtags"}},
      {{"platform": "facebook", "caption": "post copy"}}
    ]
  }},
  "canva_bulk_create": {{
    "note": "Import these rows into Canva Bulk Create. Map field names to named elements in your template.",
    "rows": [
      {{"slide": "Hero", "Headline": "main headline", "Subheadline": "subheadline", "Body": "supporting line", "CTA": "button text", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "Value Prop", "Headline": "section header", "Body": "value proposition paragraph", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "Feature 1", "Headline": "feature 1 title", "Body": "feature 1 description", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "Feature 2", "Headline": "feature 2 title", "Body": "feature 2 description", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "Feature 3", "Headline": "feature 3 title", "Body": "feature 3 description", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "Testimonial", "Headline": "quote text", "Body": "author attribution", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}},
      {{"slide": "CTA", "Headline": "closing headline", "Body": "supporting close", "CTA": "final button text", "Color1": "{bd['primary']}", "Color2": "{bd['secondary']}", "Font": "{hfont}"}}
    ]
  }}
}}

Output ONLY valid JSON. Start with {{ and end with }}. No markdown, no explanation."""

async def assemble_canva_json(outputs: dict, brief: str) -> str:
    prompt = _build_canva_json_prompt(outputs, brief)
    raw = await call_agent(
        "You extract brand and copy data into structured JSON for Canva. Output only valid JSON.",
        prompt, model="claude-sonnet-4-6", max_tokens=4096)
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.dumps(json.loads(m.group(0)), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return raw

# ── Social cascade + video brief assembly ─────────────────────────

async def assemble_social_cascade(outputs: dict, brief: str) -> str:
    """Fire Callum (LinkedIn) + Priya (Instagram/Facebook) in parallel, return social pack markdown."""
    all_content = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    social_brief = (
        f"Original brief: {brief}\n\n"
        "APPROVED MARKETING OUTPUTS (use as source of truth for all content):\n\n"
        f"{all_content}\n\n---\n\n"
        "Produce one piece of final, copy-paste-ready social content. "
        "Make it specific to the brand and campaign above — no filler, no placeholders."
    )

    async def run(name: str) -> tuple[str, str]:
        sys_ = await fetch_agent(name)
        out  = await call_agent(sys_, social_brief)
        return name, out

    results = await asyncio.gather(run("callum"), run("priya"))
    agent_outputs = dict(results)
    callum_out = agent_outputs.get("callum", "")
    priya_out  = agent_outputs.get("priya", "")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Social Content Pack\n\n"
        f"**Generated:** {ts}  \n"
        f"**Brief:** {brief}\n\n"
        "---\n\n"
        "## LinkedIn Post (Callum)\n\n"
        f"{callum_out}\n\n"
        "---\n\n"
        "## Instagram & Facebook (Priya)\n\n"
        f"{priya_out}\n"
    )


async def assemble_video_brief(outputs: dict, brief: str) -> str:
    """Fire Dante for a video brief, return PDF-ready HTML."""
    all_content = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    bd = extract_brand_data(outputs)
    video_brief_brief = (
        f"Original brief: {brief}\n\n"
        "APPROVED MARKETING OUTPUTS:\n\n"
        f"{all_content}\n\n---\n\n"
        "Produce a complete, formatted video brief document. Include:\n"
        "1. Campaign overview and creative concept\n"
        "2. Shot list (numbered, each with shot type, location/setting, action, duration)\n"
        "3. Script outline (hook → problem → solution → CTA structure)\n"
        "4. TikTok specs (aspect ratio 9:16, duration 15–60s, caption, hashtags, hook text)\n"
        "5. Instagram Reels specs (aspect ratio 9:16, duration 15–90s, cover frame, caption)\n"
        "6. Hook direction (first 3 seconds — what grabs attention, exact opening line)\n"
        "7. Talent and production notes\n\n"
        "Use the brand colors and typography from the approved outputs above. "
        "Be specific — this brief goes straight to the production team.\n\n"
        "CRITICAL: You MUST deliver complete sections 1–7 for EVERY reel requested in the brief. "
        "Do not truncate, summarise, or skip any reel. If the brief asks for 5 reels, deliver all 5 in full."
    )
    sys_ = await fetch_agent("dante")
    content = await call_agent(sys_, video_brief_brief, model="claude-sonnet-4-6", max_tokens=16000)

    # Wrap in print-ready HTML
    primary   = bd["primary"]
    secondary = bd["secondary"]
    accent    = bd["accent"]
    font_link = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    body_html = _md_to_html(content)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Video Brief — {bd.get('brand_name', 'Campaign')}</title>
  {font_link}
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: {font_stack}; background: #f8fafc; color: #1e293b; padding: 0; }}
    .toolbar {{ background: {primary}; padding: 12px 32px; display: flex; gap: 10px; align-items: center; }}
    .toolbar-brand {{ color: white; font-size: 14px; font-weight: 700; flex: 1; }}
    .btn {{ background: white; color: {primary}; border: none; padding: 8px 18px; border-radius: 6px;
            font-size: 12px; font-weight: 700; cursor: pointer; }}
    .btn-outline {{ background: transparent; color: white; border: 1px solid rgba(255,255,255,0.4); }}
    .doc {{ max-width: 820px; margin: 0 auto; padding: 48px 40px 80px; background: white;
            min-height: 100vh; box-shadow: 0 0 40px rgba(0,0,0,0.06); }}
    .doc-header {{ border-left: 4px solid {primary}; padding-left: 20px; margin-bottom: 36px; }}
    .doc-header h1 {{ font-size: 22px; font-weight: 800; color: {primary}; margin-bottom: 4px; }}
    .doc-meta {{ font-size: 12px; color: #64748b; }}
    .brief-box {{ background: {primary}0d; border: 1px solid {primary}30; border-radius: 8px;
                  padding: 12px 16px; margin-bottom: 32px; font-size: 13px; color: #475569; }}
    h1 {{ font-size: 20px; font-weight: 800; color: {primary}; margin: 32px 0 12px; }}
    h2 {{ font-size: 17px; font-weight: 700; color: {primary}; margin: 28px 0 10px;
          padding-bottom: 6px; border-bottom: 1px solid {primary}20; }}
    h3 {{ font-size: 14px; font-weight: 700; color: #334155; margin: 18px 0 8px; }}
    p  {{ font-size: 14px; line-height: 1.8; margin-bottom: 12px; color: #334155; }}
    ul {{ padding-left: 20px; margin-bottom: 14px; }}
    li {{ font-size: 14px; line-height: 1.7; margin-bottom: 5px; color: #334155; }}
    strong {{ font-weight: 700; color: #1e293b; }}
    hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }}
    @media print {{
      .toolbar {{ display: none !important; }}
      body {{ background: white; }}
      .doc {{ box-shadow: none; padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="toolbar no-print">
    <div class="toolbar-brand">📹 Video Brief</div>
    <button class="btn" onclick="window.print()">Save as PDF</button>
    <button class="btn btn-outline" onclick="window.close()">Close</button>
  </div>
  <div class="doc">
    <div class="doc-header">
      <h1>Video Production Brief</h1>
      <div class="doc-meta">Generated {ts}</div>
    </div>
    <div class="brief-box"><strong>Brief:</strong> {brief}</div>
    {body_html}
  </div>
</body>
</html>"""


# ── Apify / Zara competitor enrichment ───────────────────────────

_COMPETITOR_RE = re.compile(
    r"\b(competitor[s]?|competitive|intelligence|benchmark|audit|landscape|market research|rival[s]?)\b",
    re.IGNORECASE,
)

def _extract_handles(text: str) -> list[str]:
    """Pull @handles and names after 'competitor/rival/vs' from text."""
    handles = re.findall(r"@([\w.]{2,30})", text)
    names   = re.findall(
        r"(?:competitor[s]?|rival[s]?|vs\.?)[:\s]+([A-Za-z0-9_.]{2,30})",
        text, re.IGNORECASE,
    )
    seen: dict[str, None] = {}
    for h in handles + names:
        seen[h.lstrip("@").lower()] = None
    return list(seen.keys())

def _run_apify_sync(handles: list[str]) -> dict:
    client = ApifyClient(APIFY_API_TOKEN)
    run = client.actor("apify/instagram-scraper").call(run_input={
        "directUrls": [f"https://www.instagram.com/{h}/" for h in handles],
        "resultsType": "posts",
        "resultsLimit": 5,
    })
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return {"handles": handles, "posts": items[:20]}

async def enrich_zara_with_apify(zara_output: str, brief: str) -> str:
    """Append live Instagram competitor data to Zara's output when brief signals research."""
    if not APIFY_API_TOKEN:
        return zara_output
    if not _COMPETITOR_RE.search(brief):
        return zara_output
    handles = _extract_handles(brief + "\n" + zara_output)
    if not handles:
        return zara_output
    logging.info("Apify: scraping Instagram for handles %s", handles)
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _run_apify_sync, handles)
    except Exception as e:
        logging.error("Apify scrape failed: %s", e)
        return zara_output
    posts = data.get("posts", [])
    if not posts:
        return zara_output
    lines = ["\n\n---\n\n## Competitor Intelligence — Live Instagram Data\n"]
    for post in posts[:10]:
        handle  = post.get("ownerUsername", "unknown")
        caption = (post.get("caption") or "")[:200].replace("\n", " ")
        likes   = post.get("likesCount", 0)
        comments = post.get("commentsCount", 0)
        url     = post.get("url", "")
        lines.append(f"**@{handle}** — {likes} likes · {comments} comments")
        if caption:
            lines.append(f"> {caption.strip()}")
        if url:
            lines.append(f"[View post]({url})")
        lines.append("")
    return zara_output + "\n".join(lines)


# ── Buffer integration ────────────────────────────────────────────

def _post_to_buffer_sync(text: str) -> dict:
    profiles_r = requests.get(
        "https://api.bufferapp.com/1/profiles.json",
        params={"access_token": BUFFER_ACCESS_TOKEN},
        timeout=15,
    )
    profiles_r.raise_for_status()
    profiles = profiles_r.json()
    linkedin = [p for p in profiles if p.get("service") == "linkedin"]
    target   = linkedin if linkedin else profiles
    if not target:
        raise ValueError("No Buffer profiles found")
    profile_id = target[0]["id"]
    update_r = requests.post(
        "https://api.bufferapp.com/1/updates/create.json",
        data={
            "access_token": BUFFER_ACCESS_TOKEN,
            "profile_ids[]": profile_id,
            "text": text[:3000],
        },
        timeout=15,
    )
    update_r.raise_for_status()
    return update_r.json()

async def post_to_buffer(social_pack_md: str) -> None:
    if not BUFFER_ACCESS_TOKEN:
        return
    m = re.search(
        r"## LinkedIn Post \(Callum\)\s*\n\n(.+?)(?:\n\n---|\Z)",
        social_pack_md, re.DOTALL,
    )
    if not m:
        logging.warning("Buffer: LinkedIn post not found in social pack")
        return
    linkedin_text = m.group(1).strip()
    if not linkedin_text:
        return
    logging.info("Buffer: queuing LinkedIn post (%d chars)", len(linkedin_text))
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _post_to_buffer_sync, linkedin_text)
        logging.info("Buffer: queued successfully — %s", result)
    except Exception as e:
        logging.error("Buffer post failed: %s", e)


# ── GitHub + Anthropic ────────────────────────────────────────────

async def fetch_agent(name: str) -> str:
    if name in _agent_cache:
        return _agent_cache[name]
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/agents/{name}.md"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    async with httpx.AsyncClient() as h:
        r = await h.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    _agent_cache[name] = r.text
    return r.text

async def call_agent(system: str, message: str, model: str = "claude-sonnet-4-6",
                     max_tokens: int = 4096) -> str:
    r = await anthropic_client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": message}],
    )
    return r.content[0].text

# ── Save helpers ──────────────────────────────────────────────────

def _slug(text: str, n: int = 35) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text[:n].lower()).strip("-")

def save_markdown(job: dict) -> Path:
    ts    = job["timestamp"]
    title = job.get("title") or job["brief"]
    path  = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{_slug(title)}.md"
    client_label = {"wibiz": "WiBiz", "ai-living": "AI Living", "charter": "Club Charter"}.get(
        job.get("client", "other"), job.get("project_name") or "Other")
    lines = ["# Studio N — Job Output", "",
             f"**Title:** {title}", "",
             f"**Client:** {client_label}", "",
             f"**Date:** {ts.strftime('%Y-%m-%d %H:%M')}", "",
             f"**Brief:** {job['brief']}", "", "---", ""]
    if job.get("marcus_analysis"):
        lines += ["## Marcus — Brief Analysis", "", job["marcus_analysis"], "", "---", ""]
    for a, c in (job.get("stage1_outputs") or {}).items():
        lines += [f"## {a.capitalize()} — Stage 1", "", c, "", "---", ""]
    if job.get("marcus_review"):
        lines += ["## Marcus — Stage 1 Review", "", job["marcus_review"], "", "---", ""]
    for a, c in (job.get("cascade_outputs") or {}).items():
        lines += [f"## {a.capitalize()} — Cascade", "", c, "", "---", ""]
    if job.get("marcus_cascade_review"):
        lines += ["## Marcus — Final Review", "", job["marcus_cascade_review"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def save_assembled(job: dict, assembly_type: str, content: str) -> Path:
    ts = job["timestamp"]
    ext, label = ASSEMBLERS[assembly_type]
    path = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{_slug(job['brief'])}_{label}{ext}"
    path.write_text(content, encoding="utf-8")
    return path

# ── Markdown → HTML ───────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = t.split("**")
    t = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
    for pat, rep in [
        (r"^### (.+)$", r"<h3>\1</h3>"), (r"^## (.+)$", r"<h2>\1</h2>"),
        (r"^# (.+)$",   r"<h1>\1</h1>"), (r"^---+$",    "<hr>"),
        (r"^[-•] (.+)$", r"<li>\1</li>"),
    ]:
        t = re.sub(pat, rep, t, flags=re.MULTILINE)
    out = []
    for chunk in t.split("\n\n"):
        chunk = chunk.strip()
        if not chunk: continue
        if re.match(r"^<(h[1-6]|li|hr)", chunk):
            chunk = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", chunk, flags=re.DOTALL)
            out.append(chunk)
        else:
            out.append(f"<p>{chunk.replace(chr(10), '<br>')}</p>")
    return "\n".join(out)

# ── Shared chrome ─────────────────────────────────────────────────

_PAGE_STYLE = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; min-height: 100vh; }
  nav { background: #1e293b; padding: 14px 32px; display: flex; align-items: center; justify-content: space-between; }
  .nav-brand { color: white; font-size: 16px; font-weight: 700; text-decoration: none; letter-spacing: -0.3px; }
  .nav-brand span { color: #64748b; font-weight: 400; font-size: 13px; margin-left: 8px; }
  .nav-links { display: flex; gap: 6px; align-items: center; }
  .nav-link { color: #94a3b8; font-size: 13px; text-decoration: none; padding: 6px 12px; border-radius: 6px; transition: all 0.15s; }
  .nav-link:hover, .nav-link.active { background: rgba(255,255,255,0.1); color: white; }
  .container { max-width: 900px; margin: 0 auto; padding: 32px 24px 80px; }
  .dm-toggle { width:32px;height:32px;border-radius:7px;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.06);cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.15s;color:#94a3b8; }
  .dm-toggle:hover { background:rgba(255,255,255,0.14);color:white; }
  /* Job cards */
  .job-card { background:white;border-radius:12px;padding:20px 24px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.08); }
  .job-top { display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:12px; }
  .job-title { font-size:15px;font-weight:700;color:#1e293b; }
  .job-ts { font-size:12px;color:#94a3b8;margin-bottom:6px;margin-top:3px; }
  .job-brief-text { font-size:12px;color:#64748b;line-height:1.5; }
  .job-agents { display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px; }
  .agent-chip { font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;background:#f1f5f9;color:#475569;white-space:nowrap; }
  .job-files { display:flex;flex-wrap:wrap;gap:6px;align-items:center; }
  .job-rebrief { font-size:11px;font-weight:700;padding:5px 12px;border-radius:6px;background:#f8fafc;color:#475569;text-decoration:none;border:1px solid #e2e8f0;white-space:nowrap;cursor:pointer;transition:all 0.15s; }
  .job-rebrief:hover { background:#0f172a;color:white;border-color:#0f172a; }
  /* Tabs */
  .tab-bar { display:flex;flex-wrap:wrap;gap:6px;margin-bottom:24px; }
  .tab-link { font-size:12px;font-weight:700;padding:6px 14px;border-radius:6px;text-decoration:none;white-space:nowrap;transition:all 0.15s; }
  /* Search */
  .search-bar { margin-bottom:20px; }
  .search-input { width:100%;border:1px solid #e2e8f0;border-radius:9px;padding:10px 14px;font-size:14px;font-family:inherit;color:#1e293b;outline:none;background:white;transition:border-color 0.15s; }
  .search-input:focus { border-color:#94a3b8; }
  /* Page head */
  .page-head { display:flex;align-items:baseline;justify-content:space-between;margin-bottom:16px; }
  .page-title { font-size:20px;font-weight:700; }
  .page-count { font-size:13px;color:#94a3b8; }
  /* Star button */
  .star-btn { background:none;border:none;cursor:pointer;font-size:16px;padding:0 2px;line-height:1;color:#cbd5e1;transition:color 0.15s;flex-shrink:0; }
  .star-btn:hover { color:#f59e0b; }
  .job-card.starred .star-btn { color:#f59e0b; }
  /* Checkbox */
  .job-check { width:15px;height:15px;cursor:pointer;flex-shrink:0;accent-color:#1e293b; }
  /* Bulk archive bar */
  .bulk-bar { display:none;position:sticky;bottom:20px;left:0;right:0;margin:0 auto;max-width:400px;
    background:#1e293b;color:white;border-radius:12px;padding:12px 20px;
    align-items:center;justify-content:space-between;gap:12px;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);z-index:50; }
  .bulk-bar.visible { display:flex; }
  .bulk-bar-label { font-size:13px;font-weight:600; }
  .bulk-archive-btn { background:#ef4444;color:white;border:none;padding:7px 16px;border-radius:8px;
    font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;transition:background 0.15s; }
  .bulk-archive-btn:hover { background:#dc2626; }
  .bulk-cancel-btn { background:rgba(255,255,255,0.1);color:white;border:none;padding:7px 12px;border-radius:8px;
    font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;transition:background 0.15s; }
  .bulk-cancel-btn:hover { background:rgba(255,255,255,0.2); }
  /* ZIP button */
  .zip-btn { font-size:11px;font-weight:700;padding:5px 11px;border-radius:6px;
    background:#f0fdf4;color:#16a34a;text-decoration:none;white-space:nowrap;
    border:1px solid #86efac;transition:all 0.15s; }
  .zip-btn:hover { background:#dcfce7; }
  /* Mobile responsive */
  @media (max-width: 640px) {
    nav { padding: 12px 16px; }
    .container { padding: 16px 16px 60px; }
    .job-card { padding: 16px 18px; }
    .tab-bar { overflow-x: auto; flex-wrap: nowrap; -webkit-overflow-scrolling: touch; }
    .search-input { font-size: 16px; }
  }
  /* Dark mode */
  body.dark { background:#0b1120;color:#cbd5e1; }
  body.dark nav { background:#090f1c; }
  body.dark .page-title { color:#e2e8f0; }
  body.dark .page-count { color:#475569; }
  body.dark .job-card { background:#141e2e;box-shadow:0 1px 3px rgba(0,0,0,0.4); }
  body.dark .job-title { color:#e2e8f0; }
  body.dark .job-ts { color:#475569; }
  body.dark .job-brief-text { color:#475569; }
  body.dark .agent-chip { background:#1e293b;color:#64748b; }
  body.dark .job-rebrief { background:#1e293b;color:#64748b;border-color:#1e293b; }
  body.dark .job-rebrief:hover { background:#334155;color:#e2e8f0;border-color:#334155; }
  body.dark .search-input { background:#141e2e;border-color:#1e293b;color:#e2e8f0; }
  body.dark .search-input:focus { border-color:#334155; }
  body.dark .search-input::placeholder { color:#475569; }
"""

_PRINT_STYLE = """
  body { font-family: -apple-system, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 24px 60px; color: #1e293b; }
  .toolbar { display:flex; gap:10px; margin-bottom:28px; }
  .btn { background:#1e293b; color:white; border:none; padding:9px 20px; border-radius:7px; font-size:13px; font-weight:600; cursor:pointer; }
  .btn-outline { background:none; border:1px solid #e2e8f0; color:#475569; }
  h1 { font-size:20px; font-weight:700; margin-bottom:4px; }
  .meta { color:#64748b; font-size:13px; margin-bottom:8px; }
  .brief-box { background:#f8fafc; border-left:3px solid #1e293b; padding:11px 16px; margin-bottom:28px; font-size:13px; color:#475569; border-radius:0 6px 6px 0; }
  h2{font-size:16px;font-weight:700;margin:28px 0 10px;} h3{font-size:14px;font-weight:700;margin:20px 0 8px;}
  p{font-size:14px;line-height:1.75;margin-bottom:12px;} ul{padding-left:20px;margin-bottom:12px;}
  li{font-size:14px;line-height:1.7;margin-bottom:4px;} strong{font-weight:700;}
  hr{border:none;border-top:1px solid #e2e8f0;margin:24px 0;}
  @media print { .toolbar{display:none!important;} body{margin:20px;} }
"""

def _nav(active: str) -> str:
    def lnk(href, label, key):
        cls = "nav-link active" if key == active else "nav-link"
        return f'<a href="{href}" class="{cls}">{label}</a>'
    return f"""<nav>
      <a href="/" class="nav-brand">Studio N<span>by Marcus</span></a>
      <div class="nav-links">
        {lnk('/','Brief','brief')}
        {lnk('/outputs','Outputs','outputs')}
        <button class="dm-toggle" onclick="toggleDark()" title="Toggle dark mode">
          <svg id="dm-moon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          <svg id="dm-sun"  width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
        </button>
        <a href="/logout" class="nav-link" style="color:#6b7280;">Sign out</a>
      </div>
    </nav>
    <script>
    (function(){{if(localStorage.getItem('dm')==='1'){{document.body.classList.add('dark');var m=document.getElementById('dm-moon'),s=document.getElementById('dm-sun');if(m)m.style.display='none';if(s)s.style.display='';}} }})();
    function toggleDark(){{var d=document.body.classList.toggle('dark');localStorage.setItem('dm',d?'1':'0');var m=document.getElementById('dm-moon'),s=document.getElementById('dm-sun');if(m)m.style.display=d?'none':'';if(s)s.style.display=d?'':'none';}}
    </script>"""

def _print_page(title: str, brief: str, date: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<style>{_PRINT_STYLE}</style></head><body>
<div class="toolbar">
  <button class="btn" onclick="window.print()">Save as PDF</button>
  <button class="btn btn-outline" onclick="window.close()">Close</button>
</div>
<h1>{title}</h1><div class="meta">{date}</div>
<div class="brief-box"><strong>Brief:</strong> {brief}</div>
{body}</body></html>"""

# ── Agent extraction (lightweight second call) ────────────────────

async def extract_agents_from_analysis(analysis: str, brief: str) -> list[str]:
    """Dedicated extraction call — returns only a JSON array of agent names.
    Uses a cheap model with max_tokens=200 so it has nothing to do but output the array."""
    prompt = (
        f"Marketing brief: {brief}\n\n"
        f"Strategic analysis:\n{analysis[:2000]}\n\n"
        "Which specialist agents are needed for this project? "
        "Choose from: nadia, felix, callum, priya, dante, suki, reeva, zara\n"
        "Return only a JSON array of names. Example: [\"reeva\", \"felix\", \"nadia\"]\n"
        "No explanation. No other text. Just the array."
    )
    try:
        raw = await call_agent(
            "You extract agent names from marketing briefs. Output only a JSON array of names, nothing else.",
            prompt,
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
        )
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            names = json.loads(m.group(0))
            valid = [n for n in names if n in VALID_AGENTS]
            logging.info("extraction call returned %s → valid agents: %s", names, valid)
            return valid
        logging.warning("extraction call had no JSON array in response: %r", raw[:200])
    except Exception as e:
        logging.error("extraction call failed: %s", e)
    return []


# ── Job processor ─────────────────────────────────────────────────

async def process_job(job_id: str, brief: str, title: str = "",
                      client: str = "other", project_name: str = "",
                      allowed_agents: list[str] | None = None) -> None:
    q = _jobs[job_id]

    async def emit(ev: dict) -> None:
        await q.put(ev)

    job: dict = {
        "brief": brief,
        "timestamp": datetime.now(),
        "title": title or brief[:60],
        "client": client,
        "project_name": project_name,
    }

    try:
        # Build context-enriched brief — tells Marcus and all agents exactly which
        # client or project this is for, so they don't have to guess from tone clues.
        _CLIENT_LABELS = {"wibiz": "WiBiz", "ai-living": "AI Living", "charter": "Club Charter"}
        _label = _CLIENT_LABELS.get(client)
        if _label:
            enriched_brief = f"CLIENT: {_label}\n\nBRIEF:\n{brief}"
        elif project_name:
            enriched_brief = f"PROJECT: {project_name}\n\nBRIEF:\n{brief}"
        else:
            enriched_brief = brief

        await emit({"type": "status", "message": "Marcus is reading your brief..."})
        marcus_system = await fetch_agent("marcus")
        # Prefix puts the JSON requirement as the very first thing Marcus reads,
        # before his own identity prompt, so it can't be overridden by prose flow.
        marcus_raw = await call_agent(
            ORCHESTRATION_PREFIX + marcus_system, enriched_brief, model="claude-opus-4-6")

        # Strip JSON block from analysis text if Marcus included one.
        json_match      = re.search(r"```json\s*(\{.*?\})\s*```", marcus_raw, re.DOTALL)
        marcus_analysis = marcus_raw[:json_match.start()].strip() if json_match else marcus_raw
        agents_needed:  list[str] = []
        agent_briefs:   dict[str, str] = {}

        # Layer 1: try to parse Marcus's own JSON block (best quality — includes written briefs).
        if json_match:
            try:
                p             = json.loads(json_match.group(1))
                agents_needed = p.get("agents_needed", [])
                agent_briefs  = p.get("briefs", {})
                logging.info("job %s — layer1: parsed Marcus JSON block, agents=%s", job_id, agents_needed)
            except json.JSONDecodeError as e:
                logging.error("job %s — layer1: Marcus JSON parse failed: %s", job_id, e)
        else:
            logging.warning("job %s — layer1: no JSON block in Marcus response", job_id)

        # Layer 2: dedicated extraction call — lightweight model, max_tokens=200, returns only an array.
        # Runs whenever layer 1 produced no agents. Cannot be confused by prose.
        if not agents_needed:
            await emit({"type": "status", "message": "Extracting agent plan..."})
            agents_needed = await extract_agents_from_analysis(marcus_analysis, enriched_brief)
            if agents_needed:
                agent_briefs = {n: f"Original brief: {enriched_brief}\n\nMarcus's analysis:\n{marcus_analysis}"
                                for n in agents_needed}
                logging.info("job %s — layer2: extraction call resolved agents=%s", job_id, agents_needed)
            else:
                logging.warning("job %s — layer2: extraction call returned no agents", job_id)

        # Layer 3: last-resort text scan for agent name mentions.
        if not agents_needed:
            mentioned = [n for n in VALID_AGENTS if re.search(rf"\b{n}\b", marcus_raw, re.IGNORECASE)]
            if mentioned:
                agents_needed = mentioned
                agent_briefs  = {n: f"Original brief: {enriched_brief}\n\nMarcus's analysis:\n{marcus_analysis}"
                                 for n in mentioned}
                logging.warning("job %s — layer3: text-scan inferred agents=%s", job_id, mentioned)
            else:
                logging.warning("job %s — layer3: no agents found anywhere; pipeline will produce no output", job_id)

        logging.info("job %s — final agents_needed: %s", job_id, agents_needed)
        job["marcus_analysis"] = marcus_analysis
        await emit({"type": "marcus_analysis", "content": marcus_analysis})

        valid_s1 = [n for n in agents_needed if n in agent_briefs and n in VALID_AGENTS]
        if allowed_agents:
            valid_s1 = [n for n in valid_s1 if n in allowed_agents]
        if valid_s1:
            await emit({"type": "agents_identified", "agents": valid_s1})
            await emit({"type": "status",
                        "message": f"Running {', '.join(n.capitalize() for n in valid_s1)} in parallel..."})

        async def run_s1(name: str) -> tuple[str, str]:
            sys_ = await fetch_agent(name)
            out  = await call_agent(sys_, agent_briefs[name])
            return name, out

        stage1_outputs: dict[str, str] = {}
        if valid_s1:
            results = await asyncio.gather(*[run_s1(n) for n in valid_s1])
            stage1_outputs = dict(results)
            # Zara: append live competitor Instagram data when brief signals research
            if "zara" in stage1_outputs:
                await emit({"type": "status", "message": "Zara: running competitor intelligence scrape..."})
                stage1_outputs["zara"] = await enrich_zara_with_apify(stage1_outputs["zara"], enriched_brief)
            for name, content in stage1_outputs.items():
                logging.info("job %s — stage1 %s: %d chars", job_id, name, len(content))
                await emit({"type": "specialist", "agent": name, "content": content})
        else:
            logging.warning("job %s — no valid stage1 agents to run (valid_s1=%s)", job_id, valid_s1)

        job["stage1_outputs"] = stage1_outputs

        # Stage 1 review + cascade decision
        marcus_review = None
        cascade_plan  = {"next_agents": [], "assembly": "none", "flags": []}

        if stage1_outputs:
            await emit({"type": "status", "message": "Marcus is reviewing outputs..."})
            review_input = (
                "Review each output against the brief and brand standards. "
                "Give your verdict (Approved / Revise / Reject) for each.\n\n"
                + "".join(f"## {n.capitalize()}\n\n{o}\n\n---\n\n"
                          for n, o in stage1_outputs.items())
            )
            review_raw = await call_agent(
                marcus_system + REVIEW_CASCADE_SUFFIX,
                review_input, model="claude-opus-4-6")

            cm = re.search(r"```cascade\s*(\{.*?\})\s*```", review_raw, re.DOTALL)
            marcus_review = review_raw[:cm.start()].strip() if cm else review_raw

            if cm:
                try:
                    cascade_plan = json.loads(cm.group(1)).get("cascade", cascade_plan)
                except json.JSONDecodeError as e:
                    logging.error("job %s — failed to parse Marcus cascade JSON: %s\nRaw block: %s",
                                  job_id, e, cm.group(1)[:500])
            else:
                logging.warning("job %s — Marcus review returned no cascade block", job_id)

            logging.info("job %s — cascade plan: %s", job_id, cascade_plan)
            job["marcus_review"] = marcus_review
            await emit({"type": "review", "content": marcus_review})

        # Stage 2: cascade agents
        cascade_outputs: dict[str, str] = {}
        next_agents = [n for n in cascade_plan.get("next_agents", [])
                       if n in VALID_AGENTS and n not in stage1_outputs]
        if allowed_agents:
            next_agents = [n for n in next_agents if n in allowed_agents]

        if next_agents:
            await emit({"type": "cascade_start", "agents": next_agents})
            await emit({"type": "status",
                        "message": f"Cascade: {', '.join(n.capitalize() for n in next_agents)}..."})

            ctx = "\n\n".join(
                f"## {n.capitalize()} — Approved\n\n{o}" for n, o in stage1_outputs.items())
            base_brief = (
                f"Original brief: {enriched_brief}\n\n---\n\n"
                f"APPROVED OUTPUTS FROM PREVIOUS STAGE (use as source of truth):\n\n{ctx}\n\n---\n\n"
                f"Produce complete, final content — not a draft. This goes directly into the assembled deliverable."
            )

            async def run_cascade(name: str) -> tuple[str, str]:
                sys_ = await fetch_agent(name)
                out  = await call_agent(sys_, base_brief)
                return name, out

            results = await asyncio.gather(*[run_cascade(n) for n in next_agents])
            cascade_outputs = dict(results)
            for name, content in cascade_outputs.items():
                logging.info("job %s — cascade %s: %d chars", job_id, name, len(content))
                await emit({"type": "cascade_output", "agent": name, "content": content})

        job["cascade_outputs"] = cascade_outputs

        for flag in cascade_plan.get("flags", []):
            msg = THIRD_PARTY_MESSAGES.get(flag, f"Manual step: {flag}")
            await emit({"type": "third_party_flag", "flag": flag, "message": msg})

        if cascade_outputs:
            await emit({"type": "status", "message": "Marcus is reviewing cascade outputs..."})
            r2 = await call_agent(
                marcus_system,
                "Review these cascade outputs:\n\n" +
                "".join(f"## {n.capitalize()}\n\n{o}\n\n---\n\n"
                        for n, o in cascade_outputs.items()),
                model="claude-opus-4-6")
            job["marcus_cascade_review"] = r2
            await emit({"type": "cascade_review", "content": r2})

        # Assembly — generate and save each file sequentially so one failure
        # never blocks the others. Each step logs on error but continues.
        all_outputs = {**stage1_outputs, **cascade_outputs}
        assembled_content_for_db: dict[str, str] = {}

        logging.info(
            "job %s — all_outputs before assembly: keys=%s stage1=%s cascade=%s total_chars=%d",
            job_id,
            list(all_outputs.keys()),
            list(stage1_outputs.keys()),
            list(cascade_outputs.keys()),
            sum(len(v) for v in all_outputs.values()),
        )
        if not all_outputs:
            logging.warning("job %s — all_outputs is empty, skipping assembly entirely", job_id)

        if all_outputs:
            await emit({"type": "assembly_start", "assembly_type": "bundle"})
            bundle = {}

            # 1. HTML Website
            await emit({"type": "status", "message": "Building HTML website..."})
            try:
                website_content = await assemble_html(
                    _build_website_prompt(all_outputs, brief),
                    "You generate complete, production-ready B2B HTML websites with Google Fonts. Output only valid HTML.")
                website_path = save_assembled(job, "html_website", website_content)
                bundle["html_website"] = website_path.name
                assembled_content_for_db["html_website"] = website_content
                logging.info("Assembly saved: %s", website_path)
            except Exception as e:
                logging.error("Assembly html_website failed: %s\n%s", e, traceback.format_exc())

            # 2. A4 One-Pager
            await emit({"type": "status", "message": "Building A4 one-pager..."})
            try:
                onepager_content = await assemble_html(
                    _build_onepager_prompt(all_outputs, brief),
                    "You generate complete, print-ready A4 HTML one-pagers with Google Fonts. Output only valid HTML.")
                onepager_path = save_assembled(job, "html_onepager", onepager_content)
                bundle["html_onepager"] = onepager_path.name
                assembled_content_for_db["html_onepager"] = onepager_content
                logging.info("Assembly saved: %s", onepager_path)
            except Exception as e:
                logging.error("Assembly html_onepager failed: %s\n%s", e, traceback.format_exc())

            # 3. Canva JSON Template
            await emit({"type": "status", "message": "Building Canva JSON template..."})
            try:
                canva_json_str = await assemble_canva_json(all_outputs, brief)
                canva_path = save_assembled(job, "canva_json", canva_json_str)
                bundle["canva_json"] = canva_path.name
                assembled_content_for_db["canva_json"] = canva_json_str
                logging.info("Assembly saved: %s", canva_path)
            except Exception as e:
                logging.error("Assembly canva_json failed: %s\n%s", e, traceback.format_exc())

            # 4. Social Content Pack (Callum + Priya in parallel)
            await emit({"type": "status", "message": "Generating social content pack (Callum + Priya)..."})
            try:
                social_cascade_content = await assemble_social_cascade(all_outputs, brief)
                social_cascade_path = save_assembled(job, "social_pack_cascade", social_cascade_content)
                bundle["social_pack_cascade"] = social_cascade_path.name
                assembled_content_for_db["social_pack_cascade"] = social_cascade_content
                logging.info("Assembly saved: %s", social_cascade_path)
                await post_to_buffer(social_cascade_content)
            except Exception as e:
                logging.error("Assembly social_pack_cascade failed: %s\n%s", e, traceback.format_exc())

            # 5. Video Brief (Dante)
            await emit({"type": "status", "message": "Generating video brief (Dante)..."})
            try:
                video_brief_content = await assemble_video_brief(all_outputs, brief)
                video_brief_path = save_assembled(job, "video_brief", video_brief_content)
                bundle["video_brief"] = video_brief_path.name
                assembled_content_for_db["video_brief"] = video_brief_content
                logging.info("Assembly saved: %s", video_brief_path)
            except Exception as e:
                logging.error("Assembly video_brief failed: %s\n%s", e, traceback.format_exc())

            await emit({
                "type":  "assembly_bundle_done",
                "files": bundle,
                "labels": {
                    "html_website":        "HTML Website",
                    "html_onepager":       "A4 One-Pager (PDF-ready)",
                    "canva_json":          "Canva Template (JSON)",
                    "social_pack_cascade": "Social Content Pack",
                    "video_brief":         "Video Brief (PDF-ready)",
                },
            })

            # Optional compiled social pack (legacy social_pack assembly type)
            if cascade_plan.get("assembly") == "social_pack":
                await emit({"type": "status", "message": "Compiling full social content pack..."})
                try:
                    social_content = await assemble_social_pack(all_outputs, brief)
                    social_path = save_assembled(job, "social_pack", social_content)
                    await emit({
                        "type":          "assembly_done",
                        "assembly_type": "social_pack",
                        "label":         "Social Content Pack",
                        "filename":      social_path.name,
                    })
                except Exception as e:
                    logging.error("Assembly social_pack failed: %s", e)

        job["assembled_files"] = bundle if all_outputs else {}
        md_path = save_markdown(job)
        db_save_job(job_id, job, assembled_content_for_db)
        _completed[job_id] = job
        await emit({"type": "done", "job_id": job_id, "saved_as": md_path.name})

    except Exception as exc:
        logging.error("process_job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        await emit({"type": "error", "message": str(exc)})
    finally:
        await asyncio.sleep(7200)
        _jobs.pop(job_id, None)
        _completed.pop(job_id, None)

# ── Routes ────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")
    if (secrets.compare_digest(username, HTTP_USER) and
            secrets.compare_digest(password, HTTP_PASS)):
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": "Incorrect username or password."})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

async def parse_uploaded_file(filename: str, data: bytes) -> str:
    """Extract text content from an uploaded file. Images are described via Claude vision."""
    import io
    ext = Path(filename).suffix.lower()

    if ext in ('.txt', '.md'):
        return data.decode('utf-8', errors='replace')

    elif ext == '.pdf':
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return '\n\n'.join(parts) if parts else '[PDF: no extractable text]'
        except Exception as e:
            return f'[PDF parse error: {e}]'

    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Also extract tables
            for table in doc.tables:
                for row in table.rows:
                    paragraphs.append('\t'.join(c.text for c in row.cells if c.text.strip()))
            return '\n'.join(paragraphs)
        except Exception as e:
            return f'[DOCX parse error: {e}]'

    elif ext in ('.xlsx', '.xls'):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            parts = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                parts.append(f'Sheet: {sheet_name}')
                for row in ws.iter_rows(values_only=True):
                    row_text = '\t'.join(str(c) if c is not None else '' for c in row)
                    if row_text.strip():
                        parts.append(row_text)
            return '\n'.join(parts)
        except Exception as e:
            return f'[Spreadsheet parse error: {e}]'

    elif ext == '.csv':
        try:
            import csv
            reader = csv.reader(io.StringIO(data.decode('utf-8', errors='replace')))
            return '\n'.join('\t'.join(row) for row in reader)
        except Exception as e:
            return f'[CSV parse error: {e}]'

    elif ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp'):
        try:
            import base64
            media_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                         '.gif': 'image/gif', '.webp': 'image/webp'}
            media_type = media_map.get(ext, 'image/jpeg')
            b64 = base64.standard_b64encode(data).decode()
            resp = await anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": "Describe all content in this image in detail. Extract any text verbatim. Include layout, structure, and key visual information."}
                ]}]
            )
            return resp.content[0].text
        except Exception as e:
            return f'[Image OCR error: {e}]'

    else:
        return f'[Unsupported file type: {ext}]'


@app.post("/api/brief")
async def start_brief(
    brief:          str  = Form(...),
    title:          str  = Form(...),
    client_name:    str  = Form("other", alias="client"),
    project_name:   str  = Form(""),
    allowed_agents: str  = Form(""),
    files:          List[UploadFile] = File(default=[]),
):
    brief        = brief.strip()
    title        = title.strip()
    client_name  = client_name.strip()
    project_name = project_name.strip()

    if not brief:
        return JSONResponse({"error": "Brief is empty"}, status_code=400)
    if not title:
        return JSONResponse({"error": "Title is required"}, status_code=400)

    allowed_list: list[str] | None = None
    if allowed_agents.strip():
        allowed_list = [a.strip() for a in allowed_agents.split(",") if a.strip() in VALID_AGENTS]
        if not allowed_list:
            allowed_list = None

    # Parse uploaded files and append extracted content to the brief
    if files:
        attachments = []
        for f in files:
            if f.filename:
                data = await f.read()
                extracted = await parse_uploaded_file(f.filename, data)
                attachments.append(f"--- Attached file: {f.filename} ---\n{extracted}")
                logging.info("Parsed attachment: %s (%d bytes)", f.filename, len(data))
        if attachments:
            brief = brief + "\n\n" + "\n\n".join(attachments)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = asyncio.Queue()
    asyncio.create_task(process_job(job_id, brief, title=title, client=client_name,
                                    project_name=project_name, allowed_agents=allowed_list))
    return {"job_id": job_id}

@app.post("/api/job/{job_id}/archive")
async def archive_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE jobs SET archived=1 WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/api/job/{job_id}/unarchive")
async def unarchive_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE jobs SET archived=0 WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.get("/outputs/zip/{job_id}")
async def zip_job(job_id: str):
    import io
    import zipfile
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT title, assembled_files FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    title = row["title"] or "job"
    try:
        af = json.loads(row["assembled_files"] or "{}")
    except Exception:
        af = {}
    safe_title = re.sub(r"[^a-z0-9]+", "-", title[:40].lower()).strip("-") or "job"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, fname in af.items():
            path = OUTPUTS_DIR / fname
            if path.exists():
                zf.write(path, fname)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.zip"'},
    )

@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT title, brief, client, project_name FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    title, brief, client_val, project_name = row
    # Strip file attachment blocks — user just wants the original brief text
    if "\n\n--- Attached file:" in brief:
        brief = brief.split("\n\n--- Attached file:")[0]
    return {"title": title, "brief": brief, "client": client_val, "project_name": project_name or ""}

@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    async def generate():
        q = _jobs[job_id]
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Export routes ─────────────────────────────────────────────────

@app.get("/api/export/{job_id}/{agent}.md")
async def export_md(job_id: str, agent: str):
    job = _completed.get(job_id)
    if not job:
        return JSONResponse({"error": "Output not found or expired"}, status_code=404)
    ts = job["timestamp"]
    all_out = {**job.get("stage1_outputs",{}), **job.get("cascade_outputs",{})}
    if agent == "full":
        path = save_markdown(job); content = path.read_text(); fname = path.name
    elif agent == "review":
        content = job.get("marcus_review") or job.get("marcus_cascade_review") or ""
        fname = f"marcus-review_{ts.strftime('%Y-%m-%d_%H-%M')}.md"
    elif agent == "analysis":
        content = job.get("marcus_analysis", "")
        fname = f"marcus-analysis_{ts.strftime('%Y-%m-%d_%H-%M')}.md"
    else:
        content = all_out.get(agent, "")
        if not content:
            return JSONResponse({"error": "Not found"}, status_code=404)
        fname = f"{agent}_{ts.strftime('%Y-%m-%d_%H-%M')}.md"
    return StreamingResponse(iter([content.encode()]), media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})

@app.get("/api/export/{job_id}/{agent}/print", response_class=HTMLResponse)
async def print_export(job_id: str, agent: str):
    job = _completed.get(job_id)
    if not job:
        return HTMLResponse("<h1>Output not found or expired</h1>", status_code=404)
    ts = job["timestamp"].strftime("%Y-%m-%d %H:%M")
    all_out = {**job.get("stage1_outputs",{}), **job.get("cascade_outputs",{})}
    if agent == "full":
        title, md_body = "Full Output", ""
        for s, k in [("Brief Analysis","marcus_analysis"),("Stage 1 Review","marcus_review"),
                     ("Final Review","marcus_cascade_review")]:
            if job.get(k): md_body += f"## Marcus — {s}\n\n{job[k]}\n\n---\n\n"
        for n, o in all_out.items(): md_body += f"## {n.capitalize()}\n\n{o}\n\n---\n\n"
    elif agent in ("review","analysis"):
        k = "marcus_review" if agent == "review" else "marcus_analysis"
        title, md_body = f"Marcus — {agent.capitalize()}", job.get(k,"")
    else:
        title, md_body = f"{agent.capitalize()} — Output", all_out.get(agent,"")
    return HTMLResponse(_print_page(title, job["brief"], ts, _md_to_html(md_body)))

# ── Studio page — redirect to /outputs ───────────────────────────

@app.get("/studio")
async def studio_redirect():
    return RedirectResponse("/outputs", status_code=302)

@app.get("/studio_old", response_class=HTMLResponse)
async def studio_page():
    html_files = sorted(OUTPUTS_DIR.glob("*.html"),
                        key=lambda f: f.stat().st_mtime, reverse=True)

    if not html_files:
        cards_html = """<div style="text-align:center;padding:80px 0;color:#94a3b8;">
          <div style="font-size:40px;margin-bottom:16px;">🎨</div>
          <div style="font-size:16px;font-weight:600;color:#475569;margin-bottom:8px;">No HTML outputs yet</div>
          <div style="font-size:13px;">Submit a brand, website, deck, or one-pager brief to generate your first visual output.</div>
        </div>"""
    else:
        cards_html = '<div class="studio-grid">'
        for f in html_files:
            size_kb = round(f.stat().st_size / 1024, 1)
            mtime   = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            label   = next((l for l in ["website","one-pager","deck"] if l in f.name), "html")
            label_colors = {"website": "#dbeafe:#1d4ed8",
                            "one-pager": "#d1fae5:#065f46",
                            "deck": "#ede9fe:#6d28d9"}
            lbg, lfg = label_colors.get(label, "#f1f5f9:#475569").split(":")
            cards_html += f"""
            <div class="studio-card">
              <div class="preview-wrap">
                <iframe src="/outputs/view/{f.name}" sandbox="allow-same-origin"
                        scrolling="no" class="preview-frame"></iframe>
                <div class="preview-overlay">
                  <a href="/outputs/view/{f.name}" target="_blank" class="overlay-btn">Open →</a>
                </div>
              </div>
              <div class="card-meta">
                <div class="card-top">
                  <span class="label-badge" style="background:{lbg};color:{lfg};">{label.upper()}</span>
                  <span class="card-size">{size_kb} KB</span>
                </div>
                <div class="card-name" title="{f.name}">{f.name}</div>
                <div class="card-date">{mtime}</div>
                <div class="card-actions">
                  <a href="/outputs/view/{f.name}" target="_blank" class="ca-btn ca-view">Open</a>
                  <a href="/outputs/download/{f.name}" class="ca-btn ca-dl">⬇ Download</a>
                </div>
              </div>
            </div>"""
        cards_html += "</div>"

    count = len(html_files)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Studio — Studio N</title>
<style>
  {_PAGE_STYLE}
  .page-head {{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:24px;}}
  .page-title {{font-size:20px;font-weight:700;}}
  .page-count {{font-size:13px;color:#94a3b8;}}
  .studio-grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:20px;}}
  .studio-card {{background:white;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);transition:box-shadow 0.15s;}}
  .studio-card:hover {{box-shadow:0 4px 16px rgba(0,0,0,0.12);}}
  .preview-wrap {{position:relative;height:200px;overflow:hidden;background:#f1f5f9;cursor:pointer;}}
  .preview-frame {{width:200%;height:200%;transform:scale(0.5);transform-origin:0 0;border:none;pointer-events:none;}}
  .preview-overlay {{position:absolute;inset:0;background:rgba(30,41,59,0);display:flex;align-items:center;
                     justify-content:center;transition:background 0.2s;}}
  .preview-wrap:hover .preview-overlay {{background:rgba(30,41,59,0.5);}}
  .overlay-btn {{opacity:0;background:white;color:#1e293b;padding:8px 18px;border-radius:8px;
                 font-size:13px;font-weight:700;text-decoration:none;transition:opacity 0.2s;}}
  .preview-wrap:hover .overlay-btn {{opacity:1;}}
  .card-meta {{padding:14px 16px;}}
  .card-top {{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}}
  .label-badge {{font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:0.5px;}}
  .card-size {{font-size:11px;color:#94a3b8;}}
  .card-name {{font-size:12px;font-weight:600;color:#1e293b;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .card-date {{font-size:11px;color:#94a3b8;margin-bottom:10px;}}
  .card-actions {{display:flex;gap:6px;}}
  .ca-btn {{font-size:11px;font-weight:600;padding:5px 10px;border-radius:6px;text-decoration:none;transition:all 0.15s;border:1px solid #e2e8f0;color:#475569;}}
  .ca-btn:hover {{background:#f8fafc;color:#1e293b;}}
  .ca-view {{border-color:#bbf7d0;color:#16a34a;}}
  .ca-view:hover {{background:#f0fdf4;}}
  .ca-dl {{border-color:#bfdbfe;color:#2563eb;}}
  .ca-dl:hover {{background:#eff6ff;}}
</style></head><body>
{_nav('studio')}
<div class="container">
  <div class="page-head">
    <span class="page-title">Studio</span>
    <span class="page-count">{count} HTML file{'s' if count!=1 else ''}</span>
  </div>
  {cards_html}
</div></body></html>""")

# ── Outputs page (DB-backed) ──────────────────────────────────────

CLIENT_META = {
    "wibiz":     {"label": "WiBiz",       "bg": "#dcfce7", "fg": "#15803d"},
    "ai-living": {"label": "AI Living",   "bg": "#dbeafe", "fg": "#1d4ed8"},
    "charter":   {"label": "Club Charter","bg": "#fef9c3", "fg": "#a16207"},
    "other":     {"label": "Other",       "bg": "#f1f5f9", "fg": "#475569"},
}

FILE_META = {
    "html_website":        {"label": "Website",      "bg": "#dbeafe", "fg": "#1d4ed8"},
    "html_onepager":       {"label": "One-Pager",    "bg": "#d1fae5", "fg": "#065f46"},
    "canva_json":          {"label": "Canva JSON",   "bg": "#ede9fe", "fg": "#6d28d9"},
    "social_pack_cascade": {"label": "Social Pack",  "bg": "#fef3c7", "fg": "#92400e"},
    "video_brief":         {"label": "Video Brief",  "bg": "#fee2e2", "fg": "#b91c1c"},
    "social_pack":         {"label": "Social Pack",  "bg": "#fef3c7", "fg": "#92400e"},
}

AGENT_ROLES = {
    "marcus": "Director of Marketing",
    "reeva":  "Brand & Identity",
    "callum": "LinkedIn & Long-Form",
    "priya":  "Social & Short-Form",
    "dante":  "Video & Reels",
    "suki":   "Static & Visual",
    "felix":  "Decks & Presentations",
    "nadia":  "Web & Copy",
    "zara":   "Research & Intelligence",
    "kiara":  "AI Video Generation",
    "rex":    "Remotion & Motion Graphics",
    "nova":   "Brand Visuals (Nano Banana)",
}

@app.get("/outputs", response_class=HTMLResponse)
async def outputs_page(request: Request):
    # Ensure any files from DB that are missing on disk are recreated
    db_recover_missing_files()

    active_client = request.query_params.get("client", "all")
    show_archived = active_client == "archived"

    all_jobs = db_load_all_jobs(show_archived=False)
    archived_jobs = db_load_all_jobs(show_archived=True)

    if show_archived:
        jobs = archived_jobs
    elif active_client != "all":
        jobs = [j for j in all_jobs if j.get("client") == active_client]
    else:
        jobs = all_jobs

    # ── Tab bar ──────────────────────────────────────────────────
    tabs_html = ""
    tab_defs = [
        ("all",       "All",          "#1e293b", "white"),
        ("wibiz",     "WiBiz",        "#dcfce7", "#15803d"),
        ("ai-living", "AI Living",    "#dbeafe", "#1d4ed8"),
        ("charter",   "Club Charter", "#fef9c3", "#a16207"),
        ("other",     "Other",        "#f1f5f9", "#475569"),
        ("archived",  "Archived",     "#fee2e2", "#b91c1c"),
    ]
    for key, label, bg, fg in tab_defs:
        if key == "archived":
            cnt = len(archived_jobs)
        else:
            cnt = len([j for j in all_jobs if key == "all" or j.get("client") == key])
        is_active = key == active_client
        style = (f"background:{bg};color:{fg};border:2px solid {fg}40;"
                 if not is_active else
                 f"background:#1e293b;color:white;border:2px solid #1e293b;")
        tabs_html += (
            f'<a href="/outputs?client={key}" class="tab-link" style="{style}">'
            f'{label} <span style="font-size:10px;opacity:0.7;">({cnt})</span></a> '
        )

    # ── Job cards ────────────────────────────────────────────────
    if not jobs:
        body = """<div style="text-align:center;padding:80px 0;color:#94a3b8;">
          <div style="font-size:32px;margin-bottom:12px;">📂</div>
          <div style="font-size:15px;font-weight:600;color:#475569;">No jobs yet for this client</div>
          <div style="font-size:13px;margin-top:6px;">Submit a brief on the <a href="/" style="color:#1e293b;">Brief page</a>.</div>
        </div>"""
    else:
        body = ""
        for job in jobs:
            jid    = job["id"]
            title  = job.get("title") or job.get("brief", "")[:60]
            client = job.get("client", "other")
            pname  = job.get("project_name", "")
            brief  = job.get("brief", "")
            ts     = job.get("timestamp", job.get("created_at", ""))[:16].replace("T", " ")
            cm     = CLIENT_META.get(client, CLIENT_META["other"])
            client_label = pname if (client == "other" and pname) else cm["label"]
            is_archived = bool(job.get("archived", 0))

            # Agents used
            s1 = job.get("stage1_outputs", {})
            s2 = job.get("cascade_outputs", {})
            all_agents = list(s1.keys()) + list(s2.keys())
            agent_chips = ""
            for ag in all_agents:
                role = AGENT_ROLES.get(ag, ag.capitalize())
                agent_chips += (
                    f'<span style="font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;'
                    f'background:#f1f5f9;color:#475569;white-space:nowrap;">'
                    f'{ag.capitalize()} — {role}</span> '
                )

            # Assembled files
            af = job.get("assembled_files", {})
            file_btns = ""
            has_files = False
            for fkey, fname in af.items():
                fm = FILE_META.get(fkey, {"label": fkey, "bg": "#f1f5f9", "fg": "#475569"})
                path = OUTPUTS_DIR / fname
                if path.exists():
                    has_files = True
                    if fname.endswith(".html"):
                        file_btns += (
                            f'<a href="/outputs/view/{fname}" target="_blank" '
                            f'style="font-size:11px;font-weight:700;padding:5px 11px;border-radius:6px;'
                            f'background:{fm["bg"]};color:{fm["fg"]};text-decoration:none;white-space:nowrap;'
                            f'border:1px solid {fm["fg"]}40;">Open {fm["label"]} →</a> '
                        )
                    file_btns += (
                        f'<a href="/outputs/download/{fname}" '
                        f'style="font-size:11px;font-weight:600;padding:5px 11px;border-radius:6px;'
                        f'background:#f8fafc;color:#64748b;text-decoration:none;white-space:nowrap;'
                        f'border:1px solid #e2e8f0;">⬇ {fname.rsplit(".",1)[-1].upper()}</a> '
                    )
            # ZIP button
            if has_files:
                file_btns += f'<a href="/outputs/zip/{jid}" class="zip-btn">⬇ ZIP</a> '

            # Archive/unarchive button
            if is_archived:
                archive_btn = f'<button onclick="unarchiveJob(\'{jid}\')" class="job-rebrief" style="color:#16a34a;">↑ Unarchive</button>'
            else:
                archive_btn = f'<button onclick="archiveJob(\'{jid}\')" class="job-rebrief" style="color:#b91c1c;">Archive</button>'

            brief_snippet = brief[:140] + "…" if len(brief) > 140 else brief
            # strip file attachment blocks from snippet
            if "\n\n--- Attached file:" in brief_snippet:
                brief_snippet = brief_snippet.split("\n\n--- Attached file:")[0] + "…"

            # Escape for HTML attributes
            title_attr = title.lower().replace('"', '')
            brief_attr = brief_snippet.lower().replace('"', '')

            body += f"""
<div class="job-card" id="card-{jid}" style="border-left:4px solid {cm['fg']}40;" data-id="{jid}" data-title="{title_attr}" data-brief="{brief_attr}">
  <div class="job-top">
    <input type="checkbox" class="job-check" onchange="onCheckChange()" title="Select for bulk action">
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span class="job-title">{title}</span>
        <span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;background:{cm['bg']};color:{cm['fg']};white-space:nowrap;">{client_label.upper()}</span>
      </div>
      <div class="job-ts">{ts}</div>
      <div class="job-brief-text">{brief_snippet}</div>
    </div>
    <button class="star-btn" onclick="toggleStar('{jid}', this)" title="Star / pin this job">☆</button>
  </div>
  {"<div class='job-agents'>" + agent_chips + "</div>" if agent_chips else ""}
  <div class="job-files">
    {file_btns}
    <a href="/?rebrief={jid}" class="job-rebrief">↩ Re-brief</a>
    {archive_btn}
  </div>
</div>"""

    count = len(jobs)
    total = len(all_jobs)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Outputs — Studio N</title>
<style>{_PAGE_STYLE}</style></head><body>
{_nav('outputs')}
<div class="container">
  <div class="page-head">
    <span class="page-title">Outputs</span>
    <span class="page-count" id="result-count">{count} job{'s' if count!=1 else ''} · {total} total</span>
  </div>
  <div class="tab-bar">{tabs_html}</div>
  <div class="search-bar">
    <input class="search-input" type="text" placeholder="Search by title or brief content…" oninput="filterJobs(this.value)">
  </div>
  <div id="jobs-list">{body}</div>
</div>
<div class="bulk-bar" id="bulk-bar">
  <span class="bulk-bar-label" id="bulk-label">0 selected</span>
  <div style="display:flex;gap:8px;">
    <button class="bulk-cancel-btn" onclick="clearSelection()">Cancel</button>
    <button class="bulk-archive-btn" onclick="bulkArchive()">Archive selected</button>
  </div>
</div>
<script>
// ── Star / Pin ────────────────────────────────────────────────
var STARRED_KEY = 'studio_starred';
function getStarred() {{
  try {{ return JSON.parse(localStorage.getItem(STARRED_KEY) || '[]'); }} catch(e) {{ return []; }}
}}
function saveStarred(arr) {{ localStorage.setItem(STARRED_KEY, JSON.stringify(arr)); }}

function toggleStar(jid, btn) {{
  var starred = getStarred();
  var idx = starred.indexOf(jid);
  if (idx === -1) {{ starred.push(jid); btn.textContent = '★'; btn.closest('.job-card').classList.add('starred'); }}
  else {{ starred.splice(idx, 1); btn.textContent = '☆'; btn.closest('.job-card').classList.remove('starred'); }}
  saveStarred(starred);
  sortCards();
}}

function sortCards() {{
  var list = document.getElementById('jobs-list');
  var cards = Array.from(list.querySelectorAll('.job-card'));
  var starred = getStarred();
  cards.sort(function(a, b) {{
    var aS = starred.includes(a.dataset.id) ? 1 : 0;
    var bS = starred.includes(b.dataset.id) ? 1 : 0;
    return bS - aS;
  }});
  cards.forEach(function(c) {{ list.appendChild(c); }});
}}

function initStars() {{
  var starred = getStarred();
  document.querySelectorAll('.job-card').forEach(function(card) {{
    var jid = card.dataset.id;
    var btn = card.querySelector('.star-btn');
    if (starred.includes(jid)) {{
      card.classList.add('starred');
      if (btn) btn.textContent = '★';
    }}
  }});
  sortCards();
}}

// ── Bulk Archive ──────────────────────────────────────────────
function onCheckChange() {{
  var checked = document.querySelectorAll('.job-check:checked');
  var bar = document.getElementById('bulk-bar');
  var label = document.getElementById('bulk-label');
  if (checked.length > 0) {{
    bar.classList.add('visible');
    label.textContent = checked.length + ' selected';
  }} else {{
    bar.classList.remove('visible');
  }}
}}

function clearSelection() {{
  document.querySelectorAll('.job-check:checked').forEach(function(cb) {{ cb.checked = false; }});
  document.getElementById('bulk-bar').classList.remove('visible');
}}

async function archiveJob(jid) {{
  await fetch('/api/job/' + jid + '/archive', {{method:'POST'}});
  var card = document.getElementById('card-' + jid);
  if (card) card.remove();
  updateCount();
}}

async function unarchiveJob(jid) {{
  await fetch('/api/job/' + jid + '/unarchive', {{method:'POST'}});
  var card = document.getElementById('card-' + jid);
  if (card) card.remove();
  updateCount();
}}

async function bulkArchive() {{
  var checked = document.querySelectorAll('.job-check:checked');
  var ids = [];
  checked.forEach(function(cb) {{ ids.push(cb.closest('.job-card').dataset.id); }});
  for (var i = 0; i < ids.length; i++) {{
    await fetch('/api/job/' + ids[i] + '/archive', {{method:'POST'}});
    var card = document.getElementById('card-' + ids[i]);
    if (card) card.remove();
  }}
  document.getElementById('bulk-bar').classList.remove('visible');
  updateCount();
}}

function updateCount() {{
  var visible = document.querySelectorAll('.job-card:not([style*="display: none"])').length;
  var el = document.getElementById('result-count');
  if (el) el.textContent = visible + ' job' + (visible !== 1 ? 's' : '') + ' · {total} total';
}}

// ── Search ────────────────────────────────────────────────────
function filterJobs(q) {{
  q = q.toLowerCase().trim();
  var cards = document.querySelectorAll('.job-card');
  var shown = 0;
  cards.forEach(function(c) {{
    var match = !q || c.dataset.title.includes(q) || c.dataset.brief.includes(q);
    c.style.display = match ? '' : 'none';
    if (match) shown++;
  }});
  var el = document.getElementById('result-count');
  if (el) el.textContent = shown + ' job' + (shown !== 1 ? 's' : '') + ' · {total} total';
}}

// Init on load
initStars();
</script>
</body></html>""")

# ── Legacy file-based outputs (kept for backward compat) ──────────

def _parse_output_file(path: Path) -> dict:
    info = {"filename": path.name, "size_kb": round(path.stat().st_size/1024,1),
            "brief": path.stem, "date": "", "is_html": path.suffix == ".html",
            "is_json": path.suffix == ".json", "label": ""}
    for l in ["website", "one-pager", "social-pack", "video-brief", "canva"]:
        if l in path.name: info["label"] = l; break
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            if m := re.search(r"\*\*Brief:\*\* (.+)", text): info["brief"] = m.group(1).strip()
            if d := re.search(r"\*\*Date:\*\* (.+)",  text): info["date"]  = d.group(1).strip()
        elif path.suffix == ".html":
            if t := re.search(r"<title>(.+?)</title>", text): info["brief"] = t.group(1).strip()
        elif path.suffix == ".json":
            try:
                data = json.loads(text)
                info["brief"] = data.get("brand", {}).get("name", path.stem)
            except Exception:
                pass
    except Exception:
        pass
    return info

@app.get("/outputs", response_class=HTMLResponse)
async def outputs_page():
    files = [_parse_output_file(f) for f in
             sorted(OUTPUTS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
             if f.suffix in (".md", ".html", ".json")]
    rows = ""
    for f in files:
        label_styles = {
            "website":    "#dbeafe:#1d4ed8",
            "one-pager":  "#d1fae5:#065f46",
            "canva":      "#ede9fe:#6d28d9",
            "social-pack":"#fef3c7:#92400e",
        }
        lbg, lfg = label_styles.get(f["label"], "#f1f5f9:#475569").split(":") if f["label"] else ("#f1f5f9", "#475569")
        lb = (f'<span style="font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;'
              f'background:{lbg};color:{lfg};margin-left:8px;">{f["label"].upper()}</span>'
              if f["label"] else "")
        if f["is_html"]:
            view = f'<a class="out-btn out-btn-view" href="/outputs/view/{f["filename"]}" target="_blank">View</a>'
        elif f["is_json"]:
            view = f'<a class="out-btn out-btn-json" href="/outputs/download/{f["filename"]}">⬇ JSON</a>'
        else:
            view = f'<a class="out-btn out-btn-pdf" href="/outputs/view/{f["filename"]}" target="_blank">View</a>'
        ext = f["filename"].rsplit(".",1)[-1].upper()
        rows += f"""<div class="output-row">
          <div class="output-meta">
            <span class="output-date">{f['date'] or '—'}</span>
            <span class="output-size">{f['size_kb']} KB</span>
          </div>
          <div class="output-brief">{f['brief']}{lb}</div>
          <div class="output-actions">
            <a class="out-btn" href="/outputs/download/{f['filename']}">⬇ {ext}</a>
            {view}
          </div>
        </div>"""
    if not files:
        rows = """<div style="text-align:center;padding:60px 0;color:#94a3b8;">
          <div style="font-size:32px;margin-bottom:12px;">📂</div>
          <div style="font-size:15px;font-weight:600;color:#475569;">No outputs yet</div>
          <div style="font-size:13px;margin-top:6px;">Submit a brief on the <a href="/" style="color:#1e293b;">Brief page</a>.</div>
        </div>"""
    count = len(files)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Outputs — Studio N</title>
<style>
  {_PAGE_STYLE}
  .page-head {{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:24px;}}
  .page-title {{font-size:20px;font-weight:700;}} .page-count {{font-size:13px;color:#94a3b8;}}
  .output-row {{background:white;border-radius:10px;padding:18px 20px;margin-bottom:10px;
    box-shadow:0 1px 3px rgba(0,0,0,0.07);display:flex;align-items:center;gap:16px;transition:box-shadow 0.15s;}}
  .output-row:hover {{box-shadow:0 2px 8px rgba(0,0,0,0.1);}}
  .output-meta {{display:flex;flex-direction:column;gap:3px;min-width:130px;}}
  .output-date {{font-size:12px;font-weight:600;color:#475569;}} .output-size {{font-size:11px;color:#94a3b8;}}
  .output-brief {{flex:1;font-size:13px;color:#334155;line-height:1.5;}}
  .output-actions {{display:flex;gap:6px;flex-shrink:0;}}
  .out-btn {{font-size:11px;font-weight:600;padding:5px 11px;border-radius:6px;border:1px solid #e2e8f0;
    color:#475569;text-decoration:none;transition:all 0.15s;white-space:nowrap;}}
  .out-btn:hover {{background:#f8fafc;color:#1e293b;}}
  .out-btn-pdf {{border-color:#bfdbfe;color:#2563eb;}} .out-btn-pdf:hover {{background:#eff6ff;}}
  .out-btn-view {{border-color:#bbf7d0;color:#16a34a;font-weight:700;}} .out-btn-view:hover {{background:#f0fdf4;}}
  .out-btn-json {{border-color:#e9d5ff;color:#7c3aed;font-weight:700;}} .out-btn-json:hover {{background:#faf5ff;}}
</style></head><body>
{_nav('outputs')}
<div class="container">
  <div class="page-head">
    <span class="page-title">Outputs</span>
    <span class="page-count">{count} file{'s' if count!=1 else ''}</span>
  </div>
  {rows}
</div></body></html>""")

@app.get("/outputs/download/{filename}")
async def download_output(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix not in (".md", ".html", ".json"):
        return JSONResponse({"error": "File not found"}, status_code=404)
    mime = {"html": "text/html", "md": "text/markdown", "json": "application/json"}.get(
        path.suffix.lstrip("."), "application/octet-stream")
    return StreamingResponse(iter([path.read_bytes()]), media_type=mime,
                             headers={"Content-Disposition": f'attachment; filename="{safe}"'})

@app.get("/outputs/view/{filename}", response_class=HTMLResponse)
async def view_output(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix not in (".md", ".html", ".json"):
        return HTMLResponse("<h1>File not found</h1>", status_code=404)
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        return HTMLResponse(content)
    bm = re.search(r"\*\*Brief:\*\* (.+)", content)
    dm = re.search(r"\*\*Date:\*\* (.+)",  content)
    return HTMLResponse(_print_page(
        safe,
        bm.group(1).strip() if bm else safe,
        dm.group(1).strip() if dm else "",
        _md_to_html(content)))

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Video Studio ──────────────────────────────────────────────────

@app.get("/video-studio", response_class=HTMLResponse)
async def video_studio_page(request: Request):
    try:
        return templates.TemplateResponse(request=request, name="video_studio.html")
    except Exception as e:
        logging.error("video_studio render error: %s", traceback.format_exc())
        return HTMLResponse(f"<pre>Error: {e}</pre>", status_code=500)

@app.get("/api/video-studio/debug-auth")
async def vs_debug_auth():
    cid = ARCADS_CLIENT_ID
    sec = ARCADS_CLIENT_SECRET
    masked = sec[:3] + "..." + sec[-3:] if len(sec) > 6 else "TOO_SHORT"
    import base64
    creds = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{ARCADS_BASE_URL}/v1/products",
                             headers={"Authorization": f"Basic {creds}"}, timeout=15)
    return {
        "client_id_len": len(cid),
        "secret_len": len(sec),
        "secret_masked": masked,
        "status": r.status_code,
        "response": r.text[:200],
    }

@app.get("/api/video-studio/products")
async def vs_get_products():
    try:
        products = await arcads_get_products()
        return {"products": products}
    except Exception as e:
        return JSONResponse({"error": str(e), "products": []}, status_code=200)

@app.post("/api/video-studio/generate")
async def vs_generate(request: Request):
    try:
        body = await request.json()
        client_val = body.get("client", "other")
        model      = body.get("model", "seedance-2.0")
        prompt     = body.get("prompt", "")
        product_id = body.get("productId", "")
        formats    = body.get("formats", ["9:16"])
        variations = int(body.get("variations", 1))
        duration   = body.get("duration")

        if not ARCADS_CLIENT_ID:
            return JSONResponse({"error": "ARCADS_CLIENT_ID not configured"}, status_code=400)
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        # Auto-resolve product ID if not provided
        if not product_id:
            try:
                products = await arcads_get_products()
                logging.info("Auto-resolve products: %s", products)
                if products:
                    product_id = products[0].get("id", "")
            except Exception as pe:
                logging.error("Auto-resolve product error: %s", pe)

        logging.info("Using productId: %s", product_id)
        if not product_id:
            return JSONResponse({"error": "No product ID — add a product in your Arcads account first"}, status_code=400)

        job_id = str(uuid.uuid4())
        jobs_created = []

        for fmt in formats:
            for _ in range(variations):
                vid_job_id = str(uuid.uuid4())
                payload = {
                    "model": model,
                    "productId": product_id,
                    "prompt": prompt,
                }
                if fmt and fmt != "auto":
                    payload["aspectRatio"] = fmt
                if duration and str(duration) not in ("0", "auto", "none", "null"):
                    try:
                        payload["duration"] = int(duration)
                    except (ValueError, TypeError):
                        pass

                result = await arcads_generate_video(payload)
                arcads_id = result.get("id") or result.get("videoId", "")

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT INTO video_jobs (id, client, job_type, model, prompt, status, arcads_id, formats)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (vid_job_id, client_val, "video", model, prompt, "pending", arcads_id, json.dumps({"format": fmt})))
                conn.commit()
                conn.close()
                jobs_created.append({"jobId": vid_job_id, "arcadsId": arcads_id, "format": fmt})

        return {"success": True, "jobs": jobs_created, "sessionId": job_id}
    except Exception as e:
        logging.error("Video generate error: %s", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/video-studio/status/{arcads_id}")
async def vs_status(arcads_id: str):
    try:
        data = await arcads_poll_video(arcads_id)
        status = data.get("videoStatus") or data.get("status", "pending")
        url    = data.get("videoUrl") or data.get("url", "")
        return {"arcadsId": arcads_id, "status": status, "url": url, "raw": data}
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=200)

@app.post("/api/video-studio/mimic")
async def vs_mimic(
    referenceVideo: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("seedance-2.0"),
    productId: str = Form(""),
    client: str = Form("other"),
    formats: str = Form('["9:16"]'),
    variations: int = Form(1),
):
    try:
        if not ARCADS_CLIENT_ID:
            return JSONResponse({"error": "ARCADS_API_KEY not configured"}, status_code=400)

        file_bytes = await referenceVideo.read()
        content_type = referenceVideo.content_type or "video/mp4"
        file_path = await arcads_upload_file(file_bytes, referenceVideo.filename, content_type)

        fmt_list = json.loads(formats) if isinstance(formats, str) else formats
        jobs_created = []

        for fmt in fmt_list:
            for _ in range(variations):
                vid_job_id = str(uuid.uuid4())
                payload = {
                    "model": model,
                    "productId": productId,
                    "prompt": prompt or "Recreate the style and mood of this reference video",
                    "aspectRatio": fmt,
                    "referenceVideos": [file_path],
                }
                result = await arcads_generate_video(payload)
                arcads_id = result.get("id") or result.get("videoId", "")

                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT INTO video_jobs (id, client, job_type, model, prompt, status, arcads_id, formats)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (vid_job_id, client, "mimic", model, prompt, "pending", arcads_id, json.dumps({"format": fmt})))
                conn.commit()
                conn.close()
                jobs_created.append({"jobId": vid_job_id, "arcadsId": arcads_id, "format": fmt})

        return {"success": True, "jobs": jobs_created, "filePath": file_path}
    except Exception as e:
        logging.error("Mimic error: %s", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/video-studio/brand-visual")
async def vs_brand_visual(request: Request):
    try:
        body = await request.json()
        if not ARCADS_CLIENT_ID:
            return JSONResponse({"error": "ARCADS_API_KEY not configured"}, status_code=400)

        payload = {
            "model": body.get("model", "nano-banana-2"),
            "productId": body.get("productId", ""),
            "prompt": body.get("prompt", ""),
        }
        if body.get("referenceBase64"):
            payload["refImageAsBase64"] = body["referenceBase64"]

        result = await arcads_generate_image(payload)
        asset_id = result.get("id") or result.get("assetId", "")
        return {"success": True, "assetId": asset_id, "raw": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/video-studio/asset/{asset_id}")
async def vs_asset_status(asset_id: str):
    try:
        data = await arcads_poll_asset(asset_id)
        return data
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=200)

@app.get("/api/video-studio/jobs")
async def vs_list_jobs(client: str = "all"):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if client == "all":
        rows = conn.execute("SELECT * FROM video_jobs ORDER BY created_at DESC LIMIT 50").fetchall()
    else:
        rows = conn.execute("SELECT * FROM video_jobs WHERE client=? ORDER BY created_at DESC LIMIT 50", (client,)).fetchall()
    conn.close()
    return {"jobs": [dict(r) for r in rows]}

@app.post("/api/video-studio/save-result")
async def vs_save_result(request: Request):
    """Called by frontend when polling completes — saves result URL to DB."""
    body = await request.json()
    arcads_id  = body.get("arcadsId", "")
    status     = body.get("status", "")
    result_url = body.get("url", "")
    if arcads_id:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE video_jobs SET status=?, result_url=? WHERE arcads_id=?",
            (status, result_url, arcads_id)
        )
        conn.commit()
        conn.close()
    return {"ok": True}

async def _bg_poll_pending():
    """Background task: poll Arcads for pending video jobs and update DB."""
    while True:
        await asyncio.sleep(30)
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute(
                "SELECT arcads_id FROM video_jobs WHERE status='pending' AND arcads_id != '' LIMIT 20"
            ).fetchall()
            conn.close()
            for (arcads_id,) in rows:
                try:
                    data = await arcads_poll_video(arcads_id)
                    status = (data.get("videoStatus") or data.get("status", "pending")).lower()
                    url    = data.get("videoUrl") or data.get("url", "")
                    if status in ("done", "generated", "completed", "failed", "error"):
                        conn = sqlite3.connect(DB_PATH)
                        conn.execute(
                            "UPDATE video_jobs SET status=?, result_url=? WHERE arcads_id=?",
                            (status, url, arcads_id)
                        )
                        conn.commit()
                        conn.close()
                        logging.info("BG poll: %s → %s  url=%s", arcads_id, status, url[:80] if url else "")
                except Exception as e:
                    logging.warning("BG poll error for %s: %s", arcads_id, e)
        except Exception as e:
            logging.warning("BG poll loop error: %s", e)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_bg_poll_pending())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    uvicorn.run(app, host="0.0.0.0", port=port)
