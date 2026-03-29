import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import urllib.parse
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

load_dotenv()

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Studio N")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

GITHUB_REPO        = os.getenv("GITHUB_REPO", "nickdcruz/nicklaus-marketing-agents")
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
HTTP_USER          = os.getenv("HTTP_USER", "admin")
HTTP_PASS          = os.getenv("HTTP_PASS", "changeme")
PORT               = int(os.getenv("PORT", "5050"))
CANVA_CLIENT_ID    = os.getenv("CANVA_CLIENT_ID", "")
CANVA_CLIENT_SECRET = os.getenv("CANVA_CLIENT_SECRET", "")
CANVA_REDIRECT_URI = os.getenv("CANVA_REDIRECT_URI", "http://localhost:5050/auth/canva/callback")

CANVA_AUTH_URL  = "https://www.canva.com/api/oauth/authorize"
CANVA_TOKEN_URL = "https://api.canva.com/rest/v1/oauth/token"
CANVA_API_BASE  = "https://api.canva.com/rest/v1"
CANVA_SCOPES    = "design:content:read design:content:write asset:read"

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_agent_cache: dict[str, str]        = {}
_jobs:        dict[str, asyncio.Queue] = {}
_completed:   dict[str, dict]       = {}

# Canva OAuth state
_canva: dict = {
    "access_token":  None,
    "refresh_token": None,
    "connected_at":  None,
    "pkce":          {},   # state -> verifier
}

VALID_AGENTS = {"callum", "priya", "dante", "suki", "felix", "nadia", "zara", "reeva"}

THIRD_PARTY_MESSAGES = {
    "video_production": (
        "Dante's video concept is saved and ready. Production requires Canva, CapCut, or Adobe Premiere — "
        "use real screen recordings of the software, not AI-generated imagery. Brief saved to outputs."
    ),
    "social_images": (
        "Suki's design brief is saved. Static images require real screen recordings or photography. "
        "Connect Canva via Settings to auto-create the design skeleton."
    ),
    "publishing": (
        "Content is approved. Connect Buffer, Hootsuite, or a platform API to enable auto-scheduling."
    ),
}

# ── Auth middleware ───────────────────────────────────────────────

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/health" or path.startswith("/api/stream/") or path.startswith("/auth/canva"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Studio N"'})
        try:
            decoded   = base64.b64decode(auth[6:]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
        except Exception:
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Studio N"'})
        if not (secrets.compare_digest(user, HTTP_USER) and
                secrets.compare_digest(pwd,  HTTP_PASS)):
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Studio N"'})
        return await call_next(request)

app.add_middleware(BasicAuthMiddleware)

# ── Orchestration prompts ─────────────────────────────────────────

ORCHESTRATION_SUFFIX = """

---
SYSTEM — ORCHESTRATION MODE: After your full analysis, append ONE JSON block and nothing after it:

```json
{
  "agents_needed": ["agent1", "agent2"],
  "briefs": {
    "agent1": "Complete brief exactly as you would send it",
    "agent2": "Complete brief exactly as you would send it"
  }
}
```

Valid agent names: callum, priya, dante, suki, felix, nadia, zara, reeva
If no specialists needed: {"agents_needed": [], "briefs": {}}
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

    return {
        "hex_colors":  hex_colors,
        "fonts":       chosen_fonts,
        "brand_name":  brand_name,
        "primary":     hex_colors[0] if hex_colors else "#1e293b",
        "secondary":   hex_colors[1] if len(hex_colors) > 1 else "#64748b",
        "accent":      hex_colors[2] if len(hex_colors) > 2 else "#3b82f6",
    }

def _font_link(fonts: list[str]) -> str:
    families = "|".join(f.replace(" ", "+") + ":wght@300;400;600;700" for f in fonts)
    return f'<link href="https://fonts.googleapis.com/css2?family={families}&display=swap" rel="stylesheet">'

def _font_stack(fonts: list[str]) -> str:
    return ", ".join(f'"{f}"' for f in fonts) + ", -apple-system, sans-serif"

# ── Assembly prompts ──────────────────────────────────────────────

def _build_website_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    color_block = (
        f"Primary: {bd['primary']}\n"
        f"Secondary: {bd['secondary']}\n"
        f"Accent: {bd['accent']}\n"
        + ("\n".join(f"Additional: {c}" for c in bd["hex_colors"][3:]) if len(bd["hex_colors"]) > 3 else "")
    )
    font_link = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)

    return f"""You are generating a complete, production-ready B2B marketing website as a single HTML file.

BRAND COLORS (use these exact hex values throughout):
{color_block}

TYPOGRAPHY:
Google Fonts CDN tag to include in <head>: {font_link}
Font stack to use in CSS: {font_stack}

INPUTS FROM THE MARKETING TEAM:
{inputs}

ORIGINAL BRIEF:
{brief}

REQUIREMENTS:
- Single self-contained HTML5 file. Include the Google Fonts CDN <link> tag above in <head>.
- All CSS inline in a <style> block. Use the exact hex colors above for all brand elements.
- Fully responsive — mobile-first, CSS Grid and Flexbox layout.
- Navigation: logo/brand name, 3–4 nav links, CTA button in primary color.
- Hero section: bold headline (from copy inputs), subheadline, two CTA buttons, and a PLACEHOLDER for a screen recording:
  <div class="demo-placeholder">📹 Insert screen recording here — show the software in action</div>
- Value proposition: 3-column grid of key benefits with icons (use Unicode/emoji icons, not images).
- Features section: alternating text + demo placeholder layout. Each placeholder labeled with what to record.
- Social proof: 2–3 quote cards with placeholder names (e.g. "Property Manager, Singapore").
- CTA section: strong headline + primary button.
- Footer: brand name, tagline, 3-column links.
- Professional B2B aesthetic — clean, structured, confident. No stock photo placeholders. Use demo-placeholder divs styled with a dark dashed border and descriptive label for where real screen recordings go.
- All demo-placeholder divs styled: background: #f1f5f9; border: 2px dashed #94a3b8; border-radius: 8px; padding: 40px; text-align: center; color: #64748b; font-size: 14px;

Output ONLY valid HTML. Start with <!DOCTYPE html>. End with </html>. No explanation before or after."""

def _build_onepager_prompt(outputs: dict, brief: str) -> str:
    bd = extract_brand_data(outputs)
    font_link = _font_link(bd["fonts"])
    font_stack = _font_stack(bd["fonts"])
    inputs = "\n\n".join(f"## {k.upper()}\n\n{v}" for k, v in outputs.items() if v)
    return f"""Generate a complete, print-ready HTML one-pager for a B2B product.

BRAND COLORS: Primary: {bd['primary']} | Secondary: {bd['secondary']} | Accent: {bd['accent']}
FONTS CDN: {font_link}
Font stack: {font_stack}

INPUTS:
{inputs}

ORIGINAL BRIEF: {brief}

REQUIREMENTS:
- Self-contained HTML, Google Fonts CDN <link> in <head>, all CSS inline.
- Designed to print as one A4 page (210mm × 297mm). Use @page and @media print CSS.
- Layout: header with brand name + tagline, one-sentence value prop (large), 3 key points in columns, one demo-placeholder (labeled "📹 Product screenshot here"), one testimonial quote, clear CTA with URL placeholder, footer.
- Use exact hex colors above. Professional, premium feel — suitable for a board meeting or investor packet.
- demo-placeholder: background:#f1f5f9; border:2px dashed #94a3b8; border-radius:6px; padding:30px; text-align:center; color:#64748b;
- @media print: .no-print display none; page break controls; margins: 15mm.

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
    html = await call_agent(system_msg, prompt, model="claude-opus-4-6", max_tokens=8192)
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
    "html_website":  (".html", "website"),
    "html_onepager": (".html", "one-pager"),
    "html_deck":     (".html", "deck"),
    "social_pack":   (".md",   "social-pack"),
}

# ── Canva OAuth (PKCE) ────────────────────────────────────────────

def _pkce_pair() -> tuple[str, str]:
    verifier   = secrets.token_urlsafe(64)
    challenge  = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge

def canva_connected() -> bool:
    return bool(_canva.get("access_token"))

async def canva_create_design(title: str, design_type: str = "Presentation") -> Optional[dict]:
    """Create a blank design in Canva and return the edit URL."""
    token = _canva.get("access_token")
    if not token:
        return None
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(
                f"{CANVA_API_BASE}/designs",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"design_type": {"type": "preset", "name": design_type}, "title": title},
                timeout=15,
            )
            if r.status_code == 401:
                _canva["access_token"] = None
                return None
            r.raise_for_status()
            data = r.json()
            return {
                "design_id": data["design"]["id"],
                "edit_url":  data["design"]["urls"]["edit_url"],
                "view_url":  data["design"]["urls"].get("view_url", ""),
            }
        except Exception:
            return None

async def canva_refresh_token() -> bool:
    refresh = _canva.get("refresh_token")
    if not refresh or not CANVA_CLIENT_ID:
        return False
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(CANVA_TOKEN_URL, data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh,
                "client_id":     CANVA_CLIENT_ID,
                "client_secret": CANVA_CLIENT_SECRET,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            _canva["access_token"]  = data["access_token"]
            _canva["refresh_token"] = data.get("refresh_token", refresh)
            return True
        except Exception:
            return False

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
    ts   = job["timestamp"]
    path = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{_slug(job['brief'])}.md"
    lines = ["# Studio N — Job Output", "",
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
  .nav-canva { font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 6px; text-decoration: none; transition: all 0.15s; }
  .nav-canva.connected { background: #dcfce7; color: #16a34a; }
  .nav-canva.disconnected { background: rgba(255,255,255,0.08); color: #94a3b8; border: 1px solid rgba(255,255,255,0.1); }
  .container { max-width: 900px; margin: 0 auto; padding: 32px 24px 80px; }
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

def _nav(active: str, canva_ok: bool = False) -> str:
    def lnk(href, label, key):
        cls = "nav-link active" if key == active else "nav-link"
        return f'<a href="{href}" class="{cls}">{label}</a>'
    canva_html = (
        f'<a href="/auth/canva" class="nav-canva {"connected" if canva_ok else "disconnected"}">'
        f'{"✓ Canva" if canva_ok else "Connect Canva"}</a>'
    )
    return f"""<nav>
      <a href="/" class="nav-brand">Studio N<span>by Marcus</span></a>
      <div class="nav-links">
        {lnk('/','Brief','brief')}
        {lnk('/studio','Studio','studio')}
        {lnk('/outputs','Outputs','outputs')}
        {canva_html}
      </div>
    </nav>"""

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

# ── Job processor ─────────────────────────────────────────────────

async def process_job(job_id: str, brief: str) -> None:
    q = _jobs[job_id]

    async def emit(ev: dict) -> None:
        await q.put(ev)

    job: dict = {"brief": brief, "timestamp": datetime.now()}

    try:
        await emit({"type": "status", "message": "Marcus is reading your brief..."})
        marcus_system = await fetch_agent("marcus")
        marcus_raw = await call_agent(
            marcus_system + ORCHESTRATION_SUFFIX, brief, model="claude-opus-4-6")

        json_match      = re.search(r"```json\s*(\{.*?\})\s*```", marcus_raw, re.DOTALL)
        marcus_analysis = marcus_raw
        agents_needed:  list[str] = []
        agent_briefs:   dict[str, str] = {}

        if json_match:
            marcus_analysis = marcus_raw[:json_match.start()].strip()
            try:
                p             = json.loads(json_match.group(1))
                agents_needed = p.get("agents_needed", [])
                agent_briefs  = p.get("briefs", {})
            except json.JSONDecodeError:
                pass

        job["marcus_analysis"] = marcus_analysis
        await emit({"type": "marcus_analysis", "content": marcus_analysis})

        valid_s1 = [n for n in agents_needed if n in agent_briefs and n in VALID_AGENTS]
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
            for name, content in stage1_outputs.items():
                await emit({"type": "specialist", "agent": name, "content": content})

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
                except json.JSONDecodeError:
                    pass

            job["marcus_review"] = marcus_review
            await emit({"type": "review", "content": marcus_review})

        # Stage 2: cascade agents
        cascade_outputs: dict[str, str] = {}
        next_agents = [n for n in cascade_plan.get("next_agents", [])
                       if n in VALID_AGENTS and n not in stage1_outputs]

        if next_agents:
            await emit({"type": "cascade_start", "agents": next_agents})
            await emit({"type": "status",
                        "message": f"Cascade: {', '.join(n.capitalize() for n in next_agents)}..."})

            ctx = "\n\n".join(
                f"## {n.capitalize()} — Approved\n\n{o}" for n, o in stage1_outputs.items())
            base_brief = (
                f"Original brief: {brief}\n\n---\n\n"
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

        # Assembly
        assembly_type = cascade_plan.get("assembly", "none")
        all_outputs   = {**stage1_outputs, **cascade_outputs}

        if assembly_type in ASSEMBLERS:
            await emit({"type": "assembly_start", "assembly_type": assembly_type})
            await emit({"type": "status",
                        "message": f"Assembling {assembly_type.replace('_', ' ')}..."})

            if assembly_type == "html_website":
                content = await assemble_html(
                    _build_website_prompt(all_outputs, brief),
                    "You generate complete, production-ready B2B HTML websites with Google Fonts. Output only valid HTML.")
            elif assembly_type == "html_onepager":
                content = await assemble_html(
                    _build_onepager_prompt(all_outputs, brief),
                    "You generate complete, print-ready HTML one-pagers with Google Fonts. Output only valid HTML.")
            elif assembly_type == "html_deck":
                content = await assemble_html(
                    _build_deck_prompt(all_outputs, brief),
                    "You generate complete HTML presentation decks with Google Fonts and keyboard navigation. Output only valid HTML.")
            else:
                content = await assemble_social_pack(all_outputs, brief)

            assembled_path = save_assembled(job, assembly_type, content)
            job[f"assembled_{assembly_type}"] = content

            canva_url = None
            if assembly_type in ("html_website", "html_onepager", "html_deck") and canva_connected():
                await emit({"type": "status", "message": "Creating Canva design..."})
                label_map = {"html_website": "Presentation", "html_onepager": "A4 Document",
                             "html_deck": "Presentation"}
                bd = extract_brand_data(all_outputs)
                title = f"{bd['brand_name'] or brief[:40]} — {assembly_type.replace('html_','').replace('_',' ').title()}"
                result = await canva_create_design(title, label_map.get(assembly_type, "Presentation"))
                if result:
                    canva_url = result["edit_url"]
                    job["canva_url"] = canva_url

            await emit({
                "type":          "assembly_done",
                "assembly_type": assembly_type,
                "label":         ASSEMBLERS[assembly_type][1],
                "filename":      assembled_path.name,
                "canva_url":     canva_url,
            })

        md_path = save_markdown(job)
        _completed[job_id] = job
        await emit({"type": "done", "job_id": job_id, "saved_as": md_path.name})

    except Exception as exc:
        await emit({"type": "error", "message": str(exc)})
    finally:
        await asyncio.sleep(7200)
        _jobs.pop(job_id, None)
        _completed.pop(job_id, None)

# ── Routes ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "canva_connected": canva_connected(),
    })

@app.post("/api/brief")
async def start_brief(request: Request):
    data  = await request.json()
    brief = data.get("brief", "").strip()
    if not brief:
        return JSONResponse({"error": "Brief is empty"}, status_code=400)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = asyncio.Queue()
    asyncio.create_task(process_job(job_id, brief))
    return {"job_id": job_id}

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

@app.get("/api/canva/status")
async def canva_status():
    return {"connected": canva_connected(), "connected_at": _canva.get("connected_at")}

# ── Canva OAuth ───────────────────────────────────────────────────

@app.get("/auth/canva")
async def canva_auth_start():
    if not CANVA_CLIENT_ID:
        return HTMLResponse(
            "<h2>Canva not configured</h2>"
            "<p>Add <code>CANVA_CLIENT_ID</code> and <code>CANVA_CLIENT_SECRET</code> to your .env file.</p>"
            "<p>Get credentials at <a href='https://www.canva.com/developers' target='_blank'>canva.com/developers</a>.</p>",
            status_code=400)
    state             = secrets.token_urlsafe(16)
    verifier, challenge = _pkce_pair()
    _canva["pkce"][state] = verifier
    params = urllib.parse.urlencode({
        "response_type":         "code",
        "client_id":             CANVA_CLIENT_ID,
        "redirect_uri":          CANVA_REDIRECT_URI,
        "scope":                 CANVA_SCOPES,
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
    })
    return RedirectResponse(f"{CANVA_AUTH_URL}?{params}")

@app.get("/auth/canva/callback")
async def canva_auth_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return HTMLResponse(
            f"<h2>Canva auth failed: {error}</h2><p><a href='/'>Back to Studio N</a></p>",
            status_code=400)
    verifier = _canva["pkce"].pop(state, None)
    if not verifier:
        return HTMLResponse("<h2>Invalid state</h2><p><a href='/'>Back</a></p>", status_code=400)
    async with httpx.AsyncClient() as h:
        try:
            r = await h.post(CANVA_TOKEN_URL, data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  CANVA_REDIRECT_URI,
                "client_id":     CANVA_CLIENT_ID,
                "client_secret": CANVA_CLIENT_SECRET,
                "code_verifier": verifier,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
            _canva["access_token"]  = data["access_token"]
            _canva["refresh_token"] = data.get("refresh_token")
            _canva["connected_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
        except Exception as exc:
            return HTMLResponse(
                f"<h2>Token exchange failed</h2><pre>{exc}</pre><p><a href='/'>Back</a></p>",
                status_code=500)
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;
min-height:100vh;background:#f8fafc;}</style></head><body>
<div style="text-align:center;"><div style="font-size:48px;margin-bottom:16px;">✓</div>
<h2 style="color:#1e293b;margin-bottom:8px;">Canva connected</h2>
<p style="color:#64748b;margin-bottom:24px;">Studio N can now create Canva designs automatically.</p>
<a href="/" style="background:#1e293b;color:white;padding:10px 24px;border-radius:8px;text-decoration:none;font-weight:600;">
Back to Studio N</a></div></body></html>""")

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

# ── Studio page ───────────────────────────────────────────────────

@app.get("/studio", response_class=HTMLResponse)
async def studio_page():
    html_files = sorted(OUTPUTS_DIR.glob("*.html"),
                        key=lambda f: f.stat().st_mtime, reverse=True)
    canva_ok = canva_connected()

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

    canva_banner = "" if canva_ok else """
    <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:14px 18px;
                margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;gap:16px;">
      <span style="font-size:13px;color:#92400e;">
        <strong>Canva not connected.</strong> Connect to automatically create Canva designs when HTML files are assembled.
      </span>
      <a href="/auth/canva" style="background:#1e293b;color:white;padding:7px 16px;border-radius:7px;
                                   font-size:12px;font-weight:600;text-decoration:none;white-space:nowrap;">
        Connect Canva →
      </a>
    </div>"""

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
{_nav('studio', canva_ok)}
<div class="container">
  {canva_banner}
  <div class="page-head">
    <span class="page-title">Studio</span>
    <span class="page-count">{count} HTML file{'s' if count!=1 else ''}</span>
  </div>
  {cards_html}
</div></body></html>""")

# ── Outputs page ──────────────────────────────────────────────────

def _parse_output_file(path: Path) -> dict:
    info = {"filename": path.name, "size_kb": round(path.stat().st_size/1024,1),
            "brief": path.stem, "date": "", "is_html": path.suffix == ".html", "label": ""}
    for l in ["website","one-pager","deck","social-pack"]:
        if l in path.name: info["label"] = l; break
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            if m := re.search(r"\*\*Brief:\*\* (.+)", text): info["brief"] = m.group(1).strip()
            if d := re.search(r"\*\*Date:\*\* (.+)",  text): info["date"]  = d.group(1).strip()
        else:
            if t := re.search(r"<title>(.+?)</title>", text): info["brief"] = t.group(1).strip()
    except Exception:
        pass
    return info

@app.get("/outputs", response_class=HTMLResponse)
async def outputs_page():
    files = [_parse_output_file(f) for f in
             sorted(OUTPUTS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
             if f.suffix in (".md",".html")]
    rows = ""
    for f in files:
        lb = (f'<span style="font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;'
              f'background:#dbeafe;color:#1d4ed8;margin-left:8px;">{f["label"].upper()}</span>'
              if f["label"] else "")
        view = (f'<a class="out-btn out-btn-view" href="/outputs/view/{f["filename"]}" target="_blank">View</a>'
                if f["is_html"] else
                f'<a class="out-btn out-btn-pdf" href="/outputs/view/{f["filename"]}" target="_blank">⬇ PDF</a>')
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
</style></head><body>
{_nav('outputs', canva_connected())}
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
    if not path.exists() or path.suffix not in (".md",".html"):
        return JSONResponse({"error": "File not found"}, status_code=404)
    mime = "text/html" if path.suffix == ".html" else "text/markdown"
    return StreamingResponse(iter([path.read_bytes()]), media_type=mime,
                             headers={"Content-Disposition": f'attachment; filename="{safe}"'})

@app.get("/outputs/view/{filename}", response_class=HTMLResponse)
async def view_output(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix not in (".md",".html"):
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
    return {"status": "ok", "canva": canva_connected()}
