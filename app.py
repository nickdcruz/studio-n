import asyncio
import base64
import json
import os
import re
import secrets
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

load_dotenv()

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Marcus — Marketing Command")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

GITHUB_REPO       = os.getenv("GITHUB_REPO", "nickdcruz/nicklaus-marketing-agents")
GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HTTP_USER         = os.getenv("HTTP_USER", "admin")
HTTP_PASS         = os.getenv("HTTP_PASS", "changeme")
PORT              = int(os.getenv("PORT", "5050"))

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_agent_cache: dict[str, str]        = {}
_jobs:        dict[str, asyncio.Queue] = {}
_completed:   dict[str, dict]       = {}

VALID_AGENTS = {"callum", "priya", "dante", "suki", "felix", "nadia", "zara", "reeva"}

THIRD_PARTY_MESSAGES = {
    "video_production": (
        "Dante's video concept is saved and ready. "
        "Actual production requires Canva, CapCut, or Adobe Premiere — this step cannot be automated. "
        "The brief is in your outputs folder."
    ),
    "social_images": (
        "Suki's design brief is saved and ready. "
        "Static image creation requires Canva or an image generation API. "
        "Add DALLE_API_KEY to your .env to enable automated image generation."
    ),
    "publishing": (
        "Content is approved and ready to schedule. "
        "Connect Buffer, Hootsuite, or a platform API to enable auto-publishing from this app."
    ),
}

# ── Auth middleware ───────────────────────────────────────────────
# /health and /api/stream/* are exempt — EventSource cannot send Basic Auth headers.

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/health" or path.startswith("/api/stream/"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Marcus"'})
        try:
            decoded   = base64.b64decode(auth[6:]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
        except Exception:
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Marcus"'})
        if not (secrets.compare_digest(user, HTTP_USER) and
                secrets.compare_digest(pwd,  HTTP_PASS)):
            return Response("Unauthorized", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Marcus"'})
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

ASSEMBLY GUIDE — choose one value for assembly:
• Brand foundation delivered        → next_agents:[felix,nadia,suki]  assembly:html_website
• Website or landing page           → next_agents:[nadia,suki]        assembly:html_website
• One-pager or sales sheet          → next_agents:[felix,suki]        assembly:html_onepager
• Pitch deck or presentation        → next_agents:[felix]             assembly:html_deck
• Full social/content campaign      → next_agents:[priya,dante,suki]  assembly:social_pack  flags:[video_production,social_images]
• LinkedIn or long-form only        → next_agents:[]  assembly:none
• Research, strategy, or analysis   → next_agents:[]  assembly:none

flags options: video_production, social_images, publishing
Only include flags when those outputs are part of this job.
"""

# ── Assembly prompts ──────────────────────────────────────────────

_WEBSITE_PROMPT = """You are generating a complete, production-ready HTML website.

INPUTS FROM THE MARKETING TEAM:
{inputs}

ORIGINAL BRIEF:
{brief}

Generate a single, self-contained HTML5 file — a real website, not a mockup.

Requirements:
- All CSS inline, no external stylesheets or fonts (use system font stack)
- Fully responsive — works on mobile and desktop
- Sections: navigation, hero, value proposition, key benefits/features, social proof (placeholder cards), CTA section, footer
- Use the brand colors, tone, and positioning from the brand/strategy inputs
- Use the copy from the copy inputs verbatim where provided
- Use the visual/layout direction from any design brief inputs
- Real content throughout — no Lorem ipsum, no placeholder headings
- Clean, modern, professional aesthetic

Output ONLY valid HTML. Start with <!DOCTYPE html>. End with </html>. No explanation."""

_ONEPAGER_PROMPT = """You are generating a complete, print-ready HTML one-pager.

INPUTS FROM THE MARKETING TEAM:
{inputs}

ORIGINAL BRIEF:
{brief}

Generate a single, self-contained HTML file designed to be printed or saved as PDF.

Requirements:
- All CSS inline, print-optimised (A4 portrait)
- Single page — everything must fit on one printed page
- Sections: header with logo placeholder, headline, value proposition, 3 key points, one CTA, footer
- Brand colors and typography from any brand/strategy inputs
- Copy from any copy inputs
- Clean, premium, professional — suitable for a sales or investor meeting
- @media print CSS to hide any on-screen UI elements

Output ONLY valid HTML. Start with <!DOCTYPE html>. End with </html>. No explanation."""

_DECK_PROMPT = """You are generating a complete HTML presentation deck.

INPUTS FROM THE MARKETING TEAM:
{inputs}

ORIGINAL BRIEF:
{brief}

Generate a single, self-contained HTML5 presentation.

Requirements:
- All CSS inline, no external dependencies
- Each slide is a full-viewport <section class="slide">
- Navigation: arrow keys or on-screen prev/next buttons
- 8–12 slides covering: title, problem, solution, key features, proof/traction, team placeholder, pricing/tiers placeholder, CTA
- Brand colors and tone from any brand/strategy inputs
- Real content from any copy or strategy inputs
- Clean slide layouts — one message per slide, no walls of text
- Print CSS: each slide prints as one page

Output ONLY valid HTML. Start with <!DOCTYPE html>. End with </html>. No explanation."""

_SOCIAL_PACK_PROMPT = """Compile the following marketing team outputs into a structured, ready-to-use Social Media Content Pack.

INPUTS:
{inputs}

ORIGINAL BRIEF:
{brief}

Format as a clean markdown document with clear sections:
- Cover: client, brief summary, date
- LinkedIn Posts (numbered, ready to copy-paste)
- Instagram Captions (numbered, with hashtag sets)
- TikTok/Reels Scripts or Concepts (numbered)
- Facebook Posts (if applicable)
- Design Briefs for Static Posts (from Suki if present)
- Video Production Brief (from Dante if present, flagged as manual)
- Hashtag Master List

Output clean, final markdown. No meta-commentary."""

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

# ── Assemblers ────────────────────────────────────────────────────

def _format_inputs(outputs: dict) -> str:
    return "\n\n".join(
        f"## {name.upper()}\n\n{content}"
        for name, content in outputs.items() if content
    )

async def assemble_html_website(outputs: dict, brief: str) -> str:
    prompt = _WEBSITE_PROMPT.format(inputs=_format_inputs(outputs), brief=brief)
    html = await call_agent(
        "You generate complete, production-ready HTML websites. Output only valid HTML.",
        prompt, model="claude-opus-4-6", max_tokens=8192
    )
    # Strip any text before <!DOCTYPE
    m = re.search(r"<!DOCTYPE", html, re.IGNORECASE)
    return html[m.start():] if m else html

async def assemble_html_onepager(outputs: dict, brief: str) -> str:
    prompt = _ONEPAGER_PROMPT.format(inputs=_format_inputs(outputs), brief=brief)
    html = await call_agent(
        "You generate complete, print-ready HTML one-pagers. Output only valid HTML.",
        prompt, model="claude-opus-4-6", max_tokens=6144
    )
    m = re.search(r"<!DOCTYPE", html, re.IGNORECASE)
    return html[m.start():] if m else html

async def assemble_html_deck(outputs: dict, brief: str) -> str:
    prompt = _DECK_PROMPT.format(inputs=_format_inputs(outputs), brief=brief)
    html = await call_agent(
        "You generate complete HTML presentation decks with keyboard navigation. Output only valid HTML.",
        prompt, model="claude-opus-4-6", max_tokens=8192
    )
    m = re.search(r"<!DOCTYPE", html, re.IGNORECASE)
    return html[m.start():] if m else html

async def assemble_social_pack(outputs: dict, brief: str) -> str:
    prompt = _SOCIAL_PACK_PROMPT.format(inputs=_format_inputs(outputs), brief=brief)
    return await call_agent(
        "You compile marketing outputs into clean, structured content packs.",
        prompt, model="claude-opus-4-6", max_tokens=4096
    )

ASSEMBLERS = {
    "html_website":  (assemble_html_website,  ".html", "website"),
    "html_onepager": (assemble_html_onepager, ".html", "one-pager"),
    "html_deck":     (assemble_html_deck,     ".html", "deck"),
    "social_pack":   (assemble_social_pack,   ".md",   "social-pack"),
}

# ── Save helpers ──────────────────────────────────────────────────

def _slug(text: str, length: int = 35) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text[:length].lower()).strip("-")

def save_markdown(job: dict) -> Path:
    ts   = job["timestamp"]
    path = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{_slug(job['brief'])}.md"
    lines = [
        "# Marcus — Job Output", "",
        f"**Date:** {ts.strftime('%Y-%m-%d %H:%M')}", "",
        f"**Brief:** {job['brief']}", "",
        "---", "",
    ]
    if job.get("marcus_analysis"):
        lines += ["## Marcus — Brief Analysis", "", job["marcus_analysis"], "", "---", ""]
    for agent, content in (job.get("stage1_outputs") or {}).items():
        lines += [f"## {agent.capitalize()} — Stage 1", "", content, "", "---", ""]
    if job.get("marcus_review"):
        lines += ["## Marcus — Stage 1 Review", "", job["marcus_review"], "", "---", ""]
    for agent, content in (job.get("cascade_outputs") or {}).items():
        lines += [f"## {agent.capitalize()} — Cascade", "", content, "", "---", ""]
    if job.get("marcus_cascade_review"):
        lines += ["## Marcus — Final Review", "", job["marcus_cascade_review"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def save_assembled(job: dict, assembly_type: str, content: str) -> Path:
    ts  = job["timestamp"]
    _, ext, label = ASSEMBLERS[assembly_type]
    path = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{_slug(job['brief'])}_{label}{ext}"
    path.write_text(content, encoding="utf-8")
    return path

# ── Server-side markdown → HTML ───────────────────────────────────

def _md_to_html(text: str) -> str:
    t = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = t.split("**")
    t = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
    t = re.sub(r"^### (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = re.sub(r"^## (.+)$",  r"<h2>\1</h2>", t, flags=re.MULTILINE)
    t = re.sub(r"^# (.+)$",   r"<h1>\1</h1>", t, flags=re.MULTILINE)
    t = re.sub(r"^---+$",      "<hr>",          t, flags=re.MULTILINE)
    t = re.sub(r"^[-•] (.+)$", r"<li>\1</li>",  t, flags=re.MULTILINE)
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

# ── Shared nav + print styles ─────────────────────────────────────

_PAGE_STYLE = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; min-height: 100vh; }
  nav { background: #1e293b; padding: 14px 32px; display: flex; align-items: center; justify-content: space-between; }
  .nav-brand { color: white; font-size: 16px; font-weight: 700; text-decoration: none; }
  .nav-links { display: flex; gap: 6px; }
  .nav-link { color: #94a3b8; font-size: 13px; text-decoration: none; padding: 6px 12px; border-radius: 6px; transition: all 0.15s; }
  .nav-link:hover, .nav-link.active { background: rgba(255,255,255,0.1); color: white; }
  .container { max-width: 860px; margin: 0 auto; padding: 32px 24px 80px; }
"""

_PRINT_STYLE = """
  body { font-family: -apple-system, sans-serif; max-width: 780px; margin: 40px auto; padding: 0 24px 60px; color: #1e293b; }
  .toolbar { display:flex; gap:10px; margin-bottom:28px; }
  .btn { background:#1e293b; color:white; border:none; padding:9px 20px; border-radius:7px; font-size:13px; font-weight:600; cursor:pointer; }
  .btn-outline { background:none; border:1px solid #e2e8f0; color:#475569; }
  h1 { font-size:20px; font-weight:700; margin-bottom:4px; }
  .meta { color:#64748b; font-size:13px; margin-bottom:8px; }
  .brief-box { background:#f8fafc; border-left:3px solid #1e293b; padding:11px 16px; margin-bottom:28px; font-size:13px; color:#475569; border-radius:0 6px 6px 0; }
  h2 { font-size:16px; font-weight:700; margin:28px 0 10px; } h3 { font-size:14px; font-weight:700; margin:20px 0 8px; }
  p { font-size:14px; line-height:1.75; margin-bottom:12px; }
  ul { padding-left:20px; margin-bottom:12px; } li { font-size:14px; line-height:1.7; margin-bottom:4px; }
  strong { font-weight:700; } hr { border:none; border-top:1px solid #e2e8f0; margin:24px 0; }
  @media print { .toolbar { display:none !important; } body { margin:20px; } }
"""

def _nav(active: str) -> str:
    def lnk(href, label, key):
        cls = "nav-link active" if key == active else "nav-link"
        return f'<a href="{href}" class="{cls}">{label}</a>'
    return f"""<nav>
      <a href="/" class="nav-brand">Marcus</a>
      <div class="nav-links">{lnk('/','Brief','brief')}{lnk('/outputs','Outputs','outputs')}</div>
    </nav>"""

def _print_page(title: str, brief: str, date: str, body_html: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<style>{_PRINT_STYLE}</style></head><body>
<div class="toolbar">
  <button class="btn" onclick="window.print()">Save as PDF</button>
  <button class="btn btn-outline" onclick="window.close()">Close</button>
</div>
<h1>{title}</h1><div class="meta">{date}</div>
<div class="brief-box"><strong>Brief:</strong> {brief}</div>
{body_html}</body></html>"""

# ── Job processor ─────────────────────────────────────────────────

async def process_job(job_id: str, brief: str) -> None:
    q = _jobs[job_id]

    async def emit(event: dict) -> None:
        await q.put(event)

    job: dict = {"brief": brief, "timestamp": datetime.now()}

    try:
        # ── Stage 1: Marcus plans + initial agents ────────────────
        await emit({"type": "status", "message": "Marcus is reading your brief..."})
        marcus_system = await fetch_agent("marcus")
        marcus_raw = await call_agent(
            marcus_system + ORCHESTRATION_SUFFIX, brief, model="claude-opus-4-6")

        json_match    = re.search(r"```json\s*(\{.*?\})\s*```", marcus_raw, re.DOTALL)
        marcus_analysis = marcus_raw
        agents_needed: list[str] = []
        agent_briefs:  dict[str, str] = {}

        if json_match:
            marcus_analysis = marcus_raw[:json_match.start()].strip()
            try:
                parsed        = json.loads(json_match.group(1))
                agents_needed = parsed.get("agents_needed", [])
                agent_briefs  = parsed.get("briefs", {})
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
            sys = await fetch_agent(name)
            out = await call_agent(sys, agent_briefs[name])
            return name, out

        stage1_outputs: dict[str, str] = {}
        if valid_s1:
            results = await asyncio.gather(*[run_s1(n) for n in valid_s1])
            stage1_outputs = dict(results)
            for name, content in stage1_outputs.items():
                await emit({"type": "specialist", "agent": name, "content": content})

        job["stage1_outputs"] = stage1_outputs

        # ── Stage 1 review + cascade decision ─────────────────────
        marcus_review  = None
        cascade_plan   = {"next_agents": [], "assembly": "none", "flags": []}

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

            cascade_match = re.search(r"```cascade\s*(\{.*?\})\s*```", review_raw, re.DOTALL)
            marcus_review = review_raw[:cascade_match.start()].strip() if cascade_match else review_raw

            if cascade_match:
                try:
                    parsed = json.loads(cascade_match.group(1))
                    cascade_plan = parsed.get("cascade", cascade_plan)
                except json.JSONDecodeError:
                    pass

            job["marcus_review"] = marcus_review
            await emit({"type": "review", "content": marcus_review})

        # ── Stage 2: Cascade agents ────────────────────────────────
        cascade_outputs: dict[str, str] = {}
        next_agents = [n for n in cascade_plan.get("next_agents", [])
                       if n in VALID_AGENTS and n not in stage1_outputs]

        if next_agents:
            await emit({"type": "cascade_start", "agents": next_agents})
            await emit({"type": "status",
                        "message": f"Cascade: briefing {', '.join(n.capitalize() for n in next_agents)}..."})

            # Build context from all stage 1 approved outputs
            context_block = "\n\n".join(
                f"## {n.capitalize()} — Approved Output\n\n{o}"
                for n, o in stage1_outputs.items()
            )
            cascade_brief_base = (
                f"Original brief: {brief}\n\n---\n\n"
                f"CONTEXT FROM PREVIOUS STAGE (your source of truth — use this, do not contradict it):\n\n"
                f"{context_block}\n\n---\n\n"
                f"Your output will go directly into the final assembled deliverable. "
                f"Produce complete, final content — not a draft, not a brief."
            )

            async def run_cascade(name: str) -> tuple[str, str]:
                sys = await fetch_agent(name)
                out = await call_agent(sys, cascade_brief_base)
                return name, out

            results = await asyncio.gather(*[run_cascade(n) for n in next_agents])
            cascade_outputs = dict(results)
            for name, content in cascade_outputs.items():
                await emit({"type": "cascade_output", "agent": name, "content": content})

        job["cascade_outputs"] = cascade_outputs

        # ── Third-party flags ──────────────────────────────────────
        for flag in cascade_plan.get("flags", []):
            msg = THIRD_PARTY_MESSAGES.get(flag, f"Manual step required: {flag}")
            await emit({"type": "third_party_flag", "flag": flag, "message": msg})

        # ── Stage 2 review (if cascade ran) ───────────────────────
        if cascade_outputs:
            await emit({"type": "status", "message": "Marcus is reviewing cascade outputs..."})
            review2_input = (
                "Review the cascade stage outputs against the brief and brand standards.\n\n"
                + "".join(f"## {n.capitalize()}\n\n{o}\n\n---\n\n"
                          for n, o in cascade_outputs.items())
            )
            cascade_review_raw = await call_agent(
                marcus_system, review2_input, model="claude-opus-4-6")
            job["marcus_cascade_review"] = cascade_review_raw
            await emit({"type": "cascade_review", "content": cascade_review_raw})

        # ── Assembly ───────────────────────────────────────────────
        assembly_type = cascade_plan.get("assembly", "none")
        if assembly_type in ASSEMBLERS:
            await emit({"type": "status",
                        "message": f"Assembling final {assembly_type.replace('_', ' ')}..."})
            await emit({"type": "assembly_start", "assembly_type": assembly_type})

            all_outputs = {**stage1_outputs, **cascade_outputs}
            fn, _, label = ASSEMBLERS[assembly_type]
            assembled_content = await fn(all_outputs, brief)
            assembled_path    = save_assembled(job, assembly_type, assembled_content)

            job[f"assembled_{assembly_type}"] = assembled_content
            await emit({
                "type":          "assembly_done",
                "assembly_type": assembly_type,
                "label":         label,
                "filename":      assembled_path.name,
            })

        # ── Save full markdown ─────────────────────────────────────
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
    return templates.TemplateResponse("index.html", {"request": request})

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

@app.get("/api/export/{job_id}/{agent}.md")
async def export_md(job_id: str, agent: str):
    job = _completed.get(job_id)
    if not job:
        return JSONResponse({"error": "Output not found or expired"}, status_code=404)
    ts = job["timestamp"]
    if agent == "full":
        path = save_markdown(job); content = path.read_text(); fname = path.name
    elif agent == "review":
        content = job.get("marcus_review") or job.get("marcus_cascade_review") or ""
        fname = f"marcus-review_{ts.strftime('%Y-%m-%d_%H-%M')}.md"
    elif agent == "analysis":
        content = job.get("marcus_analysis", "")
        fname = f"marcus-analysis_{ts.strftime('%Y-%m-%d_%H-%M')}.md"
    else:
        content = ({**job.get("stage1_outputs",{}), **job.get("cascade_outputs",{})}).get(agent, "")
        if not content:
            return JSONResponse({"error": "Agent output not found"}, status_code=404)
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
        title, md = "Full Output", ""
        for sec, key in [("Brief Analysis","marcus_analysis"),("Stage 1 Review","marcus_review"),
                         ("Final Review","marcus_cascade_review")]:
            if job.get(key): md += f"## Marcus — {sec}\n\n{job[key]}\n\n---\n\n"
        for n, o in all_out.items(): md += f"## {n.capitalize()}\n\n{o}\n\n---\n\n"
    elif agent in ("review","analysis"):
        k = "marcus_review" if agent == "review" else "marcus_analysis"
        title, md = f"Marcus — {agent.capitalize()}", job.get(k,"")
    else:
        title, md = f"{agent.capitalize()} — Output", all_out.get(agent,"")
    return HTMLResponse(_print_page(title, job["brief"], ts, _md_to_html(md)))

# ── Outputs page ──────────────────────────────────────────────────

def _parse_output_file(path: Path) -> dict:
    info = {"filename": path.name, "size_kb": round(path.stat().st_size/1024,1),
            "brief": path.stem, "date": "", "is_html": path.suffix == ".html",
            "label": ""}
    for label in ["website","one-pager","deck","social-pack"]:
        if label in path.name:
            info["label"] = label
            break
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            if m := re.search(r"\*\*Brief:\*\* (.+)", text):  info["brief"] = m.group(1).strip()
            if d := re.search(r"\*\*Date:\*\* (.+)",  text):  info["date"]  = d.group(1).strip()
        else:
            if t := re.search(r"<title>(.+?)</title>", text): info["brief"] = t.group(1).strip()
    except Exception:
        pass
    return info

def _list_output_files() -> list[dict]:
    return [_parse_output_file(f) for f in
            sorted(OUTPUTS_DIR.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
            if f.suffix in (".md", ".html")]

@app.get("/outputs", response_class=HTMLResponse)
async def outputs_page():
    files = _list_output_files()
    rows = ""
    for f in files:
        label_html = (f'<span style="font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;'
                      f'background:#dbeafe;color:#1d4ed8;margin-left:8px;">{f["label"].upper()}</span>'
                      if f["label"] else "")
        view_btn = (f'<a class="out-btn out-btn-view" href="/outputs/view/{f["filename"]}" target="_blank">View</a>'
                    if f["is_html"] else
                    f'<a class="out-btn out-btn-pdf" href="/outputs/view/{f["filename"]}" target="_blank">⬇ PDF</a>')
        rows += f"""
        <div class="output-row">
          <div class="output-meta">
            <span class="output-date">{f['date'] or '—'}</span>
            <span class="output-size">{f['size_kb']} KB</span>
          </div>
          <div class="output-brief">{f['brief']}{label_html}</div>
          <div class="output-actions">
            <a class="out-btn" href="/outputs/download/{f['filename']}">⬇ {f['filename'].split('.')[-1].upper()}</a>
            {view_btn}
          </div>
        </div>"""
    if not files:
        rows = """<div style="text-align:center;padding:60px 0;color:#94a3b8;">
          <div style="font-size:32px;margin-bottom:12px;">📂</div>
          <div style="font-size:15px;font-weight:600;color:#475569;">No outputs yet</div>
          <div style="font-size:13px;margin-top:6px;">Submit a brief on the <a href="/" style="color:#1e293b;">Brief page</a>.</div>
        </div>"""
    count = len(files)
    return HTMLResponse(f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Outputs — Marcus</title>
<style>
  {_PAGE_STYLE}
  .page-head {{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:24px;}}
  .page-title {{font-size:20px;font-weight:700;}}
  .page-count {{font-size:13px;color:#94a3b8;}}
  .output-row {{background:white;border-radius:10px;padding:18px 20px;margin-bottom:10px;
    box-shadow:0 1px 3px rgba(0,0,0,0.07);display:flex;align-items:center;gap:16px;transition:box-shadow 0.15s;}}
  .output-row:hover {{box-shadow:0 2px 8px rgba(0,0,0,0.1);}}
  .output-meta {{display:flex;flex-direction:column;gap:3px;min-width:130px;}}
  .output-date {{font-size:12px;font-weight:600;color:#475569;}}
  .output-size {{font-size:11px;color:#94a3b8;}}
  .output-brief {{flex:1;font-size:13px;color:#334155;line-height:1.5;}}
  .output-actions {{display:flex;gap:6px;flex-shrink:0;}}
  .out-btn {{font-size:11px;font-weight:600;padding:5px 11px;border-radius:6px;
    border:1px solid #e2e8f0;color:#475569;text-decoration:none;transition:all 0.15s;white-space:nowrap;}}
  .out-btn:hover {{background:#f8fafc;color:#1e293b;}}
  .out-btn-pdf  {{border-color:#bfdbfe;color:#2563eb;}}
  .out-btn-pdf:hover  {{background:#eff6ff;}}
  .out-btn-view {{border-color:#bbf7d0;color:#16a34a;font-weight:700;}}
  .out-btn-view:hover {{background:#f0fdf4;}}
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
    if not path.exists() or path.suffix not in (".md", ".html"):
        return JSONResponse({"error": "File not found"}, status_code=404)
    mime = "text/html" if path.suffix == ".html" else "text/markdown"
    return StreamingResponse(iter([path.read_bytes()]), media_type=mime,
                             headers={"Content-Disposition": f'attachment; filename="{safe}"'})

@app.get("/outputs/view/{filename}", response_class=HTMLResponse)
async def view_output(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix not in (".md", ".html"):
        return HTMLResponse("<h1>File not found</h1>", status_code=404)
    content = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        return HTMLResponse(content)   # serve the assembled HTML directly
    brief_m = re.search(r"\*\*Brief:\*\* (.+)", content)
    date_m  = re.search(r"\*\*Date:\*\* (.+)",  content)
    return HTMLResponse(_print_page(
        safe, brief_m.group(1).strip() if brief_m else safe,
        date_m.group(1).strip() if date_m else "", _md_to_html(content)))

@app.get("/health")
async def health():
    return {"status": "ok"}
