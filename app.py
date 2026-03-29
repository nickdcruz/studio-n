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
MODEL_MARCUS      = "claude-opus-4-6"
MODEL_SPECIALIST  = "claude-sonnet-4-6"

anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
_agent_cache: dict[str, str]       = {}
_jobs:        dict[str, asyncio.Queue] = {}
_completed:   dict[str, dict]      = {}

# ── Basic Auth middleware ─────────────────────────────────────────
# /health and /api/stream/* are exempt:
#   - /health: monitoring
#   - /api/stream/*: EventSource cannot send auth headers

_EXEMPT_EXACT    = {"/health"}
_EXEMPT_PREFIXES = ("/api/stream/",)

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _EXEMPT_EXACT or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return Response(
                "Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Marcus"'},
            )
        try:
            decoded   = base64.b64decode(auth[6:]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
        except Exception:
            return Response(
                "Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Marcus"'},
            )

        ok_user = secrets.compare_digest(user, HTTP_USER)
        ok_pass = secrets.compare_digest(pwd,  HTTP_PASS)
        if not (ok_user and ok_pass):
            return Response(
                "Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Marcus"'},
            )
        return await call_next(request)

app.add_middleware(BasicAuthMiddleware)

# ── Orchestration prompt ──────────────────────────────────────────

ORCHESTRATION_SUFFIX = """

---
SYSTEM — ORCHESTRATION MODE: You are running inside an automated pipeline. After your full analysis and task breakdown, you MUST append a JSON block in this exact format and nothing after it:

```json
{
  "agents_needed": ["agent1", "agent2"],
  "briefs": {
    "agent1": "The complete brief for this agent, exactly as you would write it",
    "agent2": "The complete brief for this agent, exactly as you would write it"
  }
}
```

Valid agent names (lowercase only): callum, priya, dante, suki, felix, nadia, zara, reeva
If this brief requires no specialists (e.g. it is a question or planning discussion), return: {"agents_needed": [], "briefs": {}}
Do not add any text after the closing ``` of the JSON block.
"""

# ── GitHub fetch ──────────────────────────────────────────────────

async def fetch_agent(name: str) -> str:
    if name in _agent_cache:
        return _agent_cache[name]
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/agents/{name}.md"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    async with httpx.AsyncClient() as h:
        r = await h.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    _agent_cache[name] = r.text
    return r.text

# ── Anthropic call ────────────────────────────────────────────────

async def call_agent(system: str, message: str, model: str = MODEL_SPECIALIST) -> str:
    r = await anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    return r.content[0].text

# ── Markdown save ─────────────────────────────────────────────────

def save_markdown(job: dict) -> Path:
    ts   = job["timestamp"]
    slug = re.sub(r"[^a-z0-9]+", "-", job["brief"][:40].lower()).strip("-")
    path = OUTPUTS_DIR / f"{ts.strftime('%Y-%m-%d_%H-%M')}_{slug}.md"

    lines = [
        "# Marcus — Job Output", "",
        f"**Date:** {ts.strftime('%Y-%m-%d %H:%M')}", "",
        f"**Brief:** {job['brief']}", "",
        "---", "",
    ]
    if job.get("marcus_analysis"):
        lines += ["## Marcus — Brief Analysis", "", job["marcus_analysis"], "", "---", ""]
    for agent, content in (job.get("specialist_outputs") or {}).items():
        lines += [f"## {agent.capitalize()}", "", content, "", "---", ""]
    if job.get("marcus_review"):
        lines += ["## Marcus — Final Review", "", job["marcus_review"], ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

# ── Server-side markdown → HTML (for print/view pages) ───────────

def _md_to_html(text: str) -> str:
    t = (text
         .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    parts = t.split("**")
    t = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
    t = re.sub(r"^### (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = re.sub(r"^## (.+)$",  r"<h2>\1</h2>", t, flags=re.MULTILINE)
    t = re.sub(r"^# (.+)$",   r"<h1>\1</h1>", t, flags=re.MULTILINE)
    t = re.sub(r"^---+$",     "<hr>",          t, flags=re.MULTILINE)
    t = re.sub(r"^[-•] (.+)$", r"<li>\1</li>", t, flags=re.MULTILINE)
    chunks, out = t.split("\n\n"), []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.match(r"^<(h[1-6]|li|hr)", chunk):
            chunk = re.sub(r"(<li>.*</li>)", r"<ul>\1</ul>", chunk, flags=re.DOTALL)
            out.append(chunk)
        else:
            out.append(f"<p>{chunk.replace(chr(10), '<br>')}</p>")
    return "\n".join(out)

# ── Shared page chrome ────────────────────────────────────────────

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

def _nav(active: str) -> str:
    def link(href, label, key):
        cls = 'nav-link active' if key == active else 'nav-link'
        return f'<a href="{href}" class="{cls}">{label}</a>'
    return f"""
    <nav>
      <a href="/" class="nav-brand">Marcus</a>
      <div class="nav-links">
        {link('/','Brief',      'brief')}
        {link('/outputs','Outputs','outputs')}
      </div>
    </nav>"""

# ── Outputs helpers ───────────────────────────────────────────────

def _parse_output_file(path: Path) -> dict:
    info = {
        "filename": path.name,
        "size_kb":  round(path.stat().st_size / 1024, 1),
        "brief":    path.stem,
        "date":     "",
    }
    try:
        content = path.read_text(encoding="utf-8")
        m = re.search(r"\*\*Brief:\*\* (.+)", content)
        d = re.search(r"\*\*Date:\*\* (.+)", content)
        if m:
            info["brief"] = m.group(1).strip()
        if d:
            info["date"] = d.group(1).strip()
    except Exception:
        pass
    return info

def _list_output_files() -> list[dict]:
    files = sorted(OUTPUTS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [_parse_output_file(f) for f in files]

# ── Job processor ─────────────────────────────────────────────────

async def process_job(job_id: str, brief: str) -> None:
    q = _jobs[job_id]

    async def emit(event: dict) -> None:
        await q.put(event)

    marcus_analysis    = ""
    specialist_outputs: dict[str, str] = {}
    marcus_review      = None

    try:
        await emit({"type": "status", "message": "Marcus is reading your brief..."})

        marcus_system = await fetch_agent("marcus")
        marcus_raw = await call_agent(
            marcus_system + ORCHESTRATION_SUFFIX, brief, model=MODEL_MARCUS)

        json_match    = re.search(r"```json\s*(\{.*?\})\s*```", marcus_raw, re.DOTALL)
        marcus_analysis = marcus_raw
        agents_needed: list[str] = []
        agent_briefs:  dict[str, str] = {}

        if json_match:
            marcus_analysis = marcus_raw[: json_match.start()].strip()
            try:
                parsed        = json.loads(json_match.group(1))
                agents_needed = parsed.get("agents_needed", [])
                agent_briefs  = parsed.get("briefs", {})
            except json.JSONDecodeError:
                pass

        await emit({"type": "marcus_analysis", "content": marcus_analysis})

        valid_agents = [n for n in agents_needed if n in agent_briefs]
        if valid_agents:
            await emit({"type": "agents_identified", "agents": valid_agents})
            await emit({"type": "status",
                        "message": f"Running {', '.join(n.capitalize() for n in valid_agents)} in parallel..."})

        async def run_specialist(name: str) -> tuple[str, str]:
            system = await fetch_agent(name)
            output = await call_agent(system, agent_briefs[name])
            return name, output

        if valid_agents:
            results = await asyncio.gather(*[run_specialist(n) for n in valid_agents])
            specialist_outputs = dict(results)
            for name, output in specialist_outputs.items():
                await emit({"type": "specialist", "agent": name, "content": output})

        if specialist_outputs:
            await emit({"type": "status", "message": "Marcus is reviewing all outputs..."})
            review_input = (
                "Here are the outputs from your team. "
                "Review each against the brief and brand standards. "
                "Give your verdict (Approved / Revise / Reject) for each with specific feedback.\n\n"
            )
            for name, output in specialist_outputs.items():
                review_input += f"## {name.capitalize()}\n\n{output}\n\n---\n\n"
            marcus_review = await call_agent(marcus_system, review_input, model=MODEL_MARCUS)
            await emit({"type": "review", "content": marcus_review})

        job_data = {
            "brief":              brief,
            "timestamp":          datetime.now(),
            "marcus_analysis":    marcus_analysis,
            "specialist_outputs": specialist_outputs,
            "marcus_review":      marcus_review,
        }
        _completed[job_id] = job_data
        saved_path = save_markdown(job_data)

        await emit({"type": "done", "job_id": job_id, "saved_as": saved_path.name})

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

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/export/{job_id}/{agent}.md")
async def export_in_memory_md(job_id: str, agent: str):
    job = _completed.get(job_id)
    if not job:
        return JSONResponse({"error": "Output not found or expired"}, status_code=404)

    if agent == "full":
        path    = save_markdown(job)
        content = path.read_text(encoding="utf-8")
        fname   = path.name
    elif agent == "review":
        content = job.get("marcus_review") or ""
        fname   = f"marcus-review_{job['timestamp'].strftime('%Y-%m-%d_%H-%M')}.md"
    elif agent == "analysis":
        content = job.get("marcus_analysis") or ""
        fname   = f"marcus-analysis_{job['timestamp'].strftime('%Y-%m-%d_%H-%M')}.md"
    else:
        content = (job.get("specialist_outputs") or {}).get(agent, "")
        if not content:
            return JSONResponse({"error": "Agent output not found"}, status_code=404)
        fname = f"{agent}_{job['timestamp'].strftime('%Y-%m-%d_%H-%M')}.md"

    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/export/{job_id}/{agent}/print", response_class=HTMLResponse)
async def print_in_memory(job_id: str, agent: str):
    job = _completed.get(job_id)
    if not job:
        return HTMLResponse("<h1>Output not found or expired (2 hour limit)</h1>", status_code=404)
    return _build_print_page(agent, job["brief"], job["timestamp"].strftime("%Y-%m-%d %H:%M"), job, agent)


# ── Outputs page ──────────────────────────────────────────────────

@app.get("/outputs", response_class=HTMLResponse)
async def outputs_page():
    files = _list_output_files()

    if not files:
        rows_html = """
        <div style="text-align:center;padding:60px 0;color:#94a3b8;">
          <div style="font-size:32px;margin-bottom:12px;">📂</div>
          <div style="font-size:15px;font-weight:600;color:#475569;">No outputs yet</div>
          <div style="font-size:13px;margin-top:6px;">Submit a brief on the <a href="/" style="color:#1e293b;">Brief page</a> to generate your first output.</div>
        </div>"""
    else:
        rows_html = ""
        for f in files:
            rows_html += f"""
        <div class="output-row">
          <div class="output-meta">
            <span class="output-date">{f['date'] or '—'}</span>
            <span class="output-size">{f['size_kb']} KB</span>
          </div>
          <div class="output-brief">{f['brief']}</div>
          <div class="output-actions">
            <a class="out-btn" href="/outputs/download/{f['filename']}">⬇ MD</a>
            <a class="out-btn out-btn-pdf" href="/outputs/view/{f['filename']}" target="_blank">⬇ PDF</a>
          </div>
        </div>"""

    count = len(files)
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Outputs — Marcus</title>
  <style>
    {_PAGE_STYLE}
    .page-head {{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom:24px; }}
    .page-title {{ font-size:20px; font-weight:700; }}
    .page-count {{ font-size:13px; color:#94a3b8; }}
    .output-row {{
      background:white; border-radius:10px; padding:18px 20px;
      margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.07);
      display:flex; align-items:center; gap:16px;
    }}
    .output-row:hover {{ box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
    .output-meta {{ display:flex; flex-direction:column; gap:3px; min-width:130px; }}
    .output-date {{ font-size:12px; font-weight:600; color:#475569; }}
    .output-size {{ font-size:11px; color:#94a3b8; }}
    .output-brief {{ flex:1; font-size:13px; color:#334155; line-height:1.5; }}
    .output-actions {{ display:flex; gap:6px; flex-shrink:0; }}
    .out-btn {{
      font-size:11px; font-weight:600; padding:5px 11px; border-radius:6px;
      border:1px solid #e2e8f0; color:#475569; text-decoration:none;
      transition:all 0.15s; white-space:nowrap;
    }}
    .out-btn:hover {{ background:#f8fafc; color:#1e293b; }}
    .out-btn-pdf {{ border-color:#bfdbfe; color:#2563eb; }}
    .out-btn-pdf:hover {{ background:#eff6ff; }}
  </style>
</head>
<body>
  {_nav('outputs')}
  <div class="container">
    <div class="page-head">
      <span class="page-title">Outputs</span>
      <span class="page-count">{count} file{'s' if count != 1 else ''}</span>
    </div>
    {rows_html}
  </div>
</body>
</html>""")


@app.get("/outputs/download/{filename}")
async def download_output_file(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix != ".md":
        return JSONResponse({"error": "File not found"}, status_code=404)
    content = path.read_text(encoding="utf-8")
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safe}"'},
    )


@app.get("/outputs/view/{filename}", response_class=HTMLResponse)
async def view_output_file(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or path.suffix != ".md":
        return HTMLResponse("<h1>File not found</h1>", status_code=404)
    content = path.read_text(encoding="utf-8")
    brief_m = re.search(r"\*\*Brief:\*\* (.+)", content)
    date_m  = re.search(r"\*\*Date:\*\* (.+)", content)
    brief   = brief_m.group(1).strip() if brief_m else safe
    date    = date_m.group(1).strip()  if date_m  else ""
    return HTMLResponse(_build_print_page_raw("Full Output", brief, date, _md_to_html(content)))


# ── Print page builder ────────────────────────────────────────────

_PRINT_STYLE = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 780px; margin: 40px auto; padding: 0 24px 60px; color: #1e293b; }
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

def _build_print_page_raw(title: str, brief: str, date: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><title>{title}</title>
  <style>{_PRINT_STYLE}</style>
</head>
<body>
  <div class="toolbar">
    <button class="btn" onclick="window.print()">Save as PDF</button>
    <button class="btn btn-outline" onclick="window.close()">Close</button>
  </div>
  <h1>{title}</h1>
  <div class="meta">{date}</div>
  <div class="brief-box"><strong>Brief:</strong> {brief}</div>
  {body_html}
</body>
</html>"""

def _build_print_page(agent: str, brief: str, date: str, job: dict, _agent: str) -> HTMLResponse:
    if _agent == "full":
        title, content_md = "Full Output — All Agents", ""
        if job.get("marcus_analysis"):
            content_md += f"## Marcus — Brief Analysis\n\n{job['marcus_analysis']}\n\n---\n\n"
        for name, out in (job.get("specialist_outputs") or {}).items():
            content_md += f"## {name.capitalize()}\n\n{out}\n\n---\n\n"
        if job.get("marcus_review"):
            content_md += f"## Marcus — Final Review\n\n{job['marcus_review']}"
    elif _agent == "review":
        title, content_md = "Marcus — Final Review", job.get("marcus_review") or ""
    elif _agent == "analysis":
        title, content_md = "Marcus — Brief Analysis", job.get("marcus_analysis") or ""
    else:
        title       = f"{_agent.capitalize()} — Output"
        content_md  = (job.get("specialist_outputs") or {}).get(_agent, "")
    return HTMLResponse(_build_print_page_raw(title, brief, date, _md_to_html(content_md)))


@app.get("/health")
async def health():
    return {"status": "ok"}
