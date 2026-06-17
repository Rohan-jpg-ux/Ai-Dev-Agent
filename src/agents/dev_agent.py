"""
AI Agent for End-to-End Development
LangGraph pipeline: Requirements → Frontend → Backend → Database → Deploy
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

class DevState(TypedDict):
    messages: Annotated[list, add_messages]
    user_requirement: str
    project_name: str
    tech_stack: dict
    requirements: Optional[dict]       # Parsed & structured requirements
    database_schema: Optional[dict]    # Tables, fields, relationships
    backend_code: Optional[dict]       # FastAPI routes, models, main.py
    frontend_code: Optional[dict]      # HTML/CSS/JS files
    deployment_config: Optional[dict]  # Dockerfile, docker-compose, etc.
    project_summary: Optional[str]     # Final summary report
    output_path: Optional[str]         # Where files were saved
    current_step: str
    errors: List[str]


# ─── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.1):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=4096,
        api_key=api_key,
    )


def llm_json(prompt: str, system: str) -> dict:
    """Call LLM and parse JSON response"""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system + "\n\nRESPOND ONLY WITH VALID JSON. No markdown, no backticks, no explanation."),
        HumanMessage(content=prompt)
    ])
    text = response.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def llm_code(prompt: str, system: str) -> str:
    """Call LLM for code generation"""
    llm = get_llm(temperature=0.05)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt)
    ])
    return response.content.strip()


# ─── Node 1: Requirements Analysis ───────────────────────────────────────────

def requirements_node(state: DevState) -> DevState:
    logger.info("📋 Analyzing requirements...")
    try:
        result = llm_json(
            prompt=f"""
Analyze this application requirement and extract structured details:

REQUIREMENT: {state['user_requirement']}

Return JSON with exactly this structure:
{{
  "app_name": "short app name",
  "app_type": "web_app|api|dashboard|ecommerce|blog|todo|other",
  "description": "one sentence description",
  "core_features": ["feature1", "feature2", "feature3", "feature4", "feature5"],
  "user_roles": ["role1", "role2"],
  "data_entities": ["entity1", "entity2", "entity3"],
  "api_endpoints": [
    {{"method": "GET", "path": "/api/items", "description": "list all items"}},
    {{"method": "POST", "path": "/api/items", "description": "create item"}},
    {{"method": "GET", "path": "/api/items/{{id}}", "description": "get item"}},
    {{"method": "PUT", "path": "/api/items/{{id}}", "description": "update item"}},
    {{"method": "DELETE", "path": "/api/items/{{id}}", "description": "delete item"}}
  ],
  "tech_stack": {{
    "backend": "FastAPI",
    "database": "SQLite",
    "frontend": "HTML/CSS/JavaScript",
    "orm": "SQLAlchemy"
  }},
  "color_theme": {{
    "primary": "#6c63ff",
    "secondary": "#43b89c",
    "background": "#0f1117",
    "surface": "#1e2130"
  }}
}}
""",
            system="You are a senior software architect. Extract structured requirements from natural language descriptions."
        )
        state["requirements"] = result
        state["project_name"] = result.get("app_name", "my-app").lower().replace(" ", "-")
        state["tech_stack"] = result.get("tech_stack", {})
        state["current_step"] = "requirements_done"
        state["messages"].append(AIMessage(content=f"✅ Requirements analyzed: {result['app_name']} — {result['description']}"))
        logger.info(f"Requirements: {result['app_name']}")
    except Exception as e:
        state["errors"].append(f"Requirements error: {e}")
        # Fallback
        state["requirements"] = {
            "app_name": "MyApp", "app_type": "web_app",
            "description": state["user_requirement"][:100],
            "core_features": ["Create", "Read", "Update", "Delete"],
            "user_roles": ["user", "admin"],
            "data_entities": ["Item"],
            "api_endpoints": [
                {"method": "GET", "path": "/api/items", "description": "list items"},
                {"method": "POST", "path": "/api/items", "description": "create item"},
            ],
            "tech_stack": {"backend": "FastAPI", "database": "SQLite", "frontend": "HTML/CSS/JavaScript", "orm": "SQLAlchemy"},
            "color_theme": {"primary": "#6c63ff", "secondary": "#43b89c", "background": "#0f1117", "surface": "#1e2130"}
        }
        state["project_name"] = "my-app"
        state["current_step"] = "requirements_done"
    return state


# ─── Node 2: Database Schema ──────────────────────────────────────────────────

def database_node(state: DevState) -> DevState:
    logger.info("🗄️ Designing database schema...")
    req = state["requirements"]
    try:
        schema = llm_json(
            prompt=f"""
Design a SQLite database schema for this application:

App: {req['app_name']}
Description: {req['description']}
Entities: {req['data_entities']}
Features: {req['core_features']}

Return JSON with this exact structure:
{{
  "tables": [
    {{
      "name": "table_name",
      "description": "what this table stores",
      "columns": [
        {{"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"}},
        {{"name": "column_name", "type": "TEXT|INTEGER|REAL|BOOLEAN", "constraints": "NOT NULL|UNIQUE|DEFAULT value"}},
        {{"name": "created_at", "type": "DATETIME", "constraints": "DEFAULT CURRENT_TIMESTAMP"}},
        {{"name": "updated_at", "type": "DATETIME", "constraints": "DEFAULT CURRENT_TIMESTAMP"}}
      ]
    }}
  ],
  "relationships": [
    {{"from_table": "table1", "to_table": "table2", "type": "one_to_many", "foreign_key": "table2.table1_id"}}
  ]
}}

Create 2-4 tables that make sense for the application. Always include id, created_at, updated_at columns.
""",
            system="You are a database architect. Design clean, normalized SQLite schemas."
        )
        state["database_schema"] = schema
        state["current_step"] = "database_done"
        table_names = [t["name"] for t in schema.get("tables", [])]
        state["messages"].append(AIMessage(content=f"✅ Database schema designed: {len(table_names)} tables — {', '.join(table_names)}"))
    except Exception as e:
        state["errors"].append(f"Database error: {e}")
        state["database_schema"] = {
            "tables": [{"name": "items", "description": "Main items", "columns": [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "title", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "description", "type": "TEXT", "constraints": ""},
                {"name": "created_at", "type": "DATETIME", "constraints": "DEFAULT CURRENT_TIMESTAMP"},
            ]}],
            "relationships": []
        }
        state["current_step"] = "database_done"
    return state


# ─── Node 3: Backend Code ─────────────────────────────────────────────────────

def backend_node(state: DevState) -> DevState:
    logger.info("⚙️ Generating backend code...")
    req = state["requirements"]
    schema = state["database_schema"]

    try:
        # Generate SQLAlchemy models
        models_code = llm_code(
            prompt=f"""
Generate complete SQLAlchemy models for this database schema:

App: {req['app_name']}
Schema: {json.dumps(schema, indent=2)}

Generate a complete models.py file with:
1. SQLAlchemy Base and engine setup for SQLite (database.py style inline)
2. All table models as SQLAlchemy classes
3. Pydantic schemas for request/response (BaseModel)
4. Include all columns from the schema

Use SQLAlchemy 2.0 style. File should be complete and runnable.
""",
            system="You are a senior Python developer. Generate clean, production-ready SQLAlchemy models and Pydantic schemas. Return ONLY the Python code, no explanation."
        )

        # Generate FastAPI routes
        routes_code = llm_code(
            prompt=f"""
Generate a complete FastAPI application for:

App: {req['app_name']}
Description: {req['description']}
API Endpoints needed: {json.dumps(req['api_endpoints'], indent=2)}
Database tables: {[t['name'] for t in schema['tables']]}
Features: {req['core_features']}

Generate main.py with:
1. FastAPI app with CORS middleware
2. SQLite database connection (file: app.db)
3. All CRUD endpoints from the spec
4. Proper error handling with HTTPException
5. Startup event to create tables
6. Health check endpoint GET /health
7. Static files serving for frontend

Make it complete and runnable with: uvicorn main:app --reload
""",
            system="You are a senior FastAPI developer. Generate complete, production-ready API code. Return ONLY the Python code."
        )

        # Generate requirements.txt
        requirements = """fastapi==0.111.0
uvicorn[standard]==0.30.0
sqlalchemy==2.0.30
pydantic==2.7.1
python-multipart==0.0.9
aiofiles==23.2.1
"""

        state["backend_code"] = {
            "main.py": routes_code,
            "models.py": models_code,
            "requirements.txt": requirements,
        }
        state["current_step"] = "backend_done"
        state["messages"].append(AIMessage(content=f"✅ Backend generated: FastAPI app with {len(req['api_endpoints'])} endpoints"))
    except Exception as e:
        state["errors"].append(f"Backend error: {e}")
        state["backend_code"] = _fallback_backend(req, schema)
        state["current_step"] = "backend_done"
    return state


def _fallback_backend(req, schema) -> dict:
    table = schema["tables"][0]["name"] if schema["tables"] else "items"
    return {
        "main.py": f'''from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import os

app = FastAPI(title="{req["app_name"]}", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(DATABASE_URL, connect_args={{"check_same_thread": False}})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class {table.capitalize().rstrip("s")}(Base):
    __tablename__ = "{table}"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ItemCreate(BaseModel):
    title: str
    description: Optional[str] = None

class ItemResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    created_at: datetime
    class Config: from_attributes = True

@app.on_event("startup")
def startup(): Base.metadata.create_all(bind=engine)

@app.get("/health")
def health(): return {{"status": "ok", "app": "{req["app_name"]}"}}

@app.get("/api/{table}", response_model=List[ItemResponse])
def list_items():
    db = SessionLocal()
    items = db.query({table.capitalize().rstrip("s")}).all()
    db.close()
    return items

@app.post("/api/{table}", response_model=ItemResponse)
def create_item(item: ItemCreate):
    db = SessionLocal()
    obj = {table.capitalize().rstrip("s")}(**item.dict())
    db.add(obj); db.commit(); db.refresh(obj); db.close()
    return obj

@app.delete("/api/{table}/{{item_id}}")
def delete_item(item_id: int):
    db = SessionLocal()
    obj = db.query({table.capitalize().rstrip("s")}).filter_by(id=item_id).first()
    if not obj: raise HTTPException(404, "Not found")
    db.delete(obj); db.commit(); db.close()
    return {{"deleted": True}}
''',
        "models.py": "# Models defined inline in main.py",
        "requirements.txt": "fastapi==0.111.0\nuvicorn[standard]==0.30.0\nsqlalchemy==2.0.30\npydantic==2.7.1\npython-multipart==0.0.9\naiofiles==23.2.1\n",
    }


# ─── Node 4: Frontend Code ────────────────────────────────────────────────────

def frontend_node(state: DevState) -> DevState:
    logger.info("🎨 Generating frontend code...")
    req = state["requirements"]
    theme = req.get("color_theme", {"primary": "#6c63ff", "secondary": "#43b89c", "background": "#0f1117", "surface": "#1e2130"})

    try:
        html_code = llm_code(
            prompt=f"""
Generate a complete, beautiful single-page frontend for:

App Name: {req['app_name']}
Description: {req['description']}
Features: {req['core_features']}
API Endpoints: {json.dumps(req['api_endpoints'], indent=2)}
Color Theme: primary={theme['primary']}, secondary={theme['secondary']}, background={theme['background']}

Generate a SINGLE complete index.html file that:
1. Has a modern dark-theme UI using the color theme above
2. Uses vanilla JavaScript (no frameworks needed)
3. Calls the FastAPI backend at relative /api/ paths
4. Has navigation/sidebar if needed
5. Shows all CRUD operations (create form, list view, delete buttons)
6. Has loading states and error messages
7. Is fully responsive (mobile-friendly)
8. Has smooth animations and hover effects
9. Includes Font Awesome icons via CDN
10. Shows success/error toasts for operations

Make it look professional and modern. All CSS inline in <style> tag, all JS in <script> tag.
Return ONLY the complete HTML file starting with <!DOCTYPE html>
""",
            system="You are a senior frontend developer. Generate beautiful, modern, fully functional HTML/CSS/JS. Return ONLY the HTML code."
        )

        state["frontend_code"] = {"index.html": html_code}
        state["current_step"] = "frontend_done"
        state["messages"].append(AIMessage(content="✅ Frontend generated: modern dark-theme single-page app"))
    except Exception as e:
        state["errors"].append(f"Frontend error: {e}")
        state["frontend_code"] = {"index.html": _fallback_frontend(req, theme)}
        state["current_step"] = "frontend_done"
    return state


def _fallback_frontend(req, theme) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{req['app_name']}</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background:{theme['background']}; color:#e0e0e0; }}
.header {{ background:{theme['surface']}; padding:20px 32px; border-bottom:1px solid #2d3148; display:flex; align-items:center; gap:12px; }}
.header h1 {{ font-size:1.6rem; background:linear-gradient(135deg,{theme['primary']},{theme['secondary']}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.container {{ max-width:900px; margin:32px auto; padding:0 20px; }}
.card {{ background:{theme['surface']}; border:1px solid #2d3148; border-radius:12px; padding:24px; margin-bottom:20px; }}
.form-group {{ margin-bottom:16px; }}
label {{ display:block; margin-bottom:6px; color:#aaa; font-size:.9rem; }}
input, textarea {{ width:100%; padding:10px 14px; background:#0f1117; border:1px solid #2d3148; border-radius:8px; color:#e0e0e0; font-size:.95rem; }}
input:focus, textarea:focus {{ outline:none; border-color:{theme['primary']}; }}
.btn {{ padding:10px 22px; border:none; border-radius:8px; cursor:pointer; font-weight:600; transition:.2s; }}
.btn-primary {{ background:linear-gradient(135deg,{theme['primary']},{theme['secondary']}); color:#fff; }}
.btn-danger {{ background:#ef5350; color:#fff; }}
.item-list {{ list-style:none; }}
.item {{ background:#0f1117; border:1px solid #2d3148; border-radius:8px; padding:16px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; }}
.toast {{ position:fixed; bottom:24px; right:24px; padding:12px 20px; border-radius:8px; color:#fff; font-weight:500; z-index:999; opacity:0; transition:.3s; }}
.toast.show {{ opacity:1; }}
.toast.success {{ background:#43b89c; }}
.toast.error {{ background:#ef5350; }}
</style>
</head>
<body>
<div class="header"><i class="fas fa-rocket" style="color:{theme['primary']};font-size:1.4rem"></i><h1>{req['app_name']}</h1></div>
<div class="container">
  <div class="card">
    <h2 style="margin-bottom:16px">Add New Item</h2>
    <div class="form-group"><label>Title</label><input id="title" placeholder="Enter title..."></div>
    <div class="form-group"><label>Description</label><textarea id="desc" rows="3" placeholder="Enter description..."></textarea></div>
    <button class="btn btn-primary" onclick="createItem()"><i class="fas fa-plus"></i> Create</button>
  </div>
  <div class="card">
    <h2 style="margin-bottom:16px">Items <span id="count" style="color:#888;font-size:.9rem"></span></h2>
    <ul class="item-list" id="list"></ul>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const API = '/api/items';
async function load() {{
  const r = await fetch(API);
  const items = await r.json();
  document.getElementById('count').textContent = `(${{items.length}})`;
  document.getElementById('list').innerHTML = items.map(i => `
    <li class="item">
      <div><strong>${{i.title}}</strong><br><span style="color:#888;font-size:.85rem">${{i.description||''}}</span></div>
      <button class="btn btn-danger" onclick="del(${{i.id}})"><i class="fas fa-trash"></i></button>
    </li>`).join('');
}}
async function createItem() {{
  const title = document.getElementById('title').value.trim();
  if (!title) return toast('Title required', 'error');
  await fetch(API, {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{title,description:document.getElementById('desc').value}})}});
  document.getElementById('title').value=''; document.getElementById('desc').value='';
  toast('Created!','success'); load();
}}
async function del(id) {{
  await fetch(`${{API}}/${{id}}`,{{method:'DELETE'}}); toast('Deleted','success'); load();
}}
function toast(msg,type) {{
  const t=document.getElementById('toast'); t.textContent=msg; t.className=`toast ${{type}} show`;
  setTimeout(()=>t.className='toast',2500);
}}
load();
</script>
</body></html>"""


# ─── Node 5: Deployment Config ────────────────────────────────────────────────

def deployment_node(state: DevState) -> DevState:
    logger.info("🚀 Generating deployment config...")
    req = state["requirements"]
    name = state["project_name"]

    dockerfile = f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    docker_compose = f"""version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./app.db:/app/app.db
    environment:
      - ENV=production
    restart: unless-stopped
"""

    render_yaml = f"""services:
  - type: web
    name: {name}
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ENV
        value: production
"""

    readme = f"""# {req['app_name']}

{req['description']}

## Features
{chr(10).join(f'- {f}' for f in req['core_features'])}

## Tech Stack
- **Backend:** FastAPI + SQLAlchemy
- **Database:** SQLite
- **Frontend:** HTML/CSS/JavaScript

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000

## Deploy to Render.com (Free)

1. Push to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API Endpoints
{chr(10).join(f'- `{e["method"]} {e["path"]}` — {e["description"]}' for e in req['api_endpoints'])}
- `GET /health` — Health check
- `GET /docs` — Interactive API docs (Swagger)
"""

    gitignore = """__pycache__/
*.pyc
.env
app.db
*.db
venv/
.venv/
"""

    state["deployment_config"] = {
        "Dockerfile": dockerfile,
        "docker-compose.yml": docker_compose,
        "render.yaml": render_yaml,
        "README.md": readme,
        ".gitignore": gitignore,
    }
    state["current_step"] = "deployment_done"
    state["messages"].append(AIMessage(content="✅ Deployment config generated: Dockerfile + docker-compose + Render.com config"))
    return state


# ─── Node 6: Write Files ──────────────────────────────────────────────────────

def write_files_node(state: DevState) -> DevState:
    logger.info("💾 Writing project files...")
    project_name = state["project_name"]
    output_dir = f"outputs/projects/{project_name}"
    os.makedirs(f"{output_dir}/static", exist_ok=True)

    files_written = []

    # Backend files
    for filename, content in state["backend_code"].items():
        path = f"{output_dir}/{filename}"
        with open(path, "w") as f:
            f.write(content)
        files_written.append(path)

    # Frontend files → static/
    for filename, content in state["frontend_code"].items():
        path = f"{output_dir}/static/{filename}"
        with open(path, "w") as f:
            f.write(content)
        files_written.append(path)

    # Deployment files
    for filename, content in state["deployment_config"].items():
        path = f"{output_dir}/{filename}"
        with open(path, "w") as f:
            f.write(content)
        files_written.append(path)

    state["output_path"] = output_dir
    state["current_step"] = "files_written"
    state["messages"].append(AIMessage(content=f"✅ {len(files_written)} files written to `{output_dir}`"))
    logger.info(f"Wrote {len(files_written)} files to {output_dir}")
    return state


# ─── Node 7: Summary ──────────────────────────────────────────────────────────

def summary_node(state: DevState) -> DevState:
    logger.info("📝 Generating project summary...")
    req = state["requirements"]
    schema = state["database_schema"]

    summary = f"""# 🚀 {req['app_name']} — Generated by AI Dev Agent

**{req['description']}**

---

## 📋 Requirements Analysis
- **App Type:** {req['app_type']}
- **Features:** {', '.join(req['core_features'])}
- **User Roles:** {', '.join(req['user_roles'])}

## 🗄️ Database Schema
{chr(10).join(f"- **{t['name']}** — {t['description']} ({len(t['columns'])} columns)" for t in schema['tables'])}

## ⚙️ API Endpoints
{chr(10).join(f"- `{e['method']} {e['path']}` — {e['description']}" for e in req['api_endpoints'])}
- `GET /health` — Health check
- `GET /docs` — Swagger UI

## 📁 Generated Files
```
{state['project_name']}/
├── main.py              ← FastAPI app (all routes)
├── models.py            ← SQLAlchemy models + Pydantic schemas
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Docker container config
├── docker-compose.yml   ← Multi-container setup
├── render.yaml          ← Render.com deployment
├── README.md            ← Setup instructions
└── static/
    └── index.html       ← Frontend UI
```

## 🚀 Run in 3 Commands
```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

## ☁️ Deploy Free on Render.com
1. Push to GitHub
2. render.com → New Web Service → Connect repo
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---
*Generated by AI Dev Agent — LangGraph + Llama 3*
"""
    state["project_summary"] = summary
    state["current_step"] = "complete"
    state["messages"].append(AIMessage(content="✅ Project complete! All files generated and ready to run."))

    # Save summary
    with open(f"{state['output_path']}/SUMMARY.md", "w") as f:
        f.write(summary)

    return state


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_dev_graph():
    graph = StateGraph(DevState)
    graph.add_node("requirements", requirements_node)
    graph.add_node("database", database_node)
    graph.add_node("backend", backend_node)
    graph.add_node("frontend", frontend_node)
    graph.add_node("deployment", deployment_node)
    graph.add_node("write_files", write_files_node)
    graph.add_node("summary", summary_node)

    graph.set_entry_point("requirements")
    graph.add_edge("requirements", "database")
    graph.add_edge("database", "backend")
    graph.add_edge("backend", "frontend")
    graph.add_edge("frontend", "deployment")
    graph.add_edge("deployment", "write_files")
    graph.add_edge("write_files", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


def run_dev_agent(requirement: str) -> DevState:
    graph = build_dev_graph()
    initial_state: DevState = {
        "messages": [HumanMessage(content=requirement)],
        "user_requirement": requirement,
        "project_name": "my-app",
        "tech_stack": {},
        "requirements": None,
        "database_schema": None,
        "backend_code": None,
        "frontend_code": None,
        "deployment_config": None,
        "project_summary": None,
        "output_path": None,
        "current_step": "start",
        "errors": [],
    }
    return graph.invoke(initial_state)
