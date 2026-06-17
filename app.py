"""
AI Agent for End-to-End Development
Streamlit UI — LangGraph + Llama 3
"""

import os, sys, json, zipfile, io, time
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="AI Dev Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main, .stApp { background-color: #0f1117; }
.hero { font-size:2.6rem; font-weight:800;
  background:linear-gradient(135deg,#6c63ff,#43b89c);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub  { color:#888; font-size:1rem; margin-bottom:1.5rem; }
.card { background:#1e2130; border:1px solid #2d3148; border-radius:12px; padding:20px 24px; margin:10px 0; }
.step-row { display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid #2d3148; }
.step-icon { font-size:1.3rem; width:36px; text-align:center; }
.step-label { color:#e0e0e0; font-weight:500; }
.step-status { margin-left:auto; font-size:.85rem; }
.file-chip { display:inline-block; background:#0f1117; border:1px solid #2d3148;
  border-radius:6px; padding:4px 10px; margin:3px; font-size:.8rem; color:#aaa; }
div[data-testid="stSidebar"] { background:#151825; }
.stButton>button { background:linear-gradient(135deg,#6c63ff,#43b89c);
  color:#fff; border:none; border-radius:8px; padding:12px 28px;
  font-weight:700; font-size:1rem; width:100%; }
.stTextArea textarea { background:#1e2130 !important; color:#e0e0e0 !important;
  border:1px solid #2d3148 !important; border-radius:8px !important; }
.stTextInput input { background:#1e2130 !important; color:#e0e0e0 !important; border:1px solid #2d3148 !important; }
.example-pill { background:#1e2130; border:1px solid #6c63ff44; border-radius:20px;
  padding:6px 14px; margin:4px; display:inline-block; color:#aaa; font-size:.85rem; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Dev Agent")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("---")
    st.markdown("### 🔗 Pipeline")
    pipeline_steps = [
        ("📋", "Requirements Analysis"),
        ("🗄️", "Database Design"),
        ("⚙️", "Backend Generation"),
        ("🎨", "Frontend Generation"),
        ("🚀", "Deployment Config"),
        ("💾", "Write Files"),
        ("📝", "Summary"),
    ]
    for icon, label in pipeline_steps:
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown("**Output:** FastAPI + SQLite + HTML")
    st.markdown("**Deploy:** Render.com (free)")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">🤖 AI Dev Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Describe any app → get production-ready code in seconds · LangGraph + Llama 3</div>', unsafe_allow_html=True)

# ── Examples ──────────────────────────────────────────────────────────────────
examples = [
    "A task management app where users can create projects, add tasks with due dates, assign priorities, and mark them complete",
    "An inventory management system for a small store to track products, stock levels, suppliers, and generate low-stock alerts",
    "A blog platform where authors can write posts with tags and categories, readers can comment, and admins can moderate",
    "A simple expense tracker where users log daily expenses by category, set monthly budgets, and view spending summaries",
    "A restaurant menu management system with dishes, categories, pricing, availability toggles, and order tracking",
]

st.markdown("**💡 Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(examples[:3]):
    with cols[i]:
        short = ex[:55] + "..."
        if st.button(short, key=f"ex{i}"):
            st.session_state["requirement_input"] = ex

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
requirement = st.text_area(
    "📝 Describe your application",
    value=st.session_state.get("requirement_input", ""),
    height=120,
    placeholder="e.g. A task management app where users can create projects, add tasks with due dates, assign priorities, and mark them complete...",
)

run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("🚀 Generate Full App", use_container_width=True)

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run:
    if not requirement.strip():
        st.error("Please describe your application first.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Add your Groq API key in the sidebar.")
        st.stop()

    st.markdown("---")
    st.markdown("### ⚡ Generating Your App...")

    # Live step tracker
    step_labels = ["📋 Requirements", "🗄️ Database", "⚙️ Backend", "🎨 Frontend", "🚀 Deployment", "💾 Files", "📝 Summary"]
    step_placeholder = st.empty()

    def render_steps(current_idx):
        html = '<div class="card">'
        for i, label in enumerate(step_labels):
            if i < current_idx:
                status = '<span style="color:#43b89c">✅ Done</span>'
            elif i == current_idx:
                status = '<span style="color:#6c63ff">🔄 Running...</span>'
            else:
                status = '<span style="color:#555">⏳ Waiting</span>'
            html += f'<div class="step-row"><span class="step-icon">{label.split()[0]}</span><span class="step-label">{" ".join(label.split()[1:])}</span><span class="step-status">{status}</span></div>'
        html += '</div>'
        step_placeholder.markdown(html, unsafe_allow_html=True)

    render_steps(0)

    try:
        from src.agents.dev_agent import run_dev_agent

        # Patch to show progress — we hook into each node via a wrapper
        import src.agents.dev_agent as agent_module

        _original_nodes = {}
        node_order = ["requirements_node","database_node","backend_node","frontend_node","deployment_node","write_files_node","summary_node"]

        def make_wrapper(fn, idx):
            def wrapper(state):
                render_steps(idx)
                return fn(state)
            return wrapper

        for i, name in enumerate(node_order):
            orig = getattr(agent_module, name)
            _original_nodes[name] = orig
            setattr(agent_module, name, make_wrapper(orig, i))

        state = run_dev_agent(requirement)

        # Restore originals
        for name, orig in _original_nodes.items():
            setattr(agent_module, name, orig)

        render_steps(len(step_labels))  # all done

        st.success("🎉 App generated successfully!")

        # ── Results ───────────────────────────────────────────────────────────
        req = state.get("requirements", {})
        schema = state.get("database_schema", {})
        output_path = state.get("output_path", "")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Overview", "🗄️ Database", "⚙️ Backend", "🎨 Frontend", "🚀 Deploy", "📦 Download"
        ])

        with tab1:
            st.markdown(f"## {req.get('app_name','App')}")
            st.markdown(f"*{req.get('description','')}*")
            c1, c2, c3 = st.columns(3)
            c1.metric("App Type", req.get("app_type","web_app").replace("_"," ").title())
            c2.metric("API Endpoints", len(req.get("api_endpoints",[])))
            c3.metric("DB Tables", len(schema.get("tables",[])))

            st.markdown("### ✨ Features")
            for f in req.get("core_features", []):
                st.markdown(f"- {f}")

            st.markdown("### 🔌 API Endpoints")
            for e in req.get("api_endpoints", []):
                badge_color = {"GET":"#43b89c","POST":"#6c63ff","PUT":"#f9a825","DELETE":"#ef5350"}.get(e["method"],"#888")
                st.markdown(f'<span style="background:{badge_color};padding:2px 8px;border-radius:4px;font-size:.8rem;color:#fff">{e["method"]}</span> `{e["path"]}` — {e["description"]}', unsafe_allow_html=True)

        with tab2:
            st.markdown("### 🗄️ Database Schema")
            for table in schema.get("tables", []):
                with st.expander(f"📋 {table['name']} — {table.get('description','')}"):
                    cols_data = [{"Column": c["name"], "Type": c["type"], "Constraints": c.get("constraints","")} for c in table.get("columns",[])]
                    st.table(cols_data)
            if schema.get("relationships"):
                st.markdown("### 🔗 Relationships")
                for r in schema["relationships"]:
                    st.markdown(f"- `{r['from_table']}` → `{r['to_table']}` ({r['type']})")

        with tab3:
            st.markdown("### ⚙️ Backend Code")
            backend = state.get("backend_code", {})
            for fname, code in backend.items():
                with st.expander(f"📄 {fname}"):
                    lang = "python" if fname.endswith(".py") else "text"
                    st.code(code, language=lang)

        with tab4:
            st.markdown("### 🎨 Frontend Preview")
            frontend = state.get("frontend_code", {})
            if "index.html" in frontend:
                st.components.v1.html(frontend["index.html"], height=600, scrolling=True)
                with st.expander("📄 View HTML source"):
                    st.code(frontend["index.html"], language="html")

        with tab5:
            st.markdown("### 🚀 Deployment")
            deploy = state.get("deployment_config", {})
            for fname, content in deploy.items():
                with st.expander(f"📄 {fname}"):
                    lang = "dockerfile" if "Docker" in fname else "yaml" if fname.endswith(".yml") or fname.endswith(".yaml") else "markdown" if fname.endswith(".md") else "text"
                    st.code(content, language=lang)

            st.markdown("### 🌐 Deploy to Render.com (Free)")
            st.markdown("""
1. Download your project (next tab)
2. Push to GitHub  
3. Go to [render.com](https://render.com) → **New Web Service**
4. Connect your GitHub repo
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Click **Deploy** 🚀
""")

        with tab6:
            st.markdown("### 📦 Download Your Project")

            # Create zip in memory
            project_name = state.get("project_name", "my-app")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                backend = state.get("backend_code", {})
                frontend = state.get("frontend_code", {})
                deploy = state.get("deployment_config", {})
                summary = state.get("project_summary", "")

                for fname, content in backend.items():
                    zf.writestr(f"{project_name}/{fname}", content)
                for fname, content in frontend.items():
                    zf.writestr(f"{project_name}/static/{fname}", content)
                for fname, content in deploy.items():
                    zf.writestr(f"{project_name}/{fname}", content)
                if summary:
                    zf.writestr(f"{project_name}/SUMMARY.md", summary)

            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download Full Project (ZIP)",
                data=zip_buffer,
                file_name=f"{project_name}.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.markdown("### 📁 Files Included")
            file_list = list(backend.keys()) + [f"static/{k}" for k in frontend.keys()] + list(deploy.keys()) + ["SUMMARY.md"]
            for f in file_list:
                st.markdown(f'<span class="file-chip">📄 {f}</span>', unsafe_allow_html=True)

            st.markdown("### ⚡ Quick Start")
            st.code(f"""unzip {project_name}.zip
cd {project_name}
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000""", language="bash")

        if state.get("errors"):
            with st.expander("⚠️ Warnings"):
                for e in state["errors"]:
                    st.warning(e)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    # Landing
    st.markdown("---")
    st.markdown("### 🏗️ What gets generated")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
<b>⚙️ Backend (FastAPI)</b><br><br>
<span style="color:#888">• All CRUD endpoints<br>• SQLAlchemy ORM models<br>• Pydantic schemas<br>• CORS + error handling<br>• Auto Swagger docs</span>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
<b>🗄️ Database (SQLite)</b><br><br>
<span style="color:#888">• Auto-designed schema<br>• Normalized tables<br>• Relationships<br>• Created on startup<br>• No setup needed</span>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card">
<b>🎨 Frontend (HTML/JS)</b><br><br>
<span style="color:#888">• Modern dark UI<br>• Full CRUD interface<br>• Live API calls<br>• Toast notifications<br>• Mobile responsive</span>
</div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("""<div class="card">
<b>🐳 Docker</b><br><br>
<span style="color:#888">• Dockerfile<br>• docker-compose.yml<br>• Production ready</span>
</div>""", unsafe_allow_html=True)
    with c5:
        st.markdown("""<div class="card">
<b>☁️ Deploy Config</b><br><br>
<span style="color:#888">• Render.com YAML<br>• One-click deploy<br>• Free tier support</span>
</div>""", unsafe_allow_html=True)
    with c6:
        st.markdown("""<div class="card">
<b>📦 Download ZIP</b><br><br>
<span style="color:#888">• All files bundled<br>• Ready to run<br>• Push to GitHub</span>
</div>""", unsafe_allow_html=True)
