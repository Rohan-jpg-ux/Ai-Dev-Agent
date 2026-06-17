# 🤖 AI Dev Agent

> Describe any app in plain English → get production-ready FastAPI + SQLite + HTML code instantly.

## 🎯 What It Does

Type a requirement like:
> "A task management app where users can create projects, add tasks with due dates and priorities"

And get back a **complete, runnable application** with:
- `main.py` — FastAPI app with all CRUD endpoints
- `models.py` — SQLAlchemy models + Pydantic schemas
- `static/index.html` — Modern dark-theme frontend UI
- `Dockerfile` — Production container
- `docker-compose.yml` — Multi-container setup
- `render.yaml` — One-click Render.com deploy

## 🏗️ Tech Stack
- **LangGraph** — 7-node agent pipeline
- **Llama 3 70B** via Groq — Code generation
- **FastAPI** — Generated backend
- **SQLite** — Generated database
- **Streamlit** — UI

## 🚀 Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy
Streamlit Cloud → add `GROQ_API_KEY` secret → Deploy
