## Salary Predictor

This project provides:

- A FastAPI backend for salary prediction, options lookup, health checks, and retraining.
- A Streamlit dashboard for end-user prediction, insights, and historical data exploration.

## Prerequisites

- Python 3.12+
- uv package manager

## Install Dependencies

```bash
uv sync
```

## Run the FastAPI backend

From the repository root:

```bash
uv run uvicorn api.main:app --reload
```

## Run the Streamlit dashboard

In a second terminal, from the repository root:

```bash
uv run streamlit run dashboard/app.py
```

## Available API endpoints

- `GET /predict`
- `GET /history`
- `GET /options`
- `GET /health`
- `POST /train`
- `POST /analyze-text`
- `GET /analyses/history`

## Supabase setup

Create the following tables in Supabase SQL editor:

```sql
create table if not exists predictions (
	id bigint generated always as identity primary key,
	created_at timestamptz not null default now(),
	work_year int,
	experience_level text,
	employment_type text,
	job_title text,
	employee_residence text,
	remote_ratio int,
	company_location text,
	company_size text,
	predicted_salary_usd numeric
);

create table if not exists analyses (
	id bigint generated always as identity primary key,
	created_at timestamptz not null default now(),
	input_text text,
	task text,
	model text,
	narrative_title text,
	narrative text,
	story_points jsonb,
	theme_scores jsonb,
	key_phrases jsonb
);
```

If Row Level Security is enabled, add policies that allow your configured key to read and insert rows.

## Ollama setup for text analysis

1. Install Ollama locally.
2. Pull a model:

```bash
ollama pull <your-model-name>
```

Use the same model name in `.env` for `OLLAMA_MODEL`.

3. Start Ollama (if not already running) and verify:

```bash
ollama list
```

4. Use the dashboard "Text Analysis" tab or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/analyze-text \
	-H "Content-Type: application/json" \
	-d '{"text":"Summarize this role profile for a salary discussion","task":"summary"}'
```

## Environment variables

Configured in `.env`:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `API_BASE_URL`
- `MODEL_PATH`
- `ENCODERS_PATH`
- `TRAINING_DATA_PATH`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`

Use `.env.example` as the template for local setup.

## Streamlit Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Cloud, deploy using `dashboard/app.py` as the app entrypoint.
3. Add these secrets in Streamlit Cloud app settings:
	- `SUPABASE_URL`
	- `SUPABASE_KEY`
	- `API_BASE_URL` (use a hosted API URL if prediction/training/text features should work online)
	- `MODEL_PATH`, `ENCODERS_PATH`, `TRAINING_DATA_PATH` (optional for hosted dashboard visuals)
	- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (only if your API uses Ollama)
4. In the hosted dashboard, use the Records tab with source set to Supabase to load cloud data.

