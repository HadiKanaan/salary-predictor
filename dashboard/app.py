from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

try:
    from dashboard.utils.api_client import ApiClient
    from dashboard.utils.supabase_client import DashboardSupabaseClient
except ModuleNotFoundError:
    from utils.api_client import ApiClient
    from utils.supabase_client import DashboardSupabaseClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "cleaned" / "ds_salaries_clean.csv"
DEFAULT_ENCODERS_PATH = PROJECT_ROOT / "api" / "model" / "encoders.pkl"


def resolve_project_path(raw_path: str, fallback: Path) -> Path:
    """Resolves a configured path relative to the repository root."""
    if not raw_path:
        return fallback
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@st.cache_resource
def get_api_client(base_url: str) -> ApiClient:
    """Creates a cached API client instance."""
    return ApiClient(base_url=base_url)


@st.cache_resource
def get_supabase_client(url: str, key: str) -> DashboardSupabaseClient:
    """Creates a cached Supabase client wrapper for dashboard reads/writes."""
    return DashboardSupabaseClient(url=url, key=key)


@st.cache_data
def decode_dataframe(csv_path: str, encoders_path: str) -> pd.DataFrame:
    """Loads and decodes categorical columns for analytics."""
    df = pd.read_csv(csv_path)

    if Path(encoders_path).exists():
        import pickle

        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.inverse_transform(df[col].astype(int))

    return df


def get_options(client: ApiClient, refresh: bool = False) -> dict[str, list[str]]:
    """Returns option values from cache or API."""
    if refresh or "options_cache" not in st.session_state:
        st.session_state.options_cache = client.options()
    return st.session_state.options_cache


def to_currency(value: float) -> str:
    """Formats numeric salary values as USD."""
    return f"${value:,.2f}"


def render_predict_tab(
    client: ApiClient,
    decoded_df: pd.DataFrame,
    supabase_client: DashboardSupabaseClient | None,
) -> None:
    """Renders the prediction workflow tab."""
    st.subheader("Predict Salary")
    st.caption("Fill in role details and get a salary prediction in USD.")

    try:
        options = get_options(client)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            work_year = st.number_input(
                "Work Year", min_value=2020, max_value=2035, value=2023, step=1, key="predict_work_year"
            )
            experience_level = st.selectbox(
                "Experience Level", options.get("experience_level", []), key="predict_experience"
            )
            employment_type = st.selectbox(
                "Employment Type", options.get("employment_type", []), key="predict_employment"
            )
        with c2:
            job_title = st.selectbox("Job Title", options.get("job_title", []), index=0, key="predict_job")
            employee_residence = st.selectbox(
                "Employee Residence", options.get("employee_residence", []), key="predict_residence"
            )
            remote_ratio = st.selectbox("Remote Ratio", [0, 50, 100], index=2, key="predict_remote")
        with c3:
            company_location = st.selectbox(
                "Company Location", options.get("company_location", []), key="predict_company_location"
            )
            company_size = st.selectbox("Company Size", options.get("company_size", []), key="predict_size")

        save_prediction = st.checkbox(
            "Also save this prediction to Supabase from dashboard",
            value=False,
            disabled=supabase_client is None,
            key="predict_save_supabase",
        )

        submitted = st.form_submit_button("Predict Salary", type="primary")

    if not submitted:
        return

    payload = {
        "work_year": int(work_year),
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "employee_residence": employee_residence,
        "remote_ratio": int(remote_ratio),
        "company_location": company_location,
        "company_size": company_size,
    }

    try:
        with st.spinner("Running prediction..."):
            result = client.predict(payload)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    predicted = float(result["predicted_salary_usd"])
    p50 = float(decoded_df["salary_in_usd"].median())
    p75 = float(decoded_df["salary_in_usd"].quantile(0.75))

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Salary", to_currency(predicted))
    k2.metric("Dataset Median", to_currency(p50))
    k3.metric("Dataset 75th Percentile", to_currency(p75))

    if predicted >= p75:
        st.success("This prediction is in the upper salary range of the dataset.")
    elif predicted >= p50:
        st.info("This prediction is above the dataset median.")
    else:
        st.warning("This prediction is below the dataset median.")

    st.json(result)

    if save_prediction and supabase_client is not None:
        try:
            supabase_client.insert_prediction(
                {
                    "work_year": int(work_year),
                    "experience_level": experience_level,
                    "employment_type": employment_type,
                    "job_title": job_title,
                    "employee_residence": employee_residence,
                    "remote_ratio": int(remote_ratio),
                    "company_location": company_location,
                    "company_size": company_size,
                    "predicted_salary_usd": predicted,
                }
            )
            st.success("Prediction saved to Supabase.")
        except Exception as exc:
            st.warning(f"Prediction generated, but Supabase save failed: {exc}")


def render_insights_tab(decoded_df: pd.DataFrame) -> None:
    """Renders descriptive analytics and insight charts."""
    st.subheader("Data Insights")
    st.caption("Explore salary trends by year, role, and geography.")

    c1, c2, c3, c4 = st.columns(4)
    years = sorted(decoded_df["work_year"].dropna().unique().tolist())
    exp_levels = sorted(decoded_df["experience_level"].dropna().unique().tolist())
    countries = sorted(decoded_df["company_location"].dropna().unique().tolist())
    remote_vals = sorted(decoded_df["remote_ratio"].dropna().unique().tolist())

    with c1:
        years_sel = st.multiselect("Year", years, default=years, key="insights_year")
    with c2:
        exp_sel = st.multiselect("Experience", exp_levels, default=exp_levels, key="insights_experience")
    with c3:
        country_sel = st.multiselect(
            "Company Location", countries, default=countries, key="insights_company_location"
        )
    with c4:
        remote_sel = st.multiselect("Remote Ratio", remote_vals, default=remote_vals, key="insights_remote")

    filtered = decoded_df[
        decoded_df["work_year"].isin(years_sel)
        & decoded_df["experience_level"].isin(exp_sel)
        & decoded_df["company_location"].isin(country_sel)
        & decoded_df["remote_ratio"].isin(remote_sel)
    ]

    if filtered.empty:
        st.info("No rows match current filters.")
        return

    chart1, chart2 = st.columns(2)

    with chart1:
        salary_hist = px.histogram(
            filtered,
            x="salary_in_usd",
            nbins=40,
            title="Salary Distribution",
            labels={"salary_in_usd": "Salary (USD)"},
        )
        st.plotly_chart(salary_hist, width="stretch")

    with chart2:
        yearly = filtered.groupby("work_year", as_index=False)["salary_in_usd"].median()
        year_line = px.line(
            yearly,
            x="work_year",
            y="salary_in_usd",
            markers=True,
            title="Median Salary Trend by Year",
            labels={"salary_in_usd": "Median Salary (USD)", "work_year": "Work Year"},
        )
        st.plotly_chart(year_line, width="stretch")

    chart3, chart4 = st.columns(2)

    with chart3:
        top_jobs = (
            filtered.groupby("job_title", as_index=False)["salary_in_usd"]
            .mean()
            .sort_values("salary_in_usd", ascending=False)
            .head(12)
        )
        top_jobs_bar = px.bar(
            top_jobs,
            x="salary_in_usd",
            y="job_title",
            orientation="h",
            title="Top Paying Job Titles (Average Salary)",
            labels={"salary_in_usd": "Average Salary (USD)", "job_title": "Job Title"},
        )
        top_jobs_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(top_jobs_bar, width="stretch")

    with chart4:
        country_comp = (
            filtered.groupby(["employee_residence", "company_location"], as_index=False)["salary_in_usd"]
            .median()
            .sort_values("salary_in_usd", ascending=False)
            .head(20)
        )
        country_scatter = px.scatter(
            country_comp,
            x="employee_residence",
            y="company_location",
            size="salary_in_usd",
            color="salary_in_usd",
            title="Residence vs Company Location (Median Salary)",
            labels={"salary_in_usd": "Median Salary (USD)"},
        )
        st.plotly_chart(country_scatter, width="stretch")


def render_records_tab(decoded_df: pd.DataFrame, supabase_client: DashboardSupabaseClient | None) -> None:
    """Renders an interactive historical records explorer."""
    st.subheader("Historical Records")
    st.caption("Search, filter, sort, and export older entries from the CSV.")

    data_source = "Local CSV"
    if supabase_client is not None:
        data_source = st.radio(
            "Records Source",
            ["Local CSV", "Supabase"],
            horizontal=True,
            key="records_source",
        )

    if data_source == "Supabase":
        try:
            limit = st.slider("Rows to load", min_value=10, max_value=1000, value=100, step=10, key="records_limit")
            records = supabase_client.fetch_predictions(limit=limit) if supabase_client else []
            supabase_df = pd.DataFrame(records)
        except Exception as exc:
            st.error(f"Failed to load Supabase predictions: {exc}")
            return

        if supabase_df.empty:
            st.info("No prediction rows found in Supabase table 'predictions'.")
            return

        st.write(f"Showing {len(supabase_df):,} Supabase rows")
        st.dataframe(supabase_df, width="stretch", hide_index=True)
        csv_bytes = supabase_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Supabase CSV",
            data=csv_bytes,
            file_name="supabase_prediction_records.csv",
            mime="text/csv",
            type="secondary",
        )
        return

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        years = sorted(decoded_df["work_year"].dropna().unique().tolist())
        jobs = sorted(decoded_df["job_title"].dropna().unique().tolist())
        exp_levels = sorted(decoded_df["experience_level"].dropna().unique().tolist())
        locations = sorted(decoded_df["company_location"].dropna().unique().tolist())
        sizes = sorted(decoded_df["company_size"].dropna().unique().tolist())
        remotes = sorted(decoded_df["remote_ratio"].dropna().unique().tolist())

        with col1:
            f_years = st.multiselect("Year", years, default=years, key="records_year")
            f_jobs = st.multiselect("Job Title", jobs, default=[], key="records_job")
        with col2:
            f_exp = st.multiselect("Experience", exp_levels, default=exp_levels, key="records_experience")
            f_loc = st.multiselect(
                "Company Location", locations, default=locations, key="records_company_location"
            )
        with col3:
            f_size = st.multiselect("Company Size", sizes, default=sizes, key="records_size")
            f_remote = st.multiselect("Remote Ratio", remotes, default=remotes, key="records_remote")

        search = st.text_input("Search (job title, location, residence)", value="", key="records_search").strip().lower()

    filtered = decoded_df.copy()
    filtered = filtered[filtered["work_year"].isin(f_years)]
    filtered = filtered[filtered["experience_level"].isin(f_exp)]
    filtered = filtered[filtered["company_location"].isin(f_loc)]
    filtered = filtered[filtered["company_size"].isin(f_size)]
    filtered = filtered[filtered["remote_ratio"].isin(f_remote)]

    if f_jobs:
        filtered = filtered[filtered["job_title"].isin(f_jobs)]

    if search:
        search_cols = ["job_title", "company_location", "employee_residence"]
        mask = False
        for col in search_cols:
            mask = mask | filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask]

    st.write(f"Showing {len(filtered):,} of {len(decoded_df):,} rows")

    sort_col = st.selectbox("Sort By", filtered.columns.tolist(), index=0, key="records_sort_col")
    sort_asc = st.toggle("Ascending", value=True, key="records_sort_asc")
    filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    st.dataframe(filtered, width="stretch", hide_index=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered CSV",
        data=csv_bytes,
        file_name="filtered_salary_records.csv",
        mime="text/csv",
        type="secondary",
    )


def render_model_tab(client: ApiClient) -> None:
    """Renders API status and retraining controls."""
    st.subheader("Model Status")
    st.caption("Check API health and retrain the model from the dashboard.")

    left, right = st.columns(2)

    with left:
        if st.button("Check API Health", key="model_health_btn"):
            try:
                health = client.health()
                st.success(f"API status: {health.get('status', 'unknown')}")
            except RuntimeError as exc:
                st.error(str(exc))

    with right:
        if st.button("Retrain Model", type="primary", key="model_train_btn"):
            try:
                with st.spinner("Training model..."):
                    metrics = client.train()
                st.success("Model retrained successfully.")
                get_options(client, refresh=True)
                st.session_state.last_train = metrics
            except RuntimeError as exc:
                st.error(str(exc))

    if "last_train" in st.session_state:
        metrics = st.session_state.last_train
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", to_currency(float(metrics["mae"])))
        m2.metric("R2", f"{float(metrics['r2']):.4f}")
        m3.metric("Train Samples", f"{int(metrics['training_samples']):,}")
        m4.metric("Test Samples", f"{int(metrics['test_samples']):,}")
        st.caption(f"Model saved to: {metrics['model_path']}")


def render_text_tab(client: ApiClient, supabase_client: DashboardSupabaseClient | None) -> None:
    """Renders an Ollama-backed text analysis workflow."""
    st.subheader("Text Analysis")
    st.caption("Generate narrative insights and a score chart from your local Ollama model.")

    task = st.selectbox(
        "Analysis Task",
        ["general", "summary", "sentiment", "keywords", "risk-review"],
        index=0,
        key="text_task",
    )
    text = st.text_area(
        "Input Text",
        value="Paste job description, resume notes, or salary negotiation text.",
        height=200,
        key="text_input",
    )
    save_analysis = st.checkbox(
        "Also save this analysis to Supabase from dashboard",
        value=False,
        disabled=supabase_client is None,
        key="text_save_supabase",
    )

    if st.button("Analyze Text", type="primary", key="text_analyze_btn"):
        if not text.strip():
            st.warning("Please provide text to analyze.")
            return

        try:
            with st.spinner("Analyzing with Ollama..."):
                result = client.analyze_text(text=text.strip(), task=task)
        except RuntimeError as exc:
            st.error(str(exc))
            return

        st.success(f"Analysis completed using model: {result.get('model', 'unknown')}")
        st.markdown(f"### {result.get('narrative_title', 'Narrative Insight')}")
        st.write(result.get("narrative", result.get("analysis", "No analysis returned.")))

        story_points = result.get("story_points", [])
        if story_points:
            st.markdown("**Story Points**")
            for point in story_points:
                st.write(f"- {point}")

        phrases = result.get("key_phrases", [])
        if phrases:
            st.markdown("**Key Phrases**")
            st.write(", ".join(phrases))

        theme_scores = result.get("theme_scores", [])
        if theme_scores:
            chart_df = pd.DataFrame(theme_scores)
            chart_df = chart_df.dropna(subset=["theme", "score"]).copy()
            if not chart_df.empty:
                chart_df["score"] = pd.to_numeric(chart_df["score"], errors="coerce")
                chart_df = chart_df.dropna(subset=["score"]).sort_values("score", ascending=False)

            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="score",
                    y="theme",
                    orientation="h",
                    title="LLM-Generated Theme Scores",
                    labels={"score": "Score (0-100)", "theme": "Theme"},
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, width="stretch")

        if save_analysis and supabase_client is not None:
            try:
                supabase_client.insert_analysis(
                    {
                        "input_text": text.strip(),
                        "task": result.get("task"),
                        "model": result.get("model"),
                        "narrative_title": result.get("narrative_title"),
                        "narrative": result.get("narrative"),
                        "story_points": result.get("story_points"),
                        "theme_scores": result.get("theme_scores"),
                        "key_phrases": result.get("key_phrases"),
                    }
                )
                st.success("Analysis saved to Supabase.")
            except Exception as exc:
                st.warning(f"Analysis completed, but Supabase save failed: {exc}")


def main() -> None:
    """Runs the Streamlit dashboard application."""
    load_dotenv(override=True)

    st.set_page_config(
        page_title="Salary Predictor Dashboard",
        page_icon="USD",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    csv_path = resolve_project_path(os.getenv("TRAINING_DATA_PATH", ""), DEFAULT_CSV_PATH)
    encoders_path = resolve_project_path(os.getenv("ENCODERS_PATH", ""), DEFAULT_ENCODERS_PATH)

    st.title("Salary Predictor")
    st.caption("Predict salaries, inspect historical entries, and explore compensation insights.")

    with st.sidebar:
        st.header("Configuration")
        st.write(f"API Base URL: {api_base_url}")
        st.write(f"Supabase Configured: {'Yes' if supabase_url and supabase_key else 'No'}")
        st.write(f"Data Source: {csv_path}")
        st.write(f"Encoders: {encoders_path}")

    client = get_api_client(api_base_url)
    supabase_client: DashboardSupabaseClient | None = None

    if supabase_url and supabase_key:
        try:
            supabase_client = get_supabase_client(supabase_url, supabase_key)
        except Exception as exc:
            st.warning(f"Supabase is configured but could not initialize: {exc}")

    try:
        decoded_df = decode_dataframe(str(csv_path), str(encoders_path))
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    tab_predict, tab_insights, tab_records, tab_model, tab_text = st.tabs(
        ["Predict", "Insights", "Records", "Model", "Text Analysis"]
    )

    with tab_predict:
        render_predict_tab(client, decoded_df, supabase_client)
    with tab_insights:
        render_insights_tab(decoded_df)
    with tab_records:
        render_records_tab(decoded_df, supabase_client)
    with tab_model:
        render_model_tab(client)
    with tab_text:
        render_text_tab(client, supabase_client)


if __name__ == "__main__":
    main()
