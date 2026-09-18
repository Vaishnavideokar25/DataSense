import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent, generate_report

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSense AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 5%, rgba(124, 92, 255, 0.12), transparent 25%),
            radial-gradient(circle at 10% 30%, rgba(0, 198, 255, 0.08), transparent 25%),
            #080a12;
        color: #f4f5f8;
    }

    [data-testid="stSidebar"] {
        background: #0c0f19;
        border-right: 1px solid rgba(255,255,255,.07);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -.04em;
        margin-bottom: .2rem;
    }

    .brand span {
        color: #8b6cff;
    }

    .hero {
        padding: 2.2rem 2.3rem;
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(124,92,255,.19), rgba(14,17,29,.8) 48%, rgba(0,198,255,.08)),
            rgba(15,18,30,.92);
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 20px 70px rgba(0,0,0,.25);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: #a996ff;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4.1rem);
        line-height: .98;
        letter-spacing: -.055em;
        margin: 0;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #a58cff, #63d7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1.2rem 0 .7rem;
    }

    .card {
        background: rgba(16,19,31,.78);
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(18,21,34,.95), rgba(12,14,24,.95));
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }

    .metric-label {
        color: #858ca0;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: .15rem;
    }

    .pill {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        background: rgba(139,108,255,.13);
        color: #b7a8ff;
        border: 1px solid rgba(139,108,255,.22);
        font-size: .76rem;
        margin-right: .35rem;
    }

    .report {
        background: linear-gradient(145deg, rgba(17,20,33,.96), rgba(10,12,21,.96));
        border: 1px solid rgba(139,108,255,.16);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        line-height: 1.75;
    }

    .empty {
        text-align: center;
        padding: 4rem 1rem;
        color: #7f879a;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(14,17,28,.7);
        border: 1px dashed rgba(139,108,255,.35);
        border-radius: 16px;
        padding: .4rem;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(139,108,255,.35);
        font-weight: 600;
        min-height: 44px;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        border-color: #9a83ff;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(139,108,255,.14);
    }

    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand">Data<span>Sense</span> AI</div>', unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Workspace")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv"],
        help="CSV files only",
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Your key stays in this session and is not displayed.",
        )

        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key


# HERO
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered analytics workspace</div>
        <h1>Ask your data.<br><span>Let the agent investigate.</span></h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# LANDING STATE
# ─────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown(
        '<div class="empty">Upload a CSV from the sidebar to start your analysis.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# LOAD DATA
# ─────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, "Rows", f"{len(df):,}"),
    (m2, "Columns", f"{len(df.columns):,}"),
    (m3, "Missing cells", f"{int(df.isna().sum().sum()):,}"),
    (m4, "Duplicate rows", f"{int(df.duplicated().sum()):,}"),
]

for col, label, value in metrics:
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

target_options = ["None"] + list(df.columns)
target = st.selectbox(
    "Prediction target",
    target_options,
    help="Choose a column if you want the agent to train a predictive model.",
)

if target == "None":
    target = None

default_goal = (
    f"Analyze this dataset and identify the most important patterns"
    + (f" associated with {target}." if target else ".")
    + " Give me useful insights and explain the evidence clearly."
)

goal = st.text_area(
    "What do you want to discover?",
    value=default_goal,
    height=90,
    placeholder="Example: Find the strongest factors associated with customer churn and build a model to predict it.",
)

run = st.button("✦  Run DataSense Agent", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# DATA PREVIEW
# ─────────────────────────────────────────────────────────────
with st.expander("Preview dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# RUN AGENT
# ─────────────────────────────────────────────────────────────
if run:
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Add your Gemini API key in the sidebar or .env file first.")
        st.stop()

    if not goal.strip():
        st.error("Describe what you want the agent to investigate.")
        st.stop()

    with st.status("DataSense agent is working...", expanded=True) as status:
        st.write("Profiling the dataset...")
        try:
            results = run_agent(df, target, goal)

            st.write("Generating AI report...")
            try:
                report = generate_report(results)
            except Exception as e:
                error_text = str(e)
                if "429" in error_text or "rate limit" in error_text.lower():
                    report = """
### Analysis completed

The data analysis was completed, but the AI report is temporarily unavailable because the Gemini API rate limit was reached.

You can still view the results in the Statistics, Relationships, and ML Model tabs.
"""
                else:
                    raise

            status.update(
                label="Analysis complete",
                state="complete",
                expanded=False,
            )

            st.session_state["results"] = results
            st.session_state["report"] = report
        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.exception(e)
            st.stop()


# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    report = st.session_state.get("report", "")

    st.divider()

    plan = results.get("plan", {})
    selected_tools = plan.get("tools", [])

    # TABS
    tab_report, tab_stats, tab_corr, tab_ml, tab_raw = st.tabs(
        [
            "✦ AI Report",
            "▦ Statistics",
            "⌁ Relationships",
            "◎ ML Model",
            "{} Raw Results",
        ]
    )

    with tab_report:
        st.markdown(
            f'<div class="report">{report}</div>',
            unsafe_allow_html=True,
        )

    with tab_stats:
        stats = results.get("statistics")
        if stats is not None:
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("Descriptive statistics were not selected by the agent.")

    with tab_corr:
        corr = results.get("correlations")

        if corr is not None and not corr.empty:
            st.dataframe(corr, use_container_width=True, hide_index=True)

            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] >= 2:
                fig = px.imshow(
                    numeric.corr(),
                    text_auto=".2f",
                    aspect="auto",
                    title="Numeric correlation matrix",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Correlation analysis was not selected or no numeric relationships were available."
            )

    with tab_ml:
        ml = results.get("model_metrics")

        if ml:
            cols = st.columns(len(ml))
            for col, (key, value) in zip(cols, ml.items()):
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{str(key).replace("_", " ")}</div>'
                        f'<div class="metric-value">{value}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                "No predictive model was trained. Select a target column and run the agent again."
            )

    with tab_raw:
        st.json(results)

else:
    st.markdown(
        '<div class="empty">Configure your question above, then run the agent to generate insights.</div>',
        unsafe_allow_html=True,
    )
