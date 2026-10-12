"""
app.py - Streamlit Web Application for AI Review Insight Analyzer.

A modern, responsive dashboard for customer feedback intelligence.
Performs aspect-based sentiment analysis, topic extraction, and automated
executive intelligence powered by configurable LLMs:
- Google Gemini (Free Tier)
- Groq Cloud (Free Tier)
- OpenAI
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analyzer

# ==============================================================================
# Page Configuration & Theming
# ==============================================================================

st.set_page_config(
    page_title="AI Review Insight Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling, clean metrics, and badge tags
st.markdown(
    """
    <style>
    /* Global styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Sentiment Badges */
    .badge-pos {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neg {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-neu {
        background-color: #F1F5F9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Aspect Tag Pill */
    .aspect-tag {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }

    /* Executive recommendation box */
    .rec-box {
        background: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    
    /* Review Card Container */
    .review-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = None
if "analyzed_category" not in st.session_state:
    st.session_state.analyzed_category = None
if "pasted_reviews_text" not in st.session_state:
    st.session_state.pasted_reviews_text = ""
if "active_input_mode" not in st.session_state:
    st.session_state.active_input_mode = "Paste Reviews"


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_sample_file_reviews() -> str:
    """Load sample review text from sample_reviews.csv."""
    sample_path = Path("sample_reviews.csv")
    if sample_path.is_file():
        try:
            df = pd.read_csv(sample_path)
            if "review" in df.columns:
                return "\n".join(df["review"].dropna().tolist())
        except Exception as e:
            st.error(f"Failed to read sample_reviews.csv: {e}")
    return ""


def reset_analysis_session() -> None:
    """Clear analysis results and reset state."""
    st.session_state.analysis_results = None
    st.session_state.executive_summary = None
    st.session_state.analyzed_category = None
    st.session_state.pasted_reviews_text = ""


# ==============================================================================
# Sidebar: Settings & Multi-Provider API Configuration
# ==============================================================================

config = analyzer.load_config("config.yaml")
provider_cfg = config.get("provider", {})
proc_cfg = config.get("processing", {})

with st.sidebar:
    st.markdown("### ⚙️ LLM Provider & API")
    st.caption("Select your model provider. Free tier available via Google Gemini or Groq.")

    # Provider Selector
    provider_options = [
        "Google Gemini (Free Tier)",
        "Groq Cloud (Free Tier)",
        "OpenAI",
    ]
    current_provider_name = provider_cfg.get("name", "gemini").lower()
    default_provider_idx = 0
    if "groq" in current_provider_name:
        default_provider_idx = 1
    elif "openai" in current_provider_name:
        default_provider_idx = 2

    selected_provider_label = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=default_provider_idx,
        help="Choose an LLM backend. Google Gemini offers a generous free tier without credit cards.",
    )

    # Provider metadata mapping
    if "gemini" in selected_provider_label.lower():
        active_provider = "gemini"
        default_models = ["models/gemini-3.8-flash", "gemini-3.8-flash", "custom"]
        env_key_name = "GEMINI_API_KEY"
        provider_url_help = "Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)"
    elif "groq" in selected_provider_label.lower():
        active_provider = "groq"
        default_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "custom"]
        env_key_name = "GROQ_API_KEY"
        provider_url_help = "Get a free Groq API key from [Groq Console](https://console.groq.com/keys)"
    else:
        active_provider = "openai"
        default_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "custom"]
        env_key_name = "OPENAI_API_KEY"
        provider_url_help = "Requires paid credits. If you see a 429 quota error, switch to Google Gemini!"

    # API Key Input
    env_api_key = os.getenv(env_key_name, "").strip()
    sidebar_api_key = st.text_input(
        f"{selected_provider_label.split(' ')[0]} API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        help=f"Enter your API key or set {env_key_name} in your .env file.",
        placeholder=f"Enter {env_key_name}...",
    )

    st.markdown(f"<div style='font-size:0.8rem; margin-top:-8px; margin-bottom:12px;'>🔗 {provider_url_help}</div>", unsafe_allow_html=True)

    # Status badge for API key
    if sidebar_api_key.strip():
        st.success("API Key detected", icon="🔑")
    else:
        st.warning(f"No key found. Set {env_key_name} or enter above.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 🤖 Model Configuration")

    # Model selector
    configured_model = provider_cfg.get("model", default_models[0])
    model_idx = 0
    if configured_model in default_models:
        model_idx = default_models.index(configured_model)

    model_name = st.selectbox(
        "Model Name",
        options=default_models,
        index=model_idx,
        help="Select model to use for batch analysis and executive synthesis.",
    )

    if model_name == "custom":
        custom_model = st.text_input("Custom Model ID", value=default_models[0])
        config["provider"]["model"] = custom_model
    else:
        config["provider"]["model"] = model_name

    config["provider"]["name"] = active_provider
    config["provider"]["api_key_env_var"] = env_key_name

    with st.expander("🛠️ Advanced Settings", expanded=False):
        custom_base_url = st.text_input(
            "Custom Base URL",
            value=provider_cfg.get("base_url") or "",
            placeholder="Default provider endpoint",
            help="Optional base URL override (e.g. for self-hosted or proxy endpoints).",
        )
        if custom_base_url.strip():
            config["provider"]["base_url"] = custom_base_url.strip()

        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=15,
            value=int(proc_cfg.get("batch_size", 5)),
            help="Number of reviews processed per API call.",
        )
        config["processing"]["batch_size"] = batch_size

        max_reviews_allowed = st.number_input(
            "Max Reviews Limit",
            min_value=5,
            max_value=200,
            value=int(proc_cfg.get("max_reviews", 100)),
            help="Upper limit of reviews analyzed per run.",
        )
        config["processing"]["max_reviews"] = max_reviews_allowed

    st.markdown("---")
    if st.button("🔄 Reset / Start New Analysis", use_container_width=True):
        reset_analysis_session()
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.8rem; color:#64748B;'>
        <strong>AI Review Insight Analyzer</strong><br>
        College NLP Project<br>
        Aspect-Based Sentiment Analysis & Multi-Provider LLM
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Main View Header
# ==============================================================================

st.markdown('<div class="main-header">AI Review Insight Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Extract customer sentiment, granular aspects, and actionable executive insights from product '
    'reviews using Natural Language Processing and structured Large Language Models.'
    '</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# View State: Input & Configuration Screen (When no results are active)
# ==============================================================================

if st.session_state.analysis_results is None:
    # Category selection
    cat_col, sample_col = st.columns([3, 1])
    with cat_col:
        product_category = st.selectbox(
            "Product / Business Category (Optional)",
            options=[
                "Consumer Electronics & Hardware",
                "E-Commerce & Retail",
                "Software & SaaS Products",
                "Food, Dining & Hospitality",
                "Home & Kitchen Appliances",
                "Healthcare & Wellness",
                "Banking, Finance & Fintech",
                "General / Other",
            ],
            index=0,
            help="Providing a domain category helps standardize domain-specific aspects.",
        )

    with sample_col:
        st.write("")
        st.write("")
        if st.button("✨ Load Sample Reviews", use_container_width=True, help="Load 18 sample reviews for testing"):
            sample_content = load_sample_file_reviews()
            if sample_content:
                st.session_state.pasted_reviews_text = sample_content
                st.session_state.active_input_mode = "Paste Reviews"
                st.success("Loaded 18 sample reviews into the editor!")
                st.rerun()

    # Input method selection
    input_tab1, input_tab2 = st.tabs(["📋 Paste Reviews (One per line)", "📁 Upload CSV File"])

    prepared_reviews: List[Dict[str, str]] = []
    validation_warnings: List[str] = []

    # Tab 1: Paste Reviews
    with input_tab1:
        st.caption("Paste multiple reviews into the box below. Separate each review with a newline.")
        user_text = st.text_area(
            "Customer Reviews",
            value=st.session_state.pasted_reviews_text,
            height=230,
            placeholder=(
                "The noise-canceling headphones have amazing sound, but the companion app crashes often.\n"
                "Fast shipping and great packaging, but the product stopped working after 3 days.\n"
                "Customer support was incredibly helpful and solved my return request in minutes."
            ),
            key="text_area_input",
        )
        st.session_state.pasted_reviews_text = user_text

        if user_text.strip():
            prepared_reviews, validation_warnings = analyzer.prepare_reviews_from_text(
                raw_text=user_text,
                max_reviews=int(config["processing"]["max_reviews"]),
                max_chars=int(config["processing"]["max_review_chars"]),
            )

    # Tab 2: Upload CSV
    with input_tab2:
        st.caption("Upload a CSV file containing customer feedback. Must contain a 'review' column.")
        uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                csv_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded CSV has **{len(csv_df)}** rows and columns: `{list(csv_df.columns)}`")

                available_cols = list(csv_df.columns)
                default_col_idx = 0
                for i, col in enumerate(available_cols):
                    if str(col).lower() in ["review", "reviews", "feedback", "text"]:
                        default_col_idx = i
                        break

                selected_col = st.selectbox("Select Review Column", options=available_cols, index=default_col_idx)

                prepared_reviews, validation_warnings = analyzer.prepare_reviews_from_dataframe(
                    df=csv_df,
                    column_name=selected_col,
                    max_reviews=int(config["processing"]["max_reviews"]),
                    max_chars=int(config["processing"]["max_review_chars"]),
                )
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    # Display validation feedback
    if validation_warnings:
        for w in validation_warnings:
            st.warning(w, icon="⚠️")

    # Review readiness and action section
    st.markdown("---")
    action_col1, action_col2 = st.columns([3, 1])

    with action_col1:
        if prepared_reviews:
            st.info(
                f"✅ **{len(prepared_reviews)} valid review(s)** ready for NLP analysis using **{selected_provider_label}** (`{config['provider']['model']}`). "
                f"Batch size: {config['processing']['batch_size']} reviews/request "
                f"({(len(prepared_reviews) + config['processing']['batch_size'] - 1) // config['processing']['batch_size']} batch requests)."
            )
        else:
            st.info("ℹ️ Enter reviews above or load the sample dataset to proceed.")

    with action_col2:
        analyze_disabled = len(prepared_reviews) == 0 or not sidebar_api_key.strip()
        analyze_button = st.button(
            "🚀 Analyze Reviews",
            type="primary",
            use_container_width=True,
            disabled=analyze_disabled,
        )

    # Missing API Key Warning with One-Click Guide
    if len(prepared_reviews) > 0 and not sidebar_api_key.strip():
        st.error(
            f"🔑 **{selected_provider_label.split(' ')[0]} API Key Required**: Please enter your API key in the left sidebar "
            f"or set `{env_key_name}` in your `.env` file before running the analysis.\n\n"
            f"💡 **Need a free key?** {provider_url_help}",
            icon="🔒",
        )

    # ==========================================================================
    # Run Analysis Execution
    # ==========================================================================
    if analyze_button and prepared_reviews and sidebar_api_key.strip():
        st.session_state.analyzed_category = product_category

        # Initialize API client
        try:
            provider_type, client = analyzer.get_llm_client(
                provider_name=active_provider,
                api_key=sidebar_api_key,
                base_url=config["provider"].get("base_url"),
                config=config,
            )
        except Exception as e:
            st.error(f"Failed to initialize API client: {e}")
            st.stop()

        # Progress elements
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio: float, message: str) -> None:
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            status_text.text(f"⏳ {message}")

        start_time = time.time()
        try:
            with st.spinner(f"Processing reviews with {selected_provider_label}..."):
                analysis_results = analyzer.analyze_all_reviews(
                    client=client,
                    reviews=prepared_reviews,
                    config=config,
                    category=product_category,
                    progress_callback=update_progress,
                    provider=provider_type,
                )

                update_progress(1.0, "Generating AI Executive Summary & Actionable Insights...")
                executive_summary = analyzer.generate_executive_summary(
                    client=client,
                    analysis_results=analysis_results,
                    config=config,
                    category=product_category,
                    provider=provider_type,
                )

            elapsed = round(time.time() - start_time, 2)
            st.session_state.analysis_results = analysis_results
            st.session_state.executive_summary = executive_summary
            st.success(f"Successfully analyzed {len(analysis_results)} reviews in {elapsed}s with {selected_provider_label}!")
            st.rerun()

        except Exception as err:
            st.error(f"Analysis failed: {err}", icon="❌")
            # Actionable help if quota error occurred
            if "insufficient_quota" in str(err).lower() or "429" in str(err):
                st.warning(
                    "💡 **Quota Exhausted Tip**: Switch the **LLM Provider** in the left sidebar to "
                    "**Google Gemini (Free Tier)**. Google AI Studio provides free API keys without requiring "
                    "any credit card or billing setup!",
                    icon="💡",
                )


# ==============================================================================
# View State: Dashboard & Results Screen (When results are available)
# ==============================================================================

else:
    results: List[Dict[str, Any]] = st.session_state.analysis_results
    exec_summary: Optional[Dict[str, Any]] = st.session_state.executive_summary
    current_category: Optional[str] = st.session_state.analyzed_category

    # Top Status & Reset Bar
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.markdown(
            f"**Dataset:** `{len(results)} Reviews Analyzed` &nbsp;|&nbsp; "
            f"**Domain:** `{current_category or 'General'}` &nbsp;|&nbsp; "
            f"**Provider:** `{config.get('provider', {}).get('name', 'gemini').upper()}` (`{config.get('provider', {}).get('model', 'models/gemini-3.8-flash')}`)"
        )
    with top_col2:
        if st.button("➕ New Analysis", use_container_width=True):
            reset_analysis_session()
            st.rerun()

    # Calculate metrics
    metrics = analyzer.calculate_metrics(results)

    # --------------------------------------------------------------------------
    # 1. Summary Metrics Cards
    # --------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #1E293B;">{metrics['total']}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #16A34A;">{metrics['positive_count']} <span style="font-size:1.1rem; color:#4ADE80;">({metrics['positive_pct']}%)</span></div>
                <div class="metric-label">Positive Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #DC2626;">{metrics['negative_count']} <span style="font-size:1.1rem; color:#F87171;">({metrics['negative_pct']}%)</span></div>
                <div class="metric-label">Negative Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #64748B;">{metrics['neutral_count']} <span style="font-size:1.1rem; color:#94A3B8;">({metrics['neutral_pct']}%)</span></div>
                <div class="metric-label">Neutral / Mixed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. Charts Row (Plotly Sentiment Donut & Top Mentioned Aspects)
    # --------------------------------------------------------------------------
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("Sentiment Distribution")
        sentiment_labels = ["Positive", "Negative", "Neutral"]
        sentiment_values = [metrics["positive_count"], metrics["negative_count"], metrics["neutral_count"]]
        colors = ["#10B981", "#EF4444", "#94A3B8"]

        fig_pie = go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textposition="inside",
            hoverinfo="label+value+percent",
        )
        fig_pie_layout = go.Layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(go.Figure(data=[fig_pie], layout=fig_pie_layout), use_container_width=True)

    with chart_col2:
        st.subheader("Top Mentioned Aspects & Topics")
        aspect_dict = metrics.get("aspect_counts", {})
        if aspect_dict:
            top_aspects = list(aspect_dict.items())[:8]
            aspect_df = pd.DataFrame(top_aspects, columns=["Aspect", "Frequency"]).sort_values(
                by="Frequency", ascending=True
            )

            fig_bar = px.bar(
                aspect_df,
                x="Frequency",
                y="Aspect",
                orientation="h",
                color="Frequency",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                coloraxis_showscale=False,
                xaxis_title="Mentions",
                yaxis_title="",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No specific aspects extracted from this dataset.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. AI Executive Summary & Actionable Recommendations
    # --------------------------------------------------------------------------
    if exec_summary:
        st.markdown("## 🧠 AI Executive Intelligence Report")
        st.markdown(
            f"**Overall Customer Sentiment:** *{exec_summary.get('overall_sentiment', 'N/A')}*"
        )

        sum_c1, sum_c2, sum_c3 = st.columns(3)
        with sum_c1:
            st.markdown("#### 🌟 Most Appreciated Features")
            for item in exec_summary.get("appreciated_features", []):
                st.markdown(f"• **{item}**")

        with sum_c2:
            st.markdown("#### ⚠️ Common Customer Complaints")
            for item in exec_summary.get("common_complaints", []):
                st.markdown(f"• **{item}**")

        with sum_c3:
            st.markdown("#### 💡 Emerging Themes")
            for item in exec_summary.get("emerging_themes", []):
                st.markdown(f"• **{item}**")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🎯 3 Priority Actionable Recommendations")
        recs = exec_summary.get("actionable_recommendations", [])
        for rec in recs:
            st.markdown(
                f'<div class="rec-box">📌 {rec}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("📄 View Full Executive Narrative Summary", expanded=False):
            st.write(exec_summary.get("summary_text", ""))

        st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. Interactive Review Explorer & Sentiment Filters
    # --------------------------------------------------------------------------
    st.markdown("## 🔍 Review Explorer")

    filter_c1, filter_c2 = st.columns([1, 2])
    with filter_c1:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=["All", "Positive", "Negative", "Neutral"],
            index=0,
        )
    with filter_c2:
        search_query = st.text_input(
            "Search Reviews or Aspects",
            placeholder="Search keywords (e.g., 'battery', 'delivery', 'screen')...",
        )

    # Filter records
    filtered_results = results
    if sentiment_filter != "All":
        filtered_results = [
            r for r in filtered_results if r["sentiment"].lower() == sentiment_filter.lower()
        ]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_results = [
            r for r in filtered_results
            if q in r["review_text"].lower()
            or any(q in a.lower() for a in r.get("aspects", []))
            or any(q in p.lower() for p in r.get("positive_points", []))
            or any(q in n.lower() for n in r.get("negative_points", []))
        ]

    st.caption(f"Displaying **{len(filtered_results)}** of **{len(results)}** reviews.")

    # View Mode: Tabular DataFrame vs Expandable Cards
    view_tab1, view_tab2 = st.tabs(["📑 Detailed Cards View", "📊 Tabular Data View"])

    with view_tab1:
        if not filtered_results:
            st.info("No reviews match your selected filter criteria.")
        else:
            for item in filtered_results:
                sent = item["sentiment"].lower()
                badge_class = "badge-pos" if sent == "positive" else ("badge-neg" if sent == "negative" else "badge-neu")

                with st.expander(f"{item['review_id']} — {item['sentiment'].capitalize()} — \"{item['review_text'][:70]}...\"", expanded=False):
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 8px;">
                            <span class="{badge_class}">{item['sentiment'].upper()}</span>
                            <span style="color:#64748B; font-size:0.9rem; margin-left: 10px;">ID: {item['review_id']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Review Text:** *\"{item['review_text']}\"*")
                    st.markdown(f"**Classification Reason:** {item['explanation']}")

                    aspects = item.get("aspects", [])
                    if aspects:
                        aspect_html = "".join([f'<span class="aspect-tag">🏷️ {a}</span>' for a in aspects])
                        st.markdown(f"**Extracted Aspects:**<br>{aspect_html}", unsafe_allow_html=True)

                    card_c1, card_c2 = st.columns(2)
                    with card_c1:
                        pros = item.get("positive_points", [])
                        if pros:
                            st.markdown("**Positive Points:**")
                            for p in pros:
                                st.markdown(f"• {p}")
                    with card_c2:
                        cons = item.get("negative_points", [])
                        if cons:
                            st.markdown("**Complaints / Issues:**")
                            for c in cons:
                                st.markdown(f"• {c}")

    with view_tab2:
        if filtered_results:
            table_df = analyzer.results_to_dataframe(filtered_results)
            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Review ID": st.column_config.TextColumn(width="small"),
                    "Review Text": st.column_config.TextColumn(width="large"),
                    "Sentiment": st.column_config.TextColumn(width="small"),
                    "Aspects / Topics": st.column_config.TextColumn(width="medium"),
                },
            )
        else:
            st.info("No records to display.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 5. Export Center
    # --------------------------------------------------------------------------
    st.markdown("## 📥 Export Analysis Data")
    st.caption("Download structured analysis and executive reports for documentation or downstream workflows.")

    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        csv_bytes = analyzer.results_to_csv_bytes(results)
        st.download_button(
            label="📄 Download Results (CSV)",
            data=csv_bytes,
            file_name="review_insights_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_c2:
        json_bytes = analyzer.results_to_json_bytes(results)
        st.download_button(
            label="📦 Download Full Dataset (JSON)",
            data=json_bytes,
            file_name="review_insights_dataset.json",
            mime="application/json",
            use_container_width=True,
        )

    with exp_c3:
        if exec_summary:
            summary_json_bytes = analyzer.summary_to_json_bytes(exec_summary)
            st.download_button(
                label="🧠 Download Executive Summary (JSON)",
                data=summary_json_bytes,
                file_name="executive_summary_report.json",
                mime="application/json",
                use_container_width=True,
            )
