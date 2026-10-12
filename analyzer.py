"""
analyzer.py - Core NLP Analysis Engine for AI Review Insight Analyzer.

Supports configurable LLM providers:
- Google Gemini (Official google-genai SDK, Free Tier via Google AI Studio)
- Groq Cloud (Official groq SDK, Free Tier via Groq Console)
- OpenAI (Official openai SDK)

Handles:
- Multi-provider configuration and system prompt loading
- Robust batch processing with retry logic and clear error reporting
- Pydantic schema validation for LLM responses
- Executive summary generation with strict grounding
- Data export utilities (CSV, JSON) and metrics calculation
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Optional provider SDK imports
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load local environment variables from .env if present
load_dotenv()


# ==============================================================================
# Pydantic Schemas for Structured LLM Validation
# ==============================================================================

class ReviewAnalysisItem(BaseModel):
    """Schema for individual review analysis produced by LLM."""
    review_id: str = Field(description="Unique ID matching the review in the input batch")
    sentiment: str = Field(description="Must be 'positive', 'negative', or 'neutral'")
    explanation: str = Field(description="Brief 1-2 sentence evidence-based reasoning")
    aspects: List[str] = Field(default_factory=list, description="List of product/service topics mentioned")
    positive_points: List[str] = Field(default_factory=list, description="Specific features or experiences praised")
    negative_points: List[str] = Field(default_factory=list, description="Specific complaints or flaws identified")


class BatchAnalysisResponse(BaseModel):
    """Schema expected from the LLM for a batch of reviews."""
    results: List[ReviewAnalysisItem] = Field(
        description="List of analyzed reviews matching the requested batch"
    )


class ExecutiveSummaryModel(BaseModel):
    """Schema for AI-generated overall executive summary."""
    overall_sentiment: str = Field(description="High-level synthesis of customer sentiment")
    appreciated_features: List[str] = Field(
        default_factory=list, description="Top positive features/strengths highlighted by customers"
    )
    common_complaints: List[str] = Field(
        default_factory=list, description="Most frequent negative issues or pain points"
    )
    emerging_themes: List[str] = Field(
        default_factory=list, description="Subtle or emerging patterns detected across feedback"
    )
    actionable_recommendations: List[str] = Field(
        min_length=3, max_length=3, description="Exactly 3 concrete recommendations for the team"
    )
    summary_text: str = Field(
        description="A cohesive 2-3 paragraph executive summary narrative"
    )


# ==============================================================================
# Configuration & Prompt Loaders
# ==============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Relative or absolute path to config.yaml

    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.warning("Config file not found at %s. Using default fallback configuration.", config_path)
        return {
            "provider": {
                "name": "gemini",
                "model": "models/gemini-3.8-flash",
                "base_url": None,
                "api_key_env_var": "GEMINI_API_KEY",
            },
            "generation": {"temperature": 0.1, "max_tokens": 3000},
            "processing": {
                "batch_size": 5,
                "max_reviews": 100,
                "max_review_chars": 2000,
                "timeout_seconds": 60,
                "retry_limit": 2,
            },
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config or {}
    except Exception as e:
        logger.error("Failed to parse config file: %s", e)
        raise RuntimeError(f"Could not load configuration from {config_path}: {e}") from e


def load_system_prompt(prompt_path: str = "prompts/system_prompt.txt") -> str:
    """
    Load the system prompt template from file.

    Args:
        prompt_path: Relative or absolute path to system prompt file.

    Returns:
        System prompt string.
    """
    path = Path(prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"System prompt file not found at: {prompt_path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ==============================================================================
# Multi-Provider Client Setup
# ==============================================================================

def get_llm_client(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any]:
    """
    Initialize and return an LLM API client for the requested provider.

    Supported providers:
    - 'gemini': Google Gemini official SDK (google-genai, free tier available)
    - 'groq': Groq Cloud official SDK (groq, free tier available)
    - 'openai': OpenAI official SDK (openai)

    Returns:
        Tuple of (normalized_provider_name, client_instance)
    """
    cfg = config or load_config()
    provider_cfg = cfg.get("provider", {})
    processing_cfg = cfg.get("processing", {})

    target_provider = (provider_name or provider_cfg.get("name", "gemini")).strip().lower()

    # 1. Google Gemini Provider
    if target_provider in ["gemini", "google", "google-genai"]:
        if not HAS_GEMINI:
            raise ImportError(
                "The 'google-genai' package is required for Google Gemini. "
                "Install it with: pip install google-genai"
            )

        is_configured = provider_cfg.get("name", "").strip().lower() in ["gemini", "google", "google-genai"]
        env_var = provider_cfg.get("api_key_env_var") if is_configured else "GEMINI_API_KEY"
        final_key = (api_key or "").strip() or os.getenv(env_var, "").strip() or os.getenv("GEMINI_API_KEY", "").strip()

        if not final_key:
            raise ValueError(
                f"Google Gemini API key missing. Please provide your API key in the sidebar "
                f"or set 'GEMINI_API_KEY' in your .env file.\n"
                f"You can get a free API key at: https://aistudio.google.com/app/apikey"
            )

        client = genai.Client(api_key=final_key)
        return "gemini", client

    # 2. Groq Provider
    elif target_provider in ["groq", "groqcloud"]:
        if not HAS_GROQ:
            raise ImportError(
                "The 'groq' package is required for Groq. "
                "Install it with: pip install groq"
            )

        is_configured = provider_cfg.get("name", "").strip().lower() in ["groq", "groqcloud"]
        env_var = provider_cfg.get("api_key_env_var") if is_configured else "GROQ_API_KEY"
        final_key = (api_key or "").strip() or os.getenv(env_var, "").strip() or os.getenv("GROQ_API_KEY", "").strip()

        if not final_key:
            raise ValueError(
                f"Groq API key missing. Please provide your API key in the sidebar "
                f"or set 'GROQ_API_KEY' in your .env file.\n"
                f"You can get a free API key at: https://console.groq.com/keys"
            )

        client = groq.Groq(api_key=final_key)
        return "groq", client

    # 3. OpenAI Provider
    elif target_provider in ["openai", "openai-compatible"]:
        if not HAS_OPENAI:
            raise ImportError(
                "The 'openai' package is required for OpenAI. "
                "Install it with: pip install openai"
            )

        is_configured = provider_cfg.get("name", "").strip().lower() in ["openai", "openai-compatible"]
        env_var = provider_cfg.get("api_key_env_var") if is_configured else "OPENAI_API_KEY"
        final_key = (api_key or "").strip() or os.getenv(env_var, "").strip() or os.getenv("OPENAI_API_KEY", "").strip()

        if not final_key:
            raise ValueError(
                f"OpenAI API key missing. Please provide your API key in the sidebar "
                f"or set '{env_var}' in your .env file."
            )

        final_base_url = (base_url or "").strip() or provider_cfg.get("base_url") or None
        timeout = float(processing_cfg.get("timeout_seconds", 60))

        client = openai.OpenAI(
            api_key=final_key,
            base_url=final_base_url,
            timeout=timeout,
        )
        return "openai", client

    else:
        raise ValueError(
            f"Unsupported provider '{target_provider}'. "
            f"Supported providers are: 'gemini' (Free Tier), 'groq' (Free Tier), and 'openai'."
        )


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> openai.OpenAI:
    """Backwards-compatibility helper for initializing an OpenAI client directly."""
    _, client = get_llm_client(
        provider_name="openai",
        api_key=api_key,
        base_url=base_url,
        config=config,
    )
    return client


# ==============================================================================
# Input Parsing & Validation
# ==============================================================================

def validate_review_text(text: str, max_chars: int = 2000) -> Tuple[bool, str]:
    """
    Validate an individual review string.

    Returns:
        (is_valid, sanitized_text_or_error)
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "Empty review text."
    if len(cleaned) > max_chars:
        return False, f"Review exceeds maximum length of {max_chars} characters (length: {len(cleaned)})."
    return True, cleaned


def prepare_reviews_from_text(
    raw_text: str,
    max_reviews: int = 100,
    max_chars: int = 2000,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse newline-delimited text into structured review objects.

    Returns:
        (valid_reviews_list, warning_messages)
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    warnings: List[str] = []

    if not lines:
        return [], ["No reviews provided. Please paste at least one non-empty review."]

    if len(lines) > max_reviews:
        warnings.append(
            f"Input contained {len(lines)} reviews. Capped to maximum allowed of {max_reviews}."
        )
        lines = lines[:max_reviews]

    reviews: List[Dict[str, str]] = []
    for idx, text in enumerate(lines, start=1):
        is_valid, cleaned_or_err = validate_review_text(text, max_chars=max_chars)
        if is_valid:
            reviews.append({
                "review_id": f"REV-{idx:03d}",
                "review_text": cleaned_or_err,
            })
        else:
            warnings.append(f"Skipped line {idx}: {cleaned_or_err}")

    return reviews, warnings


def prepare_reviews_from_dataframe(
    df: pd.DataFrame,
    column_name: str = "review",
    max_reviews: int = 100,
    max_chars: int = 2000,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse a Pandas DataFrame into structured review objects.

    Returns:
        (valid_reviews_list, warning_messages)
    """
    warnings: List[str] = []

    if df.empty:
        return [], ["The uploaded CSV file is empty."]

    # Case-insensitive column matching fallback
    target_col = column_name
    if target_col not in df.columns:
        matched = [c for c in df.columns if str(c).strip().lower() == column_name.lower()]
        if matched:
            target_col = matched[0]
        else:
            synonyms = ["review", "reviews", "text", "feedback", "comment", "comments"]
            alt_matches = [c for c in df.columns if str(c).strip().lower() in synonyms]
            if alt_matches:
                target_col = alt_matches[0]
                warnings.append(f"Column '{column_name}' not found. Using matched column '{target_col}'.")
            else:
                return [], [
                    f"Required column '{column_name}' not found. Available columns: {list(df.columns)}"
                ]

    raw_items = df[target_col].dropna().astype(str).tolist()
    if not raw_items:
        return [], [f"No text entries found in column '{target_col}'."]

    if len(raw_items) > max_reviews:
        warnings.append(
            f"CSV contained {len(raw_items)} reviews. Capped to maximum allowed of {max_reviews}."
        )
        raw_items = raw_items[:max_reviews]

    reviews: List[Dict[str, str]] = []
    for idx, text in enumerate(raw_items, start=1):
        is_valid, cleaned_or_err = validate_review_text(text, max_chars=max_chars)
        if is_valid:
            reviews.append({
                "review_id": f"REV-{idx:03d}",
                "review_text": cleaned_or_err,
            })
        else:
            warnings.append(f"Skipped row {idx}: {cleaned_or_err}")

    return reviews, warnings


def chunk_reviews(reviews: List[Dict[str, str]], batch_size: int = 5) -> List[List[Dict[str, str]]]:
    """Divide review list into chunks of size batch_size."""
    batch_size = max(1, batch_size)
    return [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from raw LLM output, safely stripping markdown fences.
    """
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


# ==============================================================================
# Unified LLM Execution with Provider-Specific Error Handling
# ==============================================================================

def _execute_llm_call(
    provider: str,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
    retry_limit: int = 2,
) -> str:
    """
    Execute structured LLM generation across Gemini, Groq, or OpenAI.
    Provides exponential backoff for transient issues and explicit, transparent
    error reporting for quota or authentication failures.
    """
    last_exception: Optional[Exception] = None
    provider_norm = provider.strip().lower()

    for attempt in range(retry_limit + 1):
        try:
            # ------------------------------------------------------------------
            # 1. Google Gemini Execution
            # ------------------------------------------------------------------
            if provider_norm in ["gemini", "google", "google-genai"]:
                # Ensure model name is formatted properly with models/ prefix as required by Gemini API
                gemini_model = model if model.startswith("models/") else f"models/{model}"
                config_params = genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=user_prompt,
                    config=config_params,
                )
                content = response.text
                if not content:
                    raise ValueError("Google Gemini returned an empty response.")
                return content.strip()

            # ------------------------------------------------------------------
            # 2. Groq Cloud Execution
            # ------------------------------------------------------------------
            elif provider_norm in ["groq", "groqcloud"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Groq returned an empty completion response.")
                return content.strip()

            # ------------------------------------------------------------------
            # 3. OpenAI Execution
            # ------------------------------------------------------------------
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("OpenAI returned an empty completion response.")
                return content.strip()

        # ----------------------------------------------------------------------
        # Provider-Specific Exception Interception
        # ----------------------------------------------------------------------
        except Exception as err:
            last_exception = err
            err_msg = str(err)
            err_cls = err.__class__.__name__

            # Check for OpenAI 429 Insufficient Quota
            if "insufficient_quota" in err_msg.lower():
                raise RuntimeError(
                    "OpenAI API Error (429: insufficient_quota): Your OpenAI credits are exhausted. "
                    "Switch to Google Gemini (Free Tier) in the web UI sidebar or config.yaml to "
                    "continue analyzing reviews without paid OpenAI credits!"
                ) from err

            # Check for Google Gemini Quota Exceeded
            if "RESOURCE_EXHAUSTED" in err_msg or (hasattr(err, "code") and getattr(err, "code") == 429):
                if attempt < retry_limit:
                    sleep_time = (2 ** attempt) * 2.0
                    logger.warning("Gemini rate limit encountered. Retrying in %.1fs...", sleep_time)
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(
                    "Google Gemini API Error (429: RESOURCE_EXHAUSTED): Quota or rate limit exceeded. "
                    "Please wait a moment before analyzing more reviews. You can check your free tier "
                    "limits at https://aistudio.google.com."
                ) from err

            # Check for Invalid Key / Auth Errors
            if "API_KEY_INVALID" in err_msg or "INVALID_ARGUMENT" in err_msg or "401" in err_msg:
                raise ValueError(
                    f"Authentication failed for provider '{provider}': {err_msg}. "
                    "Please verify your API key in .env or the web UI sidebar."
                ) from err

            # Retry transient errors
            is_transient = any(
                keyword in err_cls.lower() or keyword in err_msg.lower()
                for keyword in ["ratelimit", "connection", "timeout", "unavailable", "503", "502"]
            )
            if is_transient and attempt < retry_limit:
                sleep_time = (2 ** attempt) * 1.5
                logger.warning(
                    "Transient error (%s) on attempt %d/%d for provider '%s'. Retrying in %.1fs...",
                    err_cls, attempt + 1, retry_limit + 1, provider, sleep_time
                )
                time.sleep(sleep_time)
            else:
                logger.error("LLM execution error with %s: %s", provider, err)
                raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Failed to complete LLM request after retries.")


# ==============================================================================
# Batch Review Analysis
# ==============================================================================

def analyze_review_batch(
    client: Any,
    batch: List[Dict[str, str]],
    config: Dict[str, Any],
    system_prompt: str,
    category: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Send a batch of reviews to the LLM for aspect-based sentiment analysis.

    Validates schema strictly using Pydantic.
    Preserves exact mapping to original review IDs.
    """
    provider_name = (provider or config.get("provider", {}).get("name", "gemini")).strip().lower()
    model_name = config.get("provider", {}).get("model", "models/gemini-3.8-flash")
    gen_cfg = config.get("generation", {})
    proc_cfg = config.get("processing", {})

    temperature = float(gen_cfg.get("temperature", 0.1))
    max_tokens = int(gen_cfg.get("max_tokens", 3000))
    retry_limit = int(proc_cfg.get("retry_limit", 2))

    # Construct user message with untrusted review payload cleanly encapsulated
    category_note = f"\nTarget Business/Product Category: {category}\n" if category else ""
    batch_payload = [
        {"review_id": item["review_id"], "text": item["review_text"]}
        for item in batch
    ]

    user_instructions = (
        f"{category_note}"
        "Analyze the following batch of customer reviews according to your system instructions.\n"
        "Return a JSON object containing a 'results' array where each item matches the schema:\n"
        "- review_id: (exact matching string)\n"
        "- sentiment: ('positive' | 'negative' | 'neutral')\n"
        "- explanation: (1-2 sentence evidence-based reason)\n"
        "- aspects: (list of 1-4 short topic/aspect strings)\n"
        "- positive_points: (list of positive phrases/features mentioned, or empty)\n"
        "- negative_points: (list of complaints/issues mentioned, or empty)\n\n"
        f"Input Reviews Batch (JSON):\n{json.dumps(batch_payload, indent=2)}"
    )

    # Call LLM
    raw_json_str = _execute_llm_call(
        provider=provider_name,
        client=client,
        model=model_name,
        system_prompt=system_prompt,
        user_prompt=user_instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_limit=retry_limit,
    )

    # Parse and validate JSON
    try:
        parsed_data = extract_json_payload(raw_json_str)
    except json.JSONDecodeError as jde:
        logger.error("Failed to parse LLM JSON output: %s\nRaw output: %s", jde, raw_json_str)
        raise ValueError(f"LLM produced malformed JSON: {jde}") from jde

    try:
        validated = BatchAnalysisResponse.model_validate(parsed_data)
    except ValidationError as ve:
        logger.error("LLM JSON did not conform to schema: %s", ve)
        raise ValueError(f"LLM output schema validation failed: {ve}") from ve

    # Build lookup by review_id
    validated_map = {item.review_id: item for item in validated.results}

    # Map back to original batch preserving order and original text
    output_items: List[Dict[str, Any]] = []
    for original in batch:
        rid = original["review_id"]
        rtext = original["review_text"]

        if rid in validated_map:
            item = validated_map[rid]
            sentiment_clean = str(item.sentiment).strip().lower()
            if sentiment_clean not in ["positive", "negative", "neutral"]:
                sentiment_clean = "neutral"

            output_items.append({
                "review_id": rid,
                "review_text": rtext,
                "sentiment": sentiment_clean,
                "explanation": item.explanation,
                "aspects": [str(a).strip().title() for a in item.aspects if str(a).strip()],
                "positive_points": [str(p).strip() for p in item.positive_points if str(p).strip()],
                "negative_points": [str(n).strip() for n in item.negative_points if str(n).strip()],
                "status": "success",
                "error_message": None,
            })
        else:
            logger.warning("Review ID %s was omitted in model response.", rid)
            output_items.append({
                "review_id": rid,
                "review_text": rtext,
                "sentiment": "neutral",
                "explanation": "Review analysis omitted by model output.",
                "aspects": [],
                "positive_points": [],
                "negative_points": [],
                "status": "partial_error",
                "error_message": f"Review {rid} was missing from the LLM batch output.",
            })

    return output_items


def analyze_all_reviews(
    client: Any,
    reviews: List[Dict[str, str]],
    config: Dict[str, Any],
    category: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    provider: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process full list of reviews in batches with real-time progress reporting.
    """
    if not reviews:
        return []

    system_prompt = load_system_prompt()
    batch_size = int(config.get("processing", {}).get("batch_size", 5))
    batches = chunk_reviews(reviews, batch_size=batch_size)
    total_batches = len(batches)

    results: List[Dict[str, Any]] = []

    for idx, batch in enumerate(batches, start=1):
        msg = f"Analyzing batch {idx} of {total_batches} ({len(batch)} reviews)..."
        logger.info(msg)
        if progress_callback:
            progress_callback((idx - 1) / total_batches, msg)

        try:
            batch_results = analyze_review_batch(
                client=client,
                batch=batch,
                config=config,
                system_prompt=system_prompt,
                category=category,
                provider=provider,
            )
            results.extend(batch_results)
        except Exception as e:
            logger.error("Failed to process batch %d: %s", idx, e)
            raise RuntimeError(
                f"Batch {idx}/{total_batches} failed during LLM analysis: {str(e)}"
            ) from e

        if progress_callback:
            progress_callback(idx / total_batches, f"Completed batch {idx} of {total_batches}.")

    return results


# ==============================================================================
# AI Executive Summary Generation
# ==============================================================================

def generate_executive_summary(
    client: Any,
    analysis_results: List[Dict[str, Any]],
    config: Dict[str, Any],
    category: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an AI-powered executive summary synthesizing the analyzed reviews.
    """
    if not analysis_results:
        raise ValueError("Cannot generate summary without analyzed reviews.")

    provider_name = (provider or config.get("provider", {}).get("name", "gemini")).strip().lower()
    model_name = config.get("provider", {}).get("model", "models/gemini-3.8-flash")
    gen_cfg = config.get("generation", {})
    proc_cfg = config.get("processing", {})

    temperature = float(gen_cfg.get("temperature", 0.1))
    max_tokens = int(gen_cfg.get("max_tokens", 3000))
    retry_limit = int(proc_cfg.get("retry_limit", 2))

    # Pre-aggregate stats for grounding
    total_count = len(analysis_results)
    pos_count = sum(1 for r in analysis_results if r["sentiment"] == "positive")
    neg_count = sum(1 for r in analysis_results if r["sentiment"] == "negative")
    neu_count = sum(1 for r in analysis_results if r["sentiment"] == "neutral")

    all_pros = [p for r in analysis_results for p in r.get("positive_points", [])][:30]
    all_cons = [c for r in analysis_results for c in r.get("negative_points", [])][:30]
    all_aspects = [a for r in analysis_results for a in r.get("aspects", [])][:40]

    category_info = f"Category: {category}\n" if category else ""

    summary_prompt = (
        "You are an executive product intelligence strategist.\n"
        "Synthesize the following analyzed customer review dataset into an executive summary report.\n\n"
        "### STRICT GROUNDING RULES:\n"
        "1. Base all observations strictly and only on the provided customer feedback data.\n"
        "2. Do NOT invent customer praise or complaints that are not reflected in the provided points.\n"
        "3. Provide exactly THREE prioritized, highly actionable recommendations for management.\n"
        "4. Return strictly valid JSON matching the schema.\n\n"
        f"{category_info}"
        f"DATASET AGGREGATION:\n"
        f"- Total Reviews Analyzed: {total_count}\n"
        f"- Positive: {pos_count} ({pos_count/total_count*100:.1f}%)\n"
        f"- Negative: {neg_count} ({neg_count/total_count*100:.1f}%)\n"
        f"- Neutral: {neu_count} ({neu_count/total_count*100:.1f}%)\n\n"
        f"KEY ASPECTS MENTIONED: {json.dumps(all_aspects[:25])}\n"
        f"SAMPLE POSITIVE POINTS: {json.dumps(all_pros[:20])}\n"
        f"SAMPLE COMPLAINTS: {json.dumps(all_cons[:20])}\n\n"
        "REQUIRED JSON OUTPUT SCHEMA:\n"
        "{\n"
        '  "overall_sentiment": "Short summary sentence of customer sentiment balance",\n'
        '  "appreciated_features": ["Feature 1", "Feature 2", ...],\n'
        '  "common_complaints": ["Complaint 1", "Complaint 2", ...],\n'
        '  "emerging_themes": ["Theme 1", "Theme 2", ...],\n'
        '  "actionable_recommendations": [\n'
        '    "1. Recommendation description",\n'
        '    "2. Recommendation description",\n'
        '    "3. Recommendation description"\n'
        "  ],\n"
        '  "summary_text": "2-3 comprehensive paragraphs providing the executive overview."\n'
        "}"
    )

    system_instruction = "You are a professional customer experience analyst who outputs valid JSON."

    raw_json_str = _execute_llm_call(
        provider=provider_name,
        client=client,
        model=model_name,
        system_prompt=system_instruction,
        user_prompt=summary_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        retry_limit=retry_limit,
    )

    try:
        parsed_summary = extract_json_payload(raw_json_str)
        validated = ExecutiveSummaryModel.model_validate(parsed_summary)
        return validated.model_dump()
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Failed to parse or validate executive summary JSON: %s\nRaw: %s", e, raw_json_str)
        raise ValueError(f"Executive summary generation produced invalid schema: {e}") from e


# ==============================================================================
# Metrics & Analytics Helpers
# ==============================================================================

def calculate_metrics(analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary counts, percentages, and frequency distributions.
    """
    total = len(analysis_results)
    if total == 0:
        return {
            "total": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "aspect_counts": {},
            "top_positive_points": [],
            "top_negative_points": [],
        }

    pos = sum(1 for r in analysis_results if r["sentiment"] == "positive")
    neg = sum(1 for r in analysis_results if r["sentiment"] == "negative")
    neu = sum(1 for r in analysis_results if r["sentiment"] == "neutral")

    # Aspect frequency
    aspect_counts: Dict[str, int] = {}
    for r in analysis_results:
        for aspect in r.get("aspects", []):
            normalized = aspect.strip().title()
            if normalized:
                aspect_counts[normalized] = aspect_counts.get(normalized, 0) + 1

    sorted_aspects = dict(sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True))

    all_pos_points = [p for r in analysis_results for p in r.get("positive_points", []) if p]
    all_neg_points = [n for r in analysis_results for n in r.get("negative_points", []) if n]

    return {
        "total": total,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "positive_pct": round((pos / total) * 100, 1),
        "negative_pct": round((neg / total) * 100, 1),
        "neutral_pct": round((neu / total) * 100, 1),
        "aspect_counts": sorted_aspects,
        "top_positive_points": list(dict.fromkeys(all_pos_points))[:15],
        "top_negative_points": list(dict.fromkeys(all_neg_points))[:15],
    }


def results_to_dataframe(analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert analysis result list to displayable Pandas DataFrame."""
    rows = []
    for r in analysis_results:
        rows.append({
            "Review ID": r["review_id"],
            "Review Text": r["review_text"],
            "Sentiment": r["sentiment"].capitalize(),
            "Explanation": r["explanation"],
            "Aspects / Topics": ", ".join(r.get("aspects", [])),
            "Positive Points": "; ".join(r.get("positive_points", [])),
            "Complaints": "; ".join(r.get("negative_points", [])),
        })
    return pd.DataFrame(rows)


def results_to_csv_bytes(analysis_results: List[Dict[str, Any]]) -> bytes:
    """Format review analysis into UTF-8 CSV bytes for download."""
    df = results_to_dataframe(analysis_results)
    return df.to_csv(index=False).encode("utf-8")


def results_to_json_bytes(analysis_results: List[Dict[str, Any]]) -> bytes:
    """Format review analysis into formatted JSON bytes for download."""
    return json.dumps(analysis_results, indent=2).encode("utf-8")


def summary_to_json_bytes(summary: Dict[str, Any]) -> bytes:
    """Format executive summary into formatted JSON bytes for download."""
    return json.dumps(summary, indent=2).encode("utf-8")
