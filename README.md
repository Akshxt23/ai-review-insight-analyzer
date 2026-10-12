# AI Review Insight Analyzer

An end-to-end Natural Language Processing (NLP) web application for customer review intelligence, aspect-based sentiment analysis (ABSA), and executive decision support. Built with **Python**, **Streamlit**, **Plotly**, and multi-provider LLM support (**Google Gemini Free Tier**, **Groq Cloud Free Tier**, and **OpenAI**).

---

## 1. Project Overview & Objectives

Customer reviews contain vital signals regarding product strengths, failure points, and user expectations. However, manually reading hundreds of reviews is unfeasible, and traditional bag-of-words or naive sentiment classifiers fail to identify *what specific aspects* users are praising or criticizing.

**AI Review Insight Analyzer** addresses this by leveraging state-of-the-art Large Language Models (LLMs) to perform:
1. **Document-Level & Aspect-Level Sentiment Analysis**: Accurately distinguishing between positive, negative, and neutral feedback while attributing sentiment to specific product facets.
2. **Fine-Grained Aspect & Topic Extraction**: Unsupervised extraction of grounded topics (e.g., *Battery Life*, *Customer Support*, *Delivery Speed*).
3. **Evidence Extraction**: Separating tangible praise from explicit complaints.
4. **AI Executive Intelligence Synthesis**: Synthesizing macro-level trends and generating three prioritized, actionable recommendations grounded solely in real customer feedback.

---

## 2. Supported LLM Providers & Free Tier Options

The application supports multiple LLM providers, making it resilient to API quota depletion:

| Provider | Tier | Recommended Model | Pricing / Access | Official SDK |
| :--- | :--- | :--- | :--- | :--- |
| **Google Gemini** *(Default)* | **100% Free Tier** | `models/gemini-3.8-flash` | Free forever via Google AI Studio (No credit card required!) | `google-genai` |
| **Groq Cloud** | **Free Tier** | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | Free tier with fast inference | `groq` |
| **OpenAI** | Paid Tier | `gpt-4o-mini`, `gpt-4o` | Pay-as-you-go (Subject to 429 `insufficient_quota` if unpaid) | `openai` |

---

## 3. How to Obtain and Configure a Free API Key

### Recommended: Google Gemini (100% Free - No Credit Card Needed)
1. Visit **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2. Sign in with your standard Google account.
3. Click **"Create API key"** (choose or create any Google Cloud project).
4. Copy your generated key (starts with `AIza...`).
5. Configure the key using either method below:
   - **Method A (Persistent `.env`)**:
     Create/open `.env` in the project root:
     ```env
     GEMINI_API_KEY=AIzaSy...your_key_here
     ```
   - **Method B (Web UI)**:
     Launch the Streamlit app, select **Google Gemini (Free Tier)** in the sidebar, and paste your key into the masked input field.

### Alternative: Groq Cloud (Free Tier)
1. Go to **[Groq Console](https://console.groq.com/keys)**.
2. Sign up and click **"Create API Key"**.
3. Set in `.env`:
   ```env
   GROQ_API_KEY=gsk_...your_key_here
   ```

---

## 4. Application Architecture

```mermaid
flowchart TD
    User([Customer / Analyst]) -->|Pastes text or uploads CSV| UI[Streamlit Web App (app.py)]
    UI --> Validator[Input Validation & Review Chunking]
    
    subgraph CoreEngine [NLP Processing Engine (analyzer.py)]
        Validator --> Batcher[Batch Processor (Configurable Batch Size)]
        SysPrompt[prompts/system_prompt.txt] --> PromptBuilder[Prompt Formatter & Injection Guard]
        Batcher --> PromptBuilder
        PromptBuilder --> Router{Provider Router}
        Router -->|Free Tier| GeminiClient[Google Gemini SDK (google-genai)]
        Router -->|Free Tier| GroqClient[Groq Cloud SDK (groq)]
        Router -->|Paid Tier| OpenAIClient[OpenAI SDK (openai)]
    end
    
    GeminiClient <-->|Structured JSON| GeminiAPI[Google Gemini API]
    GroqClient <-->|Structured JSON| GroqAPI[Groq API]
    OpenAIClient <-->|Structured JSON| OpenAIAPI[OpenAI API]
    
    subgraph Validation [Schema Enforcement & Metrics]
        GeminiAPI --> PydanticVal[Pydantic v2 Schema Validator]
        GroqAPI --> PydanticVal
        OpenAIAPI --> PydanticVal
        PydanticVal --> Aggregator[Metrics & Frequency Aggregator]
    end
    
    Aggregator --> Dashboard[Streamlit Dashboard & Plotly Charts]
    Aggregator --> Exporter[Export Center (CSV / JSON)]
    Dashboard --> User
    Exporter --> User
```

---

## 5. How the Prompt System Works

### Externalized System Prompt (`prompts/system_prompt.txt`)
Instead of hardcoding prompts inside Python code, instructions are loaded from [prompts/system_prompt.txt](file:///C:/Users/Akshat/.gemini/antigravity/scratch/review-insight-analyzer/prompts/system_prompt.txt):
- **Untrusted Input Guarding**: Directs the LLM to treat review text strictly as passive data, mitigating prompt injection or adversarial jailbreaks embedded within reviews.
- **Standardized Aspect Normalization**: Instructs the model to use concise, standardized title-case labels (e.g., `Build Quality`, `Battery Life`).
- **Evidence-Based Reasoning**: Demands a concise rationale grounded in explicit review text, preventing hallucinated explanations.

### Batch Processing & Cost Optimization
- Reviews are split into configurable batches (default: 5 reviews per request, configured in `config.yaml`).
- Each review maintains its unique identifier (`REV-001`, `REV-002`, etc.) throughout the round-trip to guarantee one-to-one mapping.

### Schema Validation & Zero-Hallucination Policy
- All responses from the LLM are parsed and validated through **Pydantic** models (`BatchAnalysisResponse`, `ExecutiveSummaryModel`).
- If an API call fails or encounters an unrecoverable error, the system records and surfaces the error. It **never replaces failed API calls with fake classifications**.

---

## 6. Project Directory Structure

```text
review-insight-analyzer/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── analyzer.py                # NLP engine, multi-provider client, batching & validation
├── config.yaml                # Application configuration & guardrails
│
├── prompts/
│   └── system_prompt.txt      # Externalized LLM instructions & security guardrails
│
├── requirements.txt           # Python package dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Files excluded from git tracking
├── sample_reviews.csv         # 18 realistic test reviews across varied domains
├── test_engine.py             # Unit test suite (15 tests)
└── README.md                  # Comprehensive documentation and setup guide
```

---

## 7. Installation & Setup Instructions (Windows)

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on Windows.
- Windows PowerShell or Command Prompt.
- An API Key (Google Gemini free key recommended).

### Step 1: Navigate to the Project Directory
```powershell
cd C:\Users\Akshat\.gemini\antigravity\scratch\review-insight-analyzer
```

### Step 2: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and paste your key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 8. Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

Streamlit will output a local URL, typically:
```
Local URL: http://localhost:8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 9. Error Handling & Quota Troubleshooting

If you encounter:
- `OpenAI Error: 429 insufficient_quota`: Your OpenAI account has exhausted its paid credits.
  **Fix**: Simply switch the **LLM Provider** in the left sidebar to **Google Gemini (Free Tier)** and provide a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- `Google Gemini Error: 429 RESOURCE_EXHAUSTED`: You have exceeded the free tier rate limit (15 requests per minute).
  **Fix**: The app automatically retries with exponential backoff. Wait 30–60 seconds, or increase the batch size in settings to send fewer overall requests.

---

## 10. Example Input & Output

### Example Input Review
> *"The wireless noise-canceling headphones have exceptional sound clarity and deep bass. Battery easily lasts 30 hours, but the companion phone app crashes frequently on launch."*

### Example Structured Output (JSON)
```json
{
  "review_id": "REV-001",
  "review_text": "The wireless noise-canceling headphones have exceptional sound clarity and deep bass. Battery easily lasts 30 hours, but the companion phone app crashes frequently on launch.",
  "sentiment": "positive",
  "explanation": "Customer expresses overall high satisfaction with sound clarity and battery life, despite minor app stability frustrations.",
  "aspects": [
    "Sound Quality",
    "Battery Life",
    "Companion App"
  ],
  "positive_points": [
    "Exceptional sound clarity and deep bass",
    "Battery lasts 30 hours"
  ],
  "negative_points": [
    "Companion phone app crashes frequently on launch"
  ],
  "status": "success",
  "error_message": null
}
```

---

## 11. Security & Quality Assurance

- **Zero Secret Leakage**: API keys are excluded from git via `.gitignore`, masked in Streamlit inputs, and excluded from trace logs.
- **Prompt Injection Defense**: Review inputs are placed inside bounded JSON structures and delimited from system instructions.
- **Input Validation**: Rejects empty files, non-string data, missing columns, and reviews exceeding character safety thresholds.
- **No Mock or Fake Data Fallbacks**: The system strictly calls the configured LLM API. If an API key is missing or quota is exhausted, it transparently informs the user rather than faking outputs.
