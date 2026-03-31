# RedactIQ - Smart PII Redaction Assistant

Hybrid PII detection and redaction system combining rule-based patterns (21 regex detectors with checksum validation), LLM inference (Qwen3-8B via vLLM), and anomaly detection. Supports 12 global and 9 India-specific PII types. Optimized for Intel Xeon CPUs with AMX/BF16 support.

## Architecture

```
Input (Text / PDF / DOCX / XLSX / CSV / TXT)
    |
    v
[File Parser] --------------> Extract text from uploaded documents
    |
    v
[Rule-Based Detection] -----> 21 regex patterns (global + India PII)
    |                          Luhn (credit cards) & Verhoeff (Aadhaar) validation
    v
[LLM Detection (vLLM)] -----> Qwen3-8B: names, addresses, quasi-identifiers
    |
    v
[Hybrid Merger] ------------> Union / Intersection / Priority strategies
    |
    v
[Redaction Engine] ----------> Mask (****) / Pseudonymize / Hash
    |
    v
[Anomaly Detection] ---------> LLM-based or ML-based (Isolation Forest / SVM / Autoencoder)
    |
    v
[Analytics Dashboard] -------> 5 interactive Plotly charts
    |
    v
Output: Redacted file (same format) + Entity list + Anomaly flags + Audit log
```

## Supported PII Types

### Global (12 patterns)

| PII Type | Detection Method | Validation |
|----------|-----------------|------------|
| Email | Regex | Domain structure |
| Phone (US/International) | Regex | Format check |
| SSN | Regex + keyword context | Requires keyword (e.g., "SSN", "Social Security") |
| Credit Card | Regex | Luhn checksum |
| IP Address | Regex | Octet range (0-255) |
| Passport (US) | Regex + keyword context | Requires keyword |
| Date of Birth | Regex + keyword context | Requires keyword |
| Address (US) | Regex | Street number + name + suffix |
| Driver License (US) | Regex + keyword context | Requires keyword |
| Bank Account | Regex + keyword context | Requires keyword |
| Name | LLM-based | Qwen3-8B contextual detection |
| Organization | LLM-based | Qwen3-8B contextual detection |

### India-Specific (9 patterns)

| PII Type | Format | Validation |
|----------|--------|------------|
| Aadhaar Number | 12 digits (starts 2-9), with/without separators | Verhoeff checksum |
| PAN Card | `ABCPD1234E` (5 letters + 4 digits + 1 letter) | 4th char holder-type check |
| Indian Passport | `A1234567` (letter + 7 digits) | Keyword context required |
| IFSC Code | `SBIN0001234` (4 letters + 0 + 6 alphanumeric) | 5th character must be `0` |
| Indian Phone/Mobile | `+91`/`0` prefix or 10 digits starting 6-9 | Digit range validation |
| Voter ID (EPIC) | `ABC1234567` (3 letters + 7 digits) | Keyword context required |
| Indian Driving License | `DL-0420110012345` (state code + digits) | Keyword context required |
| GSTIN | `22ABCDE1234F1Z5` (2 digits + PAN + 1 + Z + 1) | Embedded PAN + Z-position check |
| UPI ID | `user@bankhandle` | Whitelisted bank handles (paytm, oksbi, ybl, etc.) |

## Prerequisites

- **Python 3.10+** (tested with 3.11 and 3.14)
- **pip** (comes with Python)
- **Git** (to clone the repo)
- **Docker + Docker Compose** (optional, for containerized deployment)

## Project Structure

```
redactiq/
├── redactiq/                    # Main Python package
│   ├── __init__.py
│   ├── detection/
│   │   ├── rule_engine.py       # 21 regex patterns + Luhn/Verhoeff validation
│   │   ├── llm_detector.py      # Qwen3-8B contextual detection via vLLM
│   │   └── hybrid.py            # Merge strategies (union/intersection/priority)
│   ├── redaction/
│   │   ├── engine.py            # Mask / Pseudonymize / Hash modes
│   │   └── pipeline.py          # End-to-end orchestration
│   ├── anomaly/
│   │   └── detector.py          # LLM-based + ML-based anomaly detection
│   ├── serving/
│   │   └── api.py               # FastAPI REST endpoints + rate limiting
│   ├── data/
│   │   └── generate.py          # Synthetic PII dataset generator
│   ├── utils/
│   │   ├── config.py            # YAML config loader
│   │   ├── models.py            # Pydantic data models (21 entity types)
│   │   ├── file_parser.py       # PDF/DOCX/XLSX/CSV/TXT text extraction
│   │   └── file_writer.py       # Same-format redacted file output
│   └── ui/
│       └── app.py               # Gradio UI (5 tabs + analytics dashboard)
├── scripts/
│   ├── setup.sh                 # Environment setup
│   ├── fine_tune.py             # LoRA fine-tuning for Qwen3-8B
│   ├── train_anomaly.py         # Anomaly model training
│   └── evaluate.py              # Evaluation & benchmarking
├── tests/
│   ├── test_rule_engine.py      # 53 tests (global + India PII patterns)
│   ├── test_redaction.py        # 9 tests (mask/pseudonymize/hash modes)
│   └── test_pipeline.py         # 12 tests (pipeline, batch, filtering)
├── configs/
│   ├── default.yaml             # Main configuration (21 patterns)
│   └── prometheus.yml           # Monitoring config
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Detailed Setup & Run Guide

### Option A: Local Setup (Recommended for Development)

#### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd redactiq
```

#### Step 2 — Create a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Git Bash)
python -m venv venv
source venv/Scripts/activate
```

#### Step 3 — Install dependencies

```bash
# All dependencies + dev tools
pip install -e ".[dev]"

# If you want the Gradio UI
pip install -e ".[dev,ui]"

# If you have Intel Xeon and want IPEX optimizations
pip install -e ".[dev,intel]"
```

> **Note:** Requires Python 3.10-3.12 for full `torch`/`vllm` support.

#### Step 4 — Generate synthetic training data

```bash
python -m redactiq.data.generate
```

This creates `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, and
`data/anomaly_baseline.jsonl` with realistic synthetic PII data.

#### Step 5 — Run the tests

```bash
pytest tests/ -v
```

You should see all **74 tests** pass. The tests cover:
- **Rule engine** (53 tests): email, phone, SSN, credit card (Luhn), IP, passport, deduplication, configuration, plus India-specific — Aadhaar (Verhoeff), PAN, Indian passport, IFSC, Indian phone, Voter ID, Indian DL, GSTIN, UPI
- **Redaction engine** (9 tests): mask, pseudonymize, hash modes + per-call mode override + length preservation
- **Pipeline** (12 tests): end-to-end redaction, batch processing, entity type filtering, text segmentation

#### Step 6 — Start the API server

```bash
python -m redactiq.serving.api
```

The server starts at **http://localhost:8000**. It runs in rules-only mode by default
(no LLM model needed).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/redact` | POST | Redact a single document |
| `/redact/batch` | POST | Redact multiple documents |
| `/redact/file` | POST | Redact an uploaded file (PDF/DOCX/XLSX/CSV/TXT) |
| `/metrics` | GET | Prometheus metrics |

#### Step 7 — Try the API

**Single document redaction:**
```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith email: john@test.com, SSN: 123-45-6789", "mode": "mask"}'
```

**Batch redaction:**
```bash
curl -X POST http://localhost:8000/redact/batch \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Contact: jane@email.com, phone (555) 123-4567", "Credit card: 4532-0151-1283-0366"], "mode": "hash"}'
```

**Filter to specific entity types only:**
```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "Email john@test.com and IP 192.168.1.100", "mode": "mask", "entity_types": ["EMAIL"]}'
```

**India-specific PII:**
```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "Aadhaar: 9876 5432 1012, PAN: ABCPD1234E, UPI: user@paytm", "mode": "mask"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

#### Step 8 — Start the Gradio UI (optional)

```bash
pip install gradio    # if not already installed
python -m redactiq.ui.app
```

Opens at **http://localhost:7860** with five tabs:

| Tab | Features |
|-----|----------|
| **Redact Text** | Paste text, choose mode (mask/pseudonymize/hash), toggle anomaly detection |
| **Compare Modes** | See all three redaction modes side by side |
| **File Redaction** | Upload PDF/DOCX/XLSX/CSV/TXT, download redacted file in the same format |
| **Analytics** | 5 interactive Plotly charts — entity distribution, confidence histogram, detection source breakdown, risk heatmap, anomaly flags |
| **About** | Supported PII types, architecture overview, tech stack |

---

### Option B: Docker Deployment

#### Step 1 — Build and start all services

```bash
docker compose up -d
```

This starts four containers:

| Service | URL | Description |
|---------|-----|-------------|
| **api** | http://localhost:8000 | FastAPI redaction server |
| **ui** | http://localhost:7860 | Gradio web interface |
| **prometheus** | http://localhost:9091 | Metrics collection |
| **grafana** | http://localhost:3000 | Dashboards (login: `admin` / `redactiq`) |

#### Step 2 — Check the services

```bash
# Verify API is running
curl http://localhost:8000/health

# Check container status
docker compose ps

# View logs
docker compose logs api
docker compose logs ui
```

#### Step 3 — Stop the services

```bash
docker compose down
```

---

### Option C: Quick Script Setup (Linux/macOS)

```bash
cd redactiq
bash scripts/setup.sh
```

This automatically creates a virtual environment, installs all dependencies, generates
synthetic data, and sets Intel CPU environment variables.

---

## File Redaction

RedactIQ supports uploading and redacting documents while preserving the original format:

| Input Format | Output Format | Preservation |
|-------------|---------------|--------------|
| PDF | PDF | Page-by-page reconstruction via fpdf2 |
| DOCX | DOCX | Run formatting, paragraphs, tables preserved via python-docx |
| XLSX | XLSX | Cell-level replacement, formatting preserved via openpyxl |
| CSV | CSV | Row/column structure preserved |
| TXT | TXT | Plain text passthrough |

Upload a file in the **File Redaction** tab or via the `/redact/file` API endpoint. The redacted output maintains the same file type as the input.

---

## Analytics Dashboard

The Gradio UI includes an interactive analytics dashboard with five Plotly charts:

1. **Entity Distribution** — Bar chart showing counts of each detected PII type
2. **Confidence Histogram** — Distribution of detection confidence scores
3. **Detection Source** — Donut chart breaking down rule-based vs. LLM vs. hybrid detections
4. **Risk Heatmap** — Entity types mapped against confidence bands (low/medium/high)
5. **Anomaly Flags** — Horizontal bar chart of anomaly scores, color-coded by severity (red > 0.8, orange > 0.5, blue otherwise)

---

## Full Pipeline (with LLM + Anomaly Detection)

For the complete hybrid detection (rules + LLM + anomaly detection):

```bash
# 1. Generate training data
python -m redactiq.data.generate

# 2. Fine-tune the LLM with LoRA (requires ~16GB RAM for 8B model)
python scripts/fine_tune.py --model Qwen/Qwen3-8B

# 3. Train the anomaly detector
python scripts/train_anomaly.py

# 4. Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --device cpu --dtype bfloat16

# 5. Start RedactIQ API (another terminal)
REDACTIQ_DEVICE=cpu python -m redactiq.serving.api

# 6. Start the UI (another terminal)
python -m redactiq.ui.app
```

## Intel CPU Optimizations

Set these environment variables for best performance on Intel Xeon:

```bash
export OMP_NUM_THREADS=$(nproc)
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
```

For INT8 quantization with Intel Extension for PyTorch:
```bash
pip install intel-extension-for-pytorch
export REDACTIQ_QUANTIZATION=int8
```

## Configuration

Edit `configs/default.yaml` to customize:
- **Model**: name, quantization (bfloat16/int8/float32), LoRA adapter path
- **Detection**: which of the 21 regex patterns to enable, LLM confidence threshold
- **Merge strategy**: union / intersection / llm_priority / rule_priority
- **Anomaly**: mode (llm/isolation_forest/one_class_svm/autoencoder), threshold
- **Redaction**: mode (mask/pseudonymize/hash), mask character
- **Serving**: host, port, rate limit, CORS origins

## Evaluation

```bash
# Generate test data first
python -m redactiq.data.generate

# Run evaluation (precision, recall, F1 per entity type + throughput)
python scripts/evaluate.py

# Run unit tests (74 tests)
pytest tests/ -v
```

## Redaction Modes

| Mode | Example Input | Example Output |
|------|--------------|----------------|
| **mask** | `john@test.com` | `**************` |
| **pseudonymize** | `john@test.com` | `sarah.jones@example.net` |
| **hash** | `john@test.com` | `[HASH:a3b2c1d4]` |

## Tech Stack

| Component         | Technology                                       |
|-------------------|--------------------------------------------------|
| LLM Inference     | vLLM (PagedAttention, continuous batch)           |
| Base Model        | Qwen3-8B                                         |
| Fine-Tuning       | LoRA via PEFT                                    |
| Rule Engine       | Python regex + Luhn + Verhoeff checksum          |
| Anomaly Detection | LLM-based + scikit-learn (Isolation Forest/SVM)  |
| File Parsing      | PyPDF2, python-docx, openpyxl                    |
| File Writing      | fpdf2, python-docx, openpyxl                     |
| API               | FastAPI + Uvicorn                                |
| UI                | Gradio + Plotly (interactive charts)             |
| Monitoring        | Prometheus + Grafana                             |
| CPU Optimization  | Intel oneAPI, AMX/BF16, AVX-512                  |

## Performance Optimizations

RedactIQ includes several performance optimizations for production-grade throughput:

| Optimization | Where | Impact |
|-------------|-------|--------|
| **Single-pass string building** | `engine.py` `redact()` | O(T) instead of O(N*T) — 5-20x faster for multi-entity documents |
| **Detect once, redact thrice** | `pipeline.py` `detect_and_redact_multi()` | Compare Modes tab runs detection 1x instead of 3x |
| **Lightweight CSV cell processing** | `pipeline.py` `redact_cell()` | Skips Pydantic/logging/timing per cell — 10-20% faster for large CSVs |
| **Concurrent batch processing** | `pipeline.py` `process_batch()` | ThreadPoolExecutor when LLM enabled — near-Nx speedup for batch API |
| **HTTP connection pooling** | `llm_detector.py` | Persistent `httpx.Client` reuses TCP connections to LLM API |
| **O(1) confidence scoring** | `rule_engine.py` `_SCORERS` dispatch dict | Replaces 21-step if/elif chain with dict lookup |
| **Cached `.lower()` in LLM parser** | `llm_detector.py` `_parse_output()` | Computes `original_text.lower()` once instead of N times |
| **Precompiled regex patterns** | `rule_engine.py` `_PATTERNS` | All 21 patterns compiled at module load, not per-call |
