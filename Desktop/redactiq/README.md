# RedactIQ - Smart PII Redaction Assistant

Hybrid PII detection and redaction system combining rule-based patterns, LLM inference (via vLLM), and anomaly detection. Optimized for Intel Xeon CPUs with AMX/BF16 support.

## Architecture

```
Input Text
    |
    v
[Rule-Based Detection] --> Regex for emails, phones, SSNs, credit cards, IPs
    |
    v
[LLM Detection (vLLM)] --> Context-aware: names, addresses, quasi-identifiers
    |
    v
[Hybrid Merger] ----------> Union / Intersection / Priority strategies
    |
    v
[Redaction Engine] -------> Mask (████) / Pseudonymize / Hash
    |
    v
[Anomaly Detection] ------> Isolation Forest on text embeddings
    |
    v
Output: Redacted text + Entity list + Anomaly flags + Audit log
```

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
│   │   ├── rule_engine.py       # Regex-based PII detection (SSN, email, CC, etc.)
│   │   ├── llm_detector.py      # vLLM-based contextual detection
│   │   └── hybrid.py            # Merge strategies (union/intersection/priority)
│   ├── redaction/
│   │   ├── engine.py            # Mask / Pseudonymize / Hash
│   │   └── pipeline.py          # End-to-end orchestration
│   ├── anomaly/
│   │   └── detector.py          # Isolation Forest / One-Class SVM / Autoencoder
│   ├── serving/
│   │   └── api.py               # FastAPI REST endpoints
│   ├── data/
│   │   └── generate.py          # Synthetic PII dataset generator
│   ├── utils/
│   │   ├── config.py            # YAML config loader
│   │   └── models.py            # Pydantic data models
│   └── ui/
│       └── app.py               # Gradio demo interface
├── scripts/
│   ├── setup.sh                 # Environment setup
│   ├── fine_tune.py             # LoRA fine-tuning
│   ├── train_anomaly.py         # Anomaly model training
│   └── evaluate.py              # Evaluation & benchmarking
├── tests/
│   ├── test_rule_engine.py
│   ├── test_redaction.py
│   └── test_pipeline.py
├── configs/
│   ├── default.yaml             # Main configuration
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

You should see all 43 tests pass. The tests cover:
- **Rule engine**: email, phone, SSN (context-aware), credit card (Luhn validated), IP, passport
- **Redaction engine**: mask, pseudonymize, hash modes + per-call mode override
- **Pipeline**: end-to-end redaction, batch processing, entity type filtering, text segmentation

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

**Health check:**
```bash
curl http://localhost:8000/health
```

#### Step 8 — Start the Gradio UI (optional)

```bash
pip install gradio    # if not already installed
python -m redactiq.ui.app
```

Opens at **http://localhost:7860**. Features:
- **Redact tab** — Paste text, choose mode (mask/pseudonymize/hash), toggle anomaly detection
- **Compare Modes tab** — See all three redaction modes side by side
- **Sample texts** — Pre-loaded examples to try instantly

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
- **Detection**: which regex patterns to enable, LLM confidence threshold
- **Merge strategy**: union / intersection / llm_priority / rule_priority
- **Anomaly**: algorithm (isolation_forest/one_class_svm/autoencoder), threshold
- **Redaction**: mode (mask/pseudonymize/hash), mask character
- **Serving**: host, port, rate limit, CORS origins

## Evaluation

```bash
# Generate test data first
python -m redactiq.data.generate

# Run evaluation (precision, recall, F1 per entity type + throughput)
python scripts/evaluate.py

# Run unit tests
pytest tests/ -v
```

## Redaction Modes

| Mode | Example Input | Example Output |
|------|--------------|----------------|
| **mask** | `john@test.com` | `██████████████` |
| **pseudonymize** | `john@test.com` | `sarah.jones@example.net` |
| **hash** | `john@test.com` | `[HASH:a3b2c1d4]` |

## Tech Stack

| Component         | Technology                              |
|-------------------|-----------------------------------------|
| LLM Inference     | vLLM (PagedAttention, continuous batch) |
| Base Model        | Qwen3-8B                                |
| Fine-Tuning       | LoRA via PEFT                           |
| Rule Engine       | Python regex with Luhn validation       |
| Anomaly Detection | scikit-learn (Isolation Forest)         |
| API               | FastAPI + Uvicorn                       |
| UI                | Gradio                                  |
| Monitoring        | Prometheus + Grafana                    |
| CPU Optimization  | Intel oneAPI, AMX/BF16, AVX-512         |
