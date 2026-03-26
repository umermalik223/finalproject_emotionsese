# EmotionSense AI

> A multimodal emotion detection and therapeutic support system that analyzes facial expressions, speech patterns, and text sentiment to deliver empathetic, evidence-based mental health responses.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Final%20Year%20Project-orange?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Models & AI Pipeline](#models--ai-pipeline)
- [Emotion Fusion Engine](#emotion-fusion-engine)
- [Safety & Crisis Detection](#safety--crisis-detection)
- [Response Format](#response-format)
- [Performance & Monitoring](#performance--monitoring)
- [Frontend Interface](#frontend-interface)
- [Session Management](#session-management)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

EmotionSense AI is a final-year university project that explores the intersection of **affective computing**, **multimodal machine learning**, and **mental health technology**. The system simultaneously analyzes three human communication channels — face, voice, and text — then synthesizes these signals into a single fused emotional state and generates a therapeutically grounded response via GPT-4o-mini.

The core research questions driving the project:

- Can combining facial, vocal, and linguistic cues produce a more accurate emotion reading than any single modality alone?
- Can an AI system generate genuinely helpful, safe responses to users in emotional distress?
- How should crisis detection and intervention be handled responsibly in an automated system?

The result is a working proof-of-concept that answers all three questions with a production-quality implementation.

---

## Key Features

| Feature | Description |
|---|---|
| **Multimodal Analysis** | Simultaneous processing of video frames, audio files, and text |
| **Real-time Fusion** | Weighted fusion of 4 modalities into one high-confidence emotion reading |
| **Therapeutic Responses** | GPT-4o-mini generates evidence-based CBT/mindfulness guidance |
| **Crisis Detection** | Multi-layer keyword + risk scoring with automatic 988/911 resource routing |
| **Session Tracking** | Persistent emotional progression across a full conversation session |
| **Parallel Pipeline** | ThreadPoolExecutor-based stage execution with per-stage timeouts |
| **Performance Profiling** | Live CPU/memory monitoring, per-operation latency, bottleneck reports |
| **Graceful Degradation** | Missing modalities handled cleanly — system works with text only |
| **Web Interface** | Responsive HTML/CSS/JS frontend with WebRTC capture |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         User Input                                 │
│              Video Frame  │  Audio File  │  Text                   │
└──────────────────┬─────────────┬─────────────┬──────────────────────┘
                   │             │             │
                   ▼             ▼             ▼
          ┌────────────────────────────────────────┐
          │          Input Validator               │
          │  (format, quality, safety pre-checks)  │
          └──────────────────┬─────────────────────┘
                             │
              ───────────────┼─────────────────
             │               │                │
             ▼               ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
    │ Face Emotion │ │Speech Emotion│ │ Speech-to-Text   │
    │  (DeepFace)  │ │ (Wav2Vec2)   │ │   (Whisper)      │
    └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
           │                │                   │
           └────────────────┼───────────────────┘
                            │
              ──────────────┼──────────────────
             │                                 │
             ▼                                 ▼
    ┌──────────────────┐             ┌──────────────────┐
    │  Text Emotion    │             │  Mental State    │
    │  (Transformer)   │             │  (Transformer)   │
    └────────┬─────────┘             └────────┬─────────┘
             │                                │
             └────────────────┬───────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Emotion Fusion     │
                   │  face 55% │ speech  │
                   │  38% │ text 4%  │   │
                   │  mental 3%          │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Safety Checker    │
                   │  crisis keywords +  │
                   │  risk scoring       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Therapeutic Response│
                   │   GPT-4o-mini       │
                   │  (async, retry x5)  │
                   └──────────┬──────────┘
                              │
                              ▼
                        JSON Response
```

---

## Tech Stack

### Backend
| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (ASGI) |
| Deep Learning | PyTorch 2.0+, Hugging Face Transformers 4.30+ |
| Facial Analysis | DeepFace 0.0.79 |
| Speech Recognition | Wav2Vec2 (superb/wav2vec2-base-superb-er) |
| Transcription | Whisper (local, transformers-based) |
| LLM Responses | OpenAI GPT-4o-mini (async) |
| Audio Processing | Librosa, SoundDevice, SoundFile, Scipy |
| Video Processing | OpenCV 4.8+ |
| Logging | Loguru 0.7+ |
| Monitoring | psutil 5.9+ |

### Frontend
| Layer | Technology |
|---|---|
| UI | HTML5 + CSS3 + Vanilla JavaScript |
| Media Capture | WebRTC (getUserMedia API) |
| Communication | Fetch API (REST) |

---

## Project Structure

```
finalproject_emotionsese/
│
├── main.py                          # Entry point: API server + UI launcher
├── config.py                        # Configuration manager (env > json > defaults)
├── config.json                      # User config overrides
├── requirements.txt                 # All Python dependencies
│
├── core/
│   ├── orchestrator.py              # EmotionSenseOrchestrator — main controller
│   ├── pipeline_manager.py          # Parallel stage execution engine
│   └── emotion_fusion.py            # Weighted multimodal fusion engine
│
├── models/
│   ├── face_emotion.py              # DeepFace facial emotion detection
│   ├── speech_emotion.py            # Wav2Vec2 speech emotion recognition
│   ├── speech_to_text.py            # Whisper speech transcription
│   ├── text_emotion_analyzer.py     # Transformer-based text sentiment
│   ├── mental_state_analyzer.py     # Clinical mental state classification
│   ├── therapeutic_response_generator.py  # GPT-4o-mini async responder
│   └── therapeutic_fallbacks.py     # Emergency fallback response bank
│
├── utils/
│   ├── input_validator.py           # Input validation (video/audio/text)
│   ├── safety_checker.py            # Crisis detection and risk scoring
│   ├── logger.py                    # Multi-file logging setup
│   └── profiler.py                  # Performance metrics collector
│
├── ui/
│   ├── index.html                   # Main multimodal analysis interface
│   ├── style.css                    # Responsive styling
│   ├── script.js                    # Frontend logic + WebRTC capture
│   ├── sessions.html                # Session history viewer
│   └── about.html                   # Project information page
│
├── textmodels/
│   └── mental/                      # Mental state model files
│
└── voicemodels/                     # Not committed — see Installation
    ├── whisper/                     # Whisper STT model weights
    └── text_emotion/                # Text emotion classifier weights
```

> **Note:** Model folders (`voicemodels/`, large weight files) are excluded from this repository due to GitHub's 1 GB file size limit. See [Installation](#installation) for download instructions.

---

## Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **FFmpeg** (required by Librosa for audio decoding)
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **OpenAI API Key** — required for therapeutic response generation
- A webcam and microphone for real-time multimodal capture via the web UI

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/umermalik223/finalproject_emotionsese.git
cd finalproject_emotionsese.git
```

**2. Create and activate a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> If you have a GPU, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org) before running the above command for significantly faster inference.

**4. Download model weights**

The following models must be downloaded separately:

| Model | Target Directory | How to Download |
|---|---|---|
| Whisper (base) | `voicemodels/whisper/` | See snippet below |
| Text Emotion Classifier | `voicemodels/text_emotion/` | Hugging Face fine-tuned model |
| Mental State Classifier | `textmodels/mental/` | Add model weight files |

Download Whisper via Python:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model.save_pretrained("voicemodels/whisper/")
processor.save_pretrained("voicemodels/whisper/")
```

DeepFace and Wav2Vec2 model weights are downloaded automatically on first run.

### Configuration

**Option 1 — `.env` file (recommended)**

Create a `.env` file in the project root:

```env
EMOTIONSENSE_OPENAI_API_KEY=sk-your-openai-api-key-here
EMOTIONSENSE_API_PORT=8003
EMOTIONSENSE_LOG_LEVEL=INFO
```

**Option 2 — `config.json`**

```json
{
    "openai_api_key": "sk-your-openai-api-key-here",
    "api_port": 8003,
    "log_level": "INFO"
}
```

Configuration precedence: **environment variables > config.json > config.py defaults**

**Key configuration options:**

| Key | Default | Description |
|---|---|---|
| `openai_api_key` | `""` | OpenAI API key (required) |
| `gpt_model` | `gpt-4o-mini` | GPT model to use |
| `api_port` | `8003` | FastAPI server port |
| `ui_port` | `8501` | Web UI port |
| `max_workers` | `4` | Thread pool workers for parallel pipeline |
| `safety_enabled` | `true` | Enable crisis detection |
| `intervention_threshold` | `medium` | Risk level that triggers crisis resources |

### Running the Application

```bash
# API server only (port 8003)
python main.py api

# Web UI only (port 8501)
python main.py ui

# Both simultaneously
python main.py both
```

Once running:
- **Web Interface:** `http://localhost:8501`
- **Swagger API Docs:** `http://localhost:8003/docs`
- **Health Check:** `http://localhost:8003/health`

---

## API Reference

All endpoints are served at `http://localhost:8003`.

### `GET /health`

Returns system health and model initialization status.

```json
{
    "status": "healthy",
    "models": {
        "face_emotion": true,
        "speech_emotion": true,
        "speech_to_text": true,
        "text_emotion": true,
        "mental_state": true,
        "therapeutic": true
    }
}
```

### `POST /analyze/multimodal`

Full multimodal analysis — video + audio + transcription-derived text.

**Form Data:**

| Field | Type | Required | Description |
|---|---|---|---|
| `video_frame` | file (image) | No | JPEG/PNG video frame |
| `audio_file` | file (audio) | No | WAV/MP3 audio clip |
| `session_id` | string | No | Auto-generated if omitted |

### `POST /analyze/audio`

Speech emotion + transcription + text emotion + therapeutic response.

### `POST /analyze/video`

Facial emotion detection only.

### `POST /analyze/text`

```json
{ "text": "I have been feeling anxious lately", "session_id": "optional" }
```

Text emotion + mental state + therapeutic response.

### `GET /session/{session_id}/summary`

Emotional progression timeline and aggregate metrics for a session.

### `GET /performance/metrics`

Per-operation latency stats (min/max/avg/p95), success rates, CPU and memory usage, bottleneck analysis.

---

## Models & AI Pipeline

### Stage 1 — Parallel Execution

| Model | Input | Timeout |
|---|---|---|
| **DeepFace** | Video frame | 5s |
| **Wav2Vec2** (superb-er) | Audio waveform | 8s |
| **Whisper** (local) | Audio waveform | 20s |

**DeepFace** uses CLAHE contrast enhancement on the detected face crop. When multiple faces are present, the largest is selected.

**Wav2Vec2** normalizes and zero-pads audio to 3 seconds minimum, then maps 4-class SUPERB labels to the project's 7-class schema.

**Whisper** runs fully locally from `voicemodels/whisper/` to avoid API latency. Silent audio is rejected via RMS quality gate.

### Stage 2 — Sequential (awaits Whisper transcript)

| Model | Input | Timeout |
|---|---|---|
| **Text Emotion Transformer** | Transcribed text | 3s |
| **Mental State Transformer** | Transcribed text | 3s |

### Stage 3 — Final Synthesis

| Step | Timeout |
|---|---|
| **Emotion Fusion** | 2s |
| **GPT-4o-mini** (async, 5 retries) | 25s |

---

## Emotion Fusion Engine

The fusion engine (`core/emotion_fusion.py`) combines outputs from all active modalities using weighted scoring.

**Default weights:**

| Modality | Weight | Rationale |
|---|---|---|
| Facial | 55% | Strongest involuntary cue; hardest to suppress |
| Speech | 38% | Prosody and tone carry strong emotional content |
| Text | 4% | Overlaps with speech transcript; lower independent value |
| Mental State | 3% | Narrow label set; useful supplementary signal |

**Fusion process:**
1. Normalize all emotion labels to 7 canonical categories: `happy`, `sad`, `angry`, `fear`, `neutral`, `surprise`, `disgust`
2. Multiply each modality's confidence by its weight
3. Sum weighted scores across modalities per emotion
4. Highest score is the fused emotion
5. **Coherence score** (0–1): proportion of modalities agreeing on the top emotion
6. **Reliability score**: penalized when fewer than 2 modalities are active

---

## Safety & Crisis Detection

The `SafetyChecker` runs on every request before therapeutic response generation.

### Detection Layers

**1. Keyword Analysis** — weighted scoring for critical terms (suicide, self-harm, violence). A single high-severity keyword immediately escalates risk to HIGH.

**2. Emotional Risk Factors** — high-confidence detections (>0.75) of `sad`, `angry`, `fear`, or `disgust` increment the risk score.

**3. Mental State Risk** — `"Requires Attention"` output from the mental state model increments the risk score.

**4. Coherence Analysis** — very low coherence may signal emotional dysregulation.

### Risk Levels

| Level | Threshold | Response |
|---|---|---|
| `low` | Score < 1.0 | Normal therapeutic response |
| `medium` | Score ≥ 1.0 | Therapeutic response + crisis resources |
| `high` | Score ≥ 2.0 or suicide keyword | Crisis resources only; GPT response skipped |

### Crisis Resources Included

- **988 Suicide & Crisis Lifeline** — call or text 988
- **Crisis Text Line** — text HOME to 741741
- **Emergency Services** — 911

---

## Response Format

```json
{
    "success": true,
    "session_id": "session_abc123",
    "timestamp": "2024-11-15T14:32:10.421Z",
    "processing_time": 3.84,

    "facial_emotion": {
        "emotion": "sad",
        "confidence": 0.85,
        "emotion_scores": { "sad": 0.85, "neutral": 0.10, "angry": 0.05 },
        "face_coordinates": { "x": 120, "y": 45, "w": 180, "h": 200 }
    },
    "speech_emotion": {
        "emotion": "sad",
        "confidence": 0.72
    },
    "speech_to_text": {
        "text": "I have been feeling really down lately",
        "confidence": 0.92,
        "audio_duration": 4.2
    },
    "text_emotion": {
        "emotion": "Sadness/Melancholy",
        "confidence": 0.88
    },
    "mental_state": {
        "mental_state": "Mild Concern",
        "confidence": 0.65,
        "indicators": ["low mood", "uncertainty"]
    },
    "fused_emotion": {
        "emotion": "sad",
        "confidence": 0.79,
        "coherence_score": 0.92,
        "reliability_score": 0.95
    },
    "therapeutic_response": {
        "acknowledgment_and_support": "It sounds like you are carrying something heavy right now...",
        "proven_calming_techniques": [
            "4-7-8 breathing: inhale for 4 counts, hold for 7, exhale for 8...",
            "Body scan: starting from your feet, notice and release tension...",
            "Grounding (5-4-3-2-1): name 5 things you can see, 4 you can touch..."
        ],
        "after_calming_suggestions": "When you feel ready, try journaling just three sentences...",
        "system_analysis": {
            "detected_emotion": "sad",
            "confidence": 0.79,
            "modalities_used": ["facial", "speech", "text"],
            "coherence": "high"
        }
    },
    "safety_status": {
        "safe": true,
        "severity_level": "low",
        "crisis_resources_included": false
    }
}
```

---

## Performance & Monitoring

### Parallel Pipeline Efficiency

Stage 1 tasks (face, speech, transcription) execute concurrently, cutting latency vs. sequential:

| Scenario | Sequential Estimate | Parallel (Actual) |
|---|---|---|
| All modalities active | ~33s | ~20s |
| Audio only | ~23s | ~20s |
| Text only | ~6s | ~6s |

### System Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| GPU | Not required | CUDA-capable (speeds up all transformer models) |
| Disk | 4 GB | 8 GB |

---

## Frontend Interface

The web UI at `http://localhost:8501` provides three pages:

- **Main Analysis** (`index.html`) — Capture video/audio via WebRTC or upload files; submit for analysis and view results inline
- **Session History** (`sessions.html`) — Browse emotional progression across all interactions in a session
- **About** (`about.html`) — Project background, methodology, and known limitations

Built with plain HTML/CSS/JavaScript — no frontend framework — to keep the code readable and the dependency footprint minimal.

---

## Session Management

Each `session_id` accumulates:
- Timestamps and emotion readings per interaction
- Safety flags triggered
- Dominant emotion and progression trend
- Per-request processing times

Data is persisted to `sessions/session_{id}.json` and accessible via the `/session/{id}/summary` endpoint or the Sessions page.

---

## Environment Variables

| Variable | Description |
|---|---|
| `EMOTIONSENSE_OPENAI_API_KEY` | OpenAI API key |
| `EMOTIONSENSE_API_PORT` | FastAPI port (default: 8003) |
| `EMOTIONSENSE_UI_PORT` | Web UI port (default: 8501) |
| `EMOTIONSENSE_LOG_LEVEL` | Logging verbosity (DEBUG / INFO / WARNING) |
| `EMOTIONSENSE_MAX_WORKERS` | Thread pool size for pipeline |
| `EMOTIONSENSE_SAFETY_ENABLED` | Enable safety checker (true/false) |
| `EMOTIONSENSE_GPT_MODEL` | OpenAI model name |

---

## Contributing

This project was submitted as a final-year university project and is not actively maintained. You are welcome to fork and build on it.

Potential areas for improvement:
- Fine-tuning Wav2Vec2 on a domain-specific emotion dataset
- Replacing file-based sessions with a PostgreSQL backend
- Containerizing with Docker + Docker Compose
- Adding user authentication for multi-user deployments
- Experimenting with alternative fusion weight configurations

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [DeepFace](https://github.com/serengil/deepface) by Sefik Ilhan Serengil
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [OpenAI](https://openai.com) — GPT-4o-mini
- [FastAPI](https://fastapi.tiangolo.com)
- [SUPERB Benchmark](https://superbbenchmark.org) — wav2vec2-base-superb-er
- [Librosa](https://librosa.org)
- [OpenCV](https://opencv.org)

---

> **Disclaimer:** EmotionSense AI is a research prototype and is **not** a licensed medical or mental health product. It must not be used as a substitute for professional psychological or psychiatric care. If you or someone you know is in crisis, please call or text **988** (Suicide & Crisis Lifeline).
