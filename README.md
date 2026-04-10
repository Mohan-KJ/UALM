# UALM
### Unified Audio Language Model · vMK.1.0

*A multi-modal acoustic intelligence framework for comprehensive audio scene understanding*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0a7c42?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-ffd21e?style=flat-square)](https://huggingface.co/)
[![CUDA Optional](https://img.shields.io/badge/CUDA-Optional-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

</div>
---
## Abstract

UALM is a unified inference pipeline that orchestrates four specialized neural architectures to perform end-to-end acoustic scene analysis from a single audio input. Rather than treating speech, sound, and sentiment as isolated problems, UALM employs an **AST-Gated routing mechanism** that dynamically allocates compute based on the acoustic nature of the signal — suppressing linguistic modules entirely when non-speech is dominant to eliminate hallucinated transcriptions.

The system achieves high-fidelity results across transcription, emotional mapping, environmental sound classification, and speaker diarization — all within a single, deployable Streamlit interface.

---

## Capabilities

| Module | Output |
|---|---|
|**Acoustic Gating** | Speech vs. non-speech signal routing |
|**Transcription** | Multilingual speech-to-text across 90+ languages |
|**Affective Analysis** | 6-class emotion classification with valence scoring |
|**Speaker Diarization** | Temporal speaker segmentation on a sub-second timeline |
|**Scene Captioning** | Deterministic, grounded audio captions — zero hallucinations |

---

## Architecture

UALM operates through a sequential, gated inference pipeline:

```
Audio Input (MP3 · WAV · FLAC · OGG · M4A)
        │
        ▼
┌───────────────────────┐
│  VAD — Signal Gate    │  Energy + ZCR + Spectral analysis
│  (Rule-Based)         │  Determines signal presence
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  AST Classifier       │  Audio Spectrogram Transformer
│  (MKJ007/ast)         │  Speech ↔ Non-Speech routing
└──────────┬────────────┘
           │
     ┌─────┴──────┐
     │ Speech?    │ Non-Speech?
     ▼            ▼
┌─────────┐  ┌──────────────┐
│ Whisper │  │ AST Top-5    │
│  (ASR)  │  │ Scene Labels │
└────┬────┘  └──────────────┘
     │
     ▼
┌───────────────────────┐
│  DistilBERT Emotion   │  6-class affective classification
│  (MKJ007/distilbert)  │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  MFCC Diarization     │  Cosine-similarity speaker clustering
│  (Unsupervised)       │  Sub-second timeline segmentation
└───────────────────────┘
```

**Key design principle:** Modules are invoked conditionally. If VAD detects no signal, or AST classifies input as non-speech, downstream linguistic modules are bypassed entirely. This prevents confabulation and reduces unnecessary compute.

---

## Model Stack

| Dimension | Model / Methodology | Objective |
|---|---|---|
| **ASR & Translation** | `openai/whisper-base` | Multilingual speech-to-text, 90+ languages |
| **Acoustic Routing** | `MKJ007/ast-finetuned` | Speech vs. non-speech classification |
| **Affective Computing** | `MKJ007/distilbert-emotion` | 6-class emotion + valence extraction |
| **Speaker Diarization** | MFCC + Cosine Clustering | Temporal speaker segmentation |
| **Signal Detection** | Energy + ZCR + Spectral Gating | Voice activity detection (VAD) |

---

## Supported Formats & Constraints

- **Input formats:** MP3 · WAV · FLAC · OGG · M4A
- **Maximum duration:** 30 seconds per file
- **Hardware:** CUDA GPU recommended; CPU fallback supported

---

## Quickstart

### Prerequisites

- Python `3.10` or later
- `pip` package manager
- CUDA-compatible GPU *(optional — CPU inference supported)*

### Installation

```bash
git clone https://github.com/your-username/UALM.git
cd UALM
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The interface will be available at `http://localhost:8501`.

---

## Project Structure

```
UALM/
├── app.py                  # Streamlit entrypoint
├── pipeline/
│   ├── vad.py              # Voice activity detection
│   ├── classifier.py       # AST acoustic gating
│   ├── transcriber.py      # Whisper ASR module
│   ├── sentiment.py        # DistilBERT emotion module
│   └── diarizer.py         # MFCC speaker diarization
├── utils/
│   └── audio_utils.py      # Format handling & preprocessing
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Whisper `large-v3` upgrade for production-grade ASR
- [ ] Pyannote-based neural diarization (replacing MFCC clustering)
- [ ] Real-time streaming inference via WebSocket
- [ ] REST API endpoint with OpenAPI schema
- [ ] Multi-file batch processing support
- [ ] Exportable JSON / SRT transcript outputs

---

## Citation

If you use UALM in your research or build upon this work, please cite:

```bibtex
@software{ualm2025,
  author    = {MK},
  title     = {UALM: Unified Audio Language Model},
  year      = {2025},
  version   = {vMK.1.0},
  url       = {https://github.com/your-username/UALM}
}
```

---

## License

Released under the **MIT License**. Free to use for commercial and private purposes. See [`LICENSE`](LICENSE) for full terms.

---

<div align="center">
<sub>Built with Whisper · AST · DistilBERT · Streamlit</sub>
</div>
