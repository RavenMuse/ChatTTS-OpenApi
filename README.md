<div align="center">



# ChatTTS-OpenApi
A generative speech model for daily dialogue with standardized OpenAI API 
speech interface. 


</div>

## Introduction

ChatTTS is a text-to-speech model designed specifically for dialogue scenarios such as LLM assistant. 
OpenAI API is mainly used for AI application platforms to implement speech capabilities, such as Dify, Flowise, etc., as well as modular development.

### Supported Languages
- [x] English
- [x] Chinese


### Dataset & Model

- The main model is trained with Chinese and English audio data of 100,000+ hours.
- The open-source version on **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** is a 40,000 hours pre-trained model without SFT.


## Get Started
### Clone Repo
```bash
git clone https://github.com/RavenMuse/ChatTTS-OpenApi.git
cd ChatTTS-OpenApi
```
 
### Install requirements
#### 1. Install Directly
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Install from Uv
```bash
uv sync --upgrade
source .venv/bin/activate
```

### Quick Start
> Make sure you are under the project root directory when you execute these commands below.

#### 1. Launch WebUI
```bash
python examples/web/webui.py
```

#### 2. Infer by Command Line
> It will save audio to `./output_audio_n.mp3`

```bash
python examples/cmd/run.py "Your text 1." "Your text 2."
```

#### 3. Infer by API

```bash
python api.py --port 7006
```

### Basic Usage

```shell
curl http://localhost:7006/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model":"chat_tts",
    "input": "The quick brown fox jumped over the lazy dog.",
    "voice": "shimmer"
  }' \
  --output speech.mp3
```
