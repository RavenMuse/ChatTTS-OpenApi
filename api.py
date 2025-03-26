import argparse
import os
import sys
import wave
from io import BufferedWriter, BytesIO
from typing import Optional, Dict

import av
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ChatTTS
from tools.audio import float_to_int16
from tools.logger import get_logger

now_dir = os.getcwd()
sys.path.append(now_dir)
logger = get_logger("Command")

# man 7869 2222 6653 women 4099 5099
voice_mapping = {
    "alloy": "4099",
    "echo": "2222",
    "fable": "6653",
    "onyx": "7869",
    "nova": "5099",
    "shimmer": "4099",
}
video_format_dict: Dict[str, str] = {
    "m4a": "mp4",
}

audio_format_dict: Dict[str, str] = {
    "ogg": "libvorbis",
    "mp4": "aac",
}

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global chat
    env = os.getenv("TTS_DEVICE")
    device = None
    if env in ["npu", "cpu"]:
        device = torch.device(env)

    chat = ChatTTS.Chat(get_logger("ChatTTS"))
    logger.info("Initializing ChatTTS...")
    if chat.load(device=device, source="huggingface"):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)


class ChatTTSParams(BaseModel):
    input: str
    model: str = "chat_tts"
    voice: str = "alloy"
    temperature: Optional[float] = 0.3
    top_P: Optional[float] = 0.7
    top_K: Optional[int] = 20
    prompt: Optional[str] = "[oral_2][laugh_0][break_6]"


def speech_formatting(wav: np.ndarray):
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(24000)  # Sample rate in Hz
        wf.writeframes(float_to_int16(wav))
    buf.seek(0, 0)
    buf2 = BytesIO()
    wav2(buf, buf2, "mp3")
    buf.seek(0, 0)
    return buf2


def wav2(i: BytesIO, o: BufferedWriter, format: str):
    """
    https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI/blob/412a9950a1e371a018c381d1bfb8579c4b0de329/infer/lib/audio.py#L20
    """
    inp = av.open(i, "r")
    format = video_format_dict.get(format, format)
    out = av.open(o, "w", format=format)
    format = audio_format_dict.get(format, format)

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


@app.post("/audio/speech")
async def generate_speech(params: ChatTTSParams):
    if not params.input:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    input_text = params.input
    logger.info("Text input: %s", str(input_text))

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=chat.sample_random_speaker(),
        top_P=params.top_P,
        top_K=params.top_K,
        temperature=params.temperature,
    )

    params_refine_text = ChatTTS.Chat.RefineTextParams(prompt=params.prompt)

    voice = voice_mapping.get(params.voice, "4099")
    torch.manual_seed(voice)

    logger.info("Start voice inference.")

    wav = chat.infer(
        text=input_text,
        params_infer_code=params_infer_code,
        params_refine_text=params_refine_text,
    )

    buffer = speech_formatting(wav)

    async def stream_audio():
        chunk_size = 1024 * 1024
        buffer.seek(0)
        while chunk := buffer.read(chunk_size):
            yield chunk

    response = StreamingResponse(
        stream_audio(),
        media_type="audio/mpeg",
    )
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
