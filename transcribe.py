#!/usr/bin/env python3
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import os
import torch

# region agent log
def _agent_debug_log(hypothesis_id: str, message: str, data: dict | None = None) -> None:
    import json
    import time

    payload = {
        "sessionId": "44be33",
        "id": f"log_{int(time.time() * 1000)}_{hypothesis_id}",
        "timestamp": int(time.time() * 1000),
        "location": "transcribe.py",
        "message": message,
        "data": data or {},
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
    }

    # Try to write via the Docker volume mount (/audio -> project root on host)
    log_path = "/audio/.cursor/debug-44be33.log"

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Logging must never break the main flow
        pass
# endregion

print("🚀 Voxtral Mini Starting...")

MODEL_DIR = "/app/models"  # ← Твоя локальная папка

def main():
    if len(sys.argv) < 2:
        print("Usage: transcribe.py <audio_path>", file=sys.stderr)
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    print(f"📁 Processing: {wav_path}")
    
    if not wav_path.exists():
        print(f"❌ File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"📊 Size: {wav_path.stat().st_size} bytes")
    
    # Читаем аудио
    audio_data, sample_rate = sf.read(wav_path)
    print(f"🔊 Audio: {audio_data.shape}, {sample_rate}Hz")
    
    # Конвертируем в mono float32
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data.astype(np.float32)
    
    # region agent log
    _agent_debug_log(
        "H1",
        "Before importing mistral_common.audio",
        {
            "argv": sys.argv,
            "model_dir": str(MODEL_DIR),
        },
    )
    # endregion

    print("🤖 Loading Voxtral model...")

    # Правильный Voxtral API (Transformers + mistral-common Audio)
    from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
    from mistral_common.tokens.tokenizers.audio import Audio as MC_Audio

    # Мы работаем в полностью оффлайн-режиме, поэтому используем
    # локальную папку /app/models как "репозиторий" модели.
    local_model_dir = Path(MODEL_DIR)

    # region agent log
    _agent_debug_log(
        "H3",
        "Loading VoxtralRealtime model and processor from local directory",
        {
            "local_model_dir": str(local_model_dir),
            "dir_exists": local_model_dir.exists(),
            "dir_contents_sample": sorted(os.listdir(local_model_dir))[:20]
            if local_model_dir.exists()
            else None,
        },
    )
    # endregion

    processor = AutoProcessor.from_pretrained(
        local_model_dir,
        local_files_only=True,
    )
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        local_model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    # region agent log
    _agent_debug_log(
        "H3",
        "Loaded Transformers model and processor",
        {
            "model_dtype": str(model.dtype),
            "model_device": str(model.device),
        },
    )
    # endregion

    # Подготовка аудио через mistral-common Audio
    audio_obj = MC_Audio.from_file(str(wav_path), strict=False)
    audio_obj.resample(processor.feature_extractor.sampling_rate)

    inputs = processor(audio_obj.audio_array, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)

    print("🎤 Transcribing...")
    with torch.no_grad():
        outputs = model.generate(**inputs)

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    text = decoded[0] if decoded else ""
    print(f"\n📝 ТРАНСКРИПЦИЯ:\n{text}\n")
    
    if not text.strip():
        print("⚠️  Пустая транскрипция — проверь речь в файле")

if __name__ == "__main__":
    main()
