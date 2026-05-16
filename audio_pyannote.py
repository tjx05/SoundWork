from pyannote.audio import Pipeline

YOUR_HF_TOKEN = "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"

def run_diarization(audio_path, hf_token):
    # 加载预训练模型 (需要在Hugging Face同意使用协议并获取Token)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # 进行分割
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments