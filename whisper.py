import whisper


def run_asr(audio_path, segments):
    model = whisper.load_model("base")  # 可根据显存选择 base, small, medium

    results = []
    for seg in segments:
        # 裁剪音频片段或直接让Whisper处理特定时间段
        # Whisper的transcribe函数支持指定时间轴
        audio_segment_result = model.transcribe(
            audio_path,
            initial_prompt="以下是会议录音",
            start_time=seg["start"],
            end_time=seg["end"]
        )

        results.append({
            "time": f"[{seg['start']:.2f} - {seg['end']:.2f}]",
            "speaker": seg["speaker"],
            "text": audio_segment_result["text"]
        })
    return results