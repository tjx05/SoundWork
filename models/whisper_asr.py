import whisper

class WhisperASR:
    def __init__(self,model_size="base"):
        self.model=whisper.load_model(model_size)

    def transcribe(self,audio_path):
        result=self.model.transcribe(audio_path,fp16=False)
        return result["segments"]
    
# 测试代码
# if __name__ == "__main__":
#     asr = WhisperASR("base")
#     segments = asr.transcribe("Data/CREMA-D/raw/AudioWAV/1001_DFA_ANG_XX.wav")
    
#     for seg in segments:
#         print(f"[{seg['start']:.1f}s -> {seg['end']:.1f}s] {seg['text']}")