import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T

# 确保能导入根目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.wav2vec2_model import Wav2vec2MultiTaskModel

# 标签字典
EMOTION_DICT = {0: "愤怒", 1: "厌恶", 2: "恐惧", 3: "快乐", 4: "悲伤", 5: "中性"}
GENDER_DICT = {0: "男", 1: "女"}
AGE_DICT = {0: "青年", 1: "中年", 2: "老年"}

class Wav2vec2Recognizer:
    """
    基于 Wav2vec 2.0 的 情感/性别/年龄 多任务推理接口
    """
    def __init__(self, model_path="./emotion_checkpoints/best_wav2vec2_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"正在加载 Wav2vec 2.0 大模型到 {self.device}...")
        
        # 1. 实例化网络结构 (必须与训练时一致)
        # 注意：第一次运行推理也会检测一下 huggingface 缓存，但不会重新下载
        self.model = Wav2vec2MultiTaskModel(model_name="facebook/wav2vec2-base")
        
        # 2. 加载训练好的巅峰权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 预处理参数
        self.target_sr = 16000
        self.max_samples = int(self.target_sr * 3.0) # 3秒长度

    def preprocess_audio(self, audio_path):
        """
        预处理：重采样到 16kHz -> 单声道 -> 截断/补零到 3秒
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 重采样
        if sample_rate != self.target_sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        waveform = waveform.squeeze(0) # (Samples,)
        
        # 长度对齐
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            pad_length = self.max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
        return waveform

    def predict(self, audio_path):
        """
        端到端预测一条语音
        """
        # 预处理并增加 Batch 维度 -> (1, 48000)
        wave_tensor = self.preprocess_audio(audio_path).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_emo, out_gen, out_age = self.model(input_values=wave_tensor)
            
        # 获取最高概率索引
        pred_emo = torch.argmax(out_emo, dim=1).item()
        pred_gen = torch.argmax(out_gen, dim=1).item()
        pred_age = torch.argmax(out_age, dim=1).item()
        
        result = {
            "emotion": EMOTION_DICT.get(pred_emo, "未知"),
            "gender": GENDER_DICT.get(pred_gen, "未知"),
            "age": AGE_DICT.get(pred_age, "未知")
        }
        return result

if __name__ == "__main__":
    # ======== 推理测试 ========
    # 我们拿上次那条测试过的“厌恶”语音再测一次
    test_audio = "./data/raw/AudioWAV/1001_DFA_DIS_XX.wav" 
    
    try:
        recognizer = Wav2vec2Recognizer()
        print(f"\n正在通过 Wav2vec 2.0 识别音频: {test_audio}")
        
        res = recognizer.predict(test_audio)
        
        print("\n=== 大模型识别结果 ===")
        print(f"情绪: {res['emotion']}")
        print(f"性别: {res['gender']}")
        print(f"年龄: {res['age']}")
        print("======================")
        
    except Exception as e:
        print(f"推理测试报错: {e}")