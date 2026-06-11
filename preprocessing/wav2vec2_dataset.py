import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

class Wav2vec2MultiTaskDataset(Dataset):
    """
    四任务全能版: Wav2vec 2.0 专用数据集
    功能: 直接输出 1D 原始波形 (16kHz) 以及 4个子任务标签
    输出: waveform, emotion, gender, age, intensity
    """
    def __init__(self, csv_file, audio_dir, target_sr=16000, max_seconds=3.0):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到数据集文件: {csv_file}")
            
        self.data_df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        # 计算最大采样点数：比如 16000 * 3秒 = 48000 个点
        self.max_samples = int(target_sr * max_seconds)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        raw_path = row['audio_path']
        
        # 为了防范不同系统生成的 CSV 路径不一致，统一提取文件名并拼接正确的基础路径
        file_name = raw_path.replace('\\', '/').split('/')[-1]
        real_path = os.path.join(self.audio_dir, file_name)
        
        try:
            waveform, sample_rate = torchaudio.load(real_path)
            
            # 1. 统一转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 2. 严格重采样到 16000 Hz (Wav2vec 2.0 的硬性要求)
            if sample_rate != self.target_sr:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)
                
            # 去掉通道维度，变成纯 1D 张量 (1, Samples) -> (Samples,)
            waveform = waveform.squeeze(0)
            
        except Exception as e:
            print(f"读取音频失败: {real_path}, 错误: {e}")
            # 如果音频读取失败，返回全零张量，防止程序崩溃
            waveform = torch.zeros(self.max_samples)
            # 注意异常返回也要带上四个标签
            return waveform, int(row['emotion']), int(row['gender']), int(row['age']), int(row['intensity'])

        # 3. 长度对齐 (一维波形的补零和截断)
        if waveform.shape[0] > self.max_samples:
            # 音频过长：直接截断尾部
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            # 音频过短：在尾部补零
            pad_length = self.max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # 4. 返回处理好的波形张量和四个多任务标签
        return (
            waveform, 
            int(row['emotion']), 
            int(row['gender']), 
            int(row['age']), 
            int(row['intensity']) # ✨ 新增强度标签返回
        )
