# compute_emotion_bias.py
"""
使用CREMA-D数据集计算每种情感相对于中性语音的embedding偏移
"""

import torch
import sys
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.speaker_reco import SpeakerRecognizer
from config import config


def compute_emotion_bias():
    """
    计算每种情感相对于中性语音的平均embedding偏移
    """
    device = config.device
    
    # 加载模型
    print("加载模型...")
    recognizer = SpeakerRecognizer(
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.52
    )
    
    # 加载CREMA-D数据
    print("加载CREMA-D数据...")
    import pandas as pd
    df = pd.read_csv("Data/CREMA-D/processed/cremad_index.csv")
    
    # 按演员和情感分组
    actor_embeddings = defaultdict(lambda: defaultdict(list))
    
    print("提取embedding...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频"):
        actor = row['actor_id']
        emotion = row['emotion_name']
        audio_path = row['audio_path']
        
        if not os.path.exists(audio_path):
            continue
        
        try:
            # 提取embedding
            emb = recognizer._extract_embedding(audio_path)
            actor_embeddings[actor][emotion].append(emb)
        except Exception as e:
            print(f"处理失败: {audio_path}, {e}")
    
    # 计算每个演员的"中性"基准
    print("\n计算情感偏移...")
    emotion_offsets = defaultdict(list)
    
    for actor, emotions in actor_embeddings.items():
        if 'neutral' not in emotions or len(emotions['neutral']) == 0:
            continue
        
        # 计算该演员的中性平均embedding
        neutral_emb = torch.stack(emotions['neutral']).mean(dim=0)
        
        # 计算每种情感相对于中性的偏移
        for emotion, embs in emotions.items():
            if emotion == 'neutral':
                continue
            
            for emb in embs:
                offset = emb - neutral_emb
                emotion_offsets[emotion].append(offset)
    
    # 计算平均偏移
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    
    bias_dict = {}
    for emotion, offsets in emotion_offsets.items():
        if offsets:
            avg_offset = torch.stack(offsets).mean(dim=0)
            std_offset = torch.stack(offsets).std(dim=0)
            bias_dict[emotion] = {
                'bias': avg_offset,
                'std': std_offset,
                'sample_count': len(offsets)
            }
            print(f"\n{emotion}:")
            print(f"  样本数: {len(offsets)}")
            print(f"  偏移范数: {avg_offset.norm().item():.4f}")
            print(f"  偏移前10维: {avg_offset[:10].tolist()}")
    
    # 保存偏移量
    save_path = "speaker_checkpoints/emotion_bias.pth"
    torch.save(bias_dict, save_path)
    print(f"\n✅ 情感偏移已保存到: {save_path}")
    
    return bias_dict


if __name__ == "__main__":
    compute_emotion_bias()