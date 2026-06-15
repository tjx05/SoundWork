# find_mapping_by_wav2vec2_identify.py
"""
用 Wav2Vec2 预测的强度，找到最佳补偿系数
对比：VoxCeleb 模型 vs 你的模型
"""

import torch
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.speaker_reco import SpeakerRecognizer
from recognition.speaker_reco_vox import SpeakerRecognizerVox
from recognition.wav2vec2_reco import Wav2vec2Recognizer
from config import config


def test_baseline(recognizer, test_tasks):
    """测试无补偿 baseline"""
    correct = 0
    total = 0
    
    for task in tqdm(test_tasks, desc="Baseline"):
        audio_path = task['audio_path']
        expected_name = task['expected_name']
        
        emb = recognizer._extract_embedding(audio_path)
        
        best_name = None
        best_score = -1
        for name, data in recognizer.database.items():
            db_emb = data["embedding"]
            score = torch.dot(emb, db_emb) / (emb.norm() * db_emb.norm() + 1e-8)
            score = score.item()
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score > recognizer.threshold and best_name == expected_name:
            correct += 1
        total += 1
    
    return correct / total * 100 if total > 0 else 0


def test_mapping(lo_factor, md_factor, hi_factor, 
                 recognizer, emotion_bias, test_tasks, 
                 emotion_recognizer, base_strength=0.7):
    """测试一组强度因子映射的效果"""
    intensity_to_factor = {
        'LO': lo_factor,
        'MD': md_factor,
        'HI': hi_factor,
        'XX': 0.7
    }
    
    correct = 0
    total = 0
    
    for task in tqdm(test_tasks, desc="测试", leave=False):
        audio_path = task['audio_path']
        expected_name = task['expected_name']
        emotion = task['emotion']
        
        try:
            result = emotion_recognizer.predict(audio_path)
            intensity = result.get('intensity', 'MD')
        except:
            intensity = 'MD'
        
        factor = intensity_to_factor.get(intensity, 0.7)
        
        emb_original = recognizer._extract_embedding(audio_path)
        
        if emotion in emotion_bias:
            bias_data = emotion_bias[emotion]
            noise = bias_data['bias'] if isinstance(bias_data, dict) else bias_data
            
            adaptive_strength = base_strength * factor
            adaptive_strength = max(0.3, min(1.5, adaptive_strength))
            
            emb_enhanced = emb_original - adaptive_strength * noise
            emb_enhanced = emb_enhanced / (emb_enhanced.norm() + 1e-8)
        else:
            emb_enhanced = emb_original
        
        best_name = None
        best_score = -1
        for name, data in recognizer.database.items():
            db_emb = data["embedding"]
            score = torch.dot(emb_enhanced, db_emb) / (emb_enhanced.norm() * db_emb.norm() + 1e-8)
            score = score.item()
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score > recognizer.threshold and best_name == expected_name:
            correct += 1
        total += 1
    
    return correct / total * 100 if total > 0 else 0


def setup_recognizer_and_db(recognizer_class, model_path, threshold, db_path, df):
    """通用注册函数"""
    import shutil
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)
    
    recognizer = recognizer_class(
        model_path=model_path,
        threshold=threshold,
        auto_register_unseen=False,
        db_path=db_path
    )
    
    # 按演员分组，用中性语音注册
    actor_audios = defaultdict(list)
    for _, row in df.iterrows():
        actor = row['actor_id']
        emotion = row['emotion_name']
        audio_path = row['audio_path']
        if emotion == 'neutral' and os.path.exists(audio_path):
            actor_audios[actor].append(audio_path)
    
    for actor, audios in tqdm(actor_audios.items(), desc="注册"):
        if len(audios) >= 3:
            recognizer.enroll(f"Actor_{actor}", audios[:3])
    
    print(f"注册完成，共 {len(recognizer.database)} 人")
    return recognizer


def main():
    device = config.device
    print("=" * 70)
    print("对比测试：VoxCeleb 模型 vs 你的模型")
    print("用 Wav2Vec2 预测强度，寻找最佳补偿因子")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n加载 CREMA-D 数据...")
    df = pd.read_csv("Data/CREMA-D/processed/cremad_index.csv")
    
    # 准备测试任务
    test_tasks = []
    for _, row in df.iterrows():
        actor = row['actor_id']
        emotion = row['emotion_name']
        audio_path = row['audio_path']
        if emotion != 'neutral' and os.path.exists(audio_path):
            test_tasks.append({
                'actor': actor,
                'expected_name': f"Actor_{actor}",
                'emotion': emotion,
                'audio_path': audio_path
            })
    print(f"测试任务数: {len(test_tasks)}")
    
    # 2. 加载 Wav2Vec2
    print("\n加载 Wav2Vec2 识别器...")
    emotion_recognizer = Wav2vec2Recognizer(
        model_path="emotion_checkpoints/best_wav2vec2_model1.pth"
    )
    
    # ============================================================
    # 第一部分：测试 VoxCeleb 模型
    # ============================================================
    print("\n" + "=" * 70)
    print("第一部分：VoxCeleb 预训练模型")
    print("=" * 70)
    
    # 注册 Vox 模型
    recognizer_vox = setup_recognizer_and_db(
        recognizer_class=SpeakerRecognizerVox,
        model_path="speaker_checkpoints/pretrain.model",
        threshold=0.6,
        db_path="speaker_checkpoints/speaker_db_vox_test",
        df=df
    )
    
    # 测试 Vox 模型无补偿
    print("\n测试 VoxCeleb 模型无补偿 baseline...")
    vox_baseline = test_baseline(recognizer_vox, test_tasks)
    print(f"VoxCeleb 无补偿准确率: {vox_baseline:.2f}%")
    
    # ============================================================
    # 第二部分：测试你的模型
    # ============================================================
    print("\n" + "=" * 70)
    print("第二部分：你的 ECAPA-TDNN 模型")
    print("=" * 70)
    
    # 注册你的模型
    recognizer_your = setup_recognizer_and_db(
        recognizer_class=SpeakerRecognizer,
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.52,
        db_path="speaker_checkpoints/speaker_db_your_test",
        df=df
    )
    
    # 加载情感偏移
    print("\n加载情感偏移...")
    emotion_bias = torch.load("speaker_checkpoints/emotion_bias.pth", map_location='cpu')
    
    # 测试无补偿
    print("\n测试无补偿 baseline...")
    your_baseline = test_baseline(recognizer_your, test_tasks)
    print(f"你的模型无补偿准确率: {your_baseline:.2f}%")
    
    # 测试固定强度
    print("\n测试固定强度 (factor=0.7)...")
    fixed_acc = test_mapping(0.7, 0.7, 0.7, recognizer_your, emotion_bias, 
                              test_tasks, emotion_recognizer, base_strength=0.7)
    print(f"固定强度准确率: {fixed_acc:.2f}%")
    print(f"相比 baseline 提升: +{fixed_acc - your_baseline:.2f}%")
    
    # 网格搜索
    print("\n" + "=" * 70)
    print("网格搜索最佳强度因子（你的模型）")
    print("=" * 70)
    
    lo_factors = [0.2, 0.3, 0.5, 0.7]
    md_factors = [0.5, 0.7, 0.9]
    hi_factors = [0.9, 1.1]
    
    results = []
    total_combos = len(lo_factors) * len(md_factors) * len(hi_factors)
    combo_count = 0
    
    for lo in lo_factors:
        for md in md_factors:
            for hi in hi_factors:
                combo_count += 1
                print(f"\n[{combo_count}/{total_combos}] 测试 LO={lo}, MD={md}, HI={hi}")
                
                acc = test_mapping(lo, md, hi, recognizer_your, emotion_bias, 
                                   test_tasks, emotion_recognizer, base_strength=0.7)
                
                results.append({
                    'LO': lo, 'MD': md, 'HI': hi,
                    'accuracy': acc
                })
                print(f"  准确率: {acc:.2f}% (vs baseline: +{acc - your_baseline:.2f}%, vs fixed: +{acc - fixed_acc:.2f}%)")
    
    # 找最佳
    best = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "=" * 70)
    print("最终结果汇总")
    print("=" * 70)
    print(f"""
    【VoxCeleb 模型】
       无补偿: {vox_baseline:.1f}%
    
    【你的模型】
       无补偿: {your_baseline:.1f}%
       固定强度: {fixed_acc:.1f}%
       最佳自适应: {best['accuracy']:.1f}%
       最佳因子: LO={best['LO']}, MD={best['MD']}, HI={best['HI']}
       相比无补偿提升: +{best['accuracy'] - your_baseline:.1f}%
    """)
    
    # 保存结果
    import json
    save_path = "speaker_checkpoints/wav2vec2_intensity_mapping_identify.json"
    with open(save_path, "w") as f:
        json.dump({
            'vox_baseline': vox_baseline,
            'your_baseline': your_baseline,
            'fixed_accuracy': fixed_acc,
            'best_mapping': {'LO': best['LO'], 'MD': best['MD'], 'HI': best['HI']},
            'best_accuracy': best['accuracy'],
            'all_results': results
        }, f, indent=2)
    
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()