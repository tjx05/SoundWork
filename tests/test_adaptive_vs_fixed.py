# find_mapping_by_wav2vec2_identify.py
"""
用 Wav2Vec2 预测的强度，找到最佳补偿系数
使用 identify（固定数据库，不自动注册）
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
from recognition.wav2vec2_reco import Wav2vec2Recognizer
from config import config


def test_mapping(lo_factor, md_factor, hi_factor, 
                 recognizer, emotion_bias, test_tasks, 
                 emotion_recognizer, base_strength=0.7):
    """
    测试一组强度因子映射的效果
    使用 identify（固定数据库，不自动注册）
    """
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
        
        # 用 Wav2Vec2 预测强度
        try:
            result = emotion_recognizer.predict(audio_path)
            intensity = result.get('intensity', 'MD')
        except:
            intensity = 'MD'
        
        factor = intensity_to_factor.get(intensity, 0.7)
        
        # 提取embedding
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
        
        # ========== 使用 identify（不自动注册） ==========
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


def main():
    device = config.device
    print("=" * 70)
    print("用 Wav2Vec2 预测的强度，寻找最佳补偿因子")
    print("使用 identify（固定数据库，不自动注册）")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n加载 CREMA-D 数据...")
    df = pd.read_csv("Data/CREMA-D/processed/cremad_index.csv")
    
    # 2. 注册说话人
    print("\n注册阶段...")
    import shutil
    
    db_path = "speaker_checkpoints/speaker_db_test"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)
    
    recognizer = SpeakerRecognizer(
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.52,
        auto_register_unseen=False,  # 关闭自动注册
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
    
    # 3. 加载情感偏移
    print("\n加载情感偏移...")
    emotion_bias = torch.load("speaker_checkpoints/emotion_bias.pth", map_location='cpu')
    
    # 4. 加载 Wav2Vec2
    print("\n加载 Wav2Vec2 识别器...")
    emotion_recognizer = Wav2vec2Recognizer(
        model_path="emotion_checkpoints/best_wav2vec2_model1.pth"
    )
    
    # 5. 准备测试任务
    print("\n准备测试任务...")
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
    
    # 6. 先测试无补偿 baseline
    print("\n" + "=" * 70)
    print("测试无补偿 baseline")
    print("=" * 70)
    
    baseline_correct = 0
    for task in tqdm(test_tasks[:500], desc="Baseline"):
        audio_path = task['audio_path']
        expected_name = task['expected_name']
        
        emb_original = recognizer._extract_embedding(audio_path)
        
        best_name = None
        best_score = -1
        for name, data in recognizer.database.items():
            db_emb = data["embedding"]
            score = torch.dot(emb_original, db_emb) / (emb_original.norm() * db_emb.norm() + 1e-8)
            score = score.item()
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score > recognizer.threshold and best_name == expected_name:
            baseline_correct += 1
    
    baseline_acc = baseline_correct / len(test_tasks[:500]) * 100
    print(f"无补偿准确率: {baseline_acc:.2f}%")
    
    # 7. 测试固定强度
    print("\n" + "=" * 70)
    print("测试固定强度 (factor=0.7)")
    print("=" * 70)
    
    fixed_acc = test_mapping(0.7, 0.7, 0.7, recognizer, emotion_bias, 
                              test_tasks[:500], emotion_recognizer, base_strength=0.7)
    print(f"固定强度准确率: {fixed_acc:.2f}%")
    print(f"相比 baseline 提升: +{fixed_acc - baseline_acc:.2f}%")
    
    # 8. 网格搜索
    print("\n" + "=" * 70)
    print("网格搜索最佳强度因子")
    print("=" * 70)
    
    # 搜索范围
    lo_factors = [0.2, 0.3, 0.5, 0.7, 0.9]
    md_factors = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1]
    hi_factors = [0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    results = []
    test_tasks_subset = test_tasks[:500]  # 用500样本
    
    total_combos = len(lo_factors) * len(md_factors) * len(hi_factors)
    combo_count = 0
    
    for lo in lo_factors:
        for md in md_factors:
            for hi in hi_factors:
                combo_count += 1
                print(f"\n[{combo_count}/{total_combos}] 测试 LO={lo}, MD={md}, HI={hi}")
                
                acc = test_mapping(lo, md, hi, recognizer, emotion_bias, 
                                   test_tasks_subset, emotion_recognizer, base_strength=0.7)
                
                results.append({
                    'LO': lo, 'MD': md, 'HI': hi,
                    'accuracy': acc
                })
                print(f"  准确率: {acc:.2f}% (vs baseline: +{acc - baseline_acc:.2f}%, vs fixed: +{acc - fixed_acc:.2f}%)")
    
    # 9. 找最佳
    print("\n" + "=" * 70)
    print("搜索结果")
    print("=" * 70)
    
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n最佳强度因子:")
    print(f"  LO (预测为弱强度) → {best['LO']}")
    print(f"  MD (预测为中等强度) → {best['MD']}")
    print(f"  HI (预测为高强度) → {best['HI']}")
    print(f"  准确率: {best['accuracy']:.2f}%")
    print(f"  相比 baseline 提升: +{best['accuracy'] - baseline_acc:.2f}%")
    print(f"  相比固定强度提升: +{best['accuracy'] - fixed_acc:.2f}%")
    
    # 10. 对比 identify_or_register 的结果
    print("\n" + "=" * 70)
    print("重要对比")
    print("=" * 70)
    print(f"""
    identify 模式（固定数据库）:
       无补偿: {baseline_acc:.1f}%
       固定强度: {fixed_acc:.1f}%
       最佳自适应: {best['accuracy']:.1f}%
    
    identify_or_register 模式（动态注册）:
       之前测试结果: 约 44%
    
    差异原因:
       identify 模式: 只能用中性语音注册，情绪语音匹配差
       identify_or_register 模式: 情绪语音也能被注册，匹配更好
    """)
    
    # 11. 保存结果
    import json
    save_path = "speaker_checkpoints/wav2vec2_intensity_mapping_identify.json"
    with open(save_path, "w") as f:
        json.dump({
            'baseline_accuracy': baseline_acc,
            'fixed_accuracy': fixed_acc,
            'best_mapping': {'LO': best['LO'], 'MD': best['MD'], 'HI': best['HI']},
            'best_accuracy': best['accuracy'],
            'all_results': results
        }, f, indent=2)
    
    print(f"\n结果已保存到: {save_path}")


if __name__ == "__main__":
    main()