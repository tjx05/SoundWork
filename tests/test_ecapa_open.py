# tests/test_ecapa_open.py
import os
import sys
import json
import random
import numpy as np
import torch
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.speaker_reco import SpeakerRecognizer
from config import config

def set_seed(seed=42):
    """固定随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clear_speaker_db(db_path="speaker_checkpoints/speaker_db"):
    """清空说话人数据库，确保每次测试从零开始"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"✅ 已清空数据库: {db_path}")
    os.makedirs(db_path, exist_ok=True)

def load_open_test_data():
    """加载开放集测试数据"""
    with open("Data/TIMIT/open_test_info.json", "r") as f:
        all_data = json.load(f)
    
    # 按说话人分组
    speaker_audios = {}
    for item in all_data:
        spk = item['speaker_id']
        full_path = os.path.join("Data/TIMIT", item['filepath'])
        if os.path.exists(full_path):
            speaker_audios.setdefault(spk, []).append(full_path)
    
    return speaker_audios


def test_threshold(recognizer, registered_speakers, remaining_speakers, speaker_audios, threshold):
    """测试单个阈值的效果"""
    recognizer.threshold = threshold
    
    # 测试已注册人
    same_correct = 0
    same_total = 0
    same_scores = []
    
    for spk in registered_speakers:
        audios = speaker_audios[spk]
        if len(audios) >= 4:
            name, score = recognizer.identify(audios[3])
            same_scores.append(score)
            if name == spk:
                same_correct += 1
            same_total += 1
    
    same_rate = same_correct / same_total * 100 if same_total > 0 else 0
    
    # 测试未注册人
    cross_correct = 0
    cross_total = 0
    cross_scores = []
    
    for spk in remaining_speakers:
        audios = speaker_audios[spk]
        if audios:
            name, score = recognizer.identify(audios[0])
            cross_scores.append(score)
            if name is None:
                cross_correct += 1
            cross_total += 1
    
    cross_rate = cross_correct / cross_total * 100 if cross_total > 0 else 0
    
    return {
        'threshold': threshold,
        'same_rate': same_rate,
        'same_correct': same_correct,
        'same_total': same_total,
        'cross_rate': cross_rate,
        'cross_correct': cross_correct,
        'cross_total': cross_total,
        'same_scores': same_scores,
        'cross_scores': cross_scores
    }


def main():
    set_seed(42)
    clear_speaker_db()
    
    print("=" * 70)
    print("ECAPA-TDNN 开放集测试")
    print("=" * 70)
    
    # 1. 加载开放集数据
    print("\n【1】加载开放集测试数据...")
    speaker_audios = load_open_test_data()
    all_speakers = list(speaker_audios.keys())
    print(f"总新说话人数: {len(all_speakers)}")
    print(f"总音频数: {sum(len(v) for v in speaker_audios.values())}")
    
    # 2. 划分注册集和测试集
    print("\n【2】划分注册集和测试集...")
    registered_speakers = random.sample(all_speakers, 50)
    remaining_speakers = [s for s in all_speakers if s not in registered_speakers]
    
    print(f"注册说话人: {len(registered_speakers)} 人")
    print(f"未注册说话人: {len(remaining_speakers)} 人")
    
    # 3. 初始化识别器
    recognizer = SpeakerRecognizer(
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.6  # 初始阈值，后面会测试多个
    )
    
    # 4. 注册说话人（每人3句话）
    print("\n【3】注册说话人...")
    registered_count = 0
    for spk in registered_speakers:
        audios = speaker_audios[spk]
        if len(audios) >= 3:
            recognizer.enroll(spk, audios[:3])
            registered_count += 1
    print(f"注册完成，共 {registered_count} 人")
    
    # 5. 测试不同阈值
    print("\n【4】测试不同阈值...")
    print("-" * 70)
    print(f"{'阈值':<8} {'已注册识别率':<15} {'未注册拒绝率':<15} {'评价'}")
    print("-" * 70)
    
    results = []
    thresholds = [0.5, 0.52, 0.55, 0.58, 0.6, 0.62, 0.65, 0.7]
    
    for threshold in thresholds:
        result = test_threshold(recognizer, registered_speakers, remaining_speakers, speaker_audios, threshold)
        results.append(result)
        
        # 评价
        if result['same_rate'] >= 80 and result['cross_rate'] >= 80:
            eval_msg = "✅ 优秀"
        elif result['same_rate'] >= 70 and result['cross_rate'] >= 70:
            eval_msg = "👍 良好"
        elif result['same_rate'] >= 60 and result['cross_rate'] >= 60:
            eval_msg = "⚠️ 一般"
        else:
            eval_msg = "❌ 较差"
        
        print(f"{threshold:<8} {result['same_rate']:.1f}% ({result['same_correct']}/{result['same_total']})"
              f" {result['cross_rate']:.1f}% ({result['cross_correct']}/{result['cross_total']})"
              f" {eval_msg}")
    
    print("-" * 70)
    
    # 6. 推荐阈值
    print("\n【5】推荐阈值...")
    
    # 找两个指标都最高的阈值
    best_result = None
    best_score = -1
    
    for result in results:
        # 综合评分：几何平均数
        score = (result['same_rate'] * result['cross_rate']) ** 0.5
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result:
        print(f"推荐阈值: {best_result['threshold']}")
        print(f"  已注册识别率: {best_result['same_rate']:.1f}% ({best_result['same_correct']}/{best_result['same_total']})")
        print(f"  未注册拒绝率: {best_result['cross_rate']:.1f}% ({best_result['cross_correct']}/{best_result['cross_total']})")
    
    # 7. 相似度分布分析
    print("\n【6】相似度分布分析（基于默认阈值0.6的结果）...")
    default_result = results[thresholds.index(0.6)] if 0.6 in thresholds else None
    
    if default_result:
        same_scores = default_result['same_scores']
        cross_scores = default_result['cross_scores']
        
        if same_scores:
            print(f"已注册人相似度: 最小={min(same_scores):.4f}, 最大={max(same_scores):.4f}, 平均={sum(same_scores)/len(same_scores):.4f}")
        if cross_scores:
            print(f"未注册人相似度: 最小={min(cross_scores):.4f}, 最大={max(cross_scores):.4f}, 平均={sum(cross_scores)/len(cross_scores):.4f}")
        
        # 重叠分析
        if same_scores and cross_scores:
            overlap = len([s for s in same_scores if s < max(cross_scores)]) if cross_scores else 0
            print(f"相似度重叠: {overlap}/{len(same_scores)} = {overlap/len(same_scores)*100:.1f}%")
    
    # 8. 最终总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    print("""
【结论】
1. 模型在TIMIT开放集上的表现：
   - 适当阈值下，已注册人识别率和未注册人拒绝率可以平衡
   - 建议使用推荐阈值

2. 如果效果不理想，可以尝试：
   - 调整阈值（已在测试中覆盖）
   - 增加注册音频数量（从3句增加到5句）
   - 重新训练模型（增加训练轮数或调整学习率）

3. 最终系统建议阈值：""" + str(best_result['threshold']) if best_result else "未找到")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()