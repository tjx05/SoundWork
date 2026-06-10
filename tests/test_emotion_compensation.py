"""
对比测试：使用情感补偿 vs 不使用情感补偿
支持扫描不同的 compensation_strength
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


class EmotionCompensatedRecognizer:
    """
    带情感补偿的识别器（GPU版本）
    """
    def __init__(self, speaker_recognizer, emotion_recognizer, emotion_bias, device, compensation_strength=0.7):
        self.speaker_recognizer = speaker_recognizer
        self.emotion_recognizer = emotion_recognizer
        self.device = device
        self.threshold = speaker_recognizer.threshold
        self.database = speaker_recognizer.database
        self.compensation_strength = compensation_strength
        
        # 将bias移到GPU
        self.emotion_bias = {}
        for emotion, data in emotion_bias.items():
            self.emotion_bias[emotion] = {
                'bias': data['bias'].to(device),
                'std': data['std'].to(device),
                'sample_count': data['sample_count']
            }
        
        # 将数据库中的embedding也移到GPU
        self.database_gpu = {}
        for name, data in self.database.items():
            self.database_gpu[name] = {
                'embedding': data['embedding'].to(device),
                'gender': data['gender'],
                'age': data['age']
            }
        
        # 情感映射
        self.emotion_map = {
            '愤怒': 'angry', 'angry': 'angry',
            '快乐': 'happy', 'happy': 'happy',
            '悲伤': 'sad', 'sad': 'sad',
            '恐惧': 'fear', 'fear': 'fear',
            '厌恶': 'disgust', 'disgust': 'disgust',
            '中性': 'neutral', 'neutral': 'neutral'
        }
    
    def _extract_embedding(self, audio_path):
        """提取embedding并移到GPU"""
        emb = self.speaker_recognizer._extract_embedding(audio_path)
        return emb.to(self.device)
    
    def _apply_compensation(self, emb, emotion):
        """对embedding应用情感补偿（GPU上）"""
        if emotion and emotion in self.emotion_bias:
            bias = self.emotion_bias[emotion]['bias']
            emb = emb - self.compensation_strength * bias
            emb = emb / emb.norm()
        return emb
    
    def identify(self, audio_path):
        """识别"""
        # 1. 情感识别
        detected_emotion = 'neutral'
        wav2vec2_gender = '未知'
        wav2vec2_age = '未知'
        
        try:
            result = self.emotion_recognizer.predict(audio_path)
            raw_emotion = result.get('emotion', '中性')
            detected_emotion = self.emotion_map.get(raw_emotion, 'neutral')
            wav2vec2_gender = result.get('gender', '未知')
            wav2vec2_age = result.get('age', '未知')
        except Exception as e:
            print(f"情感识别失败: {e}")
        
        # 2. 提取embedding并移到GPU
        emb = self._extract_embedding(audio_path)
        
        # 3. 补偿
        emb = self._apply_compensation(emb, detected_emotion)
        
        # 4. 说话人识别（GPU上）
        best_name = None
        best_score = -1
        best_gender = None
        best_age = None
        
        for name, data in self.database_gpu.items():
            db_emb = data['embedding']
            score = torch.dot(emb, db_emb) / (emb.norm() * db_emb.norm())
            score = score.item()
            
            if score > best_score:
                best_name = name
                best_score = score
                best_gender = data["gender"]
                best_age = data["age"]
        
        # 5. 融合
        if best_gender == "未知" or best_gender is None:
            best_gender = wav2vec2_gender
        if best_age == "未知" or best_age is None:
            best_age = wav2vec2_age
        
        if best_score > self.threshold:
            return best_name, best_score, best_gender, best_age, detected_emotion
        else:
            return None, best_score, best_gender, best_age, detected_emotion


def load_data():
    """加载CREMA-D数据"""
    df = pd.read_csv("Data/CREMA-D/processed/cremad_index.csv")
    
    actor_emotion_audios = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        actor = row['actor_id']
        emotion = row['emotion_name']
        audio_path = row['audio_path']
        if os.path.exists(audio_path):
            actor_emotion_audios[actor][emotion].append(audio_path)
    
    valid_actors = []
    for actor, emotions in actor_emotion_audios.items():
        if 'neutral' in emotions and len(emotions) >= 2:
            valid_actors.append(actor)
    
    return actor_emotion_audios, valid_actors


def test_strength(strength, speaker_recognizer, emotion_recognizer, emotion_bias, 
                  actor_emotion_audios, test_actors, device):
    """测试单个补偿强度"""
    compensated_recognizer = EmotionCompensatedRecognizer(
        speaker_recognizer, emotion_recognizer, emotion_bias, device, 
        compensation_strength=strength
    )
    
    results_no_comp = []
    results_with_comp = []
    
    for actor in test_actors:
        expected_name = f"Actor_{actor}"
        for emotion, audio_paths in actor_emotion_audios[actor].items():
            if emotion == 'neutral':
                continue
            for audio_path in audio_paths[:2]:
                # 无补偿
                result = speaker_recognizer.identify(audio_path)
                name_no = result[0] if result and len(result) > 0 else None
                correct_no = (name_no == expected_name)
                
                # 有补偿
                result_comp = compensated_recognizer.identify(audio_path)
                name_comp = result_comp[0] if result_comp and len(result_comp) > 0 else None
                correct_comp = (name_comp == expected_name)
                
                results_no_comp.append(correct_no)
                results_with_comp.append(correct_comp)
    
    acc_no = sum(results_no_comp) / len(results_no_comp) * 100
    acc_comp = sum(results_with_comp) / len(results_with_comp) * 100
    
    return {
        'strength': strength,
        'acc_no': acc_no,
        'acc_comp': acc_comp,
        'improvement': acc_comp - acc_no,
        'relative_improvement': (acc_comp - acc_no) / acc_no * 100 if acc_no > 0 else 0,
        'total_samples': len(results_no_comp)
    }


def main():
    device = config.device
    print(f"使用设备: {device}")
    print("=" * 60)
    print("扫描不同的补偿强度")
    print("=" * 60)
    
    # 1. 加载情感偏移
    print("\n加载情感偏移...")
    bias_path = "speaker_checkpoints/emotion_bias.pth"
    if not os.path.exists(bias_path):
        print(f"❌ 情感偏移文件不存在: {bias_path}")
        return
    
    emotion_bias = torch.load(bias_path, map_location='cpu')
    print(f"✅ 加载偏移: {list(emotion_bias.keys())}")
    
    # 2. 加载基础识别器
    print("\n加载说话人识别器...")
    speaker_recognizer = SpeakerRecognizer(
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.52,
        db_path="speaker_checkpoints/speaker_db_test"
    )
    
    # 3. 加载情感识别器
    print("加载 Wav2Vec2 情感识别器...")
    emotion_recognizer = Wav2vec2Recognizer(
        model_path="emotion_checkpoints/best_wav2vec2_model.pth"
    )
    
    # 4. 加载CREMA-D数据
    print("\n加载CREMA-D数据...")
    actor_emotion_audios, valid_actors = load_data()
    print(f"有效演员: {len(valid_actors)}")
    
    # 5. 注册
    print("\n注册阶段...")
    import shutil
    db_path = "speaker_checkpoints/speaker_db_test"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)
    
    speaker_recognizer = SpeakerRecognizer(
        model_path="speaker_checkpoints/best_model.pth",
        threshold=0.52,
        db_path=db_path
    )
    
    for actor in tqdm(valid_actors, desc="注册"):
        expected_name = f"Actor_{actor}"
        neutral_audios = actor_emotion_audios[actor]['neutral'][:3]
        speaker_recognizer.enroll(expected_name, neutral_audios)
    
    print(f"注册完成，共 {len(speaker_recognizer.database)} 人")
    
    # ========== 关键：固定测试集，保证可比性 ==========
    # 使用前20个演员作为测试集（与第一次测试一致）
    test_actors = valid_actors[:20]
    print(f"\n测试集: {len(test_actors)} 个演员")
    
    # 先计算baseline（无补偿），确保稳定
    print("\n计算 baseline...")
    baseline_result = test_strength(
        0.0, speaker_recognizer, emotion_recognizer, emotion_bias,
        actor_emotion_audios, test_actors, device
    )
    print(f"Baseline 准确率: {baseline_result['acc_no']:.1f}% ({baseline_result['total_samples']} 个测试样本)")
    
    # 6. 测试不同的补偿强度
    print("\n" + "=" * 60)
    print("扫描补偿强度 (compensation_strength)")
    print("=" * 60)
    
    # 定义要测试的强度值
    strengths = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    
    results = []
    
    for strength in strengths:
        print(f"\n测试 strength = {strength}...")
        result = test_strength(
            strength, speaker_recognizer, emotion_recognizer, emotion_bias,
            actor_emotion_audios, test_actors, device
        )
        results.append(result)
        
        print(f"  无补偿: {result['acc_no']:.1f}%")
        print(f"  有补偿: {result['acc_comp']:.1f}%")
        print(f"  提升: +{result['improvement']:.1f}% ({result['relative_improvement']:.0f}%)")
    
    # 7. 打印汇总表格
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"\n{'强度':<10} {'准确率(无补偿)':<15} {'准确率(有补偿)':<15} {'提升':<12}")
    print("-" * 55)
    
    best_result = None
    best_improvement = -100
    
    for r in results:
        print(f"{r['strength']:<10} {r['acc_no']:.1f}%{'':<12} {r['acc_comp']:.1f}%{'':<12} +{r['improvement']:.1f}%")
        
        if r['improvement'] > best_improvement:
            best_improvement = r['improvement']
            best_result = r
    
    # 8. 推荐
    print("\n" + "=" * 60)
    print("推荐")
    print("=" * 60)
    
    if best_result and best_result['strength'] > 0:
        print(f"\n最佳补偿强度: {best_result['strength']}")
        print(f"  准确率: {best_result['acc_no']:.1f}% → {best_result['acc_comp']:.1f}%")
        print(f"  提升: +{best_result['improvement']:.1f}%")
        print(f"\n✅ 建议在配置中使用 compensation_strength = {best_result['strength']}")
    elif best_result and best_result['strength'] == 0:
        print("\n⚠️ 最佳强度为0，说明情感补偿无效")
    else:
        print("\n❌ 未找到有效结果")


if __name__ == "__main__":
    main()