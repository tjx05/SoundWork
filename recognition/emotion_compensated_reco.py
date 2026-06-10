# recognition/emotion_compensated_reco.py
"""
情感补偿说话人识别器
基于 CREMA-D 统计的情感偏移，对 embedding 进行补偿
"""

import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.speaker_reco import SpeakerRecognizer
from config import config


class EmotionCompensatedRecognizer(SpeakerRecognizer):
    """
    带情感补偿的说话人识别器
    继承自SpeakerRecognizer，只重写_extract_embedding方法
    """
    def __init__(self, model_path,threshold=0.52,
                 db_path="speaker_checkpoints/speaker_db",
                 emotion_bias_path="speaker_checkpoints/emotion_bias.pth",
                 emotion_recognizer=None,
                 compensation_strength=0.7,
                 use_compensation=True,
                 auto_register_unseen=True,
                 temp_update_momentum=0.7):
        """
        输入:
            model_path: ECAPA-TDNN模型路径
            threshold: 识别阈值
            db_path: 说话人数据库路径
            emotion_bias_path: 情感偏移文件路径
            emotion_recognizer: Wav2Vec2情感识别器
            compensation_strength: 补偿强度（推荐0.7）
            use_compensation: 是否启用情感补偿
            auto_register_unseen: 是否自动注册未注册的说话人
            temp_update_momentum: 临时说话人更新动量,momentum=0.7 表示70%保留历史，30%融合新特征
        """
        # 调用父类初始化
        super().__init__(
            model_path=model_path,
            threshold=threshold,
            auto_register_unseen=auto_register_unseen,
            temp_update_momentum=temp_update_momentum,
            db_path=db_path
        )
        
        self.emotion_recognizer=emotion_recognizer
        self.compensation_strength=compensation_strength
        self.use_compensation=use_compensation
        self._last_emotion_result=None  # 缓存
        
        # 加载情感偏移
        self.emotion_bias = {}
        if use_compensation and os.path.exists(emotion_bias_path):
            bias_data=torch.load(emotion_bias_path,map_location='cpu')
            for emotion,data in bias_data.items():
                self.emotion_bias[emotion]=data['bias']
            print(f"情感补偿已启用，强度={compensation_strength}")
            print(f"支持的情感: {list(self.emotion_bias.keys())}")
        else:
            print(f"情感补偿未启用")
        
        # 情感映射
        self.emotion_map = {
            '愤怒': 'angry', 'angry': 'angry',
            '快乐': 'happy', 'happy': 'happy',
            '悲伤': 'sad', 'sad': 'sad',
            '恐惧': 'fear', 'fear': 'fear',
            '厌恶': 'disgust', 'disgust': 'disgust',
            '中性': 'neutral', 'neutral': 'neutral'
        }
    
    def _extract_embedding(self,audio_path):
        """
        提取embedding并缓存情感识别结果
        """
        # 调用父类方法提取原始embedding（已经在CPU上）
        emb=super()._extract_embedding(audio_path)
        
        # 如果不使用补偿，直接返回
        if not self.use_compensation:
            return emb
        
        # 获取情感标签
        emotion='neutral'
        gender='未知'
        age='未知'

        if self.emotion_recognizer:
            try:
                result=self.emotion_recognizer.predict(audio_path)
                raw_emotion=result.get('emotion','中性')
                emotion=self.emotion_map.get(raw_emotion,'neutral')
                gender=result.get('gender','未知')
                age=result.get('age','未知')
            except Exception as e:
                print(f"情感识别失败: {e}")

        # 缓存结果，供后续使用
        self._last_emotion_result = {
            'emotion': emotion,
            'gender': gender,
            'age': age
        }
        
        # 应用补偿（减去情感偏移）
        if emotion in self.emotion_bias:
            bias=self.emotion_bias[emotion]
            emb=emb-self.compensation_strength*bias
            emb=emb/emb.norm()
        
        return emb
    
    def get_last_emotion(self):
        """获取最后一次识别的情感结果"""
        return self._last_emotion_result or {'emotion': '中性', 'gender': '未知', 'age': '未知'}
    