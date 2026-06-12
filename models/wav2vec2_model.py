import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2vec2MultiTaskModel(nn.Module):
    """
    终极四任务全能版: Wav2vec 2.0 语音识别大模型
    输出: 情绪(6类)、性别(2类)、年龄(3类)、强度(3类，LO/MD/HI，无效标签 -1 在损失函数中屏蔽)
    """
    def __init__(self, model_name="./local_base_model/wav2vec2-base", 
                 num_emotion=6, num_gender=2, num_age=3, num_intensity=3):
        super(Wav2vec2MultiTaskModel, self).__init__()
        
        print(f"正在从 HuggingFace 加载预训练基座: {model_name} ...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # 冻结 CNN 特征提取层，极大节省显存并加快收敛
        self.wav2vec2.feature_extractor._freeze_parameters()
        hidden_size = self.wav2vec2.config.hidden_size
        
        # 1. 情绪头 (最难，分配 256 维)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_emotion)
        )
        # 2. 性别头
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_gender)
        )
        # 3. 年龄头
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_age)
        )
        # 4. 强度头 ( 3 分类，分别对应 LO, MD, HI)
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_intensity)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 全局平均池化
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            pooled_features = sum_embeddings / sum_mask
        else:
            pooled_features = torch.mean(hidden_states, dim=1)
            
        # 经过四个独立的分类头
        emotion_logits = self.emotion_head(pooled_features)
        gender_logits = self.gender_head(pooled_features)
        age_logits = self.age_head(pooled_features)
        intensity_logits = self.intensity_head(pooled_features) 
        
        return emotion_logits, gender_logits, age_logits, intensity_logits

