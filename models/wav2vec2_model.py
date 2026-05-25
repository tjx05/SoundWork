import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2vec2MultiTaskModel(nn.Module):
    """
    基于 Wav2vec 2.0 的多任务语音识别大模型架构
    输入: 原始的 1D 语音波形 (采样率必须为 16kHz)
    输出: 情绪(6类)、性别(2类)、年龄(3类)
    """
    def __init__(self, model_name="./local_base_model/wav2vec2-base", num_emotion=6, num_gender=2, num_age=3):
        super(Wav2vec2MultiTaskModel, self).__init__()
        
        print(f"正在从 HuggingFace 加载预训练基座: {model_name} ...")
        # 1. 加载 HuggingFace 上的预训练 Wav2vec 2.0 基座模型
        # 这个模型已经在数万小时的无标注英语语音上进行了自监督学习
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # 优化策略：冻结 Wav2vec 2.0 底层的 CNN 卷积特征提取器
        # 原因：底层特征已经非常完美，冻结它可以极大节省 A40 的显存并加快训练速度
        self.wav2vec2.feature_extractor._freeze_parameters()
        
        # 获取 Wav2vec 2.0 输出的隐藏层维度（base版本通常是 768 维）
        hidden_size = self.wav2vec2.config.hidden_size
        
        # 2. 多任务独立输出头 (从 768 维特征映射到具体的类别)
        # 情绪头
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotion)
        )
        
        # 性别头
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_gender)
        )
        
        # 年龄头
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age)
        )

    def forward(self, input_values, attention_mask=None):
        """
        前向传播
        :param input_values: 原始语音波形 张量, 形状 (Batch, Sequence_length)
        :param attention_mask: 掩码 (告诉模型哪些是补零的静音部分)
        """
        # 1. 送入 Wav2vec 2.0 提取高维上下文特征
        # 输出的 last_hidden_state 形状: (Batch, Time_Frames, 768)
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 2. 全局平均池化 (Global Mean Pooling)
        # 将变长的时间帧维度压缩，得到一整段语音的 768 维“句向量”
        # 也可以在这里引入你之前的 Attention 机制，但 Wav2vec 强大的上下文能力通常让简单的 Mean Pooling 就足够了
        if attention_mask is not None:
            # 如果有 mask，只平均真实的语音部分，忽略补零的部分
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            pooled_features = sum_embeddings / sum_mask
        else:
            pooled_features = torch.mean(hidden_states, dim=1)
            
        # 3. 送入三个独立的分类头
        emotion_logits = self.emotion_head(pooled_features)
        gender_logits = self.gender_head(pooled_features)
        age_logits = self.age_head(pooled_features)
        
        return emotion_logits, gender_logits, age_logits


if __name__ == "__main__":
    # ======== 单元测试 ========
    print("正在测试 Wav2vec 2.0 多任务大模型架构...")
    
    # 模拟输入：4条语音，每条持续约 3 秒 (采样率 16000Hz * 3秒 = 48000 个采样点)
    # 注意：这里的输入是 1D 的波形数据，而不是 2D 的 FBank！
    fake_waveforms = torch.randn(4, 48000) 
    
    # 实例化模型
    model = Wav2vec2MultiTaskModel()
    
    # 前向传播
    emo_out, gen_out, age_out = model(input_values=fake_waveforms)
    
    print("\n--- 校验结果 ---")
    print(f"输入原始波形形状: {fake_waveforms.shape}")
    print(f"情绪输出形状: {emo_out.shape}")
    print(f"性别输出形状: {gen_out.shape}")
    print(f"年龄输出形状: {age_out.shape}")
    print("\n架构连通性测试通过！")