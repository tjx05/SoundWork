import torch

class Config:
    # ECAPA-TDNN 配置参数
    # 通用特征参数（ECAPA-TDNN 训练和微调共用）
    n_mels=40
    max_len=500
    sr=16000

    # 原TIMIT训练参数
    batch_size=32
    epochs=50
    lr=0.001


    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config=Config()
