import torch

class Config:
    # ECCAPA-TDNN 配置参数
    batch_size=32
    epochs=50
    lr=0.001
    n_mels=40
    max_len=500
    sr=16000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config=Config()
