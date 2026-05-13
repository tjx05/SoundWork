# tests/evaluate_ecapa_on_timit.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecapa_tdnn import ECAPA_TDNN
from preprocessing.timit_dataset import TIMITDataset
from config import config

def evaluate():
    # 设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 1. 加载测试集（您没用过的）
    print("加载TIMIT测试集...")
    test_json = "Data/TIMIT/test_info.json"
    test_dataset = TIMITDataset(
        test_json, 
        n_mels=config.n_mels, 
        max_len=config.max_len, 
        sr=config.sr
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_speakers = len(test_dataset.speaker_to_idx)
    print(f"测试集说话人数量: {num_speakers}")
    print(f"测试集样本数量: {len(test_dataset)}")
    
    # 2. 加载训练好的模型
    print("加载模型...")
    model = ECAPA_TDNN(
        n_mels=config.n_mels, 
        n_classes=num_speakers,
    )
    model_path = "speaker_checkpoints/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. 测试
    print("开始测试...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for fbank, label in test_loader:
            fbank = fbank.to(device)
            label = label.to(device)
            
            output = model(fbank)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(label.cpu().numpy())
    
    # 4. 结果
    acc = accuracy_score(all_targets, all_preds)
    print(f"\n{'='*50}")
    print(f"TIMIT测试集准确率: {acc*100:.2f}%")
    print(f"{'='*50}")
    
    # 5. 按说话人统计（可选）
    # print("\n按说话人统计（前10个）:")
    # speaker_names = list(test_dataset.speaker_to_idx.keys())
    # speaker_correct = {}
    # speaker_total = {}
    
    # 简单统计（需要重新跑才能按说话人分，这里简化）
    print("提示：整体准确率已足够评估模型")

if __name__ == "__main__":
    evaluate()