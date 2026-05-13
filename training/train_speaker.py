import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
import os
import sys
from tqdm import tqdm
import random
import json
import numpy as np
from sklearn.metrics import accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ecapa_tdnn import ECAPA_TDNN
from preprocessing.timit_dataset import TIMITDataset
from config import config

def set_seed(seed=42):
    """固定所有随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 加载数据集
    print("加载TIMIT数据集...")
    train_json=os.path.join("Data/TIMIT","train_info.json")
    dataset=TIMITDataset(train_json,n_mels=config.n_mels,max_len=config.max_len,sr=config.sr)
    num_speakers=len(dataset.speaker_to_idx)
    print(f"说话人数量: {num_speakers}")

    # 划分训练/验证
    train_size=int(0.9*len(dataset))
    val_size=len(dataset)-train_size
    train_ds,val_ds=random_split(dataset,[train_size,val_size])
    
    train_loader=DataLoader(train_ds,batch_size=config.batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=config.batch_size,shuffle=False)

    # 模型
    device=config.device
    model=ECAPA_TDNN(
        n_mels=config.n_mels,
        n_classes=num_speakers,
    )
    model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)

    print("\n开始训练")
    seed=66
    set_seed(seed)
    epochs=config.epochs
    train_losses=[]
    # min_loss=float('inf')
    best_val_acc=0
    val_accs=[] 

    for epoch in  range(epochs):
        # 训练阶段
        model.train()
        total_losses=0

        loop=tqdm(train_loader,desc=f"Epoch {epoch+1:2d}/{epochs}",leave=True)

        for data,target in loop:
            data=data.to(device)
            target=target.to(device)

            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_losses+=loss.item()

            loop.set_postfix(loss=loss.item())
        
        avg_loss=total_losses/len(train_loader)
        train_losses.append(avg_loss)

        # print(f"Epoch {epoch+1:2d}/{EPOCHS}, Loss: {avg_loss:.4f}")
        # 验证
        model.eval()
        correct=0
        total=0
        all_preds=[]
        all_targets=[]
        with torch.no_grad():
            for data,target in val_loader:
                data=data.to(device)
                target=target.to(device)

                output=model(data)
                pred=output.argmax(dim=1).cpu().numpy()

                all_preds.extend(pred)
                all_targets.extend(target.cpu().numpy())

        val_acc=accuracy_score(all_targets,all_preds)
        val_accs.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("seapker_checkpoints",exist_ok=True)
            torch.save(model.state_dict(),'seapker_checkpoints/best_model.pth')
            print(f"保存模型 | 验证准确率: {val_acc*100:.2f}% | 训练损失: {avg_loss:.4f}")
    
    # 2. 保存训练历史
    save_path="seapker_checkpoints"
    history={
        'train_losses': train_losses,  # list of float
        'val_accs': val_accs,          # list of float
        'best_val_acc': best_val_acc,
        'epochs': epochs,
        'config': {
            'n_mels': config.n_mels,
            'max_len': config.max_len,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'seed': seed
        }
    }
    # 保存为json
    with open(os.path.join(save_path, "training_history.json"), 'w') as f:
        json.dump(history,f,indent=4)
    
    # 保存为npy
    # np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
    # np.save(os.path.join(save_path, "val_accs.npy"), np.array(val_accs))



    print("\n训练完成")

if __name__ == "__main__":
    main()

