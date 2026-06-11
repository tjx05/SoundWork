import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.wav2vec2_model import Wav2vec2MultiTaskModel
from preprocessing.wav2vec2_dataset import Wav2vec2MultiTaskDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的计算设备: {device}")

    # 数据集与路径配置
    csv_path = "./data/processed/cremad_index.csv111.csv" 
    audio_dir = "./data/raw/AudioWAV" 
    checkpoint_dir = "./emotion_checkpoints"
    output_dir = "./output"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 既然有两张 A40，直接把 Batch Size 拉满到 16 或者 32
    BATCH_SIZE = 16          
    LEARNING_RATE = 5e-5     
    EPOCHS = 20              
    MAX_SECONDS = 3.0        

    print("正在加载 Wav2vec 2.0 四任务数据集...")
    full_dataset = Wav2vec2MultiTaskDataset(csv_file=csv_path, audio_dir=audio_dir, max_seconds=MAX_SECONDS)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = Wav2vec2MultiTaskModel()
    
    # 🔥 工业级多卡并行改造
    if torch.cuda.device_count() > 1:
        print(f"🔥 霸气！检测到 {torch.cuda.device_count()} 张 GPU，启动 DataParallel 并行训练...")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 4 个独立的 Loss 计算器
    criterion_emo = nn.CrossEntropyLoss()
    criterion_gen = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    # ✨ 核心魔法：无视 XX 标签 (索引为 3)，解决多数类坍缩
    criterion_int = nn.CrossEntropyLoss(ignore_index=3) 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ✨ 新增点 1：引入余弦退火学习率调度器
    # 保证学习率平滑下降至最低 1e-6，防止训练后期在最优解山谷两边反复横跳
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_target_score = 0.0

    # 训练历史记录
    history = {
        "train_loss": [], "val_loss": [],
        "acc_emo": [], "acc_gen": [], "acc_age": [], "acc_int": [] 
    }

    print("========== 开始大模型【四任务】巅峰微调 ==========")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # 在 tqdm 进度条右侧加入当前实时学习率 (lr) 监控
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch_wave, batch_emo, batch_gen, batch_age, batch_int in pbar:
            batch_wave = batch_wave.to(device)
            batch_emo, batch_gen = batch_emo.to(device), batch_gen.to(device)
            batch_age, batch_int = batch_age.to(device), batch_int.to(device)
            
            optimizer.zero_grad()
            out_emo, out_gen, out_age, out_int = model(input_values=batch_wave)
            
            loss_emo = criterion_emo(out_emo, batch_emo)
            loss_gen = criterion_gen(out_gen, batch_gen)
            loss_age = criterion_age(out_age, batch_age)
            
            # ✨ 新增点 2：训练集【防 nan 安全锁】
            # 只有当前 batch 包含非 XX 标签（即不全是3）时，才计算交叉熵，否则给予无梯度 0 损失
            if (batch_int != 3).any():
                loss_int = criterion_int(out_int, batch_int)
            else:
                loss_int = torch.tensor(0.0, device=device)
            
            # 🔥 逼迫模型重视强度的学习，将强度权重提升至 1.0
            total_loss = 1.5 * loss_emo + 0.2 * loss_gen + 1.0 * loss_age + 1.0 * loss_int
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            # 进度条实时展示当前的学习率大小
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        correct_emo, correct_gen, correct_age, correct_int = 0, 0, 0, 0
        total_samples = 0
        total_valid_int_samples = 0  # ✨ 专门记录不是 XX 的真实强度样本数量
        
        with torch.no_grad():
            for batch_wave, batch_emo, batch_gen, batch_age, batch_int in val_loader:
                batch_wave = batch_wave.to(device)
                batch_emo, batch_gen = batch_emo.to(device), batch_gen.to(device)
                batch_age, batch_int = batch_age.to(device), batch_int.to(device)
                
                out_emo, out_gen, out_age, out_int = model(input_values=batch_wave)
                
                loss_emo = criterion_emo(out_emo, batch_emo)
                loss_gen = criterion_gen(out_gen, batch_gen)
                loss_age = criterion_age(out_age, batch_age)
                
                # ✨ 新增点 3：验证集【防 nan 安全锁】
                if (batch_int != 3).any():
                    v_loss_int = criterion_int(out_int, batch_int)
                else:
                    v_loss_int = torch.tensor(0.0, device=device)
                
                val_loss += (1.5 * loss_emo + 0.2 * loss_gen + 1.0 * loss_age + 1.0 * v_loss_int).item()
                
                # 统计准确率
                preds_emo = torch.argmax(out_emo, dim=1)
                preds_gen = torch.argmax(out_gen, dim=1)
                preds_age = torch.argmax(out_age, dim=1)
                preds_int = torch.argmax(out_int, dim=1)
                
                correct_emo += (preds_emo == batch_emo).sum().item()
                correct_gen += (preds_gen == batch_gen).sum().item()
                correct_age += (preds_age == batch_age).sum().item()
                total_samples += batch_wave.size(0)
                
                # ✨ 核心魔法：只统计真实强度 (非XX) 的判对数量
                valid_int_mask = (batch_int != 3) 
                correct_int += (preds_int[valid_int_mask] == batch_int[valid_int_mask]).sum().item()
                total_valid_int_samples += valid_int_mask.sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        acc_emo = correct_emo / total_samples * 100
        acc_gen = correct_gen / total_samples * 100
        acc_age = correct_age / total_samples * 100
        
        # 防止除以 0 报错 (假设这批恰好全是 XX)
        acc_int = correct_int / max(total_valid_int_samples, 1) * 100
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["acc_emo"].append(acc_emo)
        history["acc_gen"].append(acc_gen)
        history["acc_age"].append(acc_age)
        history["acc_int"].append(acc_int)

        # 打印日志时追加当前 Epoch 的学习率大小
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
        print(f"-> Val Acc - 情绪: {acc_emo:.2f}% | 性别: {acc_gen:.2f}% | 年龄: {acc_age:.2f}% | 强度(排爆后): {acc_int:.2f}%")
        
        # 综合平均准确率
        avg_acc = (acc_emo + acc_gen + acc_age + acc_int) / 4.0
        
        if avg_acc > best_target_score:
            best_target_score = avg_acc
            save_path = os.path.join(checkpoint_dir, "best_wav2vec2_model111.pth")
            
            # 🔥 脱壳保存：兼容 DataParallel 多卡环境
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), save_path)
            
            print(f"*** 四任务平均准确率创新高 ({avg_acc:.2f}%)，权重已安全脱壳并保存至 {save_path} ***\n")
        else:
            print("")
            
        # ✨ 新增点 4：驱动学习率更新
        # 每个 Epoch 训练并验证完后，使其按照余弦退火平滑衰减
        scheduler.step()
            
    # ========== 绘制并保存曲线 ==========
    print("========== 正在绘制并保存四任务训练曲线 ==========")
    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['acc_emo'], label='Emotion Acc', marker='^', color='#1f77b4') 
    plt.plot(epochs_range, history['acc_gen'], label='Gender Acc', marker='s', color='#ff7f0e') 
    plt.plot(epochs_range, history['acc_age'], label='Age Acc', marker='d', color='#2ca02c') 
    plt.plot(epochs_range, history['acc_int'], label='Intensity Acc', marker='*', color='#d62728') 
    plt.title('Validation Accuracies (Quad-Task)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, "training_curves111.png")
    plt.savefig(plot_save_path, dpi=300) 
    plt.close()
    print(f"✅ 训练曲线已成功保存至: {plot_save_path}")

if __name__ == "__main__":
    main()