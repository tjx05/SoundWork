import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.wav2vec2_model import Wav2vec2MultiTaskModel
from preprocessing.wav2vec2_dataset import Wav2vec2MultiTaskDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用的计算设备: {device}")

    csv_path = "./data/processed/cremad_index.csv"
    audio_dir = "./data/raw/AudioWAV"
    checkpoint_dir = "./emotion_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Wav2vec 2.0 专用超参数 ===
    BATCH_SIZE = 16          # 大模型极占显存，A40虽然大，但建议先降到 16 试试水
    LEARNING_RATE = 5e-5     # 微调 Transformer 必须使用较小的学习率
    EPOCHS = 20              # 预训练模型收敛极快，20 轮足够了
    MAX_SECONDS = 3.0        # 统一截断到 3 秒
    # ===============================

    print("正在加载 Wav2vec 2.0 数据集...")
    full_dataset = Wav2vec2MultiTaskDataset(csv_file=csv_path, audio_dir=audio_dir, max_seconds=MAX_SECONDS)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = Wav2vec2MultiTaskModel().to(device)
    
    criterion_emo = nn.CrossEntropyLoss()
    criterion_gen = nn.CrossEntropyLoss()
    criterion_age = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_target_score = 0.0

    print("========== 开始大模型微调 ==========")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch_wave, batch_emo, batch_gen, batch_age in pbar:
            batch_wave = batch_wave.to(device)
            batch_emo, batch_gen, batch_age = batch_emo.to(device), batch_gen.to(device), batch_age.to(device)
            
            optimizer.zero_grad()
            out_emo, out_gen, out_age = model(input_values=batch_wave)
            
            loss_emo = criterion_emo(out_emo, batch_emo)
            loss_gen = criterion_gen(out_gen, batch_gen)
            loss_age = criterion_age(out_age, batch_age)
            
            # 使用你跑出来的黄金比例
            total_loss = 1.5 * loss_emo + 0.2 * loss_gen + 1.0 * loss_age
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        correct_emo, correct_gen, correct_age = 0, 0, 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_wave, batch_emo, batch_gen, batch_age in val_loader:
                batch_wave = batch_wave.to(device)
                batch_emo, batch_gen, batch_age = batch_emo.to(device), batch_gen.to(device), batch_age.to(device)
                
                out_emo, out_gen, out_age = model(input_values=batch_wave)
                
                loss_emo = criterion_emo(out_emo, batch_emo)
                loss_gen = criterion_gen(out_gen, batch_gen)
                loss_age = criterion_age(out_age, batch_age)
                val_loss += (1.5 * loss_emo + 0.2 * loss_gen + 1.0 * loss_age).item()
                
                preds_emo = torch.argmax(out_emo, dim=1)
                preds_gen = torch.argmax(out_gen, dim=1)
                preds_age = torch.argmax(out_age, dim=1)
                
                correct_emo += (preds_emo == batch_emo).sum().item()
                correct_gen += (preds_gen == batch_gen).sum().item()
                correct_age += (preds_age == batch_age).sum().item()
                total_samples += batch_wave.size(0)
                
        # ... 前面的代码保持不变 ...
        avg_val_loss = val_loss / len(val_loader)
        acc_emo = correct_emo / total_samples * 100
        acc_gen = correct_gen / total_samples * 100
        acc_age = correct_age / total_samples * 100
        
        print(f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"-> Val Acc - Emo: {acc_emo:.2f}% | Gen: {acc_gen:.2f}% | Age: {acc_age:.2f}%")
        
        # ==================== 修改部分：按准确率均值保存 ====================
        # 计算三个任务准确率的算术平均值
        avg_acc = (acc_emo + acc_gen + acc_age) / 3.0
        
        if avg_acc > best_target_score:
            best_target_score = avg_acc
            save_path = os.path.join(checkpoint_dir, "best_wav2vec2_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"*** 综合准确率均值创新高 ({avg_acc:.2f}%)，已保存大模型权重至 {save_path} ***\n")
        else:
            print("")
        # ====================================================================

# 别忘了文件最末尾一定要有这两行：
if __name__ == "__main__":
    main()