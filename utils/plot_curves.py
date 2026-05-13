"""
绘制训练曲线
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_training_curves(history_path=None, train_losses=None, val_accs=None, save_path=None):
    """
    绘制训练曲线
    可以传入历史文件路径，或直接传入列表
    """
    if history_path:
        with open(history_path, 'r') as f:
            history = json.load(f)
        train_losses = history['train_losses']
        val_accs = history['val_accs']
        best_acc = history['best_val_acc']
        epochs = history['epochs']
    else:
        epochs = len(train_losses)
        best_acc = max(val_accs) if val_accs else 0
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：Loss曲线
    ax1.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    # 右图：Accuracy曲线
    ax2.plot(range(1, epochs + 1), val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Validation Accuracy Curve (Best: {best_acc*100:.2f}%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"曲线已保存到 {save_path}")
    
    plt.show()

def plot_from_checkpoint(checkpoint_dir="seapker_checkpoints"):
    """目录加载历史并绘图"""
    history_path = os.path.join(checkpoint_dir,"training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path, save_path=os.path.join(checkpoint_dir, "training_curves.png"))
    else:
        print(f"未找到历史文件: {history_path}")

if __name__ == "__main__":
    # 直接运行此文件，从checkpoint加载并绘图
    plot_from_checkpoint()