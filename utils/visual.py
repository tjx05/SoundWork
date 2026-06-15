import json
import matplotlib.pyplot as plt
import numpy as np

# ========== 添加中文字体配置 ==========
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

with open("speaker_checkpoints/wav2vec2_intensity_mapping_identify.json", "r") as f:
    data = json.load(f)

all_results = data['all_results']

# 按 HI 分组
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, hi_val in enumerate([0.9, 1.1]):
    ax = axes[idx]
    hi_results = [r for r in all_results if r['HI'] == hi_val]
    
    labels = [f"LO={r['LO']}\nMD={r['MD']}" for r in hi_results]
    acc = [r['accuracy'] for r in hi_results]
    colors = ['#2ecc71' if a == max(acc) else '#95a5a6' for a in acc]
    
    bars = ax.bar(labels, acc, color=colors, edgecolor='black')
    ax.set_title(f'HI = {hi_val}', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_ylim(23, 25)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}%', ha='center', fontsize=9)

plt.suptitle('24组参数组合准确率对比', fontsize=14)
plt.tight_layout()
plt.savefig('grouped_bars_24.png', dpi=300)
plt.show()