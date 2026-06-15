# visual_parallel.py
"""
平行坐标图（适合稀疏数据）
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载数据
with open("speaker_checkpoints/wav2vec2_intensity_mapping_identify.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data['all_results'])

# 创建颜色分组的列（将准确率离散化）
df['accuracy_group'] = pd.cut(df['accuracy'], bins=5, labels=['23.7%', '23.9%', '24.0%', '24.1%', '24.2%'])

fig, ax = plt.subplots(figsize=(14, 8))

# 修正：使用 class_column 而不是 class_name
parallel_coordinates(df, class_column='accuracy_group',
                     cols=['LO', 'MD', 'HI'],
                     colormap='viridis', ax=ax)

ax.set_title('24组参数组合的平行坐标图（颜色=准确率）', fontsize=14)
ax.set_xlabel('参数', fontsize=12)
ax.set_ylabel('因子值', fontsize=12)
ax.grid(alpha=0.3)

plt.savefig('parallel_24.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 平行坐标图已保存：parallel_24.png")