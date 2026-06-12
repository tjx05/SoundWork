import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 全局字体与负号配置（解决中文乱码）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载JSON数据
json_path = "speaker_checkpoints/wav2vec2_intensity_mapping_identify.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_results = data["all_results"]
# 提取并排序所有参数取值
lo_list = sorted({d["LO"] for d in all_results})
md_list = sorted({d["MD"] for d in all_results})
hi_list = sorted({d["HI"] for d in all_results})

# 构建三维参数字典：LO -> MD -> HI -> 准确率
param_dict = defaultdict(lambda: defaultdict(dict))
for item in all_results:
    param_dict[item["LO"]][item["MD"]][item["HI"]] = item["accuracy"]

# 创建 1行2列 画布，接收两个坐标轴对象 ax1、ax2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ========== 左图：固定最优MD=1.1，绘制LO-HI热力图 ==========
md_best = 1.1
grid1 = np.zeros((len(lo_list), len(hi_list)))
for i, lo in enumerate(lo_list):
    for j, hi in enumerate(hi_list):
        grid1[i, j] = param_dict[lo][md_best][hi]

im1 = ax1.imshow(grid1, cmap="YlGnBu", aspect="auto")
ax1.set_title(f"固定 MD={md_best} | LO-HI 准确率分布", fontsize=12)
ax1.set_xlabel("强情绪因子 HI")
ax1.set_ylabel("弱情绪因子 LO")
ax1.set_xticks(range(len(hi_list)))
ax1.set_xticklabels(hi_list)
ax1.set_yticks(range(len(lo_list)))
ax1.set_yticklabels(lo_list)
# 修复点：传入坐标轴对象 ax1，而非数字1
plt.colorbar(im1, ax=ax1, label="准确率 (%)")

# ========== 右图：固定最优HI=1.3，绘制LO-MD热力图 ==========
hi_best = 1.3
grid2 = np.zeros((len(lo_list), len(md_list)))
for i, lo in enumerate(lo_list):
    for j, md in enumerate(md_list):
        grid2[i, j] = param_dict[lo][md][hi_best]

im2 = ax2.imshow(grid2, cmap="OrRd", aspect="auto")
ax2.set_title(f"固定 HI={hi_best} | LO-MD 准确率分布", fontsize=12)
ax2.set_xlabel("中情绪因子 MD")
ax2.set_ylabel("弱情绪因子 LO")
ax2.set_xticks(range(len(md_list)))
ax2.set_xticklabels(md_list)
ax2.set_yticks(range(len(lo_list)))
ax2.set_yticklabels(lo_list)
# 修复点：传入坐标轴对象 ax2，而非数字2
plt.colorbar(im2, ax=ax2, label="准确率 (%)")

# 布局自适应 + 保存图片
plt.tight_layout()
plt.savefig("heatmap_param_dist.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 参数热力图已成功保存：heatmap_param_dist.png")