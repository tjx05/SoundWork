import json
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
json_path = "speaker_checkpoints/wav2vec2_intensity_mapping_identify.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

baseline_acc = data["baseline_accuracy"]
fixed_acc = data["fixed_accuracy"]
best_acc = data["best_accuracy"]
all_results = data["all_results"]
lo_list = sorted({d["LO"] for d in all_results})
md_list = sorted({d["MD"] for d in all_results})
hi_list = sorted({d["HI"] for d in all_results})

param_dict = defaultdict(lambda: defaultdict(dict))
for item in all_results:
    param_dict[item["LO"]][item["MD"]][item["HI"]] = item["accuracy"]

# 2行2列画布
fig = plt.figure(figsize=(16, 10))

# 子图1：整体方案对比
ax1 = plt.subplot(2,2,1)
labels = ["无补偿", "固定0.7", "自适应最优"]
acc = [baseline_acc, fixed_acc, best_acc]
bars = ax1.bar(labels, acc, color=["#ff6b6b","#ffd166","#06d6a0"], alpha=0.8)
for bar, v in zip(bars, acc):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{v:.1f}%", ha="center")
ax1.set_title("不同补偿方案效果对比")
ax1.set_ylabel("准确率 (%)")

# 子图2：HI变化趋势（固定LO=0.2）
ax2 = plt.subplot(2,2,2)
lo_fix = 0.2
for md in md_list:
    y = [param_dict[lo_fix][md][h] for h in hi_list]
    ax2.plot(hi_list, y, marker="o", label=f"MD={md}")
ax2.set_title(f"固定 LO={lo_fix}，HI变化趋势")
ax2.set_xlabel("HI 因子")
ax2.set_ylabel("准确率 (%)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# 子图3：MD变化趋势（固定HI=1.3）
ax3 = plt.subplot(2,2,3)
hi_fix = 1.3
for lo in lo_list:
    y = [param_dict[lo][m][hi_fix] for m in md_list]
    ax3.plot(md_list, y, marker="s", label=f"LO={lo}")
ax3.set_title(f"固定 HI={hi_fix}，MD变化趋势")
ax3.set_xlabel("MD 因子")
ax3.set_ylabel("准确率 (%)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 子图4：典型组合+提升幅度
ax4 = plt.subplot(2,2,4)
comb = [
    ("最差组合", param_dict[0.7][0.3][0.7]),
    ("固定0.7", param_dict[0.7][0.7][0.7]),
    ("最优组合", param_dict[0.2][1.1][1.3])
]
names = [x[0] for x in comb]
vals = [x[1] for x in comb]
bars = ax4.bar(names, vals, alpha=0.8)
for bar, v in zip(bars, vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{v:.1f}%", ha="center")
ax4.set_title("典型参数组合对比")
ax4.set_ylabel("准确率 (%)")

plt.tight_layout()
plt.savefig("trend_contrast.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ 趋势与对比图已保存：trend_contrast.png")
