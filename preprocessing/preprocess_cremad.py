"""
CREMA-D 数据预处理 (V2 升级版: 支持情绪强度)
功能：构建情感(6类) / 性别(2类) / 年龄(3类) / 强度(4类) 标签
输出：cremad_index.csv 包含 [audio_path, emotion, gender, age, intensity]
"""
import pandas as pd
import os

# ========== 1. 配置路径 ==========
# 注意：使用相对路径并统一使用正斜杠 '/'
AUDIO_DIR = "./data/raw/AudioWAV"
DEMO_CSV = "./data/raw/VideoDemographics.csv"
OUTPUT_CSV = "./data/processed/cremad_index.csv111.csv"

# ========== 2. 标签映射字典 ==========
# 情感映射 (6分类)
emotion_map = {
    "ANG": 0, # 愤怒
    "DIS": 1, # 厌恶
    "FEA": 2, # 恐惧
    "HAP": 3, # 开心
    "SAD": 4, # 悲伤
    "NEU": 5, # 中性
}

# 强度映射 (4分类) —— 本次升级核心
intensity_map = {
    "LO": 0, # 低强度 (Low)
    "MD": 1, # 中等强度 (Medium)
    "HI": 2, # 高强度 (High)
    "XX": 3, # 未指定/自然情感 (Unspecified)
}

# 性别映射 (2分类)
gender_map = {"Male": 0, "Female": 1}

# 年龄映射 (3分类)
# CREMA-D实际年龄分布：20-70
def age_to_label(age):
    if age < 35:
        return 0  # 青年
    elif age < 55:
        return 1  # 中年
    else:
        return 2  # 老年

# ========== 3. 开始处理数据 ==========
# 加载人口统计信息
df_demo = pd.read_csv(DEMO_CSV)
data = []

print("正在扫描音频文件并提取多任务标签...")

if not os.path.exists(AUDIO_DIR):
    print(f"❌ 找不到音频文件夹: {AUDIO_DIR}，请检查路径！")
else:
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            # 文件名示例: 1001_DFA_ANG_XX.wav
            parts = filename.split("_")
            
            # 异常文件跳过
            if len(parts) < 4:
                continue
                
            actor_id = int(parts[0])
            emo_code = parts[2]
            intensity_code = parts[3].replace(".wav", "")

            # 查找演员的人口统计学信息
            demo = df_demo[df_demo["ActorID"] == actor_id]
            if demo.empty:
                continue

            age = demo.iloc[0]["Age"]
            sex = demo.iloc[0]["Sex"]

            # 转换数字标签 (如果找不到对应代码，给一个默认的安全值)
            emotion = emotion_map.get(emo_code, 5)      # 默认中性
            intensity = intensity_map.get(intensity_code, 3) # 默认XX
            gender = gender_map.get(sex, 0)
            age_label = age_to_label(age)

            # 强制使用正斜杠拼接路径，彻底解决 Windows -> Linux 的跨平台 Bug
            audio_path = f"{AUDIO_DIR.rstrip('/')}/{filename}"

            # 存入列表
            data.append({
                "audio_path": audio_path,
                "emotion": emotion,
                "gender": gender,
                "age": age_label,
                "intensity": intensity  # 新增的强度标签！
            })

# ========== 4. 构建与保存 DataFrame ==========
index_df = pd.DataFrame(data)

# 强制转换为 int 类型，防止后续传入 PyTorch 时报类型错误
index_df["emotion"] = index_df["emotion"].astype(int)
index_df["gender"] = index_df["gender"].astype(int)
index_df["age"] = index_df["age"].astype(int)
index_df["intensity"] = index_df["intensity"].astype(int)

# 确保输出目录存在并保存
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
index_df.to_csv(OUTPUT_CSV, index=False)

print("\n✨ 处理完成！")
print(f"✅ 完美格式索引已保存至: {OUTPUT_CSV}")
print(f"✅ 共成功提取数据: {len(index_df)} 条")
print("\n--- 最终喂给大模型的数据预览 (前3条) ---")
print(index_df.head(3))