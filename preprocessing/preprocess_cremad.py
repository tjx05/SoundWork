"""
CREMA-D 数据预处理
功能：构建情感/性别/年龄标签
输出：cremad_index.csv 包含 [filename,speaker_id,emotion,gender,age]
"""
import pandas as pd
import os

# ========== 配置 ==========
AUDIO_DIR="./Data/CREMA-D/raw/AudioWAV"
DEMO_CSV="./Data/CREMA-D/raw/VideoDemographics.csv"
OUTPUT_CSV="./Data/CREMA-D/processed/cremad_index.csv"

# 情感映射
emotion_map={
    "ANG":0, # 愤怒
    "DIS":1, # 厌恶
    "FEA":2, # 恐惧
    "HAP":3, # 开心
    "SAD":4, # 悲伤
    "NEU":5, # 中性
}

# 情感中文名
emotion_cn = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "neutral"
}

# 强度映射
intensity_map={
    "LO":0, # 弱
    "HI":1, # 强
}

# 性别映射
gender_map={"Male":0,"Female":1}

# CREMA-D实际年龄分布：20-70
def age_to_label(age):
    if age<35:
        return 0  # 青年
    elif age<55:
        return 1  # 中年
    else:
        return 2  # 老年

# ==========加载人口统计信息==========
df_demo=pd.read_csv(DEMO_CSV)

# ==========从音频文件名提取标签==========
data = []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        parts=filename.split("_")
        actor_id=int(parts[0])
        emo_code=parts[2]
        intensity_code = parts[3].replace(".wav", "")  # 去掉.wav后缀

        # 查找演员信息
        demo=df_demo[df_demo["ActorID"] == actor_id]
        if demo.empty:
            continue

        age=demo.iloc[0]["Age"]
        sex=demo.iloc[0]["Sex"]

        # 转换标签
        emotion=emotion_map.get(emo_code)
        intensity=intensity_map.get(intensity_code)
        gender=gender_map.get(sex)
        age_label=age_to_label(age)

        # 处理强度：如果是XX，标记为None或跳过
        if intensity_code in ["LO", "HI"]:
            intensity = intensity_map.get(intensity_code)
        else:
            intensity = -1  # 表示无强度信息


        audio_path=os.path.join(AUDIO_DIR, filename)
        data.append({
            "audio_path": audio_path,
            "actor_id": actor_id,
            "emotion": emotion,
            "emotion_name": emotion_cn[emotion],
            "intensity": intensity,
            "intensity_name": intensity_code,  # "LO" 或 "HI"
            "gender": gender,
            "gender_name": sex,
            "age": age_label,
            "age_name": ("young" if age_label == 0 else "middle" if age_label == 1 else "old")
        })

# 构建 DataFrame
index_df=pd.DataFrame(data)
# index_df=index_df.dropna()
# index_df["emotion"]=index_df["emotion"].astype(int)
# index_df["gender"]=index_df["gender"].astype(int)
# index_df["age"]=index_df["age"].astype(int)

# 保存
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
index_df.to_csv(OUTPUT_CSV, index=False)
print(f"索引已保存到{OUTPUT_CSV}，共{len(index_df)}条数据")
