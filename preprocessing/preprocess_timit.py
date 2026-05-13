# preprocessing/generate_open_test.py
import os
import json

# 配置
TEST_ROOT = "Data/TIMIT/TEST"  # TEST文件夹的路径
OUTPUT_JSON = "Data/TIMIT/open_test_info.json"

def infer_gender(speaker_id):
    if speaker_id.startswith('F'):
        return "F"
    elif speaker_id.startswith('M'):
        return "M"
    else:
        return "?"
    
def extract_speaker_id(speaker_full):
    """提取纯说话人ID（去掉性别前缀）"""
    if speaker_full.startswith(('F', 'M')):
        return speaker_full[1:]  # 去掉第一个字符
    return speaker_full

open_test_data = []

for root, dirs, files in os.walk(TEST_ROOT):
    for file in files:
        if file.upper().endswith(".WAV"):
            # 获取相对于TEST_ROOT的路径
            rel_path = os.path.relpath(os.path.join(root, file), TEST_ROOT)
            rel_path = rel_path.replace("\\", "/")
            
            if rel_path.endswith(".wav"):
                rel_path = rel_path[:-4] + ".WAV"
            
            # 关键：加上"TEST/"前缀
            full_rel_path = f"TEST/{rel_path}"
            
            # 解析路径
            parts = rel_path.split("/")
            # rel_path: "DR2/FCMR0/SA1.WAV"
            dialect_region = parts[0]
            speaker_full = parts[1]
            filename = parts[2]
            
            name_parts = filename.replace(".WAV", "").split(".")
            sentence_type = name_parts[0][:2]
            sentence_id = name_parts[0][2:]
            
            item = {
                "filepath": full_rel_path,  # "TEST/DR2/FCMR0/SA1.WAV"
                "dialect_region": dialect_region,
                "gender": infer_gender(speaker_full),
                "speaker_id": extract_speaker_id(speaker_full),
                "sentence_type": sentence_type,
                "sentence_id": sentence_id
            }
            open_test_data.append(item)

# 保存
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(open_test_data, f, indent=2, ensure_ascii=False)

print(f"✅ 生成 {len(open_test_data)} 条数据")
print(f"📁 保存到: {OUTPUT_JSON}")

print("\n示例路径:")
for i, item in enumerate(open_test_data[:3]):
    print(f"  {item['filepath']}")