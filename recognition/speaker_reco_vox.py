import torch
import os
import numpy as np
import time
import torchaudio
import torch.nn.functional as F

# 从 ecapa_tdnn_vox.py 导入模型（确保文件在同级目录）
from models.ecapa_tdnn_vox import ECAPA_TDNN as ECAPA_TDNN_VOX

# 模拟 config 配置（根据实际需求调整）
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_mels = 80  # vox模型固定用80维mel
        self.max_len = None  # vox模型自动处理长度，无需固定
        self.sr = 16000  # vox模型输入采样率必须是16000

config = Config()

class SpeakerRecognizerVox:
    def __init__(self, model_path, threshold=0.6, auto_register_unseen=True, 
                 temp_update_momentum=0.7, db_path="speaker_checkpoints/speaker_db"):
        """
        适配ecapa_tdnn_vox.py的说话人识别器
        输入：
            model_path：vox预训练模型路径
            threshold：识别阈值（建议0.6-0.8）
            auto_register_unseen：是否自动注册未注册的说话人
            temp_update_momentum：临时说话人更新动量
        """
        self.threshold = threshold
        self.device = config.device

        # 加载vox版本的ECAPA-TDNN模型
        self.model = ECAPA_TDNN_VOX(C=1024)  # vox模型默认C=1024

        # 加载权重（vox模型无classifier层，无需过滤）
        state_dict = torch.load(model_path, map_location=self.device)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('speaker_encoder.'):
                # 移除 'speaker_encoder.' 前缀
                new_key = key[len('speaker_encoder.'):]
                new_state_dict[new_key] = value
            elif key == 'speaker_loss.weight':
                print(f"⚠️ 跳过无关层: {key}")
                continue
            else:
                # 没有前缀，保持原样
                new_state_dict[key] = value
                
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ 缺失的键: {missing_keys[:5]}...")  # 只打印前5个
        if unexpected_keys:
            print(f"⚠️ 多余的键: {unexpected_keys[:5]}...")

        self.model.to(self.device)
        self.model.eval()

        # 注册数据库
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.database = self._load_database()

        # 临时说话人相关
        self.auto_register_unseen = auto_register_unseen
        self.temp_update_momentum = temp_update_momentum
        self.temp_speakers = {}
        self.next_temp_id = 1

    def _load_database(self):
        """加载已注册的说话人（兼容原有格式）"""
        db = {}
        for file in os.listdir(self.db_path):
            if file.endswith('.npy'):
                name = file.replace('.npy', '')
                data = np.load(os.path.join(self.db_path, file), allow_pickle=True).item()
                
                if isinstance(data, np.ndarray):
                    db[name] = {
                        "embedding": torch.FloatTensor(data),
                        "gender": "未知",
                        "age": "未知"
                    }
                else:
                    db[name] = {
                        "embedding": torch.FloatTensor(data["embedding"]),
                        "gender": data.get("gender", "未知"),
                        "age": data.get("age", "未知")
                    }
        return db

    def _load_audio(self, audio_path):
        """加载音频并统一采样率为16000，返回 (1, samples) 形状"""
        waveform, sr = torchaudio.load(audio_path)  # (channels, samples)
        
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # (1, samples)
        
        # 重采样到16000
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # 确保形状为 (1, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # 归一化到 [-1, 1] 范围
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform.to(self.device)  # 返回 (1, samples)

    def _extract_embedding(self, audio_path):
        """从语音中提取声纹特征"""
        # 加载并预处理音频，形状 (1, samples)
        x = self._load_audio(audio_path)
        
        # 模型期望输入 (batch, samples)，x 已经是 (1, samples)
        with torch.no_grad():
            # 注意：aug=False 表示不做 SpecAugment
            emb = self.model(x, aug=False)
        
        # 归一化 embedding
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu()

    def enroll(self, name, audio_path, gender="未知", age="未知"):
        """注册说话人（逻辑与原版本一致）"""
        embeddings = []
        for path in audio_path:
            emb = self._extract_embedding(path)
            embeddings.append(emb)
        
        # 取平均作为该说话人的声纹
        avg_emb = torch.stack(embeddings).mean(dim=0)
        
        # 保存到数据库
        self.database[name] = {
            "embedding": avg_emb,
            "gender": gender,
            "age": age
        }
        save_data = {
            "embedding": avg_emb.numpy(),
            "gender": gender,
            "age": age
        }
        save_path = os.path.join(self.db_path, f"{name}.npy")
        np.save(save_path, save_data)

        print(f"{name}注册成功(使用了{len(audio_path)}条语音)")
        return True

    def identify(self, audio_path):
        """识别说话人（逻辑与原版本一致）"""
        emb = self._extract_embedding(audio_path)

        if not self.database:
            print("数据库为空，请先注册说话人")
            return None, 0, None, None

        # 计算与数据库中每个说话人的相似度
        best_name = None
        best_score = -1
        
        for name, data in self.database.items():
            # 余弦相似度
            score = torch.dot(emb, data["embedding"]) / (emb.norm() * data["embedding"].norm())
            score = score.item()

            if score > best_score:
                best_name = name
                best_score = score

        if best_score > self.threshold:
            info = self.database[best_name]
            return best_name, best_score, info["gender"], info["age"]
        else:
            return None, best_score, None, None

    def identify_or_register(self, audio_path, segment_duration=None):
        """识别+自动注册临时说话人（完全兼容原逻辑）"""
        emb = self._extract_embedding(audio_path)
        
        # 匹配永久数据库
        best_name, best_score, best_gender, best_age = self._match_in_database(emb)
        if best_score > self.threshold:
            return best_name, best_score, "registered", best_gender, best_age
        
        # 匹配临时说话人
        best_temp, best_temp_score, temp_data = self._match_in_temp_with_data(emb)
        if best_temp_score > self.threshold:
            # 移动平均更新声纹
            old_emb = temp_data["emb"]
            count = temp_data["count"]
            
            # 动态权重平均（样本数越多，新样本权重越小）
            alpha = 1.0 / (count + 1)
            updated_emb = old_emb * (1 - alpha) + emb * alpha
            
            self.temp_speakers[best_temp] = {
                "emb": updated_emb,
                "count": count + 1,
                "first_seen": temp_data["first_seen"],
                "total_duration": temp_data.get("total_duration", 0) + (segment_duration or 0),
                "gender": temp_data["gender"],
                "age": temp_data["age"]
            }
            return best_temp, best_temp_score, "temp", temp_data["gender"], temp_data["age"]
        
        # 注册新临时说话人
        if self.auto_register_unseen:
            temp_name = f"Speaker_{self.next_temp_id:02d}"
            self.temp_speakers[temp_name] = {
                "emb": emb,
                "count": 1,
                "first_seen": time.time(),
                "total_duration": segment_duration or 0,
                "gender": "未知",
                "age": "未知"
            }
            self.next_temp_id += 1
            return temp_name, 1.0, "new_temp", "未知", "未知"
        
        return None, best_score, "unknown", None, None

    def _match_in_database(self, emb):
        """匹配永久数据库（辅助函数）"""
        best_name = None
        best_score = -1
        best_gender = None
        best_age = None
        
        for name, data in self.database.items():
            score = torch.dot(emb, data["embedding"]) / (emb.norm() * data["embedding"].norm())
            score = score.item()
            if score > best_score:
                best_name = name
                best_score = score
                best_gender = data["gender"]
                best_age = data["age"]

        return best_name, best_score, best_gender, best_age

    def _match_in_temp_with_data(self, emb):
        """匹配临时说话人（辅助函数）"""
        best_name = None
        best_score = -1
        best_data = None
        
        for name, data in self.temp_speakers.items():
            temp_emb = data["emb"]
            score = torch.dot(emb, temp_emb) / (emb.norm() * temp_emb.norm())
            score = score.item()
            if score > best_score:
                best_name = name
                best_score = score
                best_data = data
        
        return best_name, best_score, best_data

    def update_temp_attributes(self, temp_name, gender, age):
        """更新临时说话人属性"""
        if temp_name in self.temp_speakers:
            if self.temp_speakers[temp_name]["gender"] == "未知" and gender != "未知":
                self.temp_speakers[temp_name]["gender"] = gender
            if self.temp_speakers[temp_name]["age"] == "未知" and age != "未知":
                self.temp_speakers[temp_name]["age"] = age
            return True
        return False

    def get_temp_speaker_stats(self):
        """获取临时说话人统计信息"""
        stats = []
        for name, data in self.temp_speakers.items():
            stats.append({
                "name": name,
                "samples": data["count"],
                "total_duration": data["total_duration"],
                "first_seen": data["first_seen"],
                "gender": data["gender"],
                "age": data["age"]
            })
        return stats

    def clear_temp_speakers(self):
        """清空临时说话人"""
        self.temp_speakers.clear()
        self.next_temp_id = 1

    def promote_temp_to_permanent(self, temp_name, real_name):
        """临时转永久注册（带质量检查）"""
        if temp_name not in self.temp_speakers:
            return False, "临时说话人不存在"
        
        data = self.temp_speakers[temp_name]
        
        # 质量检查
        if data["count"] < 3:
            return False, f"样本数不足({data['count']}<3)，建议录制更多语音"
        
        if data["total_duration"] < 5.0:
            return False, f"语音时长不足({data['total_duration']:.1f}<5秒)，建议录制更长语音"
        
        # 保存到永久库
        emb = data["emb"]
        final_gender = data["gender"]
        final_age = data["age"]

        self.database[real_name] = {
            "embedding": emb,
            "gender": final_gender,
            "age": final_age
        }
        save_data = {
            "embedding": emb.numpy(),
            "gender": final_gender,
            "age": final_age
        }
        save_path = os.path.join(self.db_path, f"{real_name}.npy")
        np.save(save_path, save_data)
        
        # 移除临时说话人
        del self.temp_speakers[temp_name]
        
        return True, f"{real_name} 注册成功（基于{data['count']}段语音，总{data['total_duration']:.1f}秒）"

    def list_enrolled_speakers(self):
        """列出所有注册的说话人"""
        return list(self.database.keys())


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 初始化识别器
    recognizer = SpeakerRecognizerVox(
        model_path="path/to/your/vox_pretrained_model.pth",  # 替换为你的vox预训练权重路径
        threshold=0.7,
        db_path="speaker_db_vox"
    )

    # 1. 注册说话人
    # audio_paths = ["audio/zhangsan_1.wav", "audio/zhangsan_2.wav", "audio/zhangsan_3.wav"]
    # recognizer.enroll("张三", audio_paths, gender="男", age="青年")

    # 2. 识别说话人
    # result = recognizer.identify("audio/zhangsan_test.wav")
    # print(f"识别结果：{result}")

    # 3. 识别或自动注册临时说话人
    # result = recognizer.identify_or_register("audio/unknown_speaker.wav")
    # print(f"识别/注册结果：{result}")