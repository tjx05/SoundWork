import torch
import os
import numpy as np
import time

from models.ecapa_tdnn import ECAPA_TDNN
from config import config
from preprocessing.extract_fbank import extract_fbank

class SpeakerRecognizer:
    def __init__(self,model_path,threshold=0.6,auto_register_unseen=True,temp_update_momentum=0.7,db_path="speaker_checkpoints/speaker_db"):
        """
        说话人识别器

        输入：
            model_path：模型路径
            threshold：识别阈值
            auto_register_unseen：是否自动注册未注册的说话人
            temp_update_momentum：临时说话人更新动量,momentum=0.7 表示70%保留历史，30%融合新特征
        """
        self.threshold=threshold
        self.device=config.device

        # 加载模型
        self.model=ECAPA_TDNN(
            n_mels=config.n_mels,
        )

        # 加载权重时，跳过不匹配的classifier层
        state_dict=torch.load(model_path, map_location=self.device)
        
        # 过滤掉 classifier 相关的键
        filtered_dict={k: v for k, v in state_dict.items() 
                        if not k.startswith('classifier.')}
        
        # 加载过滤后的权重
        self.model.load_state_dict(filtered_dict,strict=False)
        self.model.to(self.device)
        self.model.eval()

        # 注册数据库（存储embedding + 性别 + 年龄）
        self.db_path=db_path
        os.makedirs(self.db_path,exist_ok=True)
        self.database=self._load_database()

        # 临时说话人
        self.auto_register_unseen=auto_register_unseen
        self.temp_update_momentum=temp_update_momentum # 移动平均系数
        self.temp_speakers={}  # {"emb": tensor, "count": int, "first_seen": float, "gender": str, "age": str}}}
        self.next_temp_id=1


    def _load_database(self):
        """
        从文件夹加载已注册的说话人（包含embedding+性别+年龄）
        """
        db={}
        for file in os.listdir(self.db_path):
            if file.endswith('.npy'):
                name=file.replace('.npy','')
                data=np.load(os.path.join(self.db_path,file),allow_pickle=True).item()

                if isinstance(data, np.ndarray):
                    db[name] = {
                        "embedding": torch.FloatTensor(data),
                        "gender": "未知",
                        "age": "未知"
                    }
                else:
                    db[name] = {
                        "embedding":torch.FloatTensor(data["embedding"]),
                        "gender":data.get("gender","未知"),
                        "age":data.get("age","未知")
                    }
        return db
    
    def _extract_embedding(self,audio_path):
        """
        从语音中提取声纹特征
        """
        # 提取FBank特征
        fbank=extract_fbank(audio_path,n_mels=config.n_mels,max_len=config.max_len,sr=config.sr)
        # 转换为张量
        x=torch.from_numpy(fbank).unsqueeze(0).to(self.device)
        # 前向传播
        with torch.no_grad():
            emb=self.model(x,is_train=False)
        return emb.squeeze(0).cpu()
    
    def enroll(self,name,audio_path,gender="未知",age="未知"):
        """
        注册说话人
        输入：
            name：说话人姓名
            audio_path：语音文件路径 3-5条
            gender：性别（男/女）
            age：年龄段（青年/中年/老年）
        """
        embeddings=[]
        for path in audio_path:
            emb=self._extract_embedding(path)
            embeddings.append(emb)
        
        # 取平均作为该说话人的声纹
        avg_emb=torch.stack(embeddings).mean(dim=0)
        
        # 保存到数据库
        self.database[name]={
            "embedding":avg_emb,
            "gender":gender,
            "age":age
        }
        save_data={
            "embedding":avg_emb.numpy(),
            "gender":gender,
            "age":age
        }
        save_path=os.path.join(self.db_path,f"{name}.npy")
        np.save(save_path,save_data)

        print(f"{name}注册成功(使用了{len(audio_path)}条语音)")
        return True
    
    def identify(self,audio_path):
        """
        识别说话人
        输入：
            audio_path：语音文件路径
        输出：
            best_name：识别到的说话人姓名
            best_score：相似度分数
            gender：性别
            age：年龄段
        """
        emb=self._extract_embedding(audio_path)

        if not self.database:
            print("数据库为空，请先注册说话人")
            return None,0,None,None

        # 计算与数据库中每个说话人的相似度
        best_name=None
        best_score=-1
        
        for name,data in self.database.items():
            # 计算余弦相似度
            score=torch.dot(emb,data["embedding"])/(emb.norm()*data["embedding"].norm())
            score=score.item()

            if score>best_score:
                best_name=name
                best_score=score

        if best_score>self.threshold:
            info=self.database[best_name]
            return best_name,best_score,info["gender"],info["age"]
        else:
            return None,best_score,None,None
        

    def identify_or_register(self, audio_path, segment_duration=None):
        """识别，如果是陌生人则自动注册（支持移动平均更新）"""
        emb=self._extract_embedding(audio_path)
        
        # 匹配永久数据库
        best_name,best_score,best_gender,best_age=self._match_in_database(emb)
        if best_score>self.threshold:
            return best_name,best_score,"registered",best_gender,best_age
        
        # 匹配临时说话人
        best_temp,best_temp_score,temp_data=self._match_in_temp_with_data(emb)
        if best_temp_score>self.threshold:
            # 匹配成功，用移动平均更新声纹
            old_emb=temp_data["emb"]
            count=temp_data["count"]
            
            # 移动平均公式: new_emb = momentum * old_emb + (1-momentum) * new_emb
            # 但更好的做法：根据样本数动态调整权重
            # 样本少时，新样本权重大；样本多时，新样本权重小
            alpha=1.0/(count+1)  # 等价于真实平均
            updated_emb=old_emb*(1-alpha)+emb*alpha
            
            # 或者使用固定的momentum（EMA方式）
            # updated_emb = old_emb * self.temp_update_momentum + emb * (1 - self.temp_update_momentum)
            
            self.temp_speakers[best_temp]={
                "emb": updated_emb,
                "count": count+1,
                "first_seen": temp_data["first_seen"],
                "total_duration": temp_data.get("total_duration", 0)+(segment_duration or 0),
                "gender":temp_data["gender"],
                "age":temp_data["age"]
            }
            return best_temp,best_temp_score,"temp",temp_data["gender"],temp_data["age"]
        
        # 完全陌生，注册新临时说话人
        if self.auto_register_unseen:
            temp_name=f"Speaker_{self.next_temp_id:02d}"
            self.temp_speakers[temp_name] = {
                "emb": emb,
                "count": 1,
                "first_seen": time.time(),
                "total_duration": segment_duration or 0,
                "gender":"未知",
                "age":"未知"
            }
            self.next_temp_id += 1
            return temp_name,1.0,"new_temp","未知","未知"
        
        return None,best_score,"unknown",None,None
    
    def _match_in_database(self, emb):
        """匹配永久数据库"""
        best_name=None
        best_score=-1
        for name,data in self.database.items():
            score=torch.dot(emb,data["embedding"])/(emb.norm()*data["embedding"].norm())
            score=score.item()
            if score>best_score:
                best_name=name
                best_score=score
                best_gender=data["gender"]
                best_age=data["age"]

        return best_name,best_score,best_gender,best_age
    
    def _match_in_temp_with_data(self, emb):
        """匹配临时说话人，返回详细信息"""
        best_name=None
        best_score=-1
        best_data=None
        
        for name,data in self.temp_speakers.items():
            temp_emb=data["emb"]
            score=torch.dot(emb, temp_emb)/(emb.norm()*temp_emb.norm())
            score=score.item()
            if score > best_score:
                best_name=name
                best_score=score
                best_data=data
        
        return best_name,best_score,best_data
    
    def update_temp_attributes(self,temp_name,gender,age):
        """更新临时说话人的性别年龄"""
        if temp_name in self.temp_speakers:
            # 如果已经是"未知"，才更新；如果已经有值，不覆盖
            if self.temp_speakers[temp_name]["gender"] == "未知" and gender != "未知":
                self.temp_speakers[temp_name]["gender"] = gender
            if self.temp_speakers[temp_name]["age"] == "未知" and age != "未知":
                self.temp_speakers[temp_name]["age"] = age
            return True
        return False
    
    def get_temp_speaker_stats(self):
        """获取临时说话人统计信息"""
        stats = []
        for name,data in self.temp_speakers.items():
            stats.append({
                "name": name,
                "samples": data["count"],
                "total_duration": data["total_duration"],
                "first_seen": data["first_seen"],
                "gender":data["gender"],
                "age":data["age"]
            })
        return stats
    
    def clear_temp_speakers(self):
        """清空临时说话人（新会议开始前调用）"""
        self.temp_speakers.clear()
        self.next_temp_id=1
    
    def promote_temp_to_permanent(self, temp_name, real_name):
        """将临时说话人转为永久注册（附带质量检查）"""
        if temp_name not in self.temp_speakers:
            return False,"临时说话人不存在"
        
        data=self.temp_speakers[temp_name]
        
        # 质量检查，样本数太少不给注册
        if data["count"]<3:
            return False, f"样本数不足({data['count']}<3)，建议录制更多语音"
        
        if data["total_duration"]<5.0:
            return False, f"语音时长不足({data['total_duration']:.1f}<5秒)，建议录制更长语音"
        
        # 使用累积的移动平均声纹
        emb=data["emb"]
        final_gender=data["gender"]
        final_age=data["age"]

        self.database[real_name] = {
            "embedding": emb,
            "gender": final_gender,
            "age": final_age
        }
        save_data={
            "embedding": emb.numpy(),
            "gender": final_gender,
            "age": final_age
        }

        save_path=os.path.join(self.db_path, f"{real_name}.npy")
        np.save(save_path,save_data)
        
        # 从临时中移除
        del self.temp_speakers[temp_name]
        
        return True, f"{real_name} 注册成功（基于{data['count']}段语音，总{data['total_duration']:.1f}秒）"

    def list_enrolled_speakers(self):
        """
        列出所有注册的说话人
        """
        return list(self.database.keys())

        