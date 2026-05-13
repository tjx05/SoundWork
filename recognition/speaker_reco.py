import torch
import os
import numpy as np

from models.ecapa_tdnn import ECAPA_TDNN
from config import config
from preprocessing.extract_fbank import extract_fbank

class SpeakerRecognizer:
    def __init__(self,model_path,threshold=0.6):
        """
        说话人识别器

        输入：
            model_path：模型路径
            threshold：识别阈值
        """
        self.threshold=threshold
        self.device=config.device

        # 加载模型
        self.model=ECAPA_TDNN(
            n_mels=config.n_mels,
        )
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 注册数据库
        self.db_path="speaker_checkpoints/speaker_db"
        os.makedirs(self.db_path,exist_ok=True)
        self.database=self._load_database()

    def _load_database(self):
        """
        从文件夹加载已注册的说话人
        """
        db={}
        for file in os.listdir(self.db_path):
            if file.endswith('.npy'):
                name=file.replace('.npy','')
                emb=np.load(os.path.join(self.db_path,file))
                db[name]=torch.FloatTensor(emb)
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
    
    def enroll(self,name,audio_path):
        """
        注册说话人
        输入：
            name：说话人姓名
            audio_path：语音文件路径 3-5条
        """
        embeddings=[]
        for path in audio_path:
            emb=self._extract_embedding(path)
            embeddings.append(emb)
        
        # 取平均作为该说话人的声纹
        avg_emb=torch.stack(embeddings).mean(dim=0)
        
        # 保存到数据库
        self.database[name]=avg_emb
        save_path=os.path.join(self.db_path,f"{name}.npy")
        np.save(save_path,avg_emb.numpy())

        print(f"{name}注册成功(使用了{len(audio_path)}条语音)")
        return True
    
    def identify(self,audio_path):
        """
        识别说话人
        输入：
            audio_path：语音文件路径
        """
        emb=self._extract_embedding(audio_path)

        if not self.database:
            print("数据库为空，请先注册说话人")
            return None,0

        # 计算与数据库中每个说话人的相似度
        best_name=None
        best_score=-1
        
        for name,enroll_emb in self.database.items():
            # 计算余弦相似度
            score=torch.dot(emb,enroll_emb)/(emb.norm()*enroll_emb.norm())
            score=score.item()

            if score>best_score:
                best_name=name
                best_score=score

        if best_score>self.threshold:
            return best_name,best_score
        else:
            return None,best_score
        
    def list_enrolled_speakers(self):
        """
        列出所有注册的说话人
        """
        return list(self.database.keys())

        