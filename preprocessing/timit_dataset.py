import librosa
import os
import numpy as np
import json
import pandas as pd
import torch
from preprocessing.extract_fbank import extract_fbank

class TIMITDataset:
    def __init__(self,json_path,n_mels,max_len,sr):
        self.n_mels=n_mels
        self.max_len=max_len
        self.sr=sr

        # 解析JSON
        with open(json_path,'r',encoding='utf-8') as f:
            data_info=json.load(f)

        self.data_list = []
        for item in data_info:
            raw_path=item['filepath'].replace('\\', '/')
            raw_path=raw_path.replace('_.wav', '.WAV')
            
            self.data_list.append({
                'path':raw_path,
                'speaker':item['speaker_id']
            })
        
        # 说话人标签映射
        speaker_list = sorted(set([d['speaker'] for d in self.data_list]))
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_list)}

        print(f"加载 {len(self.data_list)} 条数据，{len(speaker_list)} 个说话人")
        print(f"路径示例: {self.data_list[0]['path']}")
    
    def __getitem__(self,idx):
        item=self.data_list[idx]
        full_path=os.path.join("Data/TIMIT", item['path'])
        
        fbank=extract_fbank(full_path,self.n_mels,self.max_len,self.sr)
        fbank_tensor=torch.FloatTensor(fbank)
        label=self.speaker_to_idx[item['speaker']]
        return fbank_tensor, label
    
    def __len__(self):
        return len(self.data_list)