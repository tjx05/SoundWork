import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAPA_TDNN(nn.Module):
    def __init__(self, n_mels,n_classes=462,embedding_dim=512):
        super().__init__()
        
        self.conv1=nn.Conv1d(n_mels,512,kernel_size=5,stride=1,padding=2)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(512)
        
        # SE-Res2Blocks
        self.layer1=SE_Res2Block(512,512,scale=8)
        self.layer2=SE_Res2Block(512,512,scale=8)
        self.layer3=SE_Res2Block(512,512,scale=8)
        
        self.conv2=nn.Conv1d(512,1536,kernel_size=1)
        # 注意力统计池化
        self.attention=nn.Sequential(
            nn.Conv1d(1536,256,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256,1536,kernel_size=1),
            nn.Sigmoid()
        )
        
        self.bn2=nn.BatchNorm1d(1536*2)
        self.fc=nn.Linear(1536*2,embedding_dim)
        self.classifier=nn.Linear(embedding_dim,n_classes)
        
    def forward(self,x,is_train=True):
        # x: (batch, n_mels, time)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        
        # 核心特征提取
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        
        x=self.conv2(x)
        
        # 注意力统计池化
        attn=self.attention(x)
        mu=torch.sum(x*attn,dim=2)/torch.sum(attn,dim=2)
        sg=torch.sqrt((torch.sum((x**2)*attn,dim=2)/torch.sum(attn,dim=2))-mu**2)
        x=torch.cat((mu,sg),dim=1)
        
        x=self.bn2(x)
        x=self.fc(x)
        embedding=F.normalize(x,p=2,dim=1)
        # 注册、识别时用
        if not is_train:
            return embedding
        
        # 训练时
        logits=self.classifier(embedding)
        return logits

class SE_Res2Block(nn.Module):
    def __init__(self, in_dim, out_dim, scale=8):
        super().__init__()
        self.scale=scale
        self.conv1=nn.Conv1d(in_dim,out_dim,kernel_size=1)
        self.conv2 = nn.ModuleList([
            nn.Conv1d(out_dim//scale,out_dim//scale,kernel_size=3,padding=1)
            for _ in range(scale-1)
        ])
        self.conv3=nn.Conv1d(out_dim,out_dim,kernel_size=1)
        self.se=SEBlock(out_dim)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        residual=x
        x=self.conv1(x)
        x=self.relu(x)
        
        # Res2Net结构
        split=torch.split(x, x.size(1)//self.scale,dim=1)
        y=split[0]
        out=y
        for i in range(1,self.scale):
            y=split[i]+self.conv2[i-1](y)
            out=torch.cat((out,y),dim=1)
        x=out
        
        x=self.conv3(x)
        x=self.se(x)
        x=x+residual
        x=self.relu(x)
        return x

class SEBlock(nn.Module):
    """通道注意力"""
    def __init__(self,channels,reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)