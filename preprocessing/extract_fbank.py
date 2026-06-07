import librosa
import numpy as np

def extract_fbank(file_path,n_mels=80,max_len=250,sr=16000):
    """
    提取FBank特征

    输入：
        file_path:可以是文件路径(str) 或 波形数组(np.ndarray)
        n_mels:梅尔滤波器数量
        max_len:最大音频长度
        sr:采样率
    输出：
        log_mel: 对数FBank特征矩阵(n_mels,max_len)
    """
    # 加载音频
    if isinstance(file_path, str):
        y,sr = librosa.load(file_path, sr=sr)
    else:
        y = file_path

    # 预加重
    y=librosa.effects.preemphasis(y,coef=0.97)

    # 提取梅尔谱图
    mel_spec=librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=512,
        hop_length=256
    )

    # 转成对数FBank
    log_mel=librosa.power_to_db(mel_spec,ref=np.max)

    # 统一长度
    if log_mel.shape[1]<max_len:
        pad_len=max_len-log_mel.shape[1]
        log_mel=np.pad(log_mel,((0,0),(0,pad_len)),mode='constant')
    else:
        log_mel=log_mel[:, :max_len]
    
    return log_mel