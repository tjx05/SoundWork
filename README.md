# SoundWork
语音信息处理大作业——语音合成

## 项目结构
```
SoundWork/
├── app.py                      # Flask 入口
├── templates/                  # 前端页面
│   └── index.html
│
├── static/                     # 前端资源
│   ├── css/
│   └── js/
│
│  # ------------后端------------
│
├── main.py                    # 核心集成处理
│
├── Data  # 数据集目录
│   ├── CREMA-D
│   └── TIMIT
│
├── models                   # 模型目录
│   ├── __init__.py
│   ├── wav2vec2.py          # wav2vec2模型代码
│   ├── ecapa_tdnn.py        # ECAPA-TDNN模型代码
│   └── whisper_asr.py       # whisper ASR模型代码
│
├── preprocessing            # 数据预处理目录
│   ├── __init__.py
│   ├── compute_emotion_bias.py # 计算情感偏置
│   ├── extract_fbank.py     # 提取FBank特征
│   ├── preprocess_cremad.py # 预处理CREMA-D数据集
│   ├── preprocess_timit.py  # 预处理TIMIT数据集
│   ├── timit_dataset.py     # TIMIT数据集
│   └── wav2vec2_dataset.py  # wav2vec2数据集
│
├── recognition              # 识别/推理模块
│   ├── __init__.py
│   ├── emotion_compensated_reco.py # 情感补偿识别模块
│   ├── speaker_reco.py      # 说话人识别模块
│   └── wav2vec2_reco.py      # 情感识别模块
│
├── emotion_checkpoints      # 情感识别检查点目录
│
├── speaker_checkpoints      # 说话人识别检查点目录
│   ├── speaker_db           # 已注册的说话人数据库
│   ├── best_model.pth       # 最佳模型检查点
│   ├── training_curves.png  # 训练曲线图片
│   └── training_history.json  # 训练历史记录
│
├── tests                    # 测试目录
│   ├── test_ecapa_open.py   # 测试说话人识别模块
│   ├── test_ecapa_tdnn.py   # 测试ECAPA-TDNN模型的关闭集
│   └── test_emotion_compensation.py # 测试情感补偿识别模块
│   
├── training                 # 训练目录
│   ├── train_speaker.py     # 训练说话人识别模型
│   └── train_wav2vec2.py    # 训练情感识别模型
│
├── utils                    # 工具目录
│   ├── __init__.py
│   └── plot_curves.py       # 绘制训练曲线图片
│ 
├── download_model.py        # wav2vec2模型下载脚本
├── .gitignore
├── config.py
└── README.md
```

## 快速开始
```bash
# 1.安装依赖
pip install -r requirements.txt

```

## 注意事项
1. 自行去往官网下载ffmpeg，并添加到环境变量中。
2. 需要下载wav2vec2模型：运行 python download_model.py

