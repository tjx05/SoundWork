from pyannote.audio import Pipeline
from pyannote.audio import Model
import torch

MODEL_DIR = r"D:\SoundWork-tjx\SoundWork-tjx\models"



from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model

# 加载segmentation模型
segmentation_model = Model.from_pretrained(
    f"{MODEL_DIR}/3.0/config.yaml",
    map_location="cpu"
)

# 加载pipeline并手动注入模型
pipeline = SpeakerDiarization(
    segmentation=segmentation_model,
    embedding=f"{MODEL_DIR}/wespeaker-voxceleb-resnet34-LM",
    embedding_batch_size=32,
    segmentation_batch_size=32,
)

pipeline = pipeline.to(torch.device("cuda"))
print("✅ pyannote 本地加载成功！")