import os
from transformers import Wav2Vec2Model

def main():
    # 你想要下载的模型名称
    model_id = "facebook/wav2vec2-base"
    
    # 你想保存在本地的绝对/相对路径
    save_directory = "./local_base_model/wav2vec2-base"

    print(f"🚀 正在提取基座模型: {model_id} ...")
    print("这可能需要几秒钟（如果之前运行过）或几分钟（如果网络较慢），请稍候...")

    try:
        # 1. 从 HuggingFace (或本地缓存) 加载模型
        model = Wav2Vec2Model.from_pretrained(model_id)
        
        # 2. 将模型的所有必需文件（config.json, weights等）固化保存到指定目录
        model.save_pretrained(save_directory)
        
        print(f"\n✅ 提取成功！")
        print(f"基座模型已永久保存在本地目录: {os.path.abspath(save_directory)}")
        print("以后代码里直接写这个路径，就可以实现真正的无网离线、秒级加载了！")
        
    except Exception as e:
        print(f"\n❌ 提取失败，请检查网络或环境变量。错误详情:\n{e}")

if __name__ == "__main__":
    main()