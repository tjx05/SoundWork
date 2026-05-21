from flask import Flask, request, render_template, jsonify
import os
import uuid
import shutil
import wave
import json
from pydub import AudioSegment

from main import MeetingDiary
from recognition.wav2vec2_reco import Wav2vec2Recognizer

app=Flask(__name__)
UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs("temp",exist_ok=True)  # 用于存放临时音频片段

print("加载大模型...")
ai_model = Wav2vec2Recognizer(model_path="emotion_checkpoints/best_wav2vec2_model.pth")

# 初始化日记系统（只加载一次）
print("初始化会议日记系统...")
# 把大模型传进去
diary=MeetingDiary(emotion_recognizer=ai_model)
print("初始化完成")

# 存储注册说话人的额外信息（性别、年龄）
speakers_db = {}  # {name: {"gender": "", "age": "", "embeddings": []}}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """获取所有已注册说话人"""
    speakers = []
    registered_names=diary.recognizer.list_enrolled_speakers()
    for name in registered_names:
        info=speakers_db.get(name, {"gender": "男", "age": "青年"})
        speakers.append({
            "id": hash(name),
            "name": name,
            "gender": info.get("gender","男"),
            "age": info.get("age","青年"),
            "isReg": True
        })
    return jsonify({"speakers": speakers})


@app.route('/api/speakers', methods=['POST'])
def register_speaker():
    """注册说话人"""
    data=request.form
    name=data.get('name', '').strip()
    gender=data.get('gender', '男')
    age=data.get('age', '青年')

    if not name:
        return jsonify({"success": False, "message": "请输入姓名"})

    files=request.files.getlist('audios')
    if len(files)==0:
        return jsonify({"success": False, "message": "请上传声纹语音"})

    # 保存音频文件并提取特征
    audio_paths=[]
    for f in files:
        filename=f"{uuid.uuid4().hex}.wav"
        path=os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        audio_paths.append(path)

    # 调用识别器注册
    diary.recognizer.enroll(name,audio_paths)

    # 存储额外信息
    speakers_db[name] = {
        "gender":gender,
        "age":age,
        "audio_paths":audio_paths
    }


    return jsonify({"success": True, "message": f"{name} 注册成功"})


@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    """识别会议录音"""
    file = request.files.get('audio')
    if not file:
        return jsonify({"success": False, "message": "请上传音频文件"})

    # 保存原始文件
    original_filename = f"{uuid.uuid4().hex}"
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(original_path)
    
    # 读取文件头判断格式
    with open(original_path, 'rb') as f:
        header = f.read(16)
    
    # 判断文件类型
    if header[:4].hex() == '1a45dfa3':  # WebM 文件头
        file_type = 'webm'
        wav_path = original_path + '.wav'
        try:
            audio = AudioSegment.from_file(original_path, format="webm")
            audio.export(wav_path, format="wav")
            process_path = wav_path
        except Exception as e:
            return jsonify({"success": False, "message": f"WebM转换失败: {str(e)}"})
    else:
        # 假设是 WAV 或其他 torchaudio 能读的格式
        process_path = original_path
        wav_path = None  # 不需要清理额外文件
    
    try:
        results = diary.process(process_path)
        
        segments = []
        for r in results:
            # === 修改 3：把性别和年龄拼到名字后面给前端展示 ===
            # 例如展示为 "张三 (中年·男)"
            display_name = r['speaker']
            if 'gender' in r and 'age' in r:
                display_name = f"{r['speaker']} ({r['age']}·{r['gender']})"

            segments.append({
                "time": f"{int(r['start']//60):02d}:{int(r['start']%60):02d}",
                "start": r['start'],  # 添加原始秒数
                "end": r['end'],  # 添加原始秒数
                "person": r['speaker'],
                "mood": r['emotion'],
                "level": "MID",
                "text": r['text']
            })
        return jsonify({"success": True, "segments": segments})
    
    except Exception as e:
        print(f"识别错误: {e}")
        return jsonify({"success": False, "message": str(e)})
    
    finally:
        # 清理临时文件
        if os.path.exists(original_path):
            os.remove(original_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


@app.route('/api/rename_speaker', methods=['POST'])
def rename_speaker():
    """重命名说话人（同步后端）"""
    data = request.json
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    
    if not old_name or not new_name:
        return jsonify({"success": False, "message": "参数错误"})
    
    # 1. 更新 recognizer 的 database
    if old_name in diary.recognizer.database:
        diary.recognizer.database[new_name] = diary.recognizer.database.pop(old_name)
        
        # 更新磁盘上的 .npy 文件
        old_path = os.path.join(diary.recognizer.db_path, f"{old_name}.npy")
        new_path = os.path.join(diary.recognizer.db_path, f"{new_name}.npy")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
    
    # 2. 更新 speakers_db（额外信息）
    if old_name in speakers_db:
        speakers_db[new_name] = speakers_db.pop(old_name)
    
    return jsonify({"success": True, "message": f"{old_name} 已改名为 {new_name}"})


@app.route('/api/parse/start', methods=['POST'])
def start_parse():
    """开始实时解析（需要前端上传音频流，这里暂时保留为模拟，后续可接入WebRTC）"""
    return jsonify({"success": True, "message": "实时解析功能开发中，请先使用上传功能"})


@app.route('/api/parse/stop', methods=['POST'])
def stop_parse():
    """停止解析"""
    return jsonify({"success": True})


@app.route('/api/export', methods=['POST'])
def export_diary():
    """导出日记"""
    data = request.json
    content = data.get('content', '')
    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)