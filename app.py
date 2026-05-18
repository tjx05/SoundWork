from flask import Flask, request, render_template, jsonify
import os
import uuid
import shutil
import wave
import json

from main import MeetingDiary

app=Flask(__name__)
UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs("temp",exist_ok=True)  # 用于存放临时音频片段

# 初始化日记系统（只加载一次）
print("初始化会议日记系统...")
diary=MeetingDiary()
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
    file=request.files.get('audio')
    if not file:
        return jsonify({"success": False,"message": "请上传音频文件"})

    filename=f"{uuid.uuid4().hex}.wav"
    path=os.path.join(UPLOAD_FOLDER,filename)
    file.save(path)

    try:
        results=diary.process(path)

        # 转换为前端需要的格式
        segments=[]
        for r in results:
            segments.append({
                "time": f"{int(r['start']//60):02d}:{int(r['start']%60):02d}",  # 转换为 mm:ss 格式
                "person": r['speaker'],
                "mood": r['emotion'],
                "level": "MID",  # 情感强度暂用 MID
                "text": r['text']
            })
        return jsonify({"success": True,"segments": segments})
    
    except Exception as e:
        return jsonify({"success": False,"message": str(e)})
    
    finally:
        # 清理临时文件
        if os.path.exists(path):
            os.remove(path)


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