from flask import Flask, request, render_template, jsonify
import os
import uuid
import wave
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 模拟数据库
speakers_db = {}  # {name: {"gender": "", "age": "", "embeddings": []}}
parse_data_store = []  # 存储解析结果

# 未知说话人计数器
unknown_counter = {
    "青年女": 0, "青年男": 0,
    "中年女": 0, "中年男": 0,
    "老年女": 0, "老年男": 0
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/speakers', methods=['GET'])
def get_speakers():
    """获取所有已注册说话人"""
    speakers = []
    for name, info in speakers_db.items():
        speakers.append({
            "id": hash(name),
            "name": name,
            "gender": info["gender"],
            "age": info["age"],
            "isReg": True
        })
    return jsonify({"speakers": speakers})


@app.route('/api/speakers', methods=['POST'])
def register_speaker():
    """注册说话人"""
    data = request.form
    name = data.get('name', '').strip()
    gender = data.get('gender', '男')
    age = data.get('age', '青年')

    if not name:
        return jsonify({"success": False, "message": "请输入姓名"})

    files = request.files.getlist('audios')
    if len(files) == 0:
        return jsonify({"success": False, "message": "请上传声纹语音"})

    # 保存音频文件并提取特征
    audio_paths = []
    for f in files:
        filename = f"{uuid.uuid4().hex}.wav"
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        audio_paths.append(path)

    # 存储到数据库
    speakers_db[name] = {
        "gender": gender,
        "age": age,
        "audio_paths": audio_paths
    }

    # TODO: 调用后端特征提取和注册
    # from recognition.speaker_recognition import SpeakerRecognizer
    # recognizer.enroll(name, audio_paths)

    return jsonify({"success": True, "message": f"{name} 注册成功"})


@app.route('/api/recognize', methods=['POST'])
def recognize_audio():
    """识别会议录音"""
    file = request.files.get('audio')
    if not file:
        return jsonify({"success": False, "message": "请上传音频文件"})

    filename = f"{uuid.uuid4().hex}.wav"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # TODO: 调用后端解析模块
    # from main import process_audio
    # results = process_audio(path)

    # 模拟返回结果（与前端格式匹配）
    results = [
        {"time": "00:05", "person": "张三", "mood": "开心", "level": "HI", "text": "我们对齐一下本周项目进度"},
        {"time": "00:13", "person": "青年女1", "mood": "平静", "level": "LO", "text": "前端界面已全部开发完成"},
        {"time": "00:20", "person": "青年女1", "mood": "开心→失落", "level": "MID", "text": "进度还行，就是早上没来得及吃饭"},
        {"time": "00:28", "person": "张三", "mood": "理解", "level": "LO", "text": "没事，先把语音解析模块联调完"},
        {"time": "00:35", "person": "青年男1", "mood": "严肃", "level": "MID", "text": "后端接口我这边已经调试完毕"},
        {"time": "00:42", "person": "中年女1", "mood": "温和", "level": "LO", "text": "那我们定一下下周联调时间"},
        {"time": "00:50", "person": "张三", "mood": "果断", "level": "HI", "text": "那就下周二下午统一联调"},
        {"time": "00:58", "person": "中年男1", "mood": "认真", "level": "MID", "text": "我这边提前准备好测试用例"},
        {"time": "01:05", "person": "青年女2", "mood": "轻松", "level": "LO", "text": "好的，我也同步准备页面"},
        {"time": "01:12", "person": "张三", "mood": "总结", "level": "HI", "text": "大家各自准备，周二准时对接"}
    ]

    return jsonify({"success": True, "segments": results})


@app.route('/api/parse/start', methods=['POST'])
def start_parse():
    """开始实时解析（模拟）"""
    global parse_data_store, unknown_counter
    parse_data_store = []
    unknown_counter = {
        "青年女": 0, "青年男": 0,
        "中年女": 0, "中年男": 0,
        "老年女": 0, "老年男": 0
    }
    return jsonify({"success": True})


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