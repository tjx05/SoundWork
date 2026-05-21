// ========== 全局变量 ==========
let speakerList = [];
let parseData = [];
let mediaRecorder = null;
let isRecording = false;
let audioChunks = [];
let selectedAudioFile = null;  // 存储选中的文件
let isParsing = false; // 标记是否正在解析
let isPaused = false;  // 是否处于暂停状态

let totalOffset = 0;  // 记录当前录音已处理的总时长

// DOM 元素
const langSwitch = document.getElementById('langSwitch');
const uploadBox = document.getElementById('uploadBox');
const audioFile = document.getElementById('audioFile');
const regBtn = document.getElementById('regBtn');
const speakerListDom = document.getElementById('speakerList');
const chatBox = document.getElementById('chatBox');
const diaryBox = document.getElementById('diaryBox');
const startRec = document.getElementById('startRec');
const pauseRec = document.getElementById('pauseRec');
const stopRec = document.getElementById('stopRec');
const downBtn = document.getElementById('downBtn');
const startParseBtn = document.getElementById('startParseBtn');
const selectedFileDiv = document.getElementById('selectedFile');
const fileNameSpan = document.getElementById('fileName');

// ========== 辅助函数 ==========
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>]/g, function(m) {
    if (m === '&') return '&amp;';
    if (m === '<') return '&lt;';
    if (m === '>') return '&gt;';
    return m;
  });
}

// 生成日记
function generateDiary() {
  let txt = '【多人会议结构化日记】\n';
  txt += '——————————————\n';
  parseData.forEach(d => {
    txt += `[${d.time}] ${d.person} [${d.mood}]：${d.text}\n`;
  });
  diaryBox.innerText = txt;
  diaryBox.scrollTop = diaryBox.scrollHeight;
}

// 渲染说话人列表
function renderSpeaker() {
  speakerListDom.innerHTML = '';
  speakerList.forEach(item => {
    let div = document.createElement('div');
    div.className = 'speaker-item';
    div.dataset.id = item.id;
    div.innerHTML = `
      <div>
        <span class="name">${escapeHtml(item.name)}</span>
      </div>
      <span class="auto-tag">${item.isReg ? '已注册' : '自动识别'}</span>
    `;
    div.ondblclick = () => editSpeakerGlobal(item.id);
    speakerListDom.appendChild(div);
  });
}

// 加载已注册说话人
async function loadSpeakers() {
  try {
    const response = await fetch('/api/speakers');
    const data = await response.json();
    if (data.speakers) {
      speakerList = data.speakers;
      renderSpeaker();
    }
  } catch (err) {
    console.error('加载说话人失败:', err);
  }
}

// 编辑说话人
async function editSpeakerGlobal(id) {
  const sp = speakerList.find(s => s.id === id);
  if (!sp) return;
  
  const oldName = sp.name;
  const newName = prompt('修改姓名：', sp.name);
  if (!newName) return;
  
  // 调用后端接口同步
  try {
    const response = await fetch('/api/rename_speaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_name: oldName, new_name: newName })
    });
    const data = await response.json();
    if (!data.success) {
      alert('修改失败: ' + data.message);
      return;
    }
  } catch (err) {
    alert('网络错误: ' + err.message);
    return;
  }
  
  // 更新前端数据（与原来一样）
  sp.name = newName;
  renderSpeaker();
  
  parseData.forEach(d => {
    if (d.person === oldName) d.person = newName;
  });
  
  // 更新 DOM
  const chatItems = chatBox.querySelectorAll('.chat-item');
  chatItems.forEach((item, idx) => {
    if (idx < parseData.length && parseData[idx].person === newName) {
      const nameSpan = item.querySelector('.chat-name');
      if (nameSpan) {
        nameSpan.textContent = newName;
        nameSpan.ondblclick = () => editChatName(idx);
      }
    }
  });
  
  generateDiary();
}

// 添加单条对话
function addChatItem(d, index) {
  const item = document.createElement('div');
  item.className = 'chat-item';
  item.innerHTML = `
    <div class="chat-top-row">
      <div class="chat-left-info">
        <span class="chat-name" ondblclick="editChatName(${index})">${escapeHtml(d.person)}</span>
        <span class="chat-time">${d.time}</span>
      </div>
      <span class="chat-emotion">${d.mood} ${d.level}</span>
    </div>
    <div class="chat-bubble" ondblclick="editChatText(${index}, this)">
      <span class="text">${escapeHtml(d.text)}</span>
    </div>
  `;
  chatBox.appendChild(item);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// 添加对话
function addChat(time, person, mood, level, text) {
  parseData.push({ time, person, mood, level, text });
  addChatItem(parseData[parseData.length - 1], parseData.length - 1);
  generateDiary();
}

// 批量添加对话
function addChats(segments) {
  parseData = [];
  chatBox.innerHTML = '';
  
  for (const seg of segments) {
    addChat(seg.time, seg.person, seg.mood, seg.level, seg.text);
  }
  generateDiary();
  downBtn.disabled = false;
}

// 编辑对话姓名
window.editChatName = function(idx) {
  const oldName = parseData[idx].person;
  const newName = prompt('修改本条说话人姓名：', oldName);
  if (!newName || newName === oldName) return;

  parseData[idx].person = newName;

  const chatItems = chatBox.querySelectorAll('.chat-item');
  if (chatItems[idx]) {
    const nameSpan = chatItems[idx].querySelector('.chat-name');
    nameSpan.textContent = newName;
    nameSpan.ondblclick = () => editChatName(idx);
  }

  generateDiary();
}

// 编辑对话文本
window.editChatText = function(idx, bubbleEl) {
  const originalText = parseData[idx].text;
  const textarea = document.createElement('textarea');
  textarea.value = originalText;
  textarea.rows = 2;
  textarea.style.width = '100%';
  textarea.style.border = 'none';
  textarea.style.background = 'transparent';
  textarea.style.resize = 'none';
  textarea.style.fontSize = '14px';
  textarea.style.lineHeight = '1.5';
  textarea.style.fontFamily = 'inherit';
  textarea.style.outline = 'none';
  textarea.style.padding = '0';
  textarea.style.margin = '0';

  const save = () => {
    const newText = textarea.value.trim();
    if (newText) {
      parseData[idx].text = newText;
      bubbleEl.innerHTML = `<span class="text">${escapeHtml(newText)}</span>`;
      bubbleEl.ondblclick = () => editChatText(idx, bubbleEl);
      generateDiary();
    } else {
      bubbleEl.innerHTML = `<span class="text">${escapeHtml(originalText)}</span>`;
      bubbleEl.ondblclick = () => editChatText(idx, bubbleEl);
    }
  };

  textarea.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      save();
    }
    if (e.key === 'Escape') {
      bubbleEl.innerHTML = `<span class="text">${escapeHtml(originalText)}</span>`;
      bubbleEl.ondblclick = () => editChatText(idx, bubbleEl);
    }
  };

  textarea.onblur = save;
  bubbleEl.innerHTML = '';
  bubbleEl.appendChild(textarea);
  textarea.focus();
  textarea.select();
}

// 禁用/启用录音控制按钮
function setRecordBtnStatus(disabled) {
  startRec.disabled = disabled;
  pauseRec.disabled = disabled;
  stopRec.disabled = disabled;
}

// 执行解析
async function parseAudio() {
  if (!selectedAudioFile || isParsing) {
    alert('请先上传音频文件');
    return;
  }

  // 标记正在解析
  isParsing = true;
  // 禁用开始解析按钮
  startParseBtn.disabled = true;
  // 禁用录音控制按钮
  setRecordBtnStatus(true);
  // 修改按钮文字显示加载状态
  startParseBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 解析中...';
  
  // 在对话和日记区域显示解析中状态
  chatBox.innerHTML = '<div style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-spinner fa-spin"></i> 正在解析音频，请稍候...</div>';
  diaryBox.innerHTML = '<div style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-spinner fa-spin"></i> 正在生成会议记录，请稍候...</div>';

  const formData = new FormData();
  formData.append('audio', selectedAudioFile);

  try {
    const response = await fetch('/api/recognize', { method: 'POST', body: formData });
    const data = await response.json();
    
    if (data.success && data.segments) {
      // 解析成功，渲染数据
      addChats(data.segments);
      // 自定义友好提示（替换系统alert）
      showCustomToast(`解析完成！共识别 ${data.segments.length} 段对话`);
    } else {
      chatBox.innerHTML = '<div style="text-align:center; color:#ff6b6b; padding:20px;"><i class="fa fa-exclamation-circle"></i> 解析失败：' + (data.message || '未知错误') + '</div>';
      diaryBox.innerHTML = '<div style="text-align:center; color:#ff6b6b; padding:20px;"><i class="fa fa-exclamation-circle"></i> 解析失败，请重试</div>';
      showCustomToast('❌ 解析失败：' + (data.message || '未知错误'), 'error');
    }
  } catch (err) {
    chatBox.innerHTML = '<div style="text-align:center; color:#ff6b6b; padding:20px;"><i class="fa fa-exclamation-circle"></i> 解析失败：' + err.message + '</div>';
    diaryBox.innerHTML = '<div style="text-align:center; color:#ff6b6b; padding:20px;"><i class="fa fa-exclamation-circle"></i> 解析失败，请检查网络</div>';
    showCustomToast('❌ 解析失败：' + err.message, 'error');
  } finally {
    // 恢复状态
    isParsing = false;
    startParseBtn.disabled = false;
    startParseBtn.innerHTML = '<i class="fa fa-play"></i> 开始解析';
    setRecordBtnStatus(false);
  }
}

// 自定义友好提示框（替换系统alert）
function showCustomToast(message, type = 'success') {
  // 创建提示框DOM
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.top = '20px';
  toast.style.left = '50%';
  toast.style.transform = 'translateX(-50%)';
  toast.style.padding = '12px 24px';
  toast.style.borderRadius = '8px';
  toast.style.backgroundColor = type === 'success' ? '#67b99a' : '#ff6b6b';
  toast.style.color = '#fff';
  toast.style.fontSize = '14px';
  toast.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
  toast.style.zIndex = '9999';
  toast.style.opacity = '0';
  toast.style.transition = 'opacity 0.3s ease, top 0.3s ease';
  toast.innerHTML = `<i class="fa ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i> ${message}`;
  
  // 添加到页面
  document.body.appendChild(toast);
  
  // 显示动画
  setTimeout(() => {
    toast.style.opacity = '1';
    toast.style.top = '30px';
  }, 10);
  
  // 3秒后隐藏并移除
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.top = '20px';
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 300);
  }, 3000);
}

// 注册说话人
regBtn.onclick = async function() {
  const name = document.getElementById('userName').value.trim();
  const files = document.getElementById('spkAudioUp').files;

  if (!name) return showCustomToast('请输入姓名', 'error');
  if (files.length === 0) return showCustomToast('请上传声纹语音', 'error');

  const formData = new FormData();
  formData.append('name', name);
  for (let i = 0; i < files.length; i++) {
    formData.append('audios', files[i]);
  }

  try {
    const response = await fetch('/api/speakers', { method: 'POST', body: formData });
    const data = await response.json();
    if (data.success) {
      showCustomToast(data.message);
      document.getElementById('userName').value = '';
      document.getElementById('spkAudioUp').value = '';
      await loadSpeakers();
    } else {
      showCustomToast('注册失败: ' + data.message, 'error');
    }
  } catch (err) {
    showCustomToast('网络错误: ' + err.message, 'error');
  }
}

// 上传录音 - 只选择文件，不立即解析
uploadBox.onclick = () => audioFile.click();
audioFile.onchange = function(e) {
  if (!e.target.files[0]) return;
  
  selectedAudioFile = e.target.files[0];
  // 不再显示选中文件的小区域，只修改上传框提示
  uploadBox.innerHTML = `<i class="fa fa-check-circle"></i> 已选择文件：${selectedAudioFile.name}<br><small>点击"开始解析"按钮开始处理</small>`;
  startParseBtn.disabled = false;
}

// 开始解析按钮
startParseBtn.onclick = parseAudio;

// 实时录音
// 实时录音 - 前端 VAD 智能分段版（每次说话新建 Recorder）
let audioContext = null;
let analyserNode = null;
let sourceNode = null;
let isSpeaking = false;
let vadMonitorInterval = null;
let lastSpeechTime = 0;
let currentStream = null;

async function startRecording() {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // 创建 AudioContext 用于音量检测
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;
        sourceNode = audioContext.createMediaStreamSource(currentStream);
        sourceNode.connect(analyserNode);
        await audioContext.resume();
        
        isRecording = true;
        isSpeaking = false;
        isPaused = false;  // 重置暂停状态
        
        // 音量监测
        startVolumeMonitor();
        
        chatBox.innerHTML = '<div id="recording-status" style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-microphone"></i> 智能分段录音中...</div>';
        diaryBox.innerHTML = '<div style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-microphone"></i> 说完一段话后自动识别...</div>';
    } catch (err) {
        showCustomToast('无法访问麦克风: ' + err.message, 'error');
        return false;
    }
    return true;
}

// 提取音量监测为独立函数
function startVolumeMonitor() {
    if (vadMonitorInterval) clearInterval(vadMonitorInterval);
    
    vadMonitorInterval = setInterval(() => {
        if (!analyserNode || isPaused) return;  // 暂停时跳过检测
        const volume = getVolumeLevel();
        const now = Date.now();
        
        if (volume > 0.02) {
            lastSpeechTime = now;
            if (!isSpeaking) {
                isSpeaking = true;
                startNewSegment();
            }
        } else {
            if (isSpeaking && (now - lastSpeechTime) > 600) {
                isSpeaking = false;
                endCurrentSegment();
            }
        }
    }, 100);
}

let currentRecorder = null;
let currentSegmentChunks = [];

function startNewSegment() {
    console.log('开始新片段');
    currentSegmentChunks = [];
    currentRecorder = new MediaRecorder(currentStream, { mimeType: 'audio/webm' });
    currentRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) currentSegmentChunks.push(e.data);
    };
    currentRecorder.start();
}

function endCurrentSegment() {
    if (currentRecorder && currentRecorder.state === 'recording') {
        console.log('结束片段，准备上传');
        currentRecorder.stop();
        currentRecorder.onstop = () => {
            if (currentSegmentChunks.length > 0) {
                const blob = new Blob(currentSegmentChunks, { type: 'audio/webm' });
                if (blob.size > 5000) {
                    const formData = new FormData();
                    formData.append('audio', blob, `segment_${Date.now()}.webm`);
                    parseAudioBlob(formData);
                }
            }
            currentRecorder = null;
            currentSegmentChunks = [];
        };
    }
}

// 获取音量
function getVolumeLevel() {
    if (!analyserNode) return 0;
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteTimeDomainData(dataArray);
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        const v = (dataArray[i] - 128) / 128;
        sum += v * v;
    }
    return Math.sqrt(sum / dataArray.length);
}


// 上传片段
async function uploadSegment(blob) {
    if (blob.size < 5000) {
        console.log('片段太小，跳过');
        return;
    }
    const formData = new FormData();
    formData.append('audio', blob, `segment_${Date.now()}.webm`);
    await parseAudioBlob(formData);
}

function stopRecording(onComplete) {
    console.log('stopRecording 被调用');
    
    // 1. 立即停止音量监测
    if (vadMonitorInterval) {
        clearInterval(vadMonitorInterval);
        vadMonitorInterval = null;
    }
    
    isRecording = false;
    isSpeaking = false;
    isPaused = false;  // 重置暂停状态
    
    // 2. 停止当前录音器
    if (currentRecorder && currentRecorder.state === 'recording') {
        currentRecorder.stop();
        currentRecorder.onstop = () => {
            if (currentSegmentChunks.length > 0) {
                const blob = new Blob(currentSegmentChunks, { type: 'audio/webm' });
                if (blob.size > 5000) {
                    const formData = new FormData();
                    formData.append('audio', blob, `segment_${Date.now()}.webm`);
                    parseAudioBlob(formData).finally(() => {
                        if (onComplete) onComplete();
                    });
                } else {
                    if (onComplete) onComplete();
                }
            } else {
                if (onComplete) onComplete();
            }
            currentRecorder = null;
            currentSegmentChunks = [];
        };
    } else {
        if (onComplete) onComplete();
    }
    
    // 3. 清理音频资源
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    analyserNode = null;
}

let segmentSpeakerCounter = 0;  // 全局变量
async function parseAudioBlob(formData) {
    if (isParsing) return;
    
    isParsing = true;
    startParseBtn.disabled = true;
    setRecordBtnStatus(true);
    startParseBtn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> 解析中...';
    
    // 只在日记区域显示状态，不清空对话区域
    diaryBox.innerHTML = '<div style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-spinner fa-spin"></i> 正在生成会议记录，请稍候...</div>';
    
    try {
        const response = await fetch('/api/recognize', { method: 'POST', body: formData });
        const data = await response.json();
        
        if (data.success && data.segments) {
            // 追加显示，不清空
            for (const seg of data.segments) {
                console.log('片段原始时间:', seg.start, seg.end);
                // 加上偏移量，得到在整个录音中的真实时间
                const realStart = totalOffset + seg.start;
                const timeStr = formatTime(realStart);  // 格式化为 mm:ss
                console.log('seg.start:', seg.start, 'seg.end:', seg.end, 'totalOffset:', totalOffset);

                let speaker = seg.person;
                // 如果是未注册的 SPEAKER_XX，统一改成递增的序号
                if (speaker && speaker.startsWith('SPEAKER_')) {
                    segmentSpeakerCounter++;
                    speaker = `SPEAKER_${segmentSpeakerCounter.toString().padStart(2, '0')}`;
                }
                addChat(timeStr, speaker, seg.mood, seg.level, seg.text);
            }

            // 更新偏移量：加上最后一个片段的结束时间
            if (data.segments.length > 0) {
                totalOffset += data.segments[data.segments.length - 1].end;
            }

            showCustomToast(`识别成功！`);
        } else {
            showCustomToast('识别失败：' + (data.message || '未知错误'), 'error');
        }
    } catch (err) {
        showCustomToast('识别失败：' + err.message, 'error');
    } finally {
        isParsing = false;
        startParseBtn.disabled = false;
        startParseBtn.innerHTML = '<i class="fa fa-play"></i> 开始解析';
        setRecordBtnStatus(false);
        generateDiary();  // 刷新日记显示
    }
}

// 格式化时间，将秒转换为 mm:ss 格式
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// 使用简单的 MediaRecorder 录音（完整录音后识别）
startRec.onclick = async function() {
    if (isParsing) {
        showCustomToast('正在解析中，请稍后', 'error');
        return;
    }
    
    // 处理暂停后继续的情况
    if (isPaused && currentStream) {
        // 继续录音
        isPaused = false;
        
        // 重启音量监测
        startVolumeMonitor();
        
        // 恢复录音器
        if (currentRecorder && currentRecorder.state === 'paused') {
            currentRecorder.resume();
        }
        
        // 更新按钮状态
        startRec.disabled = true;
        pauseRec.disabled = false;
        stopRec.disabled = false;
        
        // 恢复提示
        const statusDiv = document.getElementById('recording-status');
        if (statusDiv) {
            statusDiv.innerHTML = '<i class="fa fa-microphone"></i> 智能分段录音中...';
        }
        
        showCustomToast('继续录音', 'success');
        return;
    }

    // 正常开始新录音
    if (isRecording) {
        showCustomToast('已在录音中', 'error');
        return;
    }
    
    // 清空之前的对话
    parseData = [];
    chatBox.innerHTML = '';
    diaryBox.innerText = '';
    downBtn.disabled = true;
    startParseBtn.disabled = true;
    
    startRec.disabled = true;
    pauseRec.disabled = false;
    stopRec.disabled = false;

    totalOffset = 0;  // 重置偏移量
    segmentSpeakerCounter = 0;  // 重置递增序号
    
    await startRecording();  // 调用你已有的 startRecording（使用 MediaRecorder）
}

pauseRec.onclick = function() {
    if (!isRecording || isPaused) return;
    
    // 暂停状态标记
    isPaused = true;
    
    // 停止音量监测
    if (vadMonitorInterval) {
        clearInterval(vadMonitorInterval);
        vadMonitorInterval = null;
    }
    
    // 暂停当前录音器（只调用一次）
    if (currentRecorder && currentRecorder.state === 'recording') {
        currentRecorder.pause();
    }
    
    // 更新按钮状态
    pauseRec.disabled = true;
    startRec.disabled = false;
    
    // 更新提示
    const statusDiv = document.getElementById('recording-status');
    if (statusDiv) {
        statusDiv.innerHTML = '<i class="fa fa-pause-circle"></i> 录音已暂停，点击"实时录音"继续';
    } else {
        chatBox.innerHTML = '<div id="recording-status" style="text-align:center; color:#999; padding:20px;"><i class="fa fa-pause-circle"></i> 录音已暂停</div>';
    }
    
    showCustomToast('录音已暂停', 'success');
}

stopRec.onclick = function() {
    if (!isRecording) return;

    // 如果处于暂停状态，先清除暂停标记
    isPaused = false;

    // 清除录音中的提示
    const statusDiv = document.getElementById('recording-status');
    if (statusDiv) statusDiv.remove();
    
    // 停止录音，传回调
    stopRecording(() => {
        // 最后一段上传完成后，恢复按钮
        startRec.disabled = false;
        stopRec.disabled = true;
        pauseRec.disabled = true;
        showCustomToast('录音结束，识别完成', 'success');
    });
}


// 导出TXT
downBtn.onclick = function() {
  const content = diaryBox.innerText;
  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = '会议结构化日记.txt';
  a.click();
  URL.revokeObjectURL(url);
}

// 中英文切换
langSwitch.onclick = function() {
  const isEn = langSwitch.innerText.includes('EN');
  if (!isEn) {
    langSwitch.innerText = 'CN / EN';
    document.querySelector('h2').innerHTML = '<i class="fa fa-microphone"></i> Multi-Meeting AI Analysis';
  } else {
    langSwitch.innerText = '中文 / EN';
    document.querySelector('h2').innerHTML = '<i class="fa fa-microphone"></i> 多人会议语音智能解析';
  }
}

// 初始化
loadSpeakers();