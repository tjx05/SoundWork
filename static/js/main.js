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

// ========== 队列管理相关变量 ==========
let parseQueue = [];        // 待解析队列
let isProcessing = false;   // 是否正在处理
let pendingSegments = [];   // 等待显示的片段（用于排序）
let segmentIndex = 0;       // 片段序号
let nextExpectedIndex = 0;  // 下一个期望显示的序号

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
    txt += `[${d.time}] ${d.person} [${d.mood}]：${d.text}\n\n`;
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
    div.dataset.name = item.name;
    div.dataset.isReg = item.isReg;
    div.innerHTML = `
      <div>
        <span class="name">${escapeHtml(item.name)}</span>
        <span class="attrs">(${item.age}·${item.gender})</span>
      </div>
      <span class="auto-tag">${item.isReg ? '已注册' : '自动识别'}</span>
    `;
    div.ondblclick = () => editSpeakerGlobal(item.id);
    
    // 添加右键菜单事件
    div.oncontextmenu = (e) => {
      e.preventDefault();
      showContextMenu(e, item);
    };
    
    speakerListDom.appendChild(div);
  });
}

// 显示右键菜单
function showContextMenu(event, speaker) {
  // 移除已有的菜单
  const existingMenu = document.querySelector('.context-menu');
  if (existingMenu) existingMenu.remove();
  
  // 创建菜单
  const menu = document.createElement('div');
  menu.className = 'context-menu';
  menu.style.position = 'fixed';
  menu.style.left = `${event.pageX}px`;
  menu.style.top = `${event.pageY}px`;
  
  if (!speaker.isReg) {
    // 临时说话人：显示"注册"
    menu.innerHTML = `
      <div class="menu-item" data-action="register">📝 注册为永久说话人</div>
    `;
  } else {
    // 已注册说话人：显示"删除"
    menu.innerHTML = `
      <div class="menu-item" data-action="delete">🗑️ 删除说话人</div>
    `;
  }
  
  document.body.appendChild(menu);
  
  // 绑定菜单项事件
  menu.querySelector('.menu-item').onclick = () => {
    if (!speaker.isReg) {
      registerTempSpeaker(speaker.name);
    } else {
      deleteRegisteredSpeaker(speaker.name);
    }
    menu.remove();
  };
  
  // 点击其他地方关闭菜单
  const closeMenu = (e) => {
    if (!menu.contains(e.target)) {
      menu.remove();
      document.removeEventListener('click', closeMenu);
    }
  };
  setTimeout(() => {
    document.addEventListener('click', closeMenu);
  }, 10);
}

// 注册临时说话人
async function registerTempSpeaker(tempName) {
  const newName = prompt('请输入姓名：', tempName);
  if (!newName) return;
  
  try {
    const response = await fetch('/api/promote_temp_speaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temp_name: tempName,
        real_name: newName
      })
    });
    const data = await response.json();
    if (data.success) {
      showCustomToast(data.message, 'success');
      await loadSpeakers();
      // 更新对话区的名字
      updateChatSpeakerName(tempName, newName);
    } else {
      showCustomToast('注册失败: ' + data.message, 'error');
    }
  } catch (err) {
    showCustomToast('网络错误: ' + err.message, 'error');
  }
}

// 删除已注册说话人
async function deleteRegisteredSpeaker(name) {
  if (!confirm(`确定要删除说话人 "${name}" 吗？`)) return;
  
  try {
    const response = await fetch('/api/delete_speaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name })
    });
    const data = await response.json();
    if (data.success) {
      showCustomToast(`已删除 ${name}`, 'success');
      await loadSpeakers();
    } else {
      showCustomToast('删除失败: ' + data.message, 'error');
    }
  } catch (err) {
    showCustomToast('网络错误: ' + err.message, 'error');
  }
}

// 更新对话区中的说话人名称
function updateChatSpeakerName(oldName, newName) {
  for (let i = 0; i < parseData.length; i++) {
    if (parseData[i].person.includes(oldName)) {
      const match = parseData[i].person.match(/\((.*)\)/);
      const attrs = match ? match[1] : '';
      parseData[i].person = attrs ? `${newName} (${attrs})` : newName;
    }
  }
  
  chatBox.innerHTML = '';
  for (let i = 0; i < parseData.length; i++) {
    addChatItem(parseData[i], i);
  }
  generateDiary();
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

// 编辑说话人（双击触发）- 统一处理所有说话人
async function editSpeakerGlobal(id) {
  const sp = speakerList.find(s => s.id === id);
  if (!sp) return;
  
  const oldName = sp.name;
  
  const newName = prompt('修改姓名：', sp.name);
  if (!newName) return;
  
  const newGender = prompt('修改性别（男/女）：', sp.gender);
  if (!newGender || (newGender !== '男' && newGender !== '女')) {
    alert('性别请输入"男"或"女"');
    return;
  }
  
  const newAge = prompt('修改年龄段（青年/中年/老年）：', sp.age);
  if (!newAge || (newAge !== '青年' && newAge !== '中年' && newAge !== '老年')) {
    alert('年龄段请输入"青年"、"中年"或"老年"');
    return;
  }
  
  // 调用后端接口同步
  try {
    const response = await fetch('/api/update_speaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        old_name: oldName, 
        new_name: newName,
        gender: newGender,
        age: newAge,
        is_reg: sp.isReg  // 告诉后端是否是已注册的
      })
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
  
  // 更新前端列表
  sp.name = newName;
  sp.gender = newGender;
  sp.age = newAge;
  renderSpeaker();
  
  // 构建新的显示格式
  const newDisplayName = `${newName} (${newAge}·${newGender})`;
  
  // 同步更新 parseData 中所有对话
  for (let i = 0; i < parseData.length; i++) {
    if (parseData[i].person.includes(oldName)) {
      parseData[i].person = newDisplayName;
    }
  }
  
  // 重新渲染对话区
  chatBox.innerHTML = '';
  for (let i = 0; i < parseData.length; i++) {
    addChatItem(parseData[i], i);
  }
  
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
  // 新增：清空后端临时说话人缓存
  try {
    await fetch('/api/clear_temp_speakers', { method: 'POST' });
  } catch(e) { console.log('清空缓存失败', e); }

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
      parseData = [];
      chatBox.innerHTML = '';
        
      for (const seg of data.segments) {
          // 直接使用后端返回的 person（已经是完整格式，如 "Speaker_01 (中年·女)"）
          const displayName = seg.person;
          addChat(seg.time, displayName, seg.mood, seg.level, seg.text);
      }
      generateDiary();
      downBtn.disabled = false;
      // addChats(data.segments);

      // 刷新左侧说话人列表（显示临时说话人）
      await loadSpeakers();

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
  const gender = document.getElementById('userGender').value;
  const age = document.getElementById('userAge').value;
  const files = document.getElementById('spkAudioUp').files;

  if (!name) return showCustomToast('请输入姓名', 'error');
  if (files.length === 0) return showCustomToast('请上传声纹语音', 'error');

  const formData = new FormData();
  formData.append('name', name);
  formData.append('gender', gender);
  formData.append('age', age);
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
        
        // chatBox.innerHTML = '<div id="recording-status" style="text-align:center; color:#67b99a; padding:20px;"><i class="fa fa-microphone"></i> 智能分段录音中...</div>';
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

// ========== 修改：endCurrentSegment 添加序号 ==========
function endCurrentSegment() {
    if (currentRecorder && currentRecorder.state === 'recording') {
        const currentIndex = segmentIndex++;  // 分配序号
        console.log('结束片段，准备上传，序号:', currentIndex);
        currentRecorder.stop();
        currentRecorder.onstop = () => {
            if (currentSegmentChunks.length > 0) {
                const blob = new Blob(currentSegmentChunks, { type: 'audio/webm' });
                if (blob.size > 5000) {
                    const formData = new FormData();
                    formData.append('audio', blob, `segment_${Date.now()}.webm`);
                    parseAudioBlob(formData, currentIndex);  // 传入序号
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

// ========== 修改：parseAudioBlob 添加序号参数，使用队列管理 ==========
// 按顺序显示片段
function displayInOrder() {
    // 按 index 排序
    pendingSegments.sort((a, b) => a.index - b.index);
    
    // 检查是否可以显示下一个
    let displayed = false;
    for (let i = 0; i < pendingSegments.length; i++) {
        if (pendingSegments[i].index === nextExpectedIndex) {
            // 显示这个片段
            for (const seg of pendingSegments[i].segments) {
                const displayName = seg.person;
                const timeStr = formatTime(totalOffset + seg.start);
                addChat(timeStr, displayName, seg.mood, seg.level, seg.text);
            }
            
            // 更新偏移量
            if (pendingSegments[i].segments.length > 0) {
                totalOffset += pendingSegments[i].segments[pendingSegments[i].segments.length - 1].end;
            }
            
            // 移除已显示的
            pendingSegments.splice(i, 1);
            nextExpectedIndex++;
            displayed = true;
            break;
        }
    }
    
    // 如果显示了，继续检查下一个
    if (displayed && pendingSegments.length > 0) {
        displayInOrder();
    }
}

async function parseAudioBlob(formData, idx) {
    // 加入队列
    parseQueue.push({ formData, index: idx });
    
    if (isProcessing) return;
    
    // 开始处理队列
    while (parseQueue.length > 0) {
        isProcessing = true;
        const next = parseQueue.shift();
        
        try {
            const response = await fetch('/api/recognize', { method: 'POST', body: next.formData });
            const data = await response.json();
            
            if (data.success && data.segments) {
                // 缓存结果，等待按顺序显示
                pendingSegments.push({
                    index: next.index,
                    segments: data.segments,
                    timestamp: Date.now()
                });
                
                // 按顺序显示
                displayInOrder();
                
                // 刷新左侧说话人列表（显示临时说话人）
                await loadSpeakers();
            } else {
                console.error('识别失败:', data.message);
            }
        } catch (err) {
            console.error('识别错误:', err);
        }
        
        isProcessing = false;
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
    // 新增：清空后端临时说话人缓存
    try {
      await fetch('/api/clear_temp_speakers', { method: 'POST' });
    } catch(e) { console.log('清空缓存失败', e); }

    // ========== 新增：重置队列相关变量 ==========
    segmentIndex = 0;
    nextExpectedIndex = 0;
    parseQueue = [];
    pendingSegments = [];
    isProcessing = false;

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

        // 更新按钮文字
        startRec.innerHTML="正在录音";
        
        // 恢复提示
        const statusDiv = document.getElementById('recording-status');
        if (statusDiv) {
            // statusDiv.innerHTML = '<i class="fa fa-microphone"></i> 智能分段录音中...';
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

    // 更新按钮文字
    startRec.innerHTML="正在录音";

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

    // 更新按钮文字
    startRec.innerHTML="继续录音";
    
    // 更新提示
    const statusDiv = document.getElementById('recording-status');
    if (statusDiv) {
        statusDiv.innerHTML = '<i class="fa fa-pause-circle"></i> 录音已暂停，点击"实时录音"继续';
    } else {
        // chatBox.innerHTML = '<div id="recording-status" style="text-align:center; color:#999; padding:20px;"><i class="fa fa-pause-circle"></i> 录音已暂停</div>';
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
        downBtn.disabled = false;

        // 更新按钮文字
        startRec.innerHTML='实时录音';

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

// 页面刷新时清空临时说话人
window.addEventListener('beforeunload', () => {
    fetch('/api/clear_temp_speakers', { method: 'POST', keepalive: true });
});