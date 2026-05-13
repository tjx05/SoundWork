// ========== 全局变量 ==========
let speakerList = [];
let parseData = [];
let unknownCounter = {
  "青年女": 0, "青年男": 0,
  "中年女": 0, "中年男": 0,
  "老年女": 0, "老年男": 0
};
let currentTimeoutIds = [];
let isParsing = false;

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
    txt += `[${d.time}] ${d.person}（${d.mood} ${d.level}）：${d.text}\n`;
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
        <span class="meta">（${item.gender}·${item.age}）</span>
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

/// 编辑说话人（全局）
function editSpeakerGlobal(id) {
  const sp = speakerList.find(s => s.id === id);
  if (!sp) return;
  
  const oldName = sp.name;
  const newName = prompt('修改姓名：', sp.name);
  if (!newName) return;
  const newGender = prompt('修改性别（男/女）：', sp.gender);
  if (!newGender || !['男', '女'].includes(newGender)) return alert('性别只能填 男 / 女');
  const newAge = prompt('修改年龄段（青年/中年/老年）：', sp.age);
  if (!newAge || !['青年', '中年', '老年'].includes(newAge)) return alert('年龄段只能填 青年/中年/老年');

  // 更新speakerList
  sp.name = newName;
  sp.gender = newGender;
  sp.age = newAge;
  
  // 更新左侧列表显示
  renderSpeaker();

  // 更新parseData中所有匹配的姓名
  parseData.forEach(d => {
    if (d.person === oldName) {
      d.person = newName;
    }
  });

  // 🔥 关键修复：更新右侧chatBox中所有显示该姓名的DOM元素
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

  // 重新生成日记
  generateDiary();
}

// 替换未知说话人
function replaceUnknownSpeaker(gender, age, newName) {
  const key = `${age}${gender}`;
  const modifiedIndices = [];

  parseData.forEach((d, idx) => {
    if (d.person.includes(key)) {
      d.person = newName;
      modifiedIndices.push(idx);
    }
  });

  const chatItems = chatBox.querySelectorAll('.chat-item');
  modifiedIndices.forEach(idx => {
    if (chatItems[idx]) {
      const nameSpan = chatItems[idx].querySelector('.chat-name');
      nameSpan.textContent = newName;
      nameSpan.ondblclick = () => editChatName(idx);
    }
  });

  renderSpeaker();
  generateDiary();
}

// 生成临时名字
function getUnknownName(gender, age) {
  const key = `${age}${gender}`;
  unknownCounter[key] = (unknownCounter[key] || 0) + 1;
  const tempName = `${age}${gender}${unknownCounter[key]}`;

  const has = speakerList.some(s => s.name === tempName);
  if (!has) {
    speakerList.push({
      id: Date.now() + Math.random(),
      name: tempName,
      gender: gender,
      age: age,
      isReg: false
    });
    renderSpeaker();
  }
  return tempName;
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
  const newItem = { time, person, mood, level, text };
  parseData.push(newItem);
  addChatItem(newItem, parseData.length - 1);
  generateDiary();
}

// 编辑对话姓名
window.editChatName = function(idx) {
  const oldName = parseData[idx].person;
  const newName = prompt('修改本条说话人姓名：', oldName);
  if (!newName) return;

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

// 注册说话人
regBtn.onclick = async function() {
  const name = document.getElementById('userName').value.trim();
  const gender = document.getElementById('userGender').value;
  const age = document.getElementById('userAge').value;
  const files = document.getElementById('spkAudioUp').files;

  if (!name) return alert('请输入姓名');
  if (files.length === 0) return alert('请上传声纹语音');

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
      alert(data.message);
      document.getElementById('userName').value = '';
      document.getElementById('spkAudioUp').value = '';
      loadSpeakers();
      replaceUnknownSpeaker(gender, age, name);
    } else {
      alert('注册失败: ' + data.message);
    }
  } catch (err) {
    alert('网络错误: ' + err.message);
  }
}

// 上传录音识别
uploadBox.onclick = () => audioFile.click();
audioFile.onchange = async function(e) {
  if (!e.target.files[0]) return;
  uploadBox.innerHTML = `<i class="fa fa-music"></i> 已选择：${e.target.files[0].name}`;

  const formData = new FormData();
  formData.append('audio', e.target.files[0]);

  try {
    const response = await fetch('/api/recognize', { method: 'POST', body: formData });
    const data = await response.json();
    if (data.success && data.segments) {
      parseData = [];
      chatBox.innerHTML = '';
      unknownCounter = {
        "青年女": 0, "青年男": 0,
        "中年女": 0, "中年男": 0,
        "老年女": 0, "老年男": 0
      };
      speakerList = speakerList.filter(s => s.isReg === true);
      for (const seg of data.segments) {
        let person = seg.person;
        // 自动识别未注册说话人
        if (person !== '张三' && !speakerList.some(s => s.name === person)) {
          // 简单规则：从名称推断性别年龄
          let gender = '男', age = '青年';
          if (person.includes('女')) gender = '女';
          if (person.includes('中年')) age = '中年';
          if (person.includes('老年')) age = '老年';
          person = getUnknownName(gender, age);
        }
        addChat(seg.time, person, seg.mood, seg.level, seg.text);
      }
      downBtn.disabled = false;
      generateDiary();
    }
  } catch (err) {
    alert('识别失败: ' + err.message);
  }
}

// 实时录音（模拟数据）
startRec.onclick = function() {
  currentTimeoutIds.forEach(id => clearTimeout(id));
  currentTimeoutIds = [];

  startRec.disabled = true;
  pauseRec.disabled = false;
  stopRec.disabled = false;
  downBtn.disabled = true;

  parseData = [];
  chatBox.innerHTML = '';
  diaryBox.innerText = '正在解析中...\n';
  unknownCounter = {
    "青年女": 0, "青年男": 0,
    "中年女": 0, "中年男": 0,
    "老年女": 0, "老年男": 0
  };
  speakerList = speakerList.filter(s => s.isReg === true);
  renderSpeaker();

  // 模拟解析数据
  const timers = [
    setTimeout(() => addChat('00:05', '张三', '开心', 'HI', '我们对齐一下本周项目进度'), 300),
    setTimeout(() => addChat('00:13', getUnknownName('女', '青年'), '平静', 'LO', '前端界面已全部开发完成'), 800),
    setTimeout(() => addChat('00:20', getUnknownName('女', '青年'), '开心→失落', 'MID', '进度还行，就是早上没来得及吃饭'), 1300),
    setTimeout(() => addChat('00:28', '张三', '理解', 'LO', '没事，先把语音解析模块联调完'), 1800),
    setTimeout(() => addChat('00:35', getUnknownName('男', '青年'), '严肃', 'MID', '后端接口我这边已经调试完毕'), 2300),
    setTimeout(() => addChat('00:42', getUnknownName('女', '中年'), '温和', 'LO', '那我们定一下下周联调时间'), 2800),
    setTimeout(() => addChat('00:50', '张三', '果断', 'HI', '那就下周二下午统一联调'), 3300),
    setTimeout(() => addChat('00:58', getUnknownName('男', '中年'), '认真', 'MID', '我这边提前准备好测试用例'), 3800),
    setTimeout(() => addChat('01:05', getUnknownName('女', '青年'), '轻松', 'LO', '好的，我也同步准备页面'), 4300),
    setTimeout(() => addChat('01:12', '张三', '总结', 'HI', '大家各自准备，周二准时对接'), 4800)
  ];
  currentTimeoutIds = timers;
}

pauseRec.onclick = function() {
  alert('暂停功能（模拟）');
}

stopRec.onclick = function() {
  currentTimeoutIds.forEach(id => clearTimeout(id));
  currentTimeoutIds = [];
  startRec.disabled = false;
  pauseRec.disabled = true;
  stopRec.disabled = true;
  downBtn.disabled = false;
  generateDiary();
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