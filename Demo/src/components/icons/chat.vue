<script setup>
import { ref, nextTick} from 'vue';
import { onMounted, onUnmounted } from 'vue';
// const messages = ref([]);
const userInput = ref('');
// Q
const recognition = ref(null);// 存储语音
// const recentChats = ref([]); // 存储最近聊天记录
const API_URL = 'http://172.22.235.92:5005/predict'; // 后端地址
const selectedmodels = ref('');

let messages = ref([]);
let recentChats = ref([]);
let currentChat = ref(null);

// 获取当前时间
const getCurrentTime = () => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
};
// 初始化新对话
const initNewChat = () => {
    const formattedDate = getCurrentTime();
    const newChat = {
        id: recentChats.value.length + 1,
        title: formattedDate,
        messages: []
    };
    recentChats.value.push(newChat);
    currentChat.value = newChat;
    messages.value = currentChat.value.messages;
};

// 页面加载时初始化新对话
initNewChat();
// 加载聊天记录
const loadChatsFromLocalStorage = () => {
  const storedChats = localStorage.getItem('chats');
  if (storedChats) {
    recentChats.value = JSON.parse(storedChats);
  }
};
// 保存聊天记录
const saveChatsToLocalStorage = () => {
  localStorage.setItem('chats', JSON.stringify(recentChats.value));
};
// 加载聊天记录
const loadChat = (chat) => {
    messages.value = chat.messages; // 加载已存储的消息到聊天界面
    currentChat.value = chat;
};

const deleteChat = () => {
  localStorage.removeItem('chats');
  messages = ref([]);
  recentChats = ref([]);
  currentChat = ref(null);
}

const newChat = () => {
    const formattedDate = getCurrentTime();
    const newChat = {
        id: recentChats.value.length + 1,
        title: formattedDate,
        messages: []
    };
    recentChats.value.push(newChat);
    currentChat.value = newChat;
    messages.value = currentChat.value.messages;
    saveChatsToLocalStorage();
};
// 在组件挂载时加载聊天记录
onMounted(() => {
  loadChatsFromLocalStorage();
});
// 发送消息
const sendMessage = async () => {
    if (!userInput.value.trim()) return;

    // 如果有用户输入的内容，将其加入消息列表
    if (userInput.value.trim()) {
        messages.value.push({ role: 'user', content: userInput.value });
    }

    try {
        const text = userInput.value;
        const model = selectedmodels.value;
        const response = await fetch(API_URL,{
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({text: text, model: model})
                
            })

        if (response.ok) {
            const data = await response.json();
            const assistantReply = data.prediction;
            messages.value.push({ role: 'assistant', content: assistantReply });
        } else {
            messages.value.push({ role: 'assistant', content: 'AI 无法响应，请稍后再试。' });
        }
    } catch (error) {
        messages.value.push({ role: 'assistant', content: '请求失败，请检查网络连接或稍后再试。' });
    }

    userInput.value = '';
    saveChatsToLocalStorage();
    nextTick(scrollToBottom);
};



const scrollToBottom = () => {
    const chatBox = document.querySelector('.chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
};
function startRecording() {
  if (window.webkitSpeechRecognition) {
    const recognitionInstance = new window.webkitSpeechRecognition();
    recognitionInstance.continuous = true;
    recognitionInstance.interimResults = true;
    recognitionInstance.lang = 'zh-CN';

    recognitionInstance.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map(result => result[0])
        .map(result => result.transcript)
        .join('');
      console.log(transcript);
      userInput.value= transcript;
    };

    recognition.value = recognitionInstance;
    recognitionInstance.start();
  } else {
    alert('您的浏览器不支持语音识别功能');
  }
}

onUnmounted(() => {
  // 组件卸载前停止录音
  if (recognition.value) {
    recognition.value.stop();
  }
});

</script>

<template>
    <div class="chat-container">
        <div class="menu">
            <div>
            <el-scrollbar>
                <el-text>历史聊天记录</el-text>
                <div v-for="chat in recentChats" :key="chat.id" class="menu-item" @click="loadChat(chat)">
                    {{ chat.title }}
                </div>
                <el-button @click="deleteChat()" class="delete-button">删除记录</el-button>
                <el-button @click="newChat()" class="new-button">新建对话</el-button>
            </el-scrollbar>
            </div>
            <el-divider />
            <div>
                <el-scrollbar>
                <el-text>选择模型</el-text>
                <el-select  class="model-select" v-model="selectedmodels" placeholder="请选择模型">
                 <el-option label = "通用检测" value = "general"></el-option>
                 <el-option label = "涉黄文本检测" value = "sex"></el-option>
                 <el-option label = "辱骂文本检测" value = "abuse"></el-option>
                 <el-option label = "地域/种族/性别文本检测" value = "four_offensive"></el-option>
                </el-select>
                </el-scrollbar>
            </div>
        </div>
        <div class="chat-content">
            <div class="chat-header">
                <h2>你好，<br>我是有害言论识别系统</h2>
                <p>我可以回答你的问题，为你提供有用信息，帮助你判断句子是为有害言论。</p>
            </div>
            <el-scrollbar class="chat-box">
                <div v-for="(message, index) in messages" :key="index" class="message" :class="message.role">
                    <el-card class="box-card" :body-style="{ padding: '16px' }">
                        <div class="message-content">
                            <p>{{ message.content }}</p>
                        </div>
                    </el-card>
                </div>
            </el-scrollbar>
            <div class="input-container">
                <el-button circle
                 @click="startRecording"
                 class="voice-button"
                >
                    <el-icon><Microphone /></el-icon>
                </el-button>
                <el-input 
                v-model="userInput" 
                placeholder="请输入你的消息..."
                :input-style="{ height: '108px'}"
                class="input-field" 
                @keyup.enter="sendMessage">

                    <template #append>
                        <el-button @click="sendMessage" class="send-button">发送</el-button>
                    </template>

                </el-input>
            </div>
        </div>
    </div>
</template>

<style scoped>
.chat-container {
    display: flex;
    height: 80vh; /* 占满整个页面高度 */
    background: #eef2f7;
    color: #333;
    margin-top: 20px;
    margin-bottom: 20px;
    margin-left: 20px;
    margin-right: 20px;
}
.voice-button {
    margin-right: 10px;
    background: #ffffff;
    border: none;
    color: #333;
    font-size: 24px;
    border-radius: 50%;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.menu {
    width: 300px;
    background: #ffffff;
    border-radius: 12px;
    margin: 10px;
    padding: 10px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
    /* flex-shrink: 0; */
}

.menu-item {
    padding: 10px 20px;
    border-radius: 6px;
    background: #f7f7f7;
    margin-bottom: 10px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.menu-item:hover {
    background: #e6e6e6;
}

.chat-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #eef2f7;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.05);
    height: calc(92vh - 24px);
    margin: 20px 100px;
}

.chat-header {
    text-align: left;
    padding: 20px;
    background: #fff;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    flex-shrink: 0;
}

.chat-header h2 {
    font-size: 1.5rem;
    margin-bottom: 10px;
}

.chat-header p {
    font-size: 1rem;
    color: #555;
}

.prompt-tip {
    background: #eef5ff;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    color: #4a90e2;
}

.chat-box {
    flex: 1;
    padding: 10px 20px;
    overflow-y: auto;
    background: #eef2f7;
    display: flex;
    flex-direction: column-reverse; /* 新消息会出现在底部 */
}

.message {
    margin-bottom: 15px;
}

.message.user {
    text-align: right;
    align-self: flex-end;
    border-radius: 12px;
}

.message.assistant {
    text-align: left;
    align-self: flex-start;
    border-radius: 12px;
}

.box-card {
    max-width: 70%;
    margin: 0 auto;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

.message-content {
    color: #1f1f1f;
}

.input-container {
    padding: 20px 0;
    background: #eef2f7;
    display: flex;
    align-items: center;
    border-top: 1px solid #dcdcdc;
    position: sticky;
    bottom: 0; /* 输入框固定在底部 */
    background-color: #eef2f7;
    z-index: 10;
}

.input-field {
    background: #ffffff;
    border: 1px solid #dcdcdc;
    border-radius: 30px;
    color: #1f1f1f;
    flex: 1;
}

.send-button {
    background: #4a90e2;
    color: #fff;
    border: none;
    border-radius: 30px;
    padding: 0 20px;
    cursor: pointer;
}

.send-button:hover {
    background: #357ab8;
}

.model-select {
    width: 100%;
    margin-top: 20px;
}
</style>