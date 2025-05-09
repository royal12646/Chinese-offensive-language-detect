<script setup>
import { ref, nextTick, onMounted } from 'vue';
import Papa from 'papaparse';

const getCurrentTime = () => {
  const now = new Date();
  const year = String(now.getFullYear());
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
};

const recentFiles = ref([]);
const currentFile = ref(null);
const tableData = ref([]);
const columns = ref([]);
const selectedColumnData = ref([]);
const isSending = ref(false);
const combinedData = ref([]);
const isDetectionComplete = ref(false);
const value = ref(null);
const API_URL = 'http://localhost:5000/predict'; // 后端地址

// 保存文件检测记录
const saveFilesToLocalStorage = () => {
  localStorage.setItem('files', JSON.stringify(recentFiles.value));
};

// 初始化新文件检测
const initNewFileDetection = () => {
  const formattedDate = getCurrentTime();
  const newFile = {
    id: recentFiles.value.length + 1,
    title: formattedDate,
    tableData: [],
    columns: [],
    selectedColumnData: [],
    isSending: false,
    combinedData: [],
    isDetectionComplete: false,
  };
  recentFiles.value.push(newFile);
  currentFile.value = newFile;
  // 确保状态重置
  resetFileDetectionState();
  saveFilesToLocalStorage();
};

// 加载文件检测记录
const loadFilesFromLocalStorage = () => {
  const storedFiles = localStorage.getItem('files');
  if (storedFiles) {
    recentFiles.value = JSON.parse(storedFiles);
  }
};

// 重置文件检测状态
const resetFileDetectionState = () => {
  tableData.value = [];
  columns.value = [];
  selectedColumnData.value = [];
  isSending.value = false;
  combinedData.value = [];
  isDetectionComplete.value = false;
  value.value = null;
};

// 加载文件检测记录
const loadFile = (file) => {
  currentFile.value = file;
  tableData.value = file.tableData;
  columns.value = file.columns;
  selectedColumnData.value = file.selectedColumnData;
  isSending.value = file.isSending;
  combinedData.value = file.combinedData;
  isDetectionComplete.value = file.isDetectionComplete;
  value.value = null;
};

// 删除文件检测记录
const deleteFile = () => {
  localStorage.removeItem('files');
  recentFiles.value = [];
  currentFile.value = null;
  resetFileDetectionState();
  initNewFileDetection();
};

// 新建文件检测
const newFileDetection = async () => {
  const formattedDate = getCurrentTime();
  const newFile = {
    id: recentFiles.value.length + 1,
    title: formattedDate,
    tableData: [],
    columns: [],
    selectedColumnData: [],
    isSending: false,
    combinedData: [],
    isDetectionComplete: false,
  };
  recentFiles.value.push(newFile);
  currentFile.value = newFile;

  // 重置状态
  resetFileDetectionState();
  saveFilesToLocalStorage();

  // 强制 UI 更新
  await nextTick();
};

// 处理文件上传
const handleFileChange = (file) => {
  const fileReader = new FileReader();
  fileReader.onload = (e) => {
    const csvData = e.target.result;
    Papa.parse(csvData, {
      header: true,
      complete: (results) => {
        columns.value = Object.keys(results.data[0]);
        tableData.value = results.data;
        saveFilesToLocalStorage();
      },
    });
  };
  fileReader.readAsText(file.raw);
};

// 处理列选择变化
const onColumnChange = (columnName) => {
  selectedColumnData.value = tableData.value.map(row => ({ value: row[columnName] }));
  saveFilesToLocalStorage();
};

// 发送数据到后端
const sendDataToBackend = async (data) => {
  if (isSending.value || isDetectionComplete.value) return;

  isSending.value = true;
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: data.value }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const dataResponse = await response.json();
    const responseData = dataResponse.prediction;
    handleResponse(data, responseData);
  } catch (error) {
    console.error('Error fetching data:', error);
  } finally {
    isSending.value = false;
  }
};

// 处理响应数据
const handleResponse = (originalData, responseData) => {
  const combinedItem = {
    originalData: originalData.value,
    responseData: responseData,
  };
  combinedData.value.push(combinedItem);
  saveFilesToLocalStorage();
};

// 提交数据
const onSubmit = async () => {
  for (const item of selectedColumnData.value) {
    await sendDataToBackend(item);
  }
  isDetectionComplete.value = true;
  saveFilesToLocalStorage();
};

// 下载检测结果
const downloadCombinedData = () => {
  if (combinedData.value.length === 0) {
    alert('没有检测结果可下载');
    return;
  }
  const csvContent = 'data:text/csv;charset=utf-8,'
    + Object.keys(combinedData.value[0]).join(',') + '\r\n'
    + combinedData.value.map(row => Object.values(row).join(',')).join('\r\n');

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', 'detection_results.csv');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

onMounted(() => {
  loadFilesFromLocalStorage();
  if (recentFiles.value.length > 0) {
    loadFile(recentFiles.value[recentFiles.value.length - 1]);
  } else {
    initNewFileDetection();
  }
});
</script>
<template>
  <div class="file-container">
    <!-- 左侧的菜单 -->
    <div class="menu">
      <el-scrollbar>
        <el-text>历史检测记录</el-text>
        <div v-for="file in recentFiles" :key="file.id" class="menu-item" @click="loadFile(file)">
          {{ file.title }}
        </div>
        <el-button @click="deleteFile()" class="delete-button">删除文件检测记录</el-button>
        <el-button @click="newFileDetection()" class="new-button">新建文件检测</el-button>
      </el-scrollbar>
    </div>

    <!-- 右侧的内容 -->
    <div class="content-container">
      <div class="chat-header">
        <h2>批量检测攻击性和偏见语言</h2>
        <p>上传你的csv文件</p>
      </div>
      <el-upload class="upload" drag action="" :on-change="handleFileChange" :auto-upload="false" accept=".csv">
        <el-icon size="100">
          <UploadFilled />
        </el-icon>
        <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      </el-upload>
      <el-table v-if="tableData.length > 0" :data="tableData" style="width: 80%" height="500" border stripe>
        <el-table-column v-for="(column, index) in columns" :key="index" :prop="column"
          :label="column"></el-table-column>
      </el-table>
      <el-select v-if="tableData.length > 0" class="select" v-model="value" @change="onColumnChange"
        placeholder="请选择你想要检测内容的列名">
        <el-option v-for="(column, index) in columns" :key="index" :label="column" :value="column">
        </el-option>
      </el-select>
      <el-table v-if="selectedColumnData.length > 0" :data="selectedColumnData" style="width: 80%" height="300">
        <el-table-column prop="value">
        </el-table-column>
      </el-table>
      <el-button class="submit-button" v-if="selectedColumnData.length > 0" type="primary"
        @click="onSubmit">开始检测</el-button>
      <el-table v-if="combinedData.length > 0" :data="combinedData" style="width: 80%" height="300" border stripe>
        <el-table-column prop="originalData" label="检测句子">
        </el-table-column>
        <el-table-column prop="responseData" label="检测结果">
        </el-table-column>
      </el-table>
      <el-button type="success" class="success-button" v-if="isDetectionComplete === true"
        @click="downloadCombinedData">
        检测完成, 是否保存为csv文件
      </el-button>
    </div>
  </div>
</template>
<style scoped>
.file-container {
  display: flex;

  margin-top: 20px;
  margin-bottom: 20px;
  margin-left: 20px;
  margin-right: 20px;

}

.chat-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;
}

.upload {
  width: 80%;
  margin-bottom: 10px;
  margin-top: 10px;
}

.select {
  width: 80%;
  margin-bottom: 10px;
  margin-top: 10px;
}

.submit-button {
  width: 80%;
  margin-bottom: 10px;
  margin-top: 10px;
}

.success-button {
  width: 80%;
  margin-bottom: 10px;
  margin-top: 10px;
}

.menu {
  width: 300px;
  background: #ffffff;
  border-radius: 12px;
  margin: 10px;
  padding: 10px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
  transition: width 0.3s;
  height:90vh
}

.menu-collapsed {
  width: 60px;
  background: #ffffff;
  border-radius: 12px;
  margin: 10px;
  padding: 10px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: center;
  align-items: center;
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

.content-container {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.toggle-button {
  margin-bottom: 10px;
}
</style>