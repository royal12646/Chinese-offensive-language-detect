<template>
    <div class="image-container">
        <!-- 右侧的内容 -->
        <div class="content-container">
            <div class="chat-header">
                <h2>批量检测攻击性和偏见语言</h2>
                <p>上传你的图片</p>
            </div>
            <!-- 上传图片 -->
            <el-upload 
              class="upload"
              drag action="" 
              :on-change="handleImageUpload" 
              :on-remove="handleRemove" 
              :auto-upload="false"
              accept="image/*">
                <el-icon size="100">
                    <FolderAdd />
                </el-icon>
                <div class="el-upload__text">将图片拖到此处，或<em>点击上传</em></div>
            </el-upload>
            <ul>
             <li v-for="(item, index) in images" :key="index">
               <img :src="item.url" alt="" style="width: 100px;">
             </li>
            </ul>
            <el-input 
                v-model="extractedText" 
                placeholder="图片检测内容"
                :input-style="{ height: '108px'}"
                class="input-field" 
                @keyup.enter="onSubmit">
            </el-input>
            <!-- 提交检测按钮 -->
            <el-button class="submit-button" v-if="extractedText" type="primary" @click="onSubmit">开始检测</el-button>

            <!-- 显示检测结果 -->
            <el-table v-if="combinedData.length > 0" :data="combinedData" style="width: 80%" height="300" border stripe>
                <el-table-column prop="originalData" label="检测句子">
                </el-table-column>
                <el-table-column prop="responseData" label="检测结果">
                </el-table-column>
            </el-table>

            <!-- 下载CSV按钮 -->
            <el-button type="success" class="success-button" v-if="isDetectionComplete === true"
                @click="downloadCombinedData">
                检测完成, 是否保存为csv文件
            </el-button>
        </div>
    </div>
</template>

<script>
import Tesseract from 'tesseract.js';

// 后端地址
const API_URL = 'http://localhost:5000/predict';

export default {
    data() {
        return {
            images: [],
            extractedText: '', // 提取的文字
            combinedData: [], // 检测结果
            isSending: false, // 检测中
            isDetectionComplete: false, // 检测完成
        };
    },
    methods: {
        // 上传图片并提取文字
        handleImageUpload(file) {
            const imageFile = file.raw;
            this.extractTextFromImage(imageFile);
            this.images.push({ url: URL.createObjectURL(imageFile)});
        },
        handleRemove(file) {
         // 移除文件时从预览列表移除
             this.images = [];
             this.extractedText = '';
        },

        // 使用Tesseract提取文字
        extractTextFromImage(imageFile) {
            Tesseract.recognize(
                imageFile,
                'chi_sim',
                {
                    logger: (m) => console.log(m),  // 打印识别进度
                }
            ).then(({ data: { text } }) => {
                const cleanedText = text.replace(/\s+/g, '');
                this.extractedText = cleanedText;  // 将去除空格后的文字存储到 extractedText
                console.log('提取到的文字:', cleanedText);
            }).catch((error) => {
                console.error('文字提取失败:', error);
            });
        },

        // 发送文字到后端进行检测
        async sendDataToBackend(data) {
            if (this.isSending) return;
            this.isSending = true;
            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: data }),
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const Data = await response.json();
                const responseData = Data.prediction;
                this.handleResponse(data, responseData);
            } catch (error) {
                console.error('Error fetching data:', error);
            } finally {
                this.isSending = false;
                this.isDetectionComplete = true;
            }
        },

        handleResponse(originalData, responseData) {
            const combinedItem = {
                originalData: originalData,
                responseData: responseData,
            };
            // 添加到`combinedData`数组中
            this.combinedData.push(combinedItem);
        },

        // 提交检测
        async onSubmit() {
            if (this.extractedText) {
                await this.sendDataToBackend(this.extractedText);
            }
        },

        // 下载检测结果为CSV
        downloadCombinedData() {
            if (this.combinedData.length === 0) {
                alert('没有检测结果可下载');
                return;
            }

            const csvContent = 'data:text/csv;charset=utf-8,'
                + Object.keys(this.combinedData[0]).join(',') + '\r\n'
                + this.combinedData.map(row => Object.values(row).join(',')).join('\r\n');

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement('a');
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', 'detection_results.csv');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        },
    }
};
</script>

<style scoped>
.image-container {
    display: flex;
    margin: 20px;
}

.input-field {
    background: #ffffff;
    border: 1px solid #dcdcdc;
    border-radius: 30px;
    color: #1f1f1f;
    flex: 1;
    width: 80%;
}

.chat-header {
    text-align: center;
    margin-top: 20px;
}

.upload {
    width: 80%;
    margin: 10px 0;
}

.submit-button,
.success-button {
    width: 80%;
    margin: 10px 0;
}

.content-container {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
