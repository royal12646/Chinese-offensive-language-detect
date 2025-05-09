<template>
    <el-container class="usersetting-container">
        <el-header class="header">
            <h2>编辑个人信息</h2>
        </el-header>
        <el-main>
            <el-form :model="form" ref="form" label-width="120px" class="user-form">
                <el-form-item label="用户昵称">
                    <el-input v-model="form.username" placeholder="请输入昵称"></el-input>
                </el-form-item>
                <el-form-item label="头像">
                    <el-upload class="avatar-uploader" action="http://127.0.0.1:8889/api/upload-avatar"
                        :show-file-list="false" :before-upload="beforeUpload" :on-success="handleAvatarSuccess"
                        :on-error="handleAvatarError" :data="uploadData" headers="headers">
                        <img :src="form.avatar" class="avatar" />
                        <el-button size="small" type="primary">上传头像</el-button>
                    </el-upload>
                </el-form-item>
                <el-form-item label="性别">
                    <el-radio-group v-model="form.gender">
                        <el-radio label="male">男</el-radio>
                        <el-radio label="female">女</el-radio>

                    </el-radio-group>
                </el-form-item>
                <el-form-item label="个人介绍">
                    <el-input v-model="form.introduction" placeholder="请输入个人介绍"></el-input>
                </el-form-item>
                <el-form-item label="密码">
                    <el-input v-model="form.password" type="password" placeholder="请输入新密码"></el-input>
                </el-form-item>
                <el-form-item label="邮箱">
                    <el-input v-model="form.email" placeholder="请输入邮箱"></el-input>
                </el-form-item>
                <el-form-item>
                    <el-button type="primary" @click="save">保存</el-button>
                    <el-button @click="reset">重置</el-button>
                </el-form-item>
            </el-form>
        </el-main>
    </el-container>
</template>

<script>
import { ElMessage } from 'element-plus'
import axios from 'axios'


export default {

    data() {
        return {
            form: {
                id: localStorage.getItem('userinfo') ? JSON.parse(localStorage.getItem('userinfo')).id : '', // 从localStorage获取id
                username: localStorage.getItem('userinfo') ? JSON.parse(localStorage.getItem('userinfo')).username : '',
                avatar: '',
                gender: 'male',
                password: localStorage.getItem('userinfo') ? JSON.parse(localStorage.getItem('userinfo')).password : '',
                email: localStorage.getItem('userinfo') ? JSON.parse(localStorage.getItem('userinfo')).email : '',
                introduction: localStorage.getItem('userinfo') ? JSON.parse(localStorage.getItem('userinfo')).introduction : ''
            },
            uploadData: {},
            headers: {}
        }
    },
    methods: {
        beforeUpload(file) {
            const isImage = file.type.startsWith('image/')
            if (!isImage) {
                ElMessage.error('只能上传图片文件!')
            }
            return isImage
        },
        handleAvatarSuccess(response) {
            console.log(response)
            if (response && response.url) {
                this.form.avatar = response.url;
                this.saveUserProfile();
            } else {
                ElMessage.error('头像上传成功，但未返回有效的 URL!');
            }
        },
        handleAvatarError() {
            ElMessage.error('头像上传失败!')
        },
        saveUserProfile() {
            // 确保用户ID存在
            if (!this.form.id) {
                ElMessage.error('用户 ID 丢失，请重新登录!');
                return;
            }

            axios.post('http://127.0.0.1:8889/api/update-user-profile', this.form)
                .then(() => {
                    ElMessage.success('头像上传成功!')
                })
                .catch(() => {
                    console.log(JSON.parse(localStorage.getItem('userinfo')).id)
                    ElMessage.error('保存失败!')
                })
        },
        save() {
            this.saveUserProfile()
        },
        reset() {
            this.$refs.form.resetFields()
        }
    }
}
</script>

<style scoped>
.usersetting-container {
    padding: 20px;
}

.header {
    margin-bottom: 20px;
}

.user-form {
    max-width: 600px;
}

.avatar-uploader {
    display: flex;
    align-items: center;
}

.avatar {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    margin-right: 10px;
}
</style>