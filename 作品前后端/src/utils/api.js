import axios from "axios";
import { ElMessage } from "element-plus";

const baseURL = 'http://172.22.235.92:8999/api/';

// 获取用户信息接口
export const fetchUserInfo = (email) => {
    return axios.get(baseURL + 'userinfo', {
        params: { email: email }
    }).then(response => response.data)
        .catch(error => {
            console.error('Error fetching user info:', error);
            throw error;
        });
};

// 获取全部用户信息接口
export const getUserData = (callback) => {
    axios.get(baseURL + 'getUserData').then((response) => {
        callback(response);
    }).catch((error) => {
        console.error("Error fetching user data:", error);
    });
};

// 检查用户名是否已存在
export const checkUsername = (username) => {
    return axios.post(baseURL + 'checkUsername', { username })
        .then(response => {
            console.log('Response:', response.data); // 打印响应数据
            return response.data;  // 返回后端的实际数据
        })
        .catch(error => {
            console.error('Error checking username:', error);
            throw error;
        });
};

// 上传头像的 API 函数
export const uploadAvatar = (formData, callback) => {
    axios.post(`${baseURL}upload-avatar`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    })
        .then(response => {
            callback(response);
        })
        .catch(error => {
            console.error('上传头像失败:', error);
        });
};

// 登录接口
export const login = (username, password) => {
    console.log('Sending login request with:', username, password);
    return axios.post(baseURL + 'login', new URLSearchParams({
        username: username,
        password: password
    }), {
        headers: {
            'content-type': 'application/x-www-form-urlencoded'
        }
    });
};

// 注册接口
export const register = (registerForm, callback = () => { }) => {
    return axios.post(baseURL + 'register', new URLSearchParams({
        username: registerForm.username,
        password: registerForm.password,
        email: registerForm.email,
        captchaInput: registerForm.captchaInput
    }), {
        headers: {
            'content-type': 'application/x-www-form-urlencoded'
        }
    }).then((response) => {
        callback(response);  // 调用回调函数
        return response;  // 确保返回响应结果
    }).catch((err) => {
        ElMessage.error('注册请求失败');
        console.error(err);
        throw err;  // 抛出错误供调用者处理
    });
};
