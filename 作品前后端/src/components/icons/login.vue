<script setup>
import { ref, onMounted, inject } from "vue";
import { useRouter } from "vue-router";
import { createuserinfo } from "@/stores/userinfo.js";
import { ElMessageBox } from 'element-plus';
import GVerify from '@/components/icons/gVerify.js';

const router = useRouter();
let api = inject('api');
let isloading = ref(false);

const loginForm = ref({
    username: '',
    password: ''
});

const registerForm = ref({
    username: '',
    password: '',
    email: '',
    captchaInput: ''
});

let isSignUpActive = ref(false);
let isUsernameValid = ref(true);

const toggleSignUp = (isSignUp) => {
    isSignUpActive.value = isSignUp;
};

// 初始化图形验证码
let captchaInput = null;

onMounted(() => {
    captchaInput = new GVerify("v_container");
});

let login = async () => {
    if (!loginForm.value.username || !loginForm.value.password) {
        ElMessageBox.alert('请输入用户名和密码', '提示', {
            confirmButtonText: '确定'
        });
        return;
    }
    isloading.value = true;
    try {
        const res = await api.login(loginForm.value.username, loginForm.value.password);
        if (res.data.status === 200 && res.data.data.length > 0) {
            let newuserinfo = {
                avatar: res.data.data[0].avatar,
                username: res.data.data[0].username,
                email: res.data.data[0].email,
                id: res.data.data[0].id,
                password: res.data.data[0].password,
                gender: res.data.data[0].gender
            };
            localStorage.setItem('userinfo', JSON.stringify(newuserinfo));
            router.push('/index');
        } else {
            ElMessageBox.alert('登录失败', '提示', {
                confirmButtonText: '确定'
            });
        }
    } catch (error) {
        ElMessageBox.alert('请求失败，请重试', '错误', {
            confirmButtonText: '确定'
        });
    } finally {
        isloading.value = false;
    }
};

let checkUsername = async () => {
    try {
        const res = await api.checkUsername(registerForm.value.username);
        if (res.exists) {
            ElMessageBox.alert('用户名已存在，请更换用户名', '提示', {
                confirmButtonText: '确定'
            });
            return false; // 用户名已存在，返回 false
        }
        return true; // 用户名不存在，返回 true
    } catch (error) {
        console.error('检查用户名失败:', error);
        ElMessageBox.alert('检查用户名失败，请重试', '错误', {
            confirmButtonText: '确定'
        });
        return false;
    }
};


let register = async () => {
    if (!registerForm.value.username || !registerForm.value.password || !registerForm.value.email || !registerForm.value.captchaInput) {
        ElMessageBox.alert('请填写完整的注册信息', '提示', {
            confirmButtonText: '确定'
        });
        return;
    }

    // 检查用户名是否已存在
    isUsernameValid = await checkUsername();
    console.log(isUsernameValid)
    if (!isUsernameValid) return; // 如果用户名已存在，停止注册流程

    console.log(captchaInput)
    console.log(captchaInput.options.code)
    if (!captchaInput.validate(captchaInput.options.code)) {
        ElMessageBox.alert('图形验证码错误', '提示', {
            confirmButtonText: '确定'
        });
        return;
    }
    isloading.value = true;
    try {
        const res = await api.register(registerForm.value);
        console.log(res)
        if (res.data.status === 200) {
            ElMessageBox.alert('注册成功', '提示', {
                confirmButtonText: '确定'
            }).then(() => {
                toggleSignUp(false);
            });
        } else {
            ElMessageBox.alert('注册失败', '提示', {
                confirmButtonText: '确定'
            });
        }
    } catch (error) {
        console.error('请求失败，错误信息:', error);
        ElMessageBox.alert('请求失败，请重试', '错误', {
            confirmButtonText: '确定'
        });
    } finally {
        isloading.value = false;
    }
};


</script>


<template>
    <div class="titles">
        <p class="line"><i class="fas fa-comment-alt icon"></i> PigeonLab——有害言论识别系统</p>
    </div>
    <div class="container" :class="{ 'right-panel-active': isSignUpActive }" id="container">
        <div class="form-container sign-up-container">
            <!-- 注册 -->
            <form @submit.prevent="register">
                <h1>用户注册</h1>
                <input type="text" placeholder="用户名" v-model="registerForm.username" @blur="checkUsername"
                    :class="{ 'input-error': !isUsernameValid }" />
                <input type="password" placeholder="密码" v-model="registerForm.password" :disabled="!isUsernameValid" />
                <input type="email" placeholder="邮箱" v-model="registerForm.email" />
                <div id="v_container" style="width: 200px; height: 50px; margin-bottom: 10px;"></div>
                <input type="text" id="code_input" placeholder="请输入验证码" v-model="registerForm.captchaInput" />
                <button class="submit-button" type="submit" :disabled="!isUsernameValid">注册</button>
            </form>
        </div>
        <div class="form-container sing-in-container">
            <!-- 登录 -->
            <form @submit.prevent="login">
                <h1>用户登录</h1>
                <div class="social-container">
                    <a href="#" class="social"><i class="iconfont icon-qq"></i></a>
                    <a href="#" class="social"><i class="iconfont icon-weixin"></i></a>
                    <a href="#" class="social"><i class="iconfont icon-weibo-copy"></i></a>
                    <a href="#" class="social"><i class="iconfont icon-github"></i></a>
                </div>
                <span>您可以选择以上几种方式登录您的账户!</span>
                <input type="text" placeholder="用户名" v-model="loginForm.username" />
                <input type="password" placeholder="密码" v-model="loginForm.password" />
                <a href="#">忘记密码?</a>
                <button class="submit-button" type="submit">登录</button>
            </form>
        </div>
        <!-- 侧边栏内容 -->
        <div class="overlay-container">
            <div class="overlay">
                <div class="overlay-pannel overlay-left">
                    <h1>已有账号?</h1>
                    <p>点击进行登录.</p>
                    <button class="ghost" @click="toggleSignUp(false)">登录</button>
                </div>
                <div class="overlay-pannel overlay-right">
                    <h1>没有帐号？</h1>
                    <p>去注册一个属于你的账号吧！</p>
                    <button class="ghost" @click="toggleSignUp(true)">注册</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            isSignUpActive: false,
            loginForm: {
                username: '',
                password: ''
            },
            registerForm: {
                username: '',
                password: '',
                email: '',
                captchaInput: ''
            }
        };
    },
    methods: {
        toggleSignUp(isSignUp) {
            this.isSignUpActive = isSignUp;
        },
        login() {
            // 登录逻辑
            console.log('登录信息:', this.loginForm);
            // 发送请求到服务器
        },
        register() {
            // 注册逻辑
            console.log('注册信息:', this.registerForm);
            // 发送请求到服务器
        },
        sendCode() {
            // 发送验证码逻辑
            console.log('发送验证码到:', this.registerForm.email);
            // 发送请求到服务器
        }
    }
};

</script>

<style>
* {
    margin: 0;
    padding: 0;
    /*标准盒子 */
    box-sizing: border-box;
}

body,
html {
    font-family: Arial, Helvetica, sans-serif;
    background-image: url('/src/assets/4.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    margin: 0 auto;
}

h1 {
    margin: 0.2rem;
    font-size: 1.2rem;
}

p {
    font-size: 0.9rem;
    /* line-height: 1.5rem; */
    /* 字体变淡 */
    font-weight: 100;
    margin: 1.2rem 0;
    /* 字间距 */
    letter-spacing: 0.1rem;
}

span {
    font-size: 0.8rem;
    margin: 1.2rem 0;
}

a {
    color: #333;
    font-size: 0.7rem;
    /* 下划线消失 */
    text-decoration: none;
}

.container {
    position: relative;
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 14px 28px rgba(0, 0, 0, 0.25), 0 10px 10px rgba(0, 0, 0, 0.22);
    padding: 0.6rem;
    width: 50rem;
    height: 35rem;
    overflow: hidden;
    max-width: 100vw;
    min-height: 70vh;
    /* 添加 flex 布局使内容居中 */
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0 auto;
    /* 居中显示容器 */
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

.container::after {
    content: '';
    position: absolute;
    top: -100px;
    left: -100px;
    right: -100px;
    bottom: -100px;
    background: rgba(0, 0, 0, 0.1);
    filter: blur(8px);
    z-index: 1000;
    pointer-events: none;
}

.titles {
    text-align: center;
    padding: 20px;
    color: #007bff;
    font-size: 28px;
    font-weight: bold;
    z-index: 1000;
}

.line {
    height: 43px;
    display: inline-block;
    padding: 0 20px;
    color: #202e59;
    font-size: 28px;
    font-weight: bold;
    border-right: 4px solid #0c0c0c;
    text-align: center;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0);
    /* border-radius: 30px; */
    animation: grow 4s steps(44) 1s normal both, blink 500ms steps(44) infinite normal;
}

@keyframes grow {
    from {
        width: 0;
    }

    to {
        width: 530px;
    }
}

@keyframes blink {
    from {
        border-right-color: #202e59;
    }

    to {
        border-right-color: transparent;
    }
}

.form-container form {
    background: #fff;
    /* 弹性布局 */
    display: flex;
    flex-direction: column;
    padding: 0 1.8rem;
    height: 100%;
    justify-content: center;
    align-items: center;

}

.social-container {
    margin: 0.6rem 0;
}

.social-container a {
    border: 1px solid #eee;
    border-radius: 50%;
    display: inline-flex;
    justify-content: center;
    align-items: center;
    margin: 0 5px;
    height: 1.8rem;
    width: 1.8rem;
}

.social-container a:hover {
    opacity: 0.8;
}

.form-container input {
    width: 100%;
    height: 2.2rem;
    text-indent: 1rem;
    border: 1px solid #ccc;
    /* 把input上左右边框取消 */
    border-left: none;
    border-right: none;
    border-top: none;
    /* 点击input边框消失 */
    outline: none;
    margin: 0.6rem 0;
}

/* 被选中时候缩小 */
.form-container button:active {
    transform: scale(0.95, 0.95);
}

.form-container button {
    padding: 0.4rem 1rem;
    background: #29437a8e;
    color: white;
    border: 1px solid #fff;
    outline: none;
    /* 鼠标放上变小手 */
    cursor: pointer;
    width: 5rem;
    border-radius: 8px;
    transition: all 100ms ease-in;
    margin: 0.6rem 0;
    font-size: 0.8rem;
    padding: 0.5rem 0;
}

button#send_code {
    width: 100%;
}

button.ghost {
    background: transparent;
    border-color: #fff;
    border: 1px solid #fff;
    outline: none;
    cursor: pointer;
    width: 5rem;
    border-radius: 8px;
    transition: all 800ms ease-in;
    margin: 0.6rem 0;
    padding: 0.5rem 0;
    color: white;
    font-size: 0.8rem;
}

button.ghost:active {
    transform: scale(0.95, 0.95);
}

.form-container {
    /* 绝对定位 */
    position: absolute;
    top: 0;
    height: 100%;
    transition: all 0.5s ease-in;

}

.sing-in-container {
    left: 0;
    width: 50%;
    z-index: 2;
}

.sign-up-container {
    left: 0;
    width: 50%;
    opacity: 0;
    z-index: 1;
}

.overlay {
    background: #29437a8e;
    width: 200%;
    height: 100%;
    position: relative;
    left: -100%;
    transition: all 0.6s ease-in-out;
    color: white;
}

.overlay-container {
    position: absolute;
    top: 0;
    right: 0;
    width: 50%;
    height: 100%;
    overflow: hidden;
    transition: all 0.6s ease-in-out;
    z-index: 99;
}

.overlay-pannel {
    position: absolute;
    top: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    width: 50%;
    height: 100%;
    padding: 0 2.2rem;
}

.overlay-right {
    right: 0;
}

.container.right-panel-active .overlay-container {
    transform: translateX(-100%);
}

.container.right-panel-active .sing-in-container {
    transform: translateX(100%);
}

.container.right-panel-active .sign-up-container {
    transform: translateX(100%);
    opacity: 1;
    z-index: 5;
    transition: all 0.6s ease-in-out;
}

.container.right-panel-active .overlay {
    transform: translateX(50%);
}

.container.right-panel-active .overlay-left {
    transform: translateX(0);
    transition: all 0.6s ease-in-out;
}

.container.right-panel-active .overlay-right {
    transform: translateX(20%);
    transition: all 0.6s ease-in-out;
}

.input-error {
    border-color: red;
}
</style>
