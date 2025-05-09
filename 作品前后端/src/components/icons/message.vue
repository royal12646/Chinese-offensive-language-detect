<script setup>

import { inject, ref } from "vue";
let api=inject('api');
let com_studentid=ref('');
let com_content=ref('');
let com_result=ref('');
let reply_studentid=ref('');
let reply_content=ref('');
let reply_result=ref('');
let isloading1=ref(false)
let isloading2=ref(false)
let send_query=()=>{
    isloading1.value=true;
    api.check_send(com_studentid.value,com_content.value,(res)=>{
       if(res.data.status==200)
       {
        com_result.value= res.data.data.map(item => item.from_id).join(', ');
       }
       else{
        ElMessage('获取出错！！！')
       }
    })
    isloading1.value=false
}
let reply_query=()=>{
    isloading2.value=true;
    api.check_reply(reply_studentid.value,reply_content.value,(res)=>{
       if(res.data.status==200)
       {
        reply_result.value= res.data.data.map(item => item.from_id).join(', ');
       }
       else{
        ElMessage('获取出错！！！')
       }
    })
    isloading2.value=false
}
</script>

<template>
   <div class="main">
        <div class="send">
            <el-text size="large" class="titletext">查询发出建议者</el-text>
            <div class="center">
                <el-form  label-position="top">
                    <el-form-item label="接受者邮箱" style="width: 70%;">
                        <el-input v-model="com_studentid"  type="textarea" :autosize="{ minRows: 1, maxRows: 1}" maxlength="20"/>
                        </el-form-item>
                        <el-form-item label="接受的消息" style="width: 70%;">
                        <el-input v-model="com_content" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }"/>
                        </el-form-item>
                        <el-form-item label="查询的结果" style="width: 70%;">
                        <el-input v-model="com_result" disabled="true"/>
                        </el-form-item>
                </el-form>
                <el-button :loading="isloading1"  type="primary" class="send_bt" @click="send_query">查询</el-button>
            </div>
        </div>
        <div class="accept">
            <el-text size="large" class="titletext">查询发出消息者</el-text>
            <div class="center">
                <el-form  label-position="top">
                    <el-form-item label="接受者邮箱" style="width: 70%;">
                        <el-input v-model="reply_studentid"  type="textarea" :autosize="{ minRows: 1, maxRows: 1}" maxlength="20"/>
                        </el-form-item>
                        <el-form-item label="接受的消息" style="width: 70%;">
                        <el-input v-model="reply_content" type="textarea" :autosize="{ minRows: 2, maxRows: 4 }"/>
                        </el-form-item>
                        <el-form-item label="查询的结果" style="width: 70%;">
                        <el-input v-model="reply_result" disabled="true"/>
                        </el-form-item>
                </el-form>
                <el-button :loading="isloading2"  type="primary" class="send_bt" @click="reply_query">查询</el-button>
        </div>
        </div>
   </div>
</template>
<style scoped>
.main{
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    width: 100%;
}
.send,.accept{
    width:90%;
    height: 50%;
    border: 1px solid rgba(0,0,0,.05);
    background: #fff;
    box-shadow: 4px 4px 40px rgba(0,0,0,.05);
    margin-top: 30px;
    box-sizing: content-box;
    padding: 30px;
    
}
.center{
    margin-top: 30px;
    display: flex;
    flex-direction: column;
    height: 300px;
}
.titletext{
    font-weight: 600;
}
.send_bt{
    width: 10%;
    margin-left: 100%;
    transform: translateX(-100%);
}
</style>