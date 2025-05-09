<script setup>
import { ElMessage } from "element-plus";
import { inject, onMounted, ref } from "vue";
const api = inject('api')
let datalist = ref([])

onMounted(() => {
  api.getUserData((res) => {
    if (res.data.status == 200) {
      datalist.value = res.data.data
    }
    else {
      ElMessage('获取出错')
    }
  })
});
</script>
<template>
  <div class="main">
    <div class="center">
      <el-table :data="datalist" style="width: 100%;margin-top: 20px;;" stripe border
        :cell-style="{ textAlign: 'center' }" :header-cell-style="{ 'text-align': 'center' }" height="100%" fit="false">
        <el-table-column type="index" label="序号" width="100">
        </el-table-column>
        <el-table-column prop="username" label="昵称" width="195" />
        <el-table-column prop="email" label="邮箱" width="195" />
        <el-table-column prop="introduction" label="个人介绍" width="245" />
        <el-table-column prop="UserPermission" label="用户权限" width="195" />
      </el-table>
    </div>
  </div>
</template>
<style scoped>
.main {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.center {
  margin-top: 10px;

}
</style>