import { createApp} from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import * as api from "@/utils/api.js"
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import '@/assets/iconfont/iconfont.css'

const app = createApp(App)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }
app.provide('api', api)
app.use(createPinia())
app.use(router)

app.mount('#app')
