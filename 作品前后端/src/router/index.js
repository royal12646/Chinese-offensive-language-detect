import { createRouter, createWebHistory } from 'vue-router'
import index from "@/components/icons/index.vue"
import login from "@/components/icons/login.vue"
import message from "@/components/icons/message.vue"
import chat from '@/components/icons/chat.vue'
import file from '@/components/icons/file.vue'
import Image from '@/components/icons/Image.vue'
import guide from '@/components/icons/guide.vue'
import usersetting from '@/components/icons/usersetting.vue'
import userdata from '@/components/icons/userdata.vue'

const router = createRouter({
  history: createWebHistory(),
  routes:[
    {
      path:'/',
      redirect:'/login'
    },
   {
    path:'/login',
    name:'login',
    component:login
   },
    {
      path:'/index',
      name:'index',
      component:index,
      meta: { breadcrumbName: '首页' },
      children: [
        {
          path: 'chat',
          component: chat,
          meta: {
            breadcrumbName: 'chat'
          }
        },
        {
            path:'',
            redirect:'/index/chat'
        },
        {
          path: 'file',
          component: file,
          meta: {breadcrumbName: 'File'}
        },
        {
          path: 'Image',
          component: Image,
          meta: { breadcrumbName: 'Image' }
        },
        {
          path:'message',
          component:message,
          meta:{breadcrumbName:'消息查询'}
        },
        {
          path:'userdata',
          component:userdata,
          meta:{breadcrumbName:'用户数据'}
        },
        {
          path: '/index/guide',
          component: guide,
          meta: {
            breadcrumbName: '帮助文档'
          }
        },
        {
          path: 'usersetting',
          component: usersetting,
          meta: { breadcrumbName: '编辑用户' }
        }
      ]
    }
  ]
})
router.beforeEach((to,from,next)=>{
  if(to.name!='login'&&!localStorage.getItem('userinfo'))
  {
    window.alert("请先登录");
    next('/login');
  }
  else{
    next();
  }
})
export default router