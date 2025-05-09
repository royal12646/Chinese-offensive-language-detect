import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
/* 用户信息 */
export const createuserinfo = defineStore('userinfo', () => {
  let nick_name=ref("");
  let photourl=ref("");
  function setnickname(nickname){
    this.nick_name=nickname;
  }
  function setphotourl(photourl){
    this.photourl=photourl
  }
  return {nick_name,photourl,setnickname,setphotourl}
})
