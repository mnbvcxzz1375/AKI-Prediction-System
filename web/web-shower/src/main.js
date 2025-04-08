import { createApp } from 'vue';
// 1. 引入你需要的组件
import Vant from 'vant';
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'

// 2. 引入组件样式
import 'vant/lib/index.css';
import 'element-plus/dist/index.css'

// 3. 注册你需要的组件
const app = createApp(App)
  app.use(router)
  app.use(Vant)
  app.use(ElementPlus)
  app.mount('#app')
