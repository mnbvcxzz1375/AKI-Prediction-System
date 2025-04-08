import { createRouter, createWebHistory } from 'vue-router'
import index from '../components/index.vue'
import dashboard from '../components/dashboard.vue'
import dashboard_test from '../components/dashboard_test.vue'
import CsvUploaderVue from '@/components/BatchUpload.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: index,
    },
    {
      path: '/result',
      name: 'dashboard',
      component: dashboard,
    },
    {
      path: '/dashboard_test',
      name: 'dashboard_test',
      component: dashboard_test,
    },
  ],
  scrollBehavior(to, from, savedPosition) {
    // 如果存在 savedPosition（例如浏览器的前进/后退），则使用它
    if (savedPosition) {
      return savedPosition;
    }
    // 否则滚动到顶部
    return { top: 0 };
  },
})

export default router
