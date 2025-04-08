<template>
  <div>
    <van-nav-bar title="急性肾衰竭(AKI)预测" fixed="true" placeholder="true"/>
    <router-view></router-view>
  </div>
</template>

<script>
import IndexPage from './components/index.vue';
import dashboard from './components/dashboard.vue'
import dashboard_test from './components/dashboard_test.vue';
import { ref, reactive, computed } from 'vue';
export default {
  components: {
    IndexPage
  }
}
</script>

<style>
/* 在全局或组件样式添加 */
:root {
  --van-blue: #1989fa;
  --van-gray-6: #969799;
}
</style>