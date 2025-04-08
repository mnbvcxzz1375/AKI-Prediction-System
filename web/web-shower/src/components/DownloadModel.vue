<template>
    <div class="download-container">
        <button class="download-btn" :disabled="isLoading" @click="downloadModel">
            <span v-if="isLoading">
                打包中 ({{ downloadProgress }}%)
            </span>
            <span v-else>
                一键下载全部模型
            </span>
        </button>
    </div>
    <div class="github-link">
        <el-link type="success" href="https://github.com/mnbvcxzz1375" :underline="false" target="_blank">
            更多具体预测操作及模型自定义
        </el-link>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
const isLoading = ref(false);
const downloadProgress = ref(0);

// 原生文件保存方法
const saveFile = (blob, fileName) => {
    const link = document.createElement('a');
    const url = window.URL.createObjectURL(blob);
    link.href = url;
    link.download = fileName;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }, 100);
};

const downloadModel = async () => {
  try {
    // 添加路径调试信息
    console.log('正在请求下载接口...')
    
    const response = await axios({
      method: 'GET',
      url: 'http://localhost:5000/download_model',
      responseType: 'blob',
      params: { 
        debug: true,
        ts: new Date().getTime()  // 防止缓存
      },
      onDownloadProgress: progress => {
        const percent = Math.round(
          (progress.loaded * 100) / (progress.total || 1))
        console.log(`下载进度：${percent}%`)
      }
    })
    // 响应头验证
    console.log('响应头：', response.headers)
    if (!response.headers['content-type'].includes('zip')) {
      throw new Error('返回的不是ZIP文件')
    }
    // 使用更可靠的文件名获取方式
    let filename = 'model_package.zip'
    const disposition = response.headers['content-disposition']
    if (disposition) {
      const matches = disposition.match(/filename="?(.+)"?/i)
      if (matches && matches[1]) {
        filename = matches[1].replace(/[\\/]/g, '_')  //防止路径注入
      }
    }
    // 改进的文件保存方法
    const blob = new Blob([response.data], { type: 'application/zip' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    document.body.appendChild(link)
    link.click()
    setTimeout(() => {
      document.body.removeChild(link)
      URL.revokeObjectURL(link.href)
    }, 100)
  } catch (error) {
    console.error('完整错误：', error)
    if (error.response) {
      console.error('响应数据：', error.response.data)
    }
    alert(`下载失败：${error.message}`)
  }
}
</script>

<style scoped>
.github-link {
    margin-top: 20px;
    text-align: center;
}

.github-link .el-link {
    font-size: 16px;
    font-weight: 500;
}
.download-container {
    padding: 20px;
    text-align: center;
}

.download-btn {
    padding: 12px 24px;
    background: #409eff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.3s;
}

.download-btn:disabled {
    background: #a0cfff;
    cursor: not-allowed;
}

.download-btn:hover:not(:disabled) {
    background: #79bbff;
}
</style>
