<template>
  <div class="batch-upload">
    <h3 class="section-title">多人数据上传</h3>
    <div class="csv-upload-area">
      <van-uploader v-model="csvFiles" accept=".csv" :max-count="1" preview-size="80" preview-full-image>
        <van-button icon="plus" type="primary">选择CSV文件</van-button>
      </van-uploader>
      <van-notice-bar color="#1989fa" background="#ecf9ff" left-icon="info-o" wrapable>
        csv文件中的gender列必须为0或1，0表示女，1表示男
      </van-notice-bar>
      <van-notice-bar color="#1989fa" background="#ecf9ff" left-icon="info-o" wrapable>
        若初始为CSV文件，请在Excel中使用"文件->另存为->CSV UTF-8，务必不能直接修改后缀
      </van-notice-bar>
      <van-button class="download-template-btn" type="default" plain @click="downloadCSVTemplate">
        下载CSV模板
      </van-button>
      <!-- 提交按钮 -->
      <van-button type="primary" block :disabled="!hasFiles" :loading="isSubmitting" loading-text="正在分析中..."
        @click="handleSubmit">
        提交预测
      </van-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';
import { showToast } from 'vant';
const emit = defineEmits(['upload-success']);
const csvFiles = ref([]);
const isSubmitting = ref(false);
const hasFiles = computed(() => csvFiles.value.length > 0);
const handleSubmit = async () => {
  if (!hasFiles.value) return;
  // 新增文件数量校验
  if (csvFiles.value.length > 1) {
    showToast('每次只能上传一个文件');
    return;
  }
  isSubmitting.value = true;
  
  try {
    const formData = new FormData();
    // 明确取第一个文件
    formData.append('file', csvFiles.value[0].file);
    const response = await axios.post('http://localhost:5000/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    if (response.data.success) {
      emit('upload-success', response.data);
      csvFiles.value = []; // 清空已选文件
      showToast({
        message: '预测提交成功',
        duration: 2000,
        position: 'top'
      });
    } else {
      handleApiError(response.data);
    }
  } catch (error) {
    handleNetworkError(error);
  } finally {
    isSubmitting.value = false;
  }
};

const handleApiError = (data) => {
  let message = data.message || '请求错误';
  if (data.mismatch_details) {
    message += ` (首列不匹配：${data.mismatch_details[0][0]}≠${data.mismatch_details[0][1]})`;
  }
  showToast({
    message,
    duration: 3000,
    position: 'top'
  });
};

const handleNetworkError = (error) => {
  let message = '网络错误';
  if (error.response) {
    if (error.response.status === 400) {
      message = error.response.data.message || '文件格式错误';
    } else if (error.response.status === 500) {
      message = '服务器处理失败';
    }
  }
  showToast({
    message,
    duration: 3000,
    position: 'top'
  });
};

const downloadCSVTemplate = () => {
  const link = document.createElement('a');
  link.href = 'predict_template.csv';
  link.download = '预测模板.csv';
  link.click();
};
</script>

<style scoped>
.batch-upload {
  padding: 16px;
}

.csv-upload-area {
  margin: 16px 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.download-template-btn {
  margin-top: 8px;
}

.submit-area {
  margin-top: 24px;
}
</style>
