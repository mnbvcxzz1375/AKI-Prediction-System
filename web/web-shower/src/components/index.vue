<template>
  <div class="predict-container">
    <van-notice-bar color="#1989fa" background="#ecf9ff" left-icon="info-o">
          不清楚部分可以不填写，但是部分为必须填写，尽量填写完整，以确保预测准确
        </van-notice-bar>
    <van-tabs v-model:active="activeTab" swipeable>

      <!-- 单条数据录入标签 -->
      <van-tab title="单条录入">

        <van-form ref="formRef" @submit="handleSubmit">
          <h3 class="section-title">实验室指标</h3>
          <van-cell-group inset>
            <template v-for="group in labGroups" :key="group.title">
              <div class="field-group">
                <h4 class="group-subtitle">{{ group.title }}</h4>
                <van-field v-for="field in group.fields" :key="field.key" v-model="formData.lab[field.key]"
                  :label="field.label" type="number" :placeholder="field.placeholder" :rules="field.rules" clearable >
                  <template #extra>
                    <span class="unit">{{ field.unit }}</span>
                  </template>
                </van-field>
              </div>
            </template>
          </van-cell-group>

          <!-- 微生物指标 -->
          <h3 class="section-title">微生物指标</h3>
          <van-cell-group inset>
            <div class="field-group">
              <van-field v-for="field in microFields" :key="field.key" v-model="formData.micro[field.key]"
                :label="field.label" :type="field.type" :placeholder="field.placeholder" :rules="field.rules"
                :readonly="!!field.options" :is-link="!!field.options" @click="field.options && showPicker(field)">
                <template #extra v-if="field.unit">
                  <span class="unit">{{ field.unit }}</span>
                </template>
              </van-field>
            </div>
          </van-cell-group>
        </van-form>
      </van-tab>
      <!-- 批量上传标签 -->
      <van-tab title="批量上传">
        <BatchUpload @upload-success="handleUploadSuccess" />
      </van-tab>
      <van-tab title="下载模型">
        <DownloadModel />
      </van-tab>
    </van-tabs>
    
    <!-- 公共提交按钮 -->
    <div class="submit-area">
      <van-button v-if="activeTab === 0" type="primary" block native-type="submit" :disabled="isAllEmpty"
        :loading="isSubmitting" loading-text="正在分析中..." @click="$refs.formRef.submit()">
        提交预测
      </van-button>
    </div>
    <!-- 选择器弹窗 -->
    <van-popup v-model:show="showPickerVisible" position="bottom" round>
      <van-picker :columns="currentPickerOptions" @confirm="onPickerConfirm" @cancel="showPickerVisible = false" />
    </van-popup>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue';
import { useRouter } from 'vue-router';
import { showToast } from 'vant';
import axios from 'axios';
import BatchUpload from './BatchUpload.vue';
import DownloadModel from './DownloadModel.vue';  
const router = useRouter();
const activeTab = ref(0);

// 处理批量上传成功
const handleUploadSuccess = (result) => {
  router.push({
    name: 'dashboard',
    state: { charts: result.charts }
  });
};
// 实验室指标分组配置（完整）
const labGroups = [

  {
    title: '电解质指标',
    fields: [
      { key: 'aniongap_min', label: '阴离子间隙最小值', unit: 'mmol/L', placeholder: '输入最小值' },
      { key: 'aniongap_max', label: '阴离子间隙最大值', unit: 'mmol/L', placeholder: '输入最大值' },
      { key: 'chloride_min', label: '氯最小值', unit: 'mmol/L' },
      { key: 'chloride_max', label: '氯最大值', unit: 'mmol/L' },
      { key: 'potassium_min', label: '钾最小值', unit: 'mmol/L' },
      { key: 'potassium_max', label: '钾最大值', unit: 'mmol/L' },
      { key: 'sodium_min', label: '钠最小值', unit: 'mmol/L' },
      { key: 'sodium_max', label: '钠最大值', unit: 'mmol/L' },
      { key: 'bicarbonate_min', label: '碳酸氢盐最小值', unit: 'mmol/L' },
      { key: 'bicarbonate_max', label: '碳酸氢盐最大值', unit: 'mmol/L' },
    ]
  },
  {
    title: '肾功能指标',
    fields: [
      { key: 'creatinine_min', label: '肌酐最小值', unit: 'μmol/L',rules: [
        { required: true, message: '必须填写' }, // 必填规则
        { pattern: /^\d+(\.\d+)?$/, message: '请输入有效数字' }
      ] },
      { key: 'creatinine_max', label: '肌酐最大值', unit: 'μmol/L' ,rules: [
        { required: true, message: '必须填写' }, // 必填规则
        { pattern: /^\d+(\.\d+)?$/, message: '请输入有效数字' }
      ]},
      { key: 'bun_min', label: '尿素氮最小值', unit: 'mg/dL' ,rules: [
        { required: true, message: '必须填写' }, // 必填规则
        { pattern: /^\d+(\.\d+)?$/, message: '请输入有效数字' }
      ]},
      { key: 'bun_max', label: '尿素氮最大值', unit: 'mg/dL',rules: [
        { required: true, message: '必须填写' }, // 必填规则
        { pattern: /^\d+(\.\d+)?$/, message: '请输入有效数字' }
      ] },
    ]
  },
  {
    title: '肝功能指标',
    fields: [
      { key: 'albumin_min', label: '白蛋白最小值', unit: 'g/dL' },
      { key: 'albumin_max', label: '白蛋白最大值', unit: 'g/dL' },
      { key: 'bilirubin_min', label: '胆红素最小值', unit: 'mg/dL' },
      { key: 'bilirubin_max', label: '胆红素最大值', unit: 'mg/dL' },
    ]
  },
  {
    title: '血糖与乳酸',
    fields: [
      { key: 'glucose_min', label: '葡萄糖最小值', unit: 'mg/dL' },
      { key: 'glucose_max', label: '葡萄糖最大值', unit: 'mg/dL' },
      { key: 'lactate_min', label: '乳酸最小值', unit: 'mmol/L' },
      { key: 'lactate_max', label: '乳酸最大值', unit: 'mmol/L' },
    ]
  },
  {
    title: '血液指标',
    fields: [
      { key: 'hematocrit_min', label: '血细胞比容最小值', unit: '%' },
      { key: 'hematocrit_max', label: '血细胞比容最大值', unit: '%' },
      { key: 'hemoglobin_min', label: '血红蛋白最小值', unit: 'g/dL' },
      { key: 'hemoglobin_max', label: '血红蛋白最大值', unit: 'g/dL' },
      { key: 'platelet_min', label: '血小板最小值', unit: '×10⁹/L' },
      { key: 'platelet_max', label: '血小板最大值', unit: '×10⁹/L' },
      { key: 'wbc_min', label: '白细胞最小值', unit: '×10⁹/L' },
      { key: 'wbc_max', label: '白细胞最大值', unit: '×10⁹/L' },
    ]
  },
  {
    title: '凝血功能',
    fields: [
      { key: 'ptt_min', label: '部分凝血活酶时间最小值', unit: '秒' },
      { key: 'ptt_max', label: '部分凝血活酶时间最大值', unit: '秒' },
      { key: 'inr_min', label: '国际标准化比率最小值', unit: '无单位' },
      { key: 'inr_max', label: '国际标准化比率最大值', unit: '无单位' },
      { key: 'pt_min', label: '凝血酶原时间最小值', unit: '秒' },
      { key: 'pt_max', label: '凝血酶原时间最大值', unit: '秒' },
    ]
  },
  {
    title: '其他',
    fields: [
      { key: 'bands_min', label: '带状中性粒细胞最小值', unit: '%' },
      { key: 'bands_max', label: '带状中性粒细胞最大值', unit: '%' },
    ]
  }

  // 其他指标分组...
];

// 微生物指标配置（完整）
const microFields = [
  {
    key: 'spec_itemid',
    label: '标本类型',
    type: 'number',
    // rules: [{ pattern: /^\d+$/, message: '请输入有效数字' }]
  },
  {
    key: 'org_itemid',
    label: '微生物种类',
    type: 'number',
    // rules: [{ pattern: /^\d+$/, message: '请输入有效数字' }]
  },
  {
    key: 'isolate_num',
    label: '分离株数量',
    type: 'number',
    // rules: [{ pattern: /^\d+$/, message: '请输入有效数字' }]
  },
  { key: 'ab_itemid', label: '抗生素类型', type: 'text' },
  { key: 'dilution_text', label: '稀释说明', type: 'text' },
  {
    key: 'dilution_value',
    label: '稀释值',
    type: 'number',
    // rules: [{ pattern: /^\d+$/, message: '请输入有效数值' }]
  },
  {
    key: 'urineoutput',
    label: '尿量',
    type: 'number',
    unit: 'mL',
    rules: [{ pattern: /^\d+$/, message: '请输入有效数值' }]
  },
  {
    key: 'gender',
    label: '性别',
    options: ['男', '女'],
    rules: [{ required: true, message: '请选择性别' }]
  },
];

// 初始化表单数据
const formData = reactive({
  lab: Object.fromEntries(
    labGroups.flatMap(group =>
      group.fields.map(field => [field.key, ''])
    )
  ),
  micro: Object.fromEntries(
    microFields.map(field => [field.key, ''])
  )
});

// 选择器逻辑
const showPickerVisible = ref(false);
const currentPicker = ref(null);
const currentPickerOptions = ref([]);

const showPicker = (field) => {
  currentPicker.value = field.key;
  currentPickerOptions.value = field.options.map(opt => ({ text: opt, value: opt }));
  showPickerVisible.value = true;
};

const onPickerConfirm = ({ selectedOptions }) => {
  formData.micro[currentPicker.value] = selectedOptions[0].value;
  showPickerVisible.value = false;
};

// 提交逻辑
const isSubmitting = ref(false);
const isAllEmpty = computed(() => {
  const labFilled = Object.values(formData.lab).some(v => v !== '');
  const microFilled = Object.values(formData.micro).some(v => v !== '');
  return !(labFilled || microFilled);
});

// 修正后的数据预处理（匹配你的Python处理逻辑）
const formatSubmissionData = () => {
  // 性别编码
  const genderMap = { '男': 1, '女': 0 };
  console.log('genderMap:', genderMap);
  const gender = genderMap[formData.micro.gender] || -1
  return {
    hadm_id: Date.now().toString(),
    lab: {
      ...Object.fromEntries(
        Object.entries(formData.lab)
          .map(([k, v]) => [k, v ? parseFloat(v) : null])
      ),
      gender: gender  // 将gender添加到lab数据
    },
    micro: {
      ...Object.fromEntries(
        Object.entries(formData.micro)
          .filter(([k]) => k !== 'gender')  // 从micro移除gender
          .map(([k, v]) => [k, k === 'dilution_text' ? parseFloat(v) : v])
      )
    }
  };
};
// API调用示例
const handleSubmit = async () => {
  try {
    isSubmitting.value = true;

    // 1. 格式化数据（包含结构调整）
    const payload = formatSubmissionData();

    // 2. 调用预测接口
    const response = await axios.post('http://localhost:5000/predict', payload);

    // 3. 处理结果并跳转到 dashboard 页面
    if (response.data.success) {
      router.push({ name: 'dashboard' });
    } else {
      showToast('预测失败: ' + response.data.message);
    }
  } catch (error) {
    console.error('API错误:', error);
    showToast('请求失败: ' + error.message);
  } finally {
    isSubmitting.value = false;
  }
};
const csvFiles = ref([]);
// 数据格式化工具函数
const formatLabData = (labData) => {
  return Object.keys(labData).reduce((acc, key) => {
    acc[key] = labData[key] ? parseFloat(labData[key]) : null;
    return acc;
  }, {});
};

// const formatMicroData = (microData) => {
//   return {
//     ...microData,
//     dilution_text: microData.dilution_text ? parseFloat(microData.dilution_text) : null,
//     gender: microData.gender === '男' ? 1 : 0,
//     spec_itemid: microData.spec_itemid,  // 保持原始值
//     org_itemid: microData.org_itemid     // 保持原始值
//   };
// };
// 上传CSV文件后的处理逻辑
const handleCSVUpload = async (file) => {
  try {
    const formData = new FormData();

    // 处理多文件上传
    const files = Array.isArray(file) ? file : [file];

    // 验证文件类型
    const invalidFiles = files.filter(f => {
      const ext = f.file.name.split('.').pop().toLowerCase();
      return ext !== 'csv';
    });

    if (invalidFiles.length > 0) {
      showToast(`无效文件类型: ${invalidFiles.map(f => f.file.name).join(', ')}`);
      return;
    }
    files.forEach(f => {
      formData.append('file', f.file);
    });
    const response = await axios.post('http://localhost:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'X-Client-Type': 'web-uploader'
      },
      timeout: 30000  // 30秒超时
    });
    if (response.data.success) {
      emit('upload-success', response.data);
    } else {
      handleApiError(response.data);
    }
  } catch (error) {
    handleNetworkError(error);
  }
};

const handleApiError = (data) => {
  let message = data.message;

  if (data.mismatch_details) {
    message += `\n首三个不匹配列: ${JSON.stringify(data.mismatch_details)}`;
  }

  if (data.error_columns) {
    message += `\n问题列: ${data.error_columns.join(', ')}`;
  }

  showToast({
    message,
    duration: 5000,
    forbidClick: true
  });
};
const handleNetworkError = (error) => {
  if (error.code === 'ECONNABORTED') {
    showToast('请求超时，请检查文件大小（建议<10MB）');
  } else if (error.response) {
    showToast(`服务器错误: ${error.response.status}`);
  } else {
    showToast(`网络错误: ${error.message}`);
  }
};

// 模板CSV下载
const downloadCSVTemplate = () => {
  const link = document.createElement('a');
  link.href = 'predict_template.csv'; // 确保此文件放在 public 或 static 文件夹中
  link.download = 'predict_template.csv';
  link.click();
};
</script>

<style scoped>
.predict-container {
  padding: 16px;
  padding-bottom: 80px;
  height: calc(100vh - 80px);
  overflow-y: auto;
}

.submit-area {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 16px;
  background: white;
  box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.1);
  z-index: 100;
}

:deep(.van-tabs__wrap) {
  position: sticky;
  top: 0;
  z-index: 100;
  background: white;
}

.section-title {
  margin: 24px 0 12px;
  color: var(--van-blue);
  font-size: 16px;
  padding-left: 16px;
  border-left: 4px solid var(--van-blue);
}

.group-subtitle {
  margin: 16px 0 8px;
  color: var(--van-gray-7);
  font-size: 14px;
  padding-left: 8px;
}

.unit {
  color: var(--van-gray-6);
  font-size: 12px;
  margin-left: 4px;
}

.submit-area {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 16px;
  background: white;
  box-shadow: 0 -2px 12px rgba(0, 0, 0, 0.1);
  z-index: 100;
}

.field-group {
  margin-bottom: 12px;
  border-radius: 8px;
  background: var(--van-gray-1);
  padding: 8px 0;
}

.csv-upload-area {
  margin-bottom: 24px;
}

.download-template-btn {
  margin-top: 8px;
  width: 100%;
}
</style>
