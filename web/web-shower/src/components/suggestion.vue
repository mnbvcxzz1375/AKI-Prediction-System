<template>
    <div class="aki-glass-container">
      <van-collapse v-model="activeNames" accordion>
        <van-collapse-item 
          v-for="item in adviceList" 
          :key="item.risk" 
          :name="item.risk"
          class="glass-item"
        >
          <template #title>
            <div class="glass-header">
              <div class="glass-badge" :class="item.risk">
                {{ item.riskLabel }}
              </div>
              <span class="glass-title">{{ item.title }}</span>
            </div>
          </template>
          
          <div class="glass-content">
            <p>{{ item.advice }}</p>
            <div class="glass-icon">
              <van-icon 
                :name="getRiskIcon(item.risk)" 
                size="28"
                :color="getIconColor(item.risk)"
              />
            </div>
          </div>
        </van-collapse-item>
      </van-collapse>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  
  const activeNames = ref([])
  
  const adviceList = [
    {
        risk: 'none',
        riskLabel: '无 AKI',
        title: '无需干预',
        advice:
        '当前患者肾功能正常，无急性肾损伤表现。建议继续保持良好的生活方式，包括充足饮水、规律作息、健康饮食等。同时建议定期监测肾功能指标，如血肌酐、尿素氮、电解质水平等，以便及早发现潜在的肾功能变化。避免使用肾毒性药物，控制基础慢性病如高血压、糖尿病等，预防肾功能恶化。',
        tagType: 'success',
    },
    {
        risk: 'low',
        riskLabel: '低风险',
        title: '建议注意观察',
        advice:
        '患者目前处于急性肾损伤的低风险状态，尚未出现明显的肾功能异常。此阶段应加强液体摄入与排出平衡的管理，避免脱水和电解质紊乱。建议每日监测血肌酐、电解质等基础肾功能指标。如需使用肾毒性药物，应权衡利弊并进行肾功能动态监测。同时控制好基础疾病，预防进一步肾损伤。',
        tagType: 'primary',
    },
    {
        risk: 'medium',
        riskLabel: '中风险',
        title: '建议预防性干预',
        advice:
        '中度 AKI 风险提示患者可能存在轻度肾功能损伤或有明显风险因素。建议加强监测，尤其是血肌酐、尿量和电解质等，每 8 小时评估一次情况。应避免使用肾毒性药物，合理补液以维持有效循环灌注。如存在感染、低血压等诱因应尽早干预，必要时请肾内科协助评估治疗方案，以防止肾功能进一步恶化。',
        tagType: 'warning',
    },
    {
        risk: 'high',
        riskLabel: '高风险',
        title: '需立即干预',
        advice:
        'AKI 高风险患者可能已出现急性肾功能下降、少尿或无尿，应立即采取干预措施。首先排查并纠正诱因（如感染、低血压、脱水等），暂停所有潜在肾毒性药物。加强血流动力学支持、静脉补液并密切监测尿量、电解质及酸碱平衡。必要时应立即会诊肾内科，评估是否需启动肾替代治疗（如透析）。',
        tagType: 'danger',
    },
]

  
  const getRiskIcon = (risk) => {
    const icons = {
      none: 'certificate',
      low: 'eye',
      medium: 'warning',
      high: 'fire'
    }
    return icons[risk] || 'info'
  }
  
  const getIconColor = (risk) => {
    const colors = {
      none: '#7ed56f',
      low: '#64b5f6',
      medium: '#ffb74d',
      high: '#ff5252'
    }
    return colors[risk] || '#64b5f6'
  }
  </script>
  
  <style scoped>
  /* 基础玻璃态效果 */
  .aki-glass-container {
    padding: 16px;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
  }
  
  /* 每个项目卡片样式 */
  .glass-item {
    margin-bottom: 16px;
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    transition: all 0.3s ease;
  }
  
  .glass-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  }
  
  /* 头部样式 */
  .glass-header {
    display: flex;
    align-items: center;
    padding: 16px;
  }
  
  .glass-title {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-left: 12px;
  }
  
  /* 徽章样式 */
  .glass-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 72px;
    height: 28px;
    padding: 0 14px;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
    z-index: 1;
  }
  
  .glass-badge::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, 
      rgba(255,255,255,0.4) 0%, 
      rgba(255,255,255,0) 50%, 
      rgba(0,0,0,0.05) 100%);
    z-index: -1;
  }
  
  .glass-badge.none {
    background: linear-gradient(135deg, #7ed56f, #28b485);
  }
  
  .glass-badge.low {
    background: linear-gradient(135deg, #64b5f6, #1976d2);
  }
  
  .glass-badge.medium {
    background: linear-gradient(135deg, #ffb74d, #fb8c00);
  }
  
  .glass-badge.high {
    background: linear-gradient(135deg, #ff5252, #d32f2f);
  }
  
  /* 内容区域 */
  .glass-content {
    display: flex;
    padding: 16px;
    background: rgba(255, 255, 255, 0.3);
    border-top: 1px solid rgba(255, 255, 255, 0.5);
  }
  
  .glass-content p {
    margin: 0;
    flex: 1;
    color: #555;
    line-height: 1.6;
    font-size: 14px;
  }
  
  .glass-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    margin-left: 12px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }
  
  /* 自定义折叠箭头 */
  :deep(.van-collapse-item__arrow) {
    color: rgba(0, 0, 0, 0.6);
    font-size: 18px;
    margin-left: auto;
  }
  
  /* 响应式调整 */
  @media (max-width: 480px) {
    .glass-badge {
      min-width: 60px;
      padding: 0 10px;
    }
    
    .glass-title {
      font-size: 15px;
    }
  }
  </style>
  