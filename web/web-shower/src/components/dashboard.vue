<template>
  <div class="container">
    <!-- 顶部导航栏 -->
    <van-nav-bar title="AKI 可视化展示" left-text="返回" left-arrow @click-left="onClickBack" class="nav-bar" />
    <suggestion />
    <!-- 内容区域 -->
    <div class="content">
      <!-- 风险分布图 -->
      <div class="chart-card">
        <h2 class="chart-title">风险分布图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>展示不同AKI风险等级患者的比例分布</van-swipe-item>
              <van-swipe-item>关注"high"风险区域占比（红色部分）</van-swipe-item>
              <van-swipe-item>对比各色块面积：红色>橙色>黄色>绿色需重点干预</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.risk" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="riskIframe" :data-src="`${baseUrl}/risk_dist.html`" @load="handleLoad('risk')" class="chart-iframe"
          :class="{ loaded: loadedStatus.risk }"></iframe>
      </div>

      <!-- 概率排序图 -->
      <div class="chart-card">
        <h2 class="chart-title">概率排序图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>按风险概率排序的病例分布</van-swipe-item>
              <van-swipe-item>观察超过阈值线（默认50%）的病例数量</van-swipe-item>
              <van-swipe-item>寻找右端聚集的高概率病例（条形图右侧）</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.prob" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="probIframe" :data-src="`${baseUrl}/prob_chart.html`" @load="handleLoad('prob')"
          class="chart-iframe" :class="{ loaded: loadedStatus.prob }"></iframe>
      </div>

      <!-- 仪表盘 -->
      <div class="chart-card">
        <h2 class="chart-title">仪表盘</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>直观显示特定患者的即时风险状态</van-swipe-item>
              <van-swipe-item>指针位置：红色区域（80-100%）需紧急处理</van-swipe-item>
              <van-swipe-item>颜色段占比：红色段范围反映高风险判定标准</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.gauge" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="gaugeIframe" :data-src="`${baseUrl}/gauge.html`" @load="handleLoad('gauge')" class="chart-iframe"
          :class="{ loaded: loadedStatus.gauge }"></iframe>
      </div>

      <!-- 概率直方图 -->
      <div class="chart-card">
        <h2 class="chart-title">概率直方图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>展示患者群体的风险概率分布形态</van-swipe-item>
              <van-swipe-item>双峰分布提示存在明显高危/低危分组</van-swipe-item>
              <van-swipe-item>右偏分布提示高危人群较多</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.histogram" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="histogramIframe" :data-src="`${baseUrl}/prob_histogram.html`" @load="handleLoad('histogram')"
          class="chart-iframe" :class="{ loaded: loadedStatus.histogram }"></iframe>
      </div>

      <!-- 实验室指标箱线图 -->
      <div class="chart-card">
        <h2 class="chart-title">实验室指标箱线图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>关键检验指标在不同风险组的分布差异</van-swipe-item>
              <van-swipe-item>比较各风险组箱体的中位数位置</van-swipe-item>
              <van-swipe-item>观察异常值（离散点）分布规律</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.boxplot" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="boxplotIframe" :data-src="`${baseUrl}/lab_boxplot.html`" @load="handleLoad('boxplot')"
          class="chart-iframe" :class="{ loaded: loadedStatus.boxplot }"></iframe>
      </div>

      <!-- 实验室指标相关性图 -->
      <div class="chart-card">
        <h2 class="chart-title">实验室指标相关性图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>揭示实验室指标间的相互作用关系</van-swipe-item>
              <van-swipe-item>寻找与AKI概率强相关（深色）的指标</van-swipe-item>
              <van-swipe-item>关注指标间的协同/拮抗关系（正/负相关）</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.correlation" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="correlationIframe" :data-src="`${baseUrl}/lab_correlation.html`" @load="handleLoad('correlation')"
          class="chart-iframe" :class="{ loaded: loadedStatus.correlation }"></iframe>
      </div>

      <!-- 实验室指标配对图 -->
      <div class="chart-card">
        <h2 class="chart-title">实验室指标配对图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>多维指标关系的可视化探索</van-swipe-item>
              <van-swipe-item>观察散点图的线性/非线性趋势</van-swipe-item>
              <van-swipe-item>识别高风险簇的聚集区域（右上象限）</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.pairplot" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="pairplotIframe" :data-src="`${baseUrl}/lab_pairplot.html`" @load="handleLoad('pairplot')"
          class="chart-iframe" :class="{ loaded: loadedStatus.pairplot }"></iframe>
      </div>

      <!-- 实验室指标小提琴图 -->
      <div class="chart-card">
        <h2 class="chart-title">实验室指标小提琴图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>实验室指标的密度分布比较</van-swipe-item>
              <van-swipe-item>比较各风险组分布的"腰部"位置</van-swipe-item>
              <van-swipe-item>观察分布形态的对称性</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.violin" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="violinIframe" :data-src="`${baseUrl}/lab_violin.html`" @load="handleLoad('violin')"
          class="chart-iframe" :class="{ loaded: loadedStatus.violin }"></iframe>
      </div>

      <!-- 实验室指标平行坐标图 -->
      <div class="chart-card">
        <h2 class="chart-title">实验室指标平行坐标图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>多指标联合分析工具</van-swipe-item>
              <van-swipe-item>追踪高风险病例的折线走向</van-swipe-item>
              <van-swipe-item>发现异常折线模式（多指标同时偏离）</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.parallel" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="parallelIframe" :data-src="`${baseUrl}/lab_parallel.html`" @load="handleLoad('parallel')"
          class="chart-iframe" :class="{ loaded: loadedStatus.parallel }"></iframe>
      </div>

      <!-- 风险雷达图 -->
      <div class="chart-card">
        <h2 class="chart-title">风险雷达图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>个体患者的多维度特征画像</van-swipe-item>
              <van-swipe-item>多边形的尖峰数量反映异常指标数量</van-swipe-item>
              <van-swipe-item>面积越大综合风险越高</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.radar" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="radarIframe" :data-src="`${baseUrl}/radar.html`" @load="handleLoad('radar')" class="chart-iframe"
          :class="{ loaded: loadedStatus.radar }"></iframe>
      </div>

      <!-- 患者分布旭日图 -->
      <div class="chart-card">
        <h2 class="chart-title">患者分布旭日图</h2>
        <van-notice-bar left-icon="fail" :scrollable="false" >
            <van-swipe
              vertical
              class="notice-swipe"
              :autoplay="3000"
              :touchable="false"
              :show-indicators="false"
            >
              <van-swipe-item>层级化风险因素分解</van-swipe-item>
              <van-swipe-item>从内到外观察风险因素的嵌套关系</van-swipe-item>
              <van-swipe-item>聚焦占比最大的外层因素</van-swipe-item>
            </van-swipe>
          </van-notice-bar>
        <van-loading v-show="!loadedStatus.sunburst" size="24px" class="loading">加载中...</van-loading>
        <iframe ref="sunburstIframe" :data-src="`${baseUrl}/sunburst.html`" @load="handleLoad('sunburst')"
          class="chart-iframe" :class="{ loaded: loadedStatus.sunburst }"></iframe>
      </div>

    </div>
  </div>
</template>

<script>
import suggestion from './suggestion.vue'
export default {
  components: { suggestion },
  name: 'AkiVisualization',
  data() {
    return {
      baseUrl: 'assert/html/',
      loadedStatus: {
        risk: false,
        prob: false,
        gauge: false,
        histogram: false,
        boxplot: false,
        correlation: false,
        pairplot: false,
        violin: false,
        parallel: false,
        radar: false,
        sunburst: false
      },
      observers: new Map()
    }
  },
  methods: {
    onClickBack() {
      this.$router.back()
    },
    handleLoad(chartType) {
      this.loadedStatus[chartType] = true
    },
    setupImmediateLoad(refName, chartType) {
      const element = this.$refs[refName]
      if (!element || element.src) return  // 防止重复加载

      // 立即设置src开始加载
      element.src = element.dataset.src

      // 主动触发加载状态更新
      this.$nextTick(() => {
        if (element.complete) {
          this.handleLoad(chartType)
        }
      })
    }
  },
  mounted() {
    // 按从上到下的顺序启动加载
    const loadQueue = [
      ['riskIframe', 'risk'],
      ['probIframe', 'prob'],
      ['gaugeIframe', 'gauge'],
      ['histogramIframe', 'histogram'],
      ['boxplotIframe', 'boxplot'],
      ['correlationIframe', 'correlation'],
      ['pairplotIframe', 'pairplot'],
      ['violinIframe', 'violin'],
      ['parallelIframe', 'parallel'],
      ['radarIframe', 'radar'],
      ['sunburstIframe', 'sunburst']
    ]
    // 分批次加载（每批3个）
    const BATCH_SIZE = 3
    loadQueue.forEach((args, index) => {
      setTimeout(() => {
        this.setupImmediateLoad(...args)
      }, Math.floor(index / BATCH_SIZE) * 100)  // 每批间隔300ms
    })
  },
  beforeUnmount() {
    // 清理逻辑可以保留
    this.observers.forEach(observer => observer.disconnect())
  }
}
</script>




<style scoped>
  .notice-swipe {
    height: 40px;
    line-height: 40px;
  }
.overlay {
  display: flex;
  align-items: center;
  justify-content: center;
}

.loading-wrapper {
  text-align: center;
  color: white;
}
/* 全局进度条 */
.global-progress {
  position: fixed;
  top: 46px;
  left: 0;
  right: 0;
  z-index: 100;
  padding: 12px 20px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.progress-text {
  margin-top: 8px;
  font-size: 12px;
  color: #666;
  display: flex;
  justify-content: space-between;
}

.tips {
  color: #999;
  font-style: italic;
}

/* 骨架屏 */
.skeleton-wrapper {
  padding: 16px;
}

.skeleton-title {
  height: 24px;
  width: 40%;
  background: #f0f0f0;
  border-radius: 4px;
  margin-bottom: 16px;
}

.skeleton-chart {
  height: 300px;
  background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
  border-radius: 8px;
  animation: skeleton-loading 1.5s infinite;
}

/* 错误状态 */
.error-wrapper {
  padding: 40px 20px;
  text-align: center;
  background: #fff5f5;
}

.error-icon {
  font-size: 48px;
  color: #ff4444;
  margin-bottom: 12px;
}

.retry-btn {
  margin-top: 12px;
}

/* 图表卡片 */
.chart-card {
  background: white;
  border-radius: 12px;
  margin: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  min-height: 400px;
}

.chart-iframe {
  width: 100%;
  border: none;
  transition: height 0.3s ease;
}

@keyframes skeleton-loading {
  0% {
    background-position: 200% 0;
  }

  100% {
    background-position: -200% 0;
  }
}

/* 响应式调整 */
@media (max-width: 768px) {
  .chart-card {
    margin: 12px;
    min-height: 300px;
  }
}

.container {
  background: #f5f7fa;
  min-height: 100vh;
}

.nav-bar {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.content {
  padding: 16px;
}

.chart-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  margin-bottom: 16px;
  overflow: hidden;
  position: relative;
  transition: transform 0.2s, box-shadow 0.2s;
}

.chart-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
}

.chart-title {
  font-size: 16px;
  color: #2c3e50;
  padding: 16px;
  margin: 0;
  border-bottom: 1px solid #ebedf0;
  font-weight: 600;
}

.loading {
  padding: 40px 0;
  text-align: center;
  color: #969799;
  font-size: 14px;
}

.chart-iframe {
  width: 100%;
  height: 480px;
  border: none;
  opacity: 0;
  transition: opacity 0.3s ease-in;
  background: transparent;
}

.chart-iframe.loaded {
  opacity: 1;
}

@media (min-width: 768px) {
  .content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
  }

  .chart-card {
    margin-bottom: 24px;
  }

  .chart-iframe {
    height: 600px;
  }

  .chart-title {
    font-size: 18px;
    padding: 20px;
  }
}

@media (min-width: 1200px) {
  .chart-card {
    margin-bottom: 32px;
  }
}
</style>