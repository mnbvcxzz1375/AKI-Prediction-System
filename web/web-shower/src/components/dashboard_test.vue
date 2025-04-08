<template>
    <div class="container">
        <!-- 顶部导航栏 -->
        <van-nav-bar title="AKI 可视化展示" left-text="返回" left-arrow @click-left="onClickBack" class="nav-bar" />
        <suggestion />

        <!-- 加载提示 -->
        <van-overlay :show="!isModelReady" class="overlay">
            <div class="loading-wrapper">
                <van-loading size="24px">模型预测中...</van-loading>
            </div>
        </van-overlay>

        <!-- 内容区域 -->
        <div v-if="isModelReady" class="content">
            <!-- 风险分布图 -->
            <div class="chart-card">
                <h2 class="chart-title">风险分布图</h2>
                
                <van-loading v-show="!loadedStatus.risk" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="riskIframe" :data-src="`${baseUrl}/risk_dist.html`" @load="handleLoad('risk')"
                    @error="handleError('risk')" class="chart-iframe" :class="{ loaded: loadedStatus.risk }"></iframe>
            </div>

            <!-- 概率排序图 -->
            <div class="chart-card">
                <h2 class="chart-title">概率排序图</h2>
                <van-loading v-show="!loadedStatus.prob" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="probIframe" :data-src="`${baseUrl}/prob_chart.html`" @load="handleLoad('prob')"
                    @error="handleError('prob')" class="chart-iframe" :class="{ loaded: loadedStatus.prob }"></iframe>
            </div>

            <!-- 仪表盘 -->
            <div class="chart-card">
                <h2 class="chart-title">仪表盘</h2>
                <van-loading v-show="!loadedStatus.gauge" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="gaugeIframe" :data-src="`${baseUrl}/gauge.html`" @load="handleLoad('gauge')"
                    @error="handleError('gauge')" class="chart-iframe" :class="{ loaded: loadedStatus.gauge }"></iframe>
            </div>

            <!-- 概率直方图 -->
            <div class="chart-card">
                <h2 class="chart-title">概率直方图</h2>
                <van-loading v-show="!loadedStatus.histogram" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="histogramIframe" :data-src="`${baseUrl}/prob_histogram.html`"
                    @load="handleLoad('histogram')" @error="handleError('histogram')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.histogram }"></iframe>
            </div>

            <!-- 实验室指标箱线图 -->
            <div class="chart-card">
                <h2 class="chart-title">实验室指标箱线图</h2>
                <van-loading v-show="!loadedStatus.boxplot" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="boxplotIframe" :data-src="`${baseUrl}/lab_boxplot.html`" @load="handleLoad('boxplot')"
                    @error="handleError('boxplot')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.boxplot }"></iframe>
            </div>

            <!-- 实验室指标相关性图 -->
            <div class="chart-card">
                <h2 class="chart-title">实验室指标相关性图</h2>
                <van-loading v-show="!loadedStatus.correlation" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="correlationIframe" :data-src="`${baseUrl}/lab_correlation.html`"
                    @load="handleLoad('correlation')" @error="handleError('correlation')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.correlation }"></iframe>
            </div>

            <!-- 实验室指标配对图 -->
            <div class="chart-card">
                <h2 class="chart-title">实验室指标配对图</h2>
                <van-loading v-show="!loadedStatus.pairplot" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="pairplotIframe" :data-src="`${baseUrl}/lab_pairplot.html`" @load="handleLoad('pairplot')"
                    @error="handleError('pairplot')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.pairplot }"></iframe>
            </div>

            <!-- 实验室指标小提琴图 -->
            <div class="chart-card">
                <h2 class="chart-title">实验室指标小提琴图</h2>
                <van-loading v-show="!loadedStatus.violin" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="violinIframe" :data-src="`${baseUrl}/lab_violin.html`" @load="handleLoad('violin')"
                    @error="handleError('violin')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.violin }"></iframe>
            </div>

            <!-- 实验室指标平行坐标图 -->
            <div class="chart-card">
                <h2 class="chart-title">实验室指标平行坐标图</h2>
                <van-loading v-show="!loadedStatus.parallel" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="parallelIframe" :data-src="`${baseUrl}/lab_parallel.html`" @load="handleLoad('parallel')"
                    @error="handleError('parallel')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.parallel }"></iframe>
            </div>

            <!-- 风险雷达图 -->
            <div class="chart-card">
                <h2 class="chart-title">风险雷达图</h2>
                <van-loading v-show="!loadedStatus.radar" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="radarIframe" :data-src="`${baseUrl}/radar.html`" @load="handleLoad('radar')"
                    @error="handleError('radar')" class="chart-iframe" :class="{ loaded: loadedStatus.radar }"></iframe>
            </div>

            <!-- 患者分布旭日图 -->
            <div class="chart-card">
                <h2 class="chart-title">患者分布旭日图</h2>
                <van-loading v-show="!loadedStatus.sunburst" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="sunburstIframe" :data-src="`${baseUrl}/sunburst.html`" @load="handleLoad('sunburst')"
                    @error="handleError('sunburst')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.sunburst }"></iframe>
            </div>

            <!-- 总览仪表盘 -->
            <div class="chart-card">
                <h2 class="chart-title">总览仪表盘</h2>
                <van-loading v-show="!loadedStatus.dashboard" size="24px" class="loading">加载中...</van-loading>
                <iframe ref="dashboardIframe" src="http://localhost:5000/dashboard" @load="handleLoad('dashboard')"
                    @error="handleError('dashboard')" class="chart-iframe"
                    :class="{ loaded: loadedStatus.dashboard }"></iframe>
            </div>
        </div>

    </div>
</template>
<script>
import suggestion from './suggestion.vue'
// import { getModelPrediction } from '@/api' // 假设的预测接口
export default {
    components: { suggestion },
    name: 'AkiVisualization',
    data() {
        return {
            baseUrl: 'assert/html/',
            isModelReady: false,      // 模型预测完成标志
            loadError: false,         // 加载错误标志
            loadedStatus: {
                // [原有加载状态对象]
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
        handleError(chartType) {
            console.error(`${chartType}图表加载失败`)
            this.loadError = true
            this.loadedStatus[chartType] = true // 隐藏加载状态
        },
        async runModelPrediction() {
            try {
                const { data } = await getModelPrediction(/* 传递预测参数 */)
                if (data.success) {
                    this.isModelReady = true
                    return true
                }
                throw new Error('模型预测失败')
            } catch (error) {
                console.error('模型预测错误:', error)
                this.$notify({ type: 'danger', message: '预测结果获取失败' })
                return false
            }
        },
        setupLazyLoad(refName, chartType) {
            const element = this.$refs[refName]
            if (!element || element.src) return
            // 使用IntersectionObserver实现懒加载
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        element.src = element.dataset.src
                        observer.unobserve(element)
                    }
                })
            }, { rootMargin: '100px' })
            this.observers.set(chartType, observer)
            observer.observe(element)
        },
        initVisualization() {
            const loadQueue = [
                // [原有加载队列配置]
            ]
            // 分批次懒加载（每批3个）
            const BATCH_SIZE = 3
            loadQueue.forEach((args, index) => {
                setTimeout(() => {
                    this.setupLazyLoad(...args)
                }, Math.floor(index / BATCH_SIZE) * 300)
            })
        }
    },
    async mounted() {
        const predictionSuccess = await this.runModelPrediction()
        if (predictionSuccess) {
            this.$nextTick(() => {
                this.initVisualization()
            })
        }
    },
    beforeUnmount() {
        this.observers.forEach(observer => observer.disconnect())
    }
}
</script>

<style scoped>
.overlay {
    display: flex;
    align-items: center;
    justify-content: center;
}

.loading-wrapper {
    text-align: center;
    color: white;
}
</style>



<style scoped>
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
