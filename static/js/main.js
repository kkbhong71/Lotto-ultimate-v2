// =============================================================================
// 🎰 Lotto Ultimate v2.0 - Frontend JavaScript
// =============================================================================

// === Global Variables ===
let charts = {};
let currentData = null;

// === DOM Elements ===
const elements = {
    btnQuickPredict: document.getElementById('btnQuickPredict'),
    btnFullAnalysis: document.getElementById('btnFullAnalysis'),
    btnCopyAll: document.getElementById('btnCopyAll'),
    numPredictions: document.getElementById('numPredictions'),
    runBacktest: document.getElementById('runBacktest'),
    runOptimization: document.getElementById('runOptimization'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    resultsContainer: document.getElementById('resultsContainer'),
    emptyState: document.getElementById('emptyState'),
    totalRounds: document.getElementById('totalRounds'),
    lastDraw: document.getElementById('lastDraw'),
    predictionsGrid: document.getElementById('predictionsGrid'),
    hotNumbers: document.getElementById('hotNumbers'),
    coldNumbers: document.getElementById('coldNumbers'),
    risingNumbers: document.getElementById('risingNumbers'),
    topPairs: document.getElementById('topPairs'),
    clusterGrid: document.getElementById('clusterGrid'),
    backtestSection: document.getElementById('backtestSection'),
    backtestContent: document.getElementById('backtestContent'),
    optimizationSection: document.getElementById('optimizationSection'),
    optimizationContent: document.getElementById('optimizationContent'),
    toast: document.getElementById('toast')
};

// === Initialize ===
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadDataInfo();
});

// === Event Listeners ===
function initEventListeners() {
    elements.btnQuickPredict.addEventListener('click', handleQuickPredict);
    elements.btnFullAnalysis.addEventListener('click', handleFullAnalysis);
    elements.btnCopyAll.addEventListener('click', handleCopyAll);
}

// === API Calls ===
async function loadDataInfo() {
    try {
        const response = await fetch('/api/data-info');
        const result = await response.json();
        
        if (result.success) {
            elements.totalRounds.textContent = result.data.total_rounds.toLocaleString() + '회';
            elements.lastDraw.textContent = result.data.last_draw.join(', ');
        }
    } catch (error) {
        console.error('Failed to load data info:', error);
    }
}

async function handleQuickPredict() {
    const numPredictions = parseInt(elements.numPredictions.value);
    
    showLoading('빠른 예측 중...');
    
    try {
        const response = await fetch('/api/quick-predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_predictions: numPredictions })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayQuickResults(result);
            showToast('예측이 완료되었습니다!', 'success');
        } else {
            showToast('오류: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('네트워크 오류가 발생했습니다.', 'error');
    } finally {
        hideLoading();
    }
}

async function handleFullAnalysis() {
    const numPredictions = parseInt(elements.numPredictions.value);
    const runBacktest = elements.runBacktest.checked;
    const runOptimization = elements.runOptimization.checked;
    
    let loadingMessage = '전체 분석 중...';
    if (runOptimization) loadingMessage = '가중치 최적화 중... (약 2-3분 소요)';
    else if (runBacktest) loadingMessage = '백테스팅 실행 중...';
    
    showLoading(loadingMessage);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_predictions: numPredictions,
                run_backtest: runBacktest,
                run_optimization: runOptimization
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentData = result.data;
            displayFullResults(result.data);
            showToast('분석이 완료되었습니다!', 'success');
        } else {
            showToast('오류: ' + result.error, 'error');
        }
    } catch (error) {
        showToast('네트워크 오류가 발생했습니다.', 'error');
    } finally {
        hideLoading();
    }
}

// === Display Functions ===
function displayQuickResults(result) {
    elements.emptyState.classList.add('hidden');
    elements.resultsContainer.classList.add('active');
    
    // Update header info
    elements.totalRounds.textContent = result.total_rounds.toLocaleString() + '회';
    elements.lastDraw.textContent = result.last_draw.join(', ');
    
    // Display predictions
    displayPredictions(result.predictions);
    
    // Hide other sections
    document.getElementById('analysisSection').style.display = 'none';
    document.getElementById('chartsSection').style.display = 'none';
    document.getElementById('clusterSection').style.display = 'none';
    elements.backtestSection.style.display = 'none';
    elements.optimizationSection.style.display = 'none';
}

function displayFullResults(data) {
    elements.emptyState.classList.add('hidden');
    elements.resultsContainer.classList.add('active');
    
    // Update header info
    if (data.analysis) {
        elements.totalRounds.textContent = data.analysis.total_rounds.toLocaleString() + '회';
        elements.lastDraw.textContent = data.analysis.last_draw.join(', ');
    }
    
    // Display predictions
    if (data.predictions) {
        displayPredictions(data.predictions);
    }
    
    // Display analysis
    if (data.analysis) {
        document.getElementById('analysisSection').style.display = 'block';
        displayAnalysis(data.analysis);
        
        document.getElementById('chartsSection').style.display = 'block';
        displayCharts(data.analysis);
    }
    
    // Display clusters
    if (data.clusters) {
        document.getElementById('clusterSection').style.display = 'block';
        displayClusters(data.clusters);
    }
    
    // Display backtest results
    if (data.backtest) {
        elements.backtestSection.style.display = 'block';
        displayBacktest(data.backtest);
    } else {
        elements.backtestSection.style.display = 'none';
    }
    
    // Display optimization results
    if (data.optimization) {
        elements.optimizationSection.style.display = 'block';
        displayOptimization(data.optimization);
    } else {
        elements.optimizationSection.style.display = 'none';
    }
}

function displayPredictions(predictions) {
    elements.predictionsGrid.innerHTML = predictions.map((pred, index) => {
        const rankClass = index === 0 ? 'rank-1' : (index < 3 ? 'rank-2' : '');
        const numbers = pred.numbers.map(n => {
            const ballClass = getBallClass(n);
            return `<div class="lotto-ball ${ballClass}">${n}</div>`;
        }).join('');
        
        return `
            <div class="prediction-card ${rankClass}" data-numbers="${pred.numbers.join(',')}">
                <div class="prediction-rank">#${pred.rank}</div>
                <div class="prediction-numbers">${numbers}</div>
                <div class="prediction-stats">
                    <div class="stat-item">
                        <span class="stat-label">합계</span>
                        <span class="stat-value">${pred.sum}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">AC</span>
                        <span class="stat-value">${pred.ac}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">홀:고</span>
                        <span class="stat-value">${pred.odd}:${pred.high}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">점수</span>
                        <span class="stat-value">${pred.total_score.toFixed(3)}</span>
                    </div>
                </div>
                <div class="prediction-actions">
                    <button class="btn-copy" onclick="copyNumbers('${pred.numbers.join(', ')}')" title="복사">
                        📋
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

function displayAnalysis(analysis) {
    // Hot numbers
    elements.hotNumbers.innerHTML = analysis.hot_numbers.map(item => `
        <div class="number-badge">
            <span class="num">${item.number}</span>
            <span class="score">(${item.score.toFixed(3)})</span>
        </div>
    `).join('');
    
    // Cold numbers
    elements.coldNumbers.innerHTML = analysis.cold_numbers.map(item => `
        <div class="number-badge">
            <span class="num">${item.number}</span>
            <span class="score">(${item.score.toFixed(3)})</span>
        </div>
    `).join('');
    
    // Rising numbers
    elements.risingNumbers.innerHTML = analysis.rising_numbers.map(item => `
        <div class="number-badge">
            <span class="num">${item.number}</span>
            <span class="trend">↑</span>
        </div>
    `).join('');
    
    // Top pairs
    elements.topPairs.innerHTML = analysis.top_pairs.map(item => `
        <div class="pair-badge">
            <span class="nums">${item.pair[0]}-${item.pair[1]}</span>
            <span class="count">${item.count}회</span>
        </div>
    `).join('');
}

function displayCharts(analysis) {
    // Destroy existing charts
    Object.values(charts).forEach(chart => chart.destroy());
    charts = {};
    
    // Score Chart
    const scoreCtx = document.getElementById('scoreChart').getContext('2d');
    const scoreData = Object.entries(analysis.number_scores).map(([num, score]) => ({
        x: parseInt(num),
        y: score
    }));
    
    charts.score = new Chart(scoreCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 45}, (_, i) => i + 1),
            datasets: [{
                label: '점수',
                data: scoreData.map(d => d.y),
                backgroundColor: scoreData.map(d => getBarColor(d.y)),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { grid: { display: false } }
            }
        }
    });
    
    // Gap Chart
    const gapCtx = document.getElementById('gapChart').getContext('2d');
    const gapData = Object.entries(analysis.gap_data).map(([num, gap]) => ({
        x: parseInt(num),
        y: gap
    }));
    
    charts.gap = new Chart(gapCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 45}, (_, i) => i + 1),
            datasets: [{
                label: '미출현 간격',
                data: gapData.map(d => d.y),
                backgroundColor: gapData.map(d => d.y > 15 ? '#ef4444' : d.y < 5 ? '#3b82f6' : '#10b981'),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { grid: { display: false } }
            }
        }
    });
    
    // Momentum Chart
    const momentumCtx = document.getElementById('momentumChart').getContext('2d');
    const momentumData = Object.entries(analysis.momentum_data).map(([num, mom]) => ({
        x: parseInt(num),
        y: mom
    }));
    
    charts.momentum = new Chart(momentumCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 45}, (_, i) => i + 1),
            datasets: [{
                label: '모멘텀',
                data: momentumData.map(d => d.y),
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function displayClusters(clusters) {
    // Cluster cards
    elements.clusterGrid.innerHTML = clusters.map(cluster => `
        <div class="cluster-card">
            <div class="cluster-id">C${cluster.id}</div>
            <div class="cluster-ratio">${cluster.ratio}%</div>
            <div class="cluster-stats">
                <div>출현: ${cluster.count}회</div>
                <div>평균 합계: ${cluster.avg_sum}</div>
                <div>평균 범위: ${cluster.avg_spread}</div>
            </div>
        </div>
    `).join('');
    
    // Cluster pie chart
    const clusterCtx = document.getElementById('clusterChart').getContext('2d');
    if (charts.cluster) charts.cluster.destroy();
    
    charts.cluster = new Chart(clusterCtx, {
        type: 'doughnut',
        data: {
            labels: clusters.map(c => `군집 ${c.id}`),
            datasets: [{
                data: clusters.map(c => c.count),
                backgroundColor: [
                    '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                    '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#14b8a6'
                ].slice(0, clusters.length)
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

function displayBacktest(backtest) {
    const maxCount = Math.max(...Object.values(backtest.match_distribution));
    
    elements.backtestContent.innerHTML = `
        <div class="backtest-summary">
            <div class="summary-card">
                <div class="value">${backtest.test_rounds}</div>
                <div class="label">테스트 회차</div>
            </div>
            <div class="summary-card">
                <div class="value">${backtest.total_predictions}</div>
                <div class="label">총 예측 수</div>
            </div>
            <div class="summary-card">
                <div class="value">${backtest.best_match}</div>
                <div class="label">최대 적중</div>
            </div>
            <div class="summary-card">
                <div class="value">${backtest.avg_match}</div>
                <div class="label">평균 적중</div>
            </div>
            <div class="summary-card">
                <div class="value" style="color: #10b981;">+${backtest.improvement}%</div>
                <div class="label">랜덤 대비</div>
            </div>
        </div>
        
        <div class="match-distribution">
            <h4>📊 적중 분포</h4>
            <div class="distribution-bars">
                ${[6, 5, 4, 3, 2, 1, 0].map(matches => {
                    const count = backtest.match_distribution[matches] || 0;
                    const pct = maxCount > 0 ? (count / maxCount * 100) : 0;
                    const totalPct = backtest.total_predictions > 0 
                        ? (count / backtest.total_predictions * 100).toFixed(1) : 0;
                    return `
                        <div class="dist-row">
                            <div class="dist-label">${matches}개 일치</div>
                            <div class="dist-bar-container">
                                <div class="dist-bar" style="width: ${pct}%"></div>
                            </div>
                            <div class="dist-value">${count}회 (${totalPct}%)</div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
        
        <div class="prize-achievements">
            <h4>🏆 등수 달성</h4>
            <div class="prize-grid">
                <div class="prize-item">
                    <div class="prize-name">3등 (5개)</div>
                    <div class="prize-count">${backtest.prize_count['3등'] || 0}회</div>
                </div>
                <div class="prize-item">
                    <div class="prize-name">4등 (4개)</div>
                    <div class="prize-count">${backtest.prize_count['4등'] || 0}회</div>
                </div>
                <div class="prize-item">
                    <div class="prize-name">5등 (3개)</div>
                    <div class="prize-count">${backtest.prize_count['5등'] || 0}회</div>
                </div>
            </div>
        </div>
    `;
}

function displayOptimization(optimization) {
    elements.optimizationContent.innerHTML = `
        <div class="summary-card" style="margin-bottom: 1.5rem;">
            <div class="value">${optimization.best_score}</div>
            <div class="label">최적화 점수</div>
        </div>
        
        <h4 style="margin-bottom: 1rem;">🎯 최적 가중치</h4>
        <div class="weights-grid">
            ${Object.entries(optimization.best_weights).map(([name, value]) => `
                <div class="weight-card">
                    <div class="name">${name}</div>
                    <div class="value">${value}</div>
                </div>
            `).join('')}
        </div>
        
        <div style="margin-top: 1.5rem; color: var(--text-secondary); font-size: 0.875rem;">
            총 ${optimization.total_tests}개의 설정 테스트 완료
        </div>
    `;
}

// === Utility Functions ===
function getBallClass(num) {
    if (num <= 10) return 'ball-1-10';
    if (num <= 20) return 'ball-11-20';
    if (num <= 30) return 'ball-21-30';
    if (num <= 40) return 'ball-31-40';
    return 'ball-41-45';
}

function getBarColor(score) {
    if (score > 0.03) return '#ef4444';
    if (score > 0.025) return '#f59e0b';
    if (score > 0.02) return '#10b981';
    return '#6366f1';
}

function showLoading(message = '로딩 중...') {
    elements.loadingText.textContent = message;
    elements.loadingOverlay.classList.add('active');
    elements.btnQuickPredict.disabled = true;
    elements.btnFullAnalysis.disabled = true;
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
    elements.btnQuickPredict.disabled = false;
    elements.btnFullAnalysis.disabled = false;
}

function showToast(message, type = 'success') {
    const toast = elements.toast;
    toast.className = `toast ${type} show`;
    toast.querySelector('.toast-icon').textContent = type === 'success' ? '✅' : '❌';
    toast.querySelector('.toast-message').textContent = message;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function copyNumbers(numbers) {
    navigator.clipboard.writeText(numbers).then(() => {
        showToast('번호가 복사되었습니다!', 'success');
    }).catch(() => {
        showToast('복사에 실패했습니다.', 'error');
    });
}

function handleCopyAll() {
    const cards = document.querySelectorAll('.prediction-card');
    const allNumbers = Array.from(cards).map((card, index) => {
        return `#${index + 1}: ${card.dataset.numbers}`;
    }).join('\n');
    
    navigator.clipboard.writeText(allNumbers).then(() => {
        showToast('전체 번호가 복사되었습니다!', 'success');
    }).catch(() => {
        showToast('복사에 실패했습니다.', 'error');
    });
}
