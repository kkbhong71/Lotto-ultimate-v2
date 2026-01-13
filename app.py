# =============================================================================
# 🎰 Lotto Ultimate v2.0 - Flask Web Application
# =============================================================================

from flask import Flask, render_template, jsonify, request
import os
import json
from lotto_engine import (
    Config, DataEngine, GeometricAnalyzer, EnsembleGenerator,
    BacktestEngine, HyperparameterOptimizer, run_full_analysis
)

app = Flask(__name__)

# CSV 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'new_1206.csv')

# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """분석 및 예측 API"""
    try:
        data = request.get_json() or {}
        num_predictions = data.get('num_predictions', 5)
        run_backtest = data.get('run_backtest', False)
        run_optimization = data.get('run_optimization', False)
        
        result = run_full_analysis(
            csv_path=CSV_PATH,
            num_predictions=num_predictions,
            run_backtest=run_backtest,
            run_optimization=run_optimization
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/quick-predict', methods=['POST'])
def quick_predict():
    """빠른 예측 API (분석 없이)"""
    try:
        data = request.get_json() or {}
        num_predictions = data.get('num_predictions', 5)
        
        config = Config()
        engine = DataEngine(CSV_PATH, config)
        
        if not engine.is_loaded:
            return jsonify({
                'success': False,
                'error': engine.error_msg
            }), 400
        
        geometric = GeometricAnalyzer(engine, config.N_CLUSTERS)
        generator = EnsembleGenerator(engine, geometric)
        predictions = generator.generate(count=num_predictions)
        
        result = {
            'success': True,
            'predictions': [
                {
                    'rank': i + 1,
                    'numbers': pred['nums'],
                    'sum': pred['stat']['sum'],
                    'ac': pred['stat']['ac'],
                    'odd': pred['stat']['odd'],
                    'high': pred['stat']['high'],
                    'total_score': pred['total_score'],
                    'algorithm': pred['algorithm']
                }
                for i, pred in enumerate(predictions)
            ],
            'last_draw': engine.last_draw,
            'total_rounds': engine.total_rounds
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """백테스팅 API"""
    try:
        data = request.get_json() or {}
        test_rounds = min(data.get('test_rounds', 10), 20)  # 최대 20회차
        predictions_per_round = min(data.get('predictions_per_round', 5), 10)
        
        config = Config()
        backtest_engine = BacktestEngine(CSV_PATH, config)
        result = backtest_engine.run_backtest(
            test_rounds=test_rounds,
            predictions_per_round=predictions_per_round
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """하이퍼파라미터 최적화 API"""
    try:
        data = request.get_json() or {}
        test_rounds = min(data.get('test_rounds', 5), 10)
        predictions_per_round = min(data.get('predictions_per_round', 5), 5)
        
        config = Config()
        optimizer = HyperparameterOptimizer(CSV_PATH, config)
        result = optimizer.optimize(
            test_rounds=test_rounds,
            predictions_per_round=predictions_per_round
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data-info', methods=['GET'])
def data_info():
    """데이터 정보 API"""
    try:
        config = Config()
        engine = DataEngine(CSV_PATH, config)
        
        if not engine.is_loaded:
            return jsonify({
                'success': False,
                'error': engine.error_msg
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'total_rounds': engine.total_rounds,
                'last_draw': engine.last_draw,
                'csv_path': CSV_PATH
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/number-analysis', methods=['GET'])
def number_analysis():
    """번호별 분석 데이터 API"""
    try:
        config = Config()
        engine = DataEngine(CSV_PATH, config)
        
        if not engine.is_loaded:
            return jsonify({
                'success': False,
                'error': engine.error_msg
            }), 400
        
        geometric = GeometricAnalyzer(engine, config.N_CLUSTERS)
        
        return jsonify({
            'success': True,
            'data': {
                'analysis': engine.get_analysis_summary(),
                'clusters': geometric.get_cluster_summary()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
