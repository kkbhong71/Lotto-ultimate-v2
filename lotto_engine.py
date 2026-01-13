# =============================================================================
# 🎰 Lotto Analytics Engine - ULTIMATE EDITION v2.0 (Web Version)
# =============================================================================
# 웹앱용으로 최적화된 분석 엔진
# - 백테스팅, 자동 최적화, 기하학적 군집화 통합
# =============================================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from copy import deepcopy
import os

# 소수 리스트
PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    """분석 및 필터링 설정"""
    
    SUM_RANGE: Tuple[int, int] = (100, 195)
    SUM_OPTIMAL: Tuple[int, int] = (115, 185)
    MIN_AC: int = 7
    OPTIMAL_AC: Tuple[int, int] = (8, 10)
    ODD_RATES: List[int] = field(default_factory=lambda: [2, 3, 4])
    HIGH_RATES: List[int] = field(default_factory=lambda: [2, 3, 4])
    MIN_ZONES: int = 3
    MAX_PER_ZONE: int = 3
    MIN_UNIQUE_ENDINGS: int = 4
    MAX_SAME_ENDING: int = 2
    PRIME_RANGE: Tuple[int, int] = (1, 4)
    MAX_CONSECUTIVE: int = 2
    PREV_MATCH_RANGE: Tuple[int, int] = (0, 3)
    
    # 가중치
    WEIGHT_FREQUENCY: float = 0.20
    WEIGHT_RECENCY: float = 0.20
    WEIGHT_GAP: float = 0.15
    WEIGHT_PAIR: float = 0.15
    WEIGHT_MOMENTUM: float = 0.15
    WEIGHT_ZONE: float = 0.15
    
    RECENT_PERIODS: List[int] = field(default_factory=lambda: [10, 30, 50, 100])
    N_CLUSTERS: int = 5
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.7
    
    def get_weights_dict(self) -> Dict[str, float]:
        return {
            'frequency': self.WEIGHT_FREQUENCY,
            'recency': self.WEIGHT_RECENCY,
            'gap': self.WEIGHT_GAP,
            'pair': self.WEIGHT_PAIR,
            'momentum': self.WEIGHT_MOMENTUM,
            'zone': self.WEIGHT_ZONE
        }
    
    def set_weights_from_dict(self, weights: Dict[str, float]):
        self.WEIGHT_FREQUENCY = weights.get('frequency', self.WEIGHT_FREQUENCY)
        self.WEIGHT_RECENCY = weights.get('recency', self.WEIGHT_RECENCY)
        self.WEIGHT_GAP = weights.get('gap', self.WEIGHT_GAP)
        self.WEIGHT_PAIR = weights.get('pair', self.WEIGHT_PAIR)
        self.WEIGHT_MOMENTUM = weights.get('momentum', self.WEIGHT_MOMENTUM)
        self.WEIGHT_ZONE = weights.get('zone', self.WEIGHT_ZONE)

# =============================================================================
# Data Engine
# =============================================================================
class DataEngine:
    """데이터 로드 및 다차원 통계 분석 엔진"""
    
    def __init__(self, csv_path: str, config: Config = None, exclude_last_n: int = 0):
        self.csv_path = csv_path
        self.config = config or Config()
        self.exclude_last_n = exclude_last_n
        self.is_loaded = False
        
        try:
            self._load_data()
            self._analyze_all()
            self.is_loaded = True
        except Exception as e:
            self.error_msg = str(e)
    
    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        all_history = df.iloc[:, 2:8].values.tolist()
        all_history = [[int(n) for n in row] for row in all_history]
        
        if self.exclude_last_n > 0:
            self.history = all_history[:-self.exclude_last_n]
            self.hidden_draws = all_history[-self.exclude_last_n:]
        else:
            self.history = all_history
            self.hidden_draws = []
        
        self.total_rounds = len(self.history)
        self.last_draw = self.history[-1] if self.history else []
        self.historical_sets = [frozenset(row) for row in self.history]
    
    def _analyze_all(self):
        self.frequency_analysis = self._analyze_frequency()
        self.gap_analysis = self._analyze_gaps()
        self.pair_analysis = self._analyze_pairs()
        self.momentum_analysis = self._analyze_momentum()
        self.zone_analysis = self._analyze_zones()
        self.pattern_stats = self._analyze_patterns()
        self.final_scores = self._calculate_final_scores()
    
    def _analyze_frequency(self) -> Dict:
        result = {
            'total': Counter(),
            'by_period': {},
            'weighted': np.zeros(46)
        }
        
        all_nums = [n for row in self.history for n in row]
        result['total'] = Counter(all_nums)
        
        period_weights = {10: 4.0, 30: 2.5, 50: 1.5, 100: 1.0}
        
        for period in self.config.RECENT_PERIODS:
            if period <= self.total_rounds:
                recent = self.history[-period:]
                nums = [n for row in recent for n in row]
                result['by_period'][period] = Counter(nums)
        
        for num in range(1, 46):
            score = 0
            for period, weight in period_weights.items():
                if period in result['by_period']:
                    expected = (period * 6) / 45
                    actual = result['by_period'][period].get(num, 0)
                    score += (actual / expected) * weight if expected > 0 else 0
            result['weighted'][num] = score
        
        w = result['weighted'][1:]
        if w.max() > 0:
            result['weighted'][1:] = w / w.max()
        
        return result
    
    def _analyze_gaps(self) -> Dict:
        result = {
            'last_seen': {},
            'gap': {},
            'avg_gap': {},
            'deviation': {},
            'score': np.zeros(46)
        }
        
        appearances = defaultdict(list)
        for idx, row in enumerate(self.history):
            for n in row:
                appearances[n].append(idx)
        
        for num in range(1, 46):
            if num in appearances and len(appearances[num]) > 0:
                result['last_seen'][num] = appearances[num][-1]
                result['gap'][num] = self.total_rounds - 1 - appearances[num][-1]
                
                if len(appearances[num]) > 1:
                    gaps = [appearances[num][i+1] - appearances[num][i] 
                            for i in range(len(appearances[num])-1)]
                    result['avg_gap'][num] = np.mean(gaps)
                else:
                    result['avg_gap'][num] = self.total_rounds / 2
                
                result['deviation'][num] = result['gap'][num] - result['avg_gap'][num]
            else:
                result['last_seen'][num] = 0
                result['gap'][num] = self.total_rounds
                result['avg_gap'][num] = self.total_rounds / 2
                result['deviation'][num] = self.total_rounds / 2
        
        for num in range(1, 46):
            avg = result['avg_gap'][num]
            if avg > 0:
                ratio = result['gap'][num] / avg
                if 0.8 <= ratio <= 1.5:
                    result['score'][num] = 1.0 - abs(ratio - 1.0) * 0.5
                elif ratio < 0.8:
                    result['score'][num] = ratio * 0.5
                else:
                    result['score'][num] = max(0, 1.0 - (ratio - 1.5) * 0.3)
        
        s = result['score'][1:]
        if s.max() > 0:
            result['score'][1:] = s / s.max()
        
        return result
    
    def _analyze_pairs(self) -> Dict:
        result = {
            'pair_count': Counter(),
            'pair_score': defaultdict(lambda: defaultdict(float)),
            'top_pairs': [],
            'number_pair_strength': np.zeros(46)
        }
        
        for row in self.history:
            for pair in combinations(sorted(row), 2):
                result['pair_count'][pair] += 1
        
        result['top_pairs'] = result['pair_count'].most_common(50)
        
        for (a, b), count in result['pair_count'].items():
            strength = count / self.total_rounds
            result['pair_score'][a][b] = strength
            result['pair_score'][b][a] = strength
            result['number_pair_strength'][a] += strength
            result['number_pair_strength'][b] += strength
        
        s = result['number_pair_strength'][1:]
        if s.max() > 0:
            result['number_pair_strength'][1:] = s / s.max()
        
        return result
    
    def _analyze_momentum(self) -> Dict:
        result = {
            'short_term': np.zeros(46),
            'mid_term': np.zeros(46),
            'momentum': np.zeros(46),
            'trend': {}
        }
        
        if self.total_rounds >= 30:
            recent_10 = Counter([n for row in self.history[-10:] for n in row])
            recent_30 = Counter([n for row in self.history[-30:] for n in row])
            
            for num in range(1, 46):
                short = recent_10.get(num, 0) / 10
                mid = recent_30.get(num, 0) / 30
                
                result['short_term'][num] = short
                result['mid_term'][num] = mid
                
                if mid > 0:
                    momentum = short / mid
                    result['momentum'][num] = momentum
                    
                    if momentum > 1.3:
                        result['trend'][num] = 'UP'
                    elif momentum < 0.7:
                        result['trend'][num] = 'DOWN'
                    else:
                        result['trend'][num] = 'STABLE'
                else:
                    result['momentum'][num] = 1.0 if short > 0 else 0.5
                    result['trend'][num] = 'UP' if short > 0 else 'DORMANT'
        
        m = result['momentum'][1:]
        if m.max() > 0 and m.max() != m.min():
            m_normalized = (m - m.min()) / (m.max() - m.min())
            result['momentum'][1:] = m_normalized
        
        return result
    
    def _analyze_zones(self) -> Dict:
        zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        result = {
            'zone_freq': {z: Counter() for z in zones},
            'zone_score': np.zeros(46)
        }
        
        recent = self.history[-50:] if self.total_rounds >= 50 else self.history
        
        for row in recent:
            for n in row:
                for z_start, z_end in zones:
                    if z_start <= n <= z_end:
                        result['zone_freq'][(z_start, z_end)][n] += 1
                        break
        
        for z_start, z_end in zones:
            zone_nums = list(range(z_start, z_end + 1))
            counts = [result['zone_freq'][(z_start, z_end)].get(n, 0) for n in zone_nums]
            
            if max(counts) > 0:
                for i, n in enumerate(zone_nums):
                    result['zone_score'][n] = counts[i] / max(counts)
        
        return result
    
    def _analyze_patterns(self) -> Dict:
        result = {
            'sum_dist': [],
            'ac_dist': [],
            'odd_dist': Counter(),
            'high_dist': Counter(),
            'consecutive_dist': Counter(),
            'ending_dist': Counter()
        }
        
        for row in self.history:
            nums = sorted(row)
            result['sum_dist'].append(sum(nums))
            
            diffs = set()
            for i in range(6):
                for j in range(i+1, 6):
                    diffs.add(nums[j] - nums[i])
            result['ac_dist'].append(len(diffs) - 5)
            
            odd_count = sum(1 for n in nums if n % 2 == 1)
            result['odd_dist'][odd_count] += 1
            
            high_count = sum(1 for n in nums if n >= 23)
            result['high_dist'][high_count] += 1
            
            consec = 0
            for i in range(5):
                if nums[i+1] == nums[i] + 1:
                    consec += 1
            result['consecutive_dist'][consec] += 1
            
            endings = tuple(sorted(set(n % 10 for n in nums)))
            result['ending_dist'][len(endings)] += 1
        
        return result
    
    def _calculate_final_scores(self) -> np.ndarray:
        scores = np.zeros(46)
        cfg = self.config
        
        for num in range(1, 46):
            score = 0
            score += self.frequency_analysis['weighted'][num] * cfg.WEIGHT_FREQUENCY
            
            recent_10 = self.frequency_analysis['by_period'].get(10, Counter())
            recency = min(recent_10.get(num, 0) / 3, 1.0)
            score += recency * cfg.WEIGHT_RECENCY
            
            score += self.gap_analysis['score'][num] * cfg.WEIGHT_GAP
            score += self.pair_analysis['number_pair_strength'][num] * cfg.WEIGHT_PAIR
            score += self.momentum_analysis['momentum'][num] * cfg.WEIGHT_MOMENTUM
            score += self.zone_analysis['zone_score'][num] * cfg.WEIGHT_ZONE
            
            scores[num] = score
        
        s = scores[1:]
        if s.sum() > 0:
            scores[1:] = s / s.sum()
        
        return scores
    
    def get_hot_numbers(self, n: int = 10) -> List[Dict]:
        scored = [(num, float(self.final_scores[num])) for num in range(1, 46)]
        sorted_nums = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
        return [{'number': num, 'score': score} for num, score in sorted_nums]
    
    def get_cold_numbers(self, n: int = 10) -> List[Dict]:
        scored = [(num, float(self.final_scores[num])) for num in range(1, 46)]
        sorted_nums = sorted(scored, key=lambda x: x[1])[:n]
        return [{'number': num, 'score': score} for num, score in sorted_nums]
    
    def get_rising_numbers(self, n: int = 10) -> List[Dict]:
        rising = [(num, trend) for num, trend in self.momentum_analysis['trend'].items() 
                  if trend == 'UP']
        sorted_nums = sorted(rising, key=lambda x: self.momentum_analysis['momentum'][x[0]], 
                            reverse=True)[:n]
        return [{'number': num, 'trend': trend} for num, trend in sorted_nums]
    
    def get_analysis_summary(self) -> Dict:
        """웹 표시용 분석 요약"""
        return {
            'total_rounds': self.total_rounds,
            'last_draw': self.last_draw,
            'hot_numbers': self.get_hot_numbers(10),
            'cold_numbers': self.get_cold_numbers(10),
            'rising_numbers': self.get_rising_numbers(10),
            'top_pairs': [{'pair': list(p), 'count': c} for p, c in self.pair_analysis['top_pairs'][:10]],
            'number_scores': {i: float(self.final_scores[i]) for i in range(1, 46)},
            'gap_data': {i: int(self.gap_analysis['gap'].get(i, 0)) for i in range(1, 46)},
            'momentum_data': {i: float(self.momentum_analysis['momentum'][i]) for i in range(1, 46)},
            'sum_stats': {
                'mean': float(np.mean(self.pattern_stats['sum_dist'])),
                'min': int(min(self.pattern_stats['sum_dist'])),
                'max': int(max(self.pattern_stats['sum_dist']))
            }
        }

# =============================================================================
# Geometric Analyzer
# =============================================================================
class GeometricAnalyzer:
    """기하학적 군집화 분석기"""
    
    def __init__(self, engine: DataEngine, n_clusters: int = 5):
        self.engine = engine
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_stats = {}
        
        self._build_clusters()
    
    def _extract_features(self, nums: List[int]) -> np.ndarray:
        nums = sorted(nums)
        features = []
        
        features.append(sum(nums))
        features.append(np.mean(nums))
        features.append(np.std(nums))
        features.append(max(nums) - min(nums))
        
        gaps = [nums[i+1] - nums[i] for i in range(5)]
        features.append(np.mean(gaps))
        features.append(np.std(gaps))
        features.append(max(gaps))
        features.append(min(gaps))
        
        features.append(nums[0])
        features.append(nums[5])
        features.append(nums[2] + nums[3])
        
        zones = [0] * 5
        for n in nums:
            if n <= 10: zones[0] += 1
            elif n <= 20: zones[1] += 1
            elif n <= 30: zones[2] += 1
            elif n <= 40: zones[3] += 1
            else: zones[4] += 1
        features.extend(zones)
        
        features.append(sum(1 for n in nums if n % 2 == 1))
        features.append(sum(1 for n in nums if n >= 23))
        
        diffs = set()
        for i in range(6):
            for j in range(i+1, 6):
                diffs.add(nums[j] - nums[i])
        features.append(len(diffs) - 5)
        
        endings = set(n % 10 for n in nums)
        features.append(len(endings))
        features.append(sum(1 for n in nums if n in PRIME_NUMBERS))
        
        consec = sum(1 for i in range(5) if nums[i+1] == nums[i] + 1)
        features.append(consec)
        
        features.append(sum(i * nums[i] for i in range(6)) / sum(nums))
        
        return np.array(features)
    
    def _build_clusters(self):
        all_features = []
        for row in self.engine.history:
            features = self._extract_features(row)
            all_features.append(features)
        
        self.feature_matrix = np.array(all_features)
        self.scaled_features = self.scaler.fit_transform(self.feature_matrix)
        
        best_score = -1
        best_k = self.n_clusters
        
        for k in range(3, min(10, len(self.engine.history) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_features)
            score = silhouette_score(self.scaled_features, labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        self.n_clusters = best_k
        self.kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        
        self._calculate_cluster_stats()
    
    def _calculate_cluster_stats(self):
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_draws = [self.engine.history[i] for i in range(len(self.engine.history)) if mask[i]]
            
            self.cluster_stats[cluster_id] = {
                'count': len(cluster_draws),
                'ratio': len(cluster_draws) / len(self.engine.history),
                'avg_sum': float(np.mean([sum(row) for row in cluster_draws])),
                'avg_spread': float(np.mean([max(row) - min(row) for row in cluster_draws])),
                'recent_count': sum(1 for i in range(-min(50, len(self.engine.history)), 0) 
                                   if self.cluster_labels[i] == cluster_id)
            }
    
    def predict_cluster(self, nums: List[int]) -> int:
        features = self._extract_features(nums)
        scaled = self.scaler.transform([features])
        return int(self.kmeans.predict(scaled)[0])
    
    def get_cluster_similarity_score(self, nums: List[int]) -> float:
        cluster_id = self.predict_cluster(nums)
        cluster_ratio = self.cluster_stats[cluster_id]['ratio']
        recent_ratio = self.cluster_stats[cluster_id]['recent_count'] / min(50, len(self.engine.history))
        return cluster_ratio * 0.6 + recent_ratio * 0.4
    
    def get_cluster_summary(self) -> List[Dict]:
        return [
            {
                'id': cid,
                'count': stats['count'],
                'ratio': round(stats['ratio'] * 100, 1),
                'avg_sum': round(stats['avg_sum'], 0),
                'avg_spread': round(stats['avg_spread'], 0)
            }
            for cid, stats in sorted(self.cluster_stats.items(), 
                                     key=lambda x: x[1]['ratio'], reverse=True)
        ]

# =============================================================================
# Filter System
# =============================================================================
class FilterSystem:
    """13단계 필터링 시스템"""
    
    def __init__(self, engine: DataEngine, geometric_analyzer: GeometricAnalyzer = None):
        self.engine = engine
        self.config = engine.config
        self.geometric_analyzer = geometric_analyzer
        self.filter_stats = defaultdict(int)
    
    def apply_all_filters(self, nums: List[int], use_cluster_filter: bool = True) -> Tuple[bool, str]:
        nums = sorted(nums)
        
        filters = [
            (self._filter_sum, "합계"),
            (self._filter_ac, "AC값"),
            (self._filter_odd_even, "홀짝"),
            (self._filter_high_low, "고저"),
            (self._filter_zones, "구간분포"),
            (self._filter_consecutive, "연속번호"),
            (self._filter_endings, "끝수"),
            (self._filter_primes, "소수"),
            (self._filter_previous, "이전회차"),
            (self._filter_historical, "역대중복"),
            (self._filter_spread, "번호간격"),
            (self._filter_edge, "경계번호"),
        ]
        
        if use_cluster_filter and self.geometric_analyzer:
            filters.append((self._filter_cluster, "군집패턴"))
        
        for filter_func, name in filters:
            if not filter_func(nums):
                self.filter_stats[name] += 1
                return False, name
        
        return True, ""
    
    def _filter_sum(self, nums): 
        s = sum(nums)
        return self.config.SUM_RANGE[0] <= s <= self.config.SUM_RANGE[1]
    
    def _filter_ac(self, nums):
        diffs = set()
        for i in range(6):
            for j in range(i+1, 6):
                diffs.add(nums[j] - nums[i])
        return (len(diffs) - 5) >= self.config.MIN_AC
    
    def _filter_odd_even(self, nums):
        return sum(1 for n in nums if n % 2 == 1) in self.config.ODD_RATES
    
    def _filter_high_low(self, nums):
        return sum(1 for n in nums if n >= 23) in self.config.HIGH_RATES
    
    def _filter_zones(self, nums):
        zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        zone_counts = [0] * 5
        for n in nums:
            for i, (start, end) in enumerate(zones):
                if start <= n <= end:
                    zone_counts[i] += 1
                    break
        return sum(1 for c in zone_counts if c > 0) >= self.config.MIN_ZONES and max(zone_counts) <= self.config.MAX_PER_ZONE
    
    def _filter_consecutive(self, nums):
        max_consec = 1
        current_consec = 1
        for i in range(5):
            if nums[i+1] == nums[i] + 1:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 1
        return max_consec <= self.config.MAX_CONSECUTIVE
    
    def _filter_endings(self, nums):
        endings = [n % 10 for n in nums]
        ending_counts = Counter(endings)
        return len(set(endings)) >= self.config.MIN_UNIQUE_ENDINGS and max(ending_counts.values()) <= self.config.MAX_SAME_ENDING
    
    def _filter_primes(self, nums):
        prime_count = sum(1 for n in nums if n in PRIME_NUMBERS)
        return self.config.PRIME_RANGE[0] <= prime_count <= self.config.PRIME_RANGE[1]
    
    def _filter_previous(self, nums):
        if not self.engine.last_draw:
            return True
        match_count = len(set(nums) & set(self.engine.last_draw))
        return self.config.PREV_MATCH_RANGE[0] <= match_count <= self.config.PREV_MATCH_RANGE[1]
    
    def _filter_historical(self, nums):
        return frozenset(nums) not in self.engine.historical_sets
    
    def _filter_spread(self, nums):
        gaps = [nums[i+1] - nums[i] for i in range(5)]
        return 4 <= np.mean(gaps) <= 12
    
    def _filter_edge(self, nums):
        if 1 in nums and 45 in nums:
            return sum(1 for n in nums if 20 <= n <= 25) >= 1
        return True
    
    def _filter_cluster(self, nums):
        score = self.geometric_analyzer.get_cluster_similarity_score(nums)
        return score >= self.config.CLUSTER_SIMILARITY_THRESHOLD * 0.3

# =============================================================================
# Ensemble Generator
# =============================================================================
class EnsembleGenerator:
    """4종 앙상블 생성기"""
    
    def __init__(self, engine: DataEngine, geometric_analyzer: GeometricAnalyzer = None):
        self.engine = engine
        self.geometric_analyzer = geometric_analyzer
        self.filter_system = FilterSystem(engine, geometric_analyzer)
    
    def generate(self, count: int = 5) -> List[Dict]:
        all_candidates = []
        
        all_candidates.extend(self._algorithm_weighted(count * 3))
        all_candidates.extend(self._algorithm_balanced(count * 3))
        all_candidates.extend(self._algorithm_pattern(count * 3))
        
        if self.geometric_analyzer:
            all_candidates.extend(self._algorithm_cluster(count * 3))
        
        unique_candidates = self._deduplicate_and_score(all_candidates)
        return sorted(unique_candidates, key=lambda x: x['total_score'], reverse=True)[:count]
    
    def _algorithm_weighted(self, count: int) -> List[Dict]:
        results = []
        attempts = 0
        weights = self.engine.final_scores[1:].copy()
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(45) / 45
        
        while len(results) < count and attempts < 50000:
            attempts += 1
            try:
                nums = sorted([int(n) for n in np.random.choice(range(1, 46), size=6, replace=False, p=weights)])
                if self.filter_system.apply_all_filters(nums)[0]:
                    results.append({'nums': nums, 'stat': self._calculate_stats(nums), 'algorithm': 'weighted'})
            except:
                continue
        return results
    
    def _algorithm_balanced(self, count: int) -> List[Dict]:
        results = []
        attempts = 0
        zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        
        while len(results) < count and attempts < 50000:
            attempts += 1
            nums = []
            zone_picks = [1, 1, 1, 1, 1]
            zone_picks[np.random.randint(0, 5)] += 1
            
            for i, (start, end) in enumerate(zones):
                zone_nums = list(range(start, end + 1))
                zone_weights = [self.engine.final_scores[n] for n in zone_nums]
                zone_weights = np.array(zone_weights)
                zone_weights = zone_weights / zone_weights.sum() if zone_weights.sum() > 0 else np.ones(len(zone_nums)) / len(zone_nums)
                
                try:
                    picked = np.random.choice(zone_nums, size=zone_picks[i], replace=False, p=zone_weights)
                    nums.extend([int(n) for n in picked])
                except:
                    picked = np.random.choice(zone_nums, size=zone_picks[i], replace=False)
                    nums.extend([int(n) for n in picked])
            
            nums = sorted(nums)
            if len(nums) == 6 and self.filter_system.apply_all_filters(nums)[0]:
                results.append({'nums': nums, 'stat': self._calculate_stats(nums), 'algorithm': 'balanced'})
        return results
    
    def _algorithm_pattern(self, count: int) -> List[Dict]:
        results = []
        attempts = 0
        top_pairs = self.engine.pair_analysis['top_pairs'][:20]
        
        while len(results) < count and attempts < 50000:
            attempts += 1
            
            if top_pairs and np.random.random() < 0.7:
                base_pair = list(top_pairs[np.random.randint(0, min(10, len(top_pairs)))][0])
            else:
                base_pair = []
            
            nums = set(base_pair)
            weights = self.engine.final_scores[1:].copy()
            for n in nums:
                weights[n-1] = 0
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
                try:
                    additional = np.random.choice(range(1, 46), size=6-len(nums), replace=False, p=weights)
                    nums.update([int(n) for n in additional])
                except:
                    continue
            
            if len(nums) == 6:
                nums = sorted(list(nums))
                if self.filter_system.apply_all_filters(nums)[0]:
                    results.append({'nums': nums, 'stat': self._calculate_stats(nums), 'algorithm': 'pattern'})
        return results
    
    def _algorithm_cluster(self, count: int) -> List[Dict]:
        results = []
        attempts = 0
        
        best_cluster = max(self.geometric_analyzer.cluster_stats.items(), key=lambda x: x[1]['ratio'])[0]
        cluster_stats = self.geometric_analyzer.cluster_stats[best_cluster]
        target_sum = int(cluster_stats['avg_sum'])
        target_spread = int(cluster_stats['avg_spread'])
        
        while len(results) < count and attempts < 50000:
            attempts += 1
            
            first_num = np.random.randint(1, 15)
            last_num = min(45, first_num + target_spread + np.random.randint(-5, 6))
            nums = [first_num, last_num]
            
            weights = self.engine.final_scores[1:].copy()
            weights[first_num-1] = 0
            weights[last_num-1] = 0
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
                try:
                    additional = np.random.choice(range(1, 46), size=4, replace=False, p=weights)
                    nums.extend([int(n) for n in additional])
                except:
                    continue
            
            nums = sorted(list(set(nums)))
            if len(nums) == 6 and abs(sum(nums) - target_sum) <= 30:
                if self.filter_system.apply_all_filters(nums)[0]:
                    results.append({'nums': nums, 'stat': self._calculate_stats(nums), 'algorithm': 'cluster'})
        return results
    
    def _calculate_stats(self, nums: List[int]) -> Dict:
        diffs = set()
        for i in range(6):
            for j in range(i+1, 6):
                diffs.add(nums[j] - nums[i])
        
        zones = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]
        zone_dist = [sum(1 for n in nums if start <= n <= end) for start, end in zones]
        
        pair_score = sum(self.engine.pair_analysis['pair_score'][a][b] for a, b in combinations(nums, 2))
        cluster_score = self.geometric_analyzer.get_cluster_similarity_score(nums) if self.geometric_analyzer else 0.5
        
        return {
            'sum': sum(nums),
            'ac': len(diffs) - 5,
            'odd': sum(1 for n in nums if n % 2 == 1),
            'high': sum(1 for n in nums if n >= 23),
            'prime': sum(1 for n in nums if n in PRIME_NUMBERS),
            'zone_dist': zone_dist,
            'num_score': sum(self.engine.final_scores[n] for n in nums),
            'pair_score': pair_score,
            'cluster_score': cluster_score
        }
    
    def _deduplicate_and_score(self, candidates: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        
        for cand in candidates:
            key = tuple(cand['nums'])
            if key not in seen:
                seen.add(key)
                stat = cand['stat']
                
                sum_score = 1.0 if 115 <= stat['sum'] <= 185 else 0.7
                ac_score = min(stat['ac'] / 10, 1.0)
                num_score = stat['num_score'] * 10
                pair_score = stat['pair_score'] * 5
                balance_score = 1.0 / (1 + np.var(stat['zone_dist']))
                cluster_score = stat.get('cluster_score', 0.5) * 2
                
                cand['total_score'] = round(
                    sum_score * 0.12 + ac_score * 0.12 + num_score * 0.30 + 
                    pair_score * 0.18 + balance_score * 0.13 + cluster_score * 0.15, 4
                )
                unique.append(cand)
        
        return unique

# =============================================================================
# Backtest Engine
# =============================================================================
class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, csv_path: str, config: Config = None):
        self.csv_path = csv_path
        self.config = config or Config()
    
    def run_backtest(self, test_rounds: int = 10, predictions_per_round: int = 5) -> Dict:
        df = pd.read_csv(self.csv_path)
        all_history = [[int(n) for n in row] for row in df.iloc[:, 2:8].values.tolist()]
        total_rounds = len(all_history)
        
        if test_rounds > total_rounds - 100:
            test_rounds = max(5, total_rounds - 100)
        
        results = {
            'test_rounds': test_rounds,
            'predictions_per_round': predictions_per_round,
            'round_results': [],
            'match_distribution': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
            'prize_count': {'3등': 0, '4등': 0, '5등': 0},
            'total_predictions': 0,
            'best_match': 0,
            'avg_match': 0
        }
        
        all_matches = []
        
        for i in range(test_rounds):
            round_idx = total_rounds - test_rounds + i
            actual_draw = set(all_history[round_idx])
            exclude_count = test_rounds - i
            
            engine = DataEngine(self.csv_path, self.config, exclude_last_n=exclude_count)
            if not engine.is_loaded:
                continue
            
            geometric = GeometricAnalyzer(engine, self.config.N_CLUSTERS)
            generator = EnsembleGenerator(engine, geometric)
            predictions = generator.generate(count=predictions_per_round)
            
            round_result = {
                'round': round_idx + 1,
                'actual': sorted(list(actual_draw)),
                'predictions': [],
                'best_match': 0
            }
            
            for pred in predictions:
                match_count = len(set(pred['nums']) & actual_draw)
                
                round_result['predictions'].append({
                    'nums': pred['nums'],
                    'match_count': match_count,
                    'score': pred['total_score']
                })
                
                all_matches.append(match_count)
                results['match_distribution'][match_count] = results['match_distribution'].get(match_count, 0) + 1
                results['total_predictions'] += 1
                
                if match_count >= 5:
                    results['prize_count']['3등'] += 1
                elif match_count >= 4:
                    results['prize_count']['4등'] += 1
                elif match_count >= 3:
                    results['prize_count']['5등'] += 1
                
                round_result['best_match'] = max(round_result['best_match'], match_count)
                results['best_match'] = max(results['best_match'], match_count)
            
            results['round_results'].append(round_result)
        
        if all_matches:
            results['avg_match'] = round(np.mean(all_matches), 2)
        
        # 랜덤 대비 개선율
        random_expected = 6 * 6 / 45
        results['improvement'] = round((results['avg_match'] / random_expected - 1) * 100, 1)
        
        return results

# =============================================================================
# Hyperparameter Optimizer
# =============================================================================
class HyperparameterOptimizer:
    """하이퍼파라미터 자동 최적화기"""
    
    def __init__(self, csv_path: str, base_config: Config = None):
        self.csv_path = csv_path
        self.base_config = base_config or Config()
        self.best_config = None
        self.optimization_history = []
    
    def optimize(self, test_rounds: int = 5, predictions_per_round: int = 5) -> Dict:
        weight_names = ['frequency', 'recency', 'gap', 'pair', 'momentum', 'zone']
        base_weights = self.base_config.get_weights_dict()
        
        best_score = -1
        best_weights = base_weights.copy()
        progress = []
        
        for target_weight in weight_names:
            best_local_score = -1
            best_local_value = base_weights[target_weight]
            
            for value in [0.10, 0.15, 0.20, 0.25, 0.30]:
                test_weights = base_weights.copy()
                test_weights[target_weight] = value
                
                other_sum = sum(v for k, v in test_weights.items() if k != target_weight)
                if other_sum > 0:
                    scale = (1 - value) / other_sum
                    for k in test_weights:
                        if k != target_weight:
                            test_weights[k] *= scale
                
                test_config = deepcopy(self.base_config)
                test_config.set_weights_from_dict(test_weights)
                
                backtest = BacktestEngine(self.csv_path, test_config)
                results = backtest.run_backtest(test_rounds=test_rounds, predictions_per_round=predictions_per_round)
                
                score = results['avg_match']
                score += results['match_distribution'].get(4, 0) * 0.5
                score += results['match_distribution'].get(5, 0) * 2.0
                score += results['match_distribution'].get(6, 0) * 10.0
                
                self.optimization_history.append({
                    'weights': test_weights.copy(),
                    'score': score,
                    'avg_match': results['avg_match']
                })
                
                if score > best_local_score:
                    best_local_score = score
                    best_local_value = value
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
            
            base_weights[target_weight] = best_local_value
            total = sum(base_weights.values())
            for k in base_weights:
                base_weights[k] /= total
            
            progress.append({
                'parameter': target_weight,
                'best_value': round(best_local_value, 2)
            })
        
        self.best_config = deepcopy(self.base_config)
        self.best_config.set_weights_from_dict(best_weights)
        
        return {
            'best_weights': {k: round(v, 3) for k, v in best_weights.items()},
            'best_score': round(best_score, 4),
            'progress': progress,
            'total_tests': len(self.optimization_history)
        }

# =============================================================================
# Main Analysis Function
# =============================================================================
def run_full_analysis(csv_path: str, num_predictions: int = 5, 
                     run_backtest: bool = False, run_optimization: bool = False) -> Dict:
    """전체 분석 실행 및 결과 반환"""
    
    result = {
        'success': False,
        'error': None,
        'data': {}
    }
    
    try:
        config = Config()
        
        # 최적화 (선택)
        if run_optimization:
            optimizer = HyperparameterOptimizer(csv_path, config)
            opt_result = optimizer.optimize(test_rounds=5, predictions_per_round=5)
            result['data']['optimization'] = opt_result
            config = optimizer.best_config
        
        # 백테스팅 (선택)
        if run_backtest:
            backtest = BacktestEngine(csv_path, config)
            bt_result = backtest.run_backtest(test_rounds=10, predictions_per_round=5)
            result['data']['backtest'] = bt_result
        
        # 메인 분석
        engine = DataEngine(csv_path, config)
        if not engine.is_loaded:
            result['error'] = engine.error_msg
            return result
        
        # 군집화 분석
        geometric = GeometricAnalyzer(engine, config.N_CLUSTERS)
        
        # 예측 생성
        generator = EnsembleGenerator(engine, geometric)
        predictions = generator.generate(count=num_predictions)
        
        # 결과 구성
        result['success'] = True
        result['data']['analysis'] = engine.get_analysis_summary()
        result['data']['clusters'] = geometric.get_cluster_summary()
        result['data']['predictions'] = [
            {
                'rank': i + 1,
                'numbers': pred['nums'],
                'sum': pred['stat']['sum'],
                'ac': pred['stat']['ac'],
                'odd': pred['stat']['odd'],
                'high': pred['stat']['high'],
                'cluster_score': round(pred['stat']['cluster_score'], 3),
                'total_score': pred['total_score'],
                'algorithm': pred['algorithm']
            }
            for i, pred in enumerate(predictions)
        ]
        result['data']['filter_stats'] = dict(generator.filter_system.filter_stats)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result
