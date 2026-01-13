# 🎰 Lotto Ultimate v2.0

AI 기반 로또 분석 시스템 - 백테스팅, 자동 최적화, 기하학적 군집화 통합

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 주요 기능

### 🔬 분석 기능
- **다중 구간 빈도 분석**: 10/30/50/100회차 구간별 가중 분석
- **출현 주기 이탈도 분석**: 평균 주기 대비 현재 상태 분석
- **동반 출현 쌍 분석**: 번호 쌍의 동반 출현 패턴 분석
- **모멘텀 분석**: 단기/중기 추세 지표

### 🔮 기하학적 군집화 (K-Means)
- 21개 특성 벡터 추출
- 자동 최적 클러스터 수 결정
- 패턴 유사도 기반 필터링

### 📊 백테스팅
- 과거 N회차 숨기고 알고리즘 검증
- 적중률, 등수별 달성 통계
- 랜덤 대비 성능 비교

### 🔧 하이퍼파라미터 자동 최적화
- Grid Search 기반 최적 가중치 탐색
- 백테스팅 결과 기반 성능 최적화

### 🎯 13단계 필터링 시스템
1. 합계 필터 (100~195)
2. AC값 필터 (≥7)
3. 홀짝 비율 (2~4개)
4. 고저 비율 (2~4개)
5. 구간 분포 (최소 3구간)
6. 연속번호 (최대 2연속)
7. 끝수 필터
8. 소수 필터
9. 이전 회차 필터
10. 역대 중복 배제
11. 번호 간격 필터
12. 경계 번호 필터
13. 군집 패턴 필터

## 🚀 설치 및 실행

### 로컬 실행

```bash
# 저장소 클론
git clone https://github.com/kkbhong71/Lotto-ultimate-v2.git
cd Lotto-ultimate-v2

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python app.py
```

브라우저에서 `http://localhost:5000` 접속

### Render.com 배포

1. GitHub에 저장소 푸시
2. Render.com에서 New Web Service 생성
3. GitHub 저장소 연결
4. 자동 배포 완료

## 📁 프로젝트 구조

```
Lotto-ultimate-v2/
├── app.py                 # Flask 메인 애플리케이션
├── lotto_engine.py        # 분석 엔진 (핵심 로직)
├── templates/
│   └── index.html         # 메인 페이지
├── static/
│   ├── css/
│   │   └── style.css      # 스타일시트
│   └── js/
│       └── main.js        # 프론트엔드 JavaScript
├── data/
│   └── new_1206.csv       # 당첨번호 데이터
├── requirements.txt       # 의존성
├── render.yaml            # Render 배포 설정
└── README.md              # 프로젝트 설명
```

## 📊 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 메인 페이지 |
| `/api/analyze` | POST | 전체 분석 및 예측 |
| `/api/quick-predict` | POST | 빠른 예측 |
| `/api/backtest` | POST | 백테스팅 실행 |
| `/api/optimize` | POST | 하이퍼파라미터 최적화 |
| `/api/data-info` | GET | 데이터 정보 조회 |
| `/api/number-analysis` | GET | 번호별 분석 데이터 |

## 🎨 스크린샷

### 메인 대시보드
- 모던 다크 테마 디자인
- 반응형 레이아웃
- 직관적인 번호 표시 (구간별 색상)

### 분석 결과
- HOT/COLD 번호 표시
- 상승 추세 번호
- 동반 출현 쌍
- 번호별 점수/간격/모멘텀 차트

### 백테스팅 결과
- 적중 분포 그래프
- 등수별 달성 통계
- 랜덤 대비 성능 비교

## ⚠️ 면책 조항

본 시스템은 통계적 분석 도구이며, 로또 당첨을 보장하지 않습니다.
도박 중독 예방을 위해 적정 금액 내에서 즐기시기 바랍니다.

## 📄 라이선스

MIT License

## 👨‍💻 개발자

- GitHub: [@kkbhong71](https://github.com/kkbhong71)
