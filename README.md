# Systematic Alpha Trader (KR + US)

KIS OpenAPI 모의투자용 자동매매 엔진입니다.  
전략은 **Larry Williams 변동성 돌파 + 당일청산**으로 고정되어 있으며, `agent_lab`/LLM 파이프라인은 제거되었습니다.

## 핵심 규칙
- `budget_per_trade`는 매일 **08:40 KST 1회 스냅샷**으로 확정 후 하루 고정
- 진입 주문은 시장가
- 포지션 제한: KR 최대 3, US 최대 3, 합계 최대 6
- 신규 진입 감시는 후보군만 수행
- 운영 감시/청산은 `후보군 ∪ 서버 보유종목`
- 서버(KIS API) 상태가 단일 진실원천(SoT), 로컬은 덮어씀
- 텔레그램은 즉시 알림만 사용(30분 배치 요약 없음)
- US S&P500 유니버스는 **원격 소스(datahub) 성공 시에만** 사용(폴백 없음, 실패 시 precompute 실패)

## 현재 필터링 기준 (적용 중)
- KR 원천 유니버스: `KOSPI200 + KOSDAQ150`
- US 원천 유니버스: `S&P500`
- 공통 1차 필터:
  - 전일 통계 조회 성공(`prev_close`/`prev_turnover` 유효)
  - `prev_close > 0`
  - `prev_turnover > 0` (전일 거래대금 존재)
  - 실패/누락 종목은 유니버스 단계에서 제외(불량 데이터 제외)
- 공통 랭킹 기준:
  - `prev_turnover` 내림차순
- 공통 최종 후보:
  - 시장별 상위 `20`개 (`TRADER_CANDIDATES_MAX_KR/US`)
  - 유효 종목이 부족하면 20개 미만으로 확정될 수 있음
- API 호출 제어:
  - `TRADER_RATE_LIMIT_RETRIES`, `TRADER_RATE_LIMIT_BACKOFF_SEC`, `TRADER_RATE_LIMIT_BACKOFF_MAX_SEC` 기반 재시도/백오프
  - US는 거래소 순회 사이에 `TRADER_US_EXCHANGE_SPACING_SEC` 간격 적용
- 참고 산출물:
  - 유효 유니버스 캐시: `out/trader/{kr|us}/{YYYYMMDD}/cache/{kr|us}_valid_universe.csv`
  - 유동성 랭킹 캐시(풀): `out/trader/{kr|us}/{YYYYMMDD}/cache/{kr|us}_universe_liquidity.csv`
  - 최종 후보 캐시(실제 실행 기준): `out/trader/{kr|us}/{YYYYMMDD}/cache/{kr|us}_final_candidates.csv`

## 필터링 실험 후보 (미적용, 다음 단계)
아래 항목은 현재 전략에 적용되지 않았고, 이후 실험 대상으로만 문서화합니다.

- 전일 거래대금 하한값 강화:
  - 단순 `> 0` 대신 시장별 최소 임계값 적용
- 최근 N일 평균 거래대금 기반 필터:
  - 단일 전일값 노이즈 완화
- 가격/거래정지/관리종목 가드 강화:
  - 실행 가능성 중심 필터
- 변동성 품질 필터:
  - 최근 N일 ATR/평균 range 기반 하한
- 시장 레짐 연동 필터:
  - 지수 필터 ON 시에만 후보 확정, OFF면 후보 생성은 하되 주문 차단 유지
- 슬리피지/체결품질 사전 필터:
  - 과거 체결 품질이 낮은 종목 제외

## 설정 파일
- 민감정보: `.env`
- 비민감 운용값: `config/trader.config`
- 샘플: `.env.example`, `config/trader.config.example`

## 주요 명령
```bash
# 상태 확인
./scripts/run_trader_wsl.sh --action status

# 계좌 동기화
./scripts/run_trader_wsl.sh --action sync-account --market ALL

# 사전 계산 (KR/US)
./scripts/run_trader_wsl.sh --action precompute --market ALL

# 08:40 예산 스냅샷
./scripts/run_trader_wsl.sh --action snapshot-budget

# 단일 사이클 실행
./scripts/run_trader_wsl.sh --action run-cycle --market KR
./scripts/run_trader_wsl.sh --action run-cycle --market US

# 청산
./scripts/run_trader_wsl.sh --action liquidate --market KR --phase primary
./scripts/run_trader_wsl.sh --action liquidate --market US --phase retry

# 일일 리포트
./scripts/run_trader_wsl.sh --action report --date YYYYMMDD

# 데몬 실행
./scripts/run_trader_wsl.sh --action daemon

# 테스트/검증 실행(로그는 logs/trader_test/*로 분리)
./scripts/run_trader_test_wsl.sh --action status

# 단위 테스트
/home/heesu/anaconda3/envs/systematic-alpha/bin/python -m pytest -q tests/test_trader_algorithm.py tests/test_trader_morning_prep.py

# 실 API/원격소스 헬스체크 테스트 (DNS/API 실패 시 테스트 실패)
/home/heesu/anaconda3/envs/systematic-alpha/bin/python -m pytest -q -m live_api tests/test_trader_live_api_health.py

# 테스트 잔존물 정리 + 전체 테스트(단위+live_api) 1회 실행
./scripts/run_all_tests_wsl.sh

# live_api 제외 실행(빠른 로컬 회귀)
./scripts/run_all_tests_wsl.sh --skip-live
```

## 크론 작업
```bash
# 등록(즉시 daemon 기동 포함)
./scripts/register_trader_tasks_wsl.sh

# 제거
./scripts/remove_trader_tasks_wsl.sh

# 재설정
./scripts/reset_trader_tasks_wsl.sh
```

## 모니터링
```bash
./scripts/monitor_trader_wsl.sh --once
./scripts/monitor_trader_wsl.sh

# 테스트 로그 모니터링
./scripts/monitor_trader_wsl.sh --log-profile trader_test --once
# 테스트 DB + 로그 모니터링
./scripts/monitor_trader_wsl.sh --profile test --once
```

## 로그 분리
- 운영 로그: `logs/trader/<YYYYMMDD>/...`
- 테스트 로그: `logs/trader_test/<YYYYMMDD>/...`
- `run_trader_wsl.sh`에서 `--log-profile`로 경로를 명시할 수 있습니다.
  - 예: `./scripts/run_trader_wsl.sh --log-profile trader_test --action sync-account --market US`

## DB 요약 조회
```bash
# test 프로파일, 특정 일자
/home/heesu/anaconda3/envs/systematic-alpha/bin/python scripts/trader_db_summary.py --profile test --date YYYYMMDD --limit 10

# prod 프로파일, 최근 상태
/home/heesu/anaconda3/envs/systematic-alpha/bin/python scripts/trader_db_summary.py --profile prod --limit 10
```

## 상태/산출물 분리(Profile)
- 운영 기본 프로파일: `prod`
  - DB: `state/trader/trader.sqlite`
  - 산출물: `out/trader/...`
  - 로그: `logs/trader/...`
- 테스트 프로파일: `test`
  - DB: `state/trader_test/trader.sqlite`
  - 산출물: `out/trader_test/...`
  - 로그: `logs/trader_test/...`
- 지정 실행 예시:
  - `./scripts/run_trader_wsl.sh --profile test --log-profile trader_test --action status`

## 컷오버/리셋
```bash
# 기존 agent_lab/trader 상태와 로그를 archive로 이동 후 trader 상태 재초기화
./scripts/run_trader_wsl.sh --action archive-reset
```

아카이브 경로: `archive/cutover_trader_<timestamp>/`
