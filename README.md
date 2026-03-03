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
