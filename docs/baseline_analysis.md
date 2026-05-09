# Baseline 分析

## 対象データ

2026-05-09 時点の baseline は、JRA-VAN `RACE` 日次 backfill から作った replay dataset を使う。

- Raw run: `backfill_daily_RACE_20250509_20260509_v1`
- 取得期間: 2025-05-09 から 2026-05-09
- Staging: `races=3564`, `entries=48173`, `results=47660`, `odds=47487`
- Replay dataset v1: `races=3283`, `feature_rows=45287`, `results=45287`, `odds=45287`
- 再生成版: `jravan-replay-v2`

`v1` は Phase 2/3 の基準線として有効。`v2` は `odds_timeseries.csv`、`payouts.csv`、market movement、expanded past-performance、jockey/trainer stats、categorical-safe person IDs を含む。

## Market-Implied Baseline

全期間での market-implied baseline:

- Races: 3283
- Observations: 45287
- Winners: 3286
- Mean predicted probability: 0.07249
- Empirical win rate: 0.07256
- Log loss: 0.20508
- Brier score: 0.05783
- Expected calibration error: 0.00138
- Bet records: 0

`bet_records=0` は正常。単勝オッズの逆数を race 内で正規化しているため、同じ market odds に対して控除率込みの positive edge は基本的に出ない。この baseline は収益モデルではなく、calibration、replay、data integrity、backtest accounting の基準線。

## LightGBM Baseline

初回 LightGBM は以下の split で実行した。

- Train: 2025-05-09 から 2026-02-28
- Validation: 2026-03-01 から 2026-05-08
- Train rows: 36788
- Validation rows: 8499

同一 validation window の結果:

| Model | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: |
| Market-implied | 0.20398 | 0.05713 | 0.00523 |
| LightGBM v1 | 0.20706 | 0.05753 | 0.00901 |
| LightGBM v2 | 0.20816 | 0.05807 | 0.00966 |

現時点では market-implied baseline の方が良い。これは自然な結果で、単勝 market は騎手、馬場、過去走、調教、直前気配、資金流入などの集合知を既に含む。

LightGBM v1 の feature importance では `entry_win_odds` が gain の 57.8% を占めた。LightGBM v2 では 53.3% まで下がり、`jockey_past_win_rate`、`trainer_past_win_rate`、`trainer_past_run_count`、`jockey_past_run_count` が上位に入った。一方で validation metrics は v1 より少し悪化したため、追加特徴量はまだ calibration/tuning/ablation 前の素材という位置づけ。

`jockey_id` / `trainer_id` は v2 で categorical として認識されるようになった。ID の大小を数値として学習する問題は解消した。

## Phase 3.5 が必要な理由

Phase 4 の ensemble に入る前に、各 Level 0 が同じ market signal を再学習するだけの状態を避ける必要がある。

優先して追加するデータ:

- Odds time series: opening/latest/min/max、snapshot count、implied probability movement、CLV。
- Payout / pool: settlement proxy、将来の official payout mapper、控除率・市場効率の検証。
- Expanded past performance: top3 rate、平均賞金、平均走破時計、同距離成績、距離変化、前走条件、馬体重。
- Person stats: 騎手/調教師の target race 前 run count / win rate。
- Categorical-safe IDs: `jockey:01020` / `trainer:04050` のように ID を順序数として扱わせない。

現 `RACE` daily backfill では O1 odds が runner あたり 1 snapshot のため、`odds_open`、`odds_latest`、`odds_min`、`odds_max` が同値になり、movement 系特徴量は実質ゼロ情報になる。CLV と market movement を評価するには、`0B31/0B41` の realtime odds を日次で蓄積する必要がある。

## Phase Status

- Phase 2: 完了。market baseline と LightGBM baseline を同じ time-series split で比較できる。
- Phase 3: 完了。Kelly/backtest/paper trading artifact を出力できる。
- Phase 3.5: 実装完了。`jravan-replay-v2` の実データ smoke build と LightGBM training が通る。

Phase 4 に入る条件:

- realtime odds を継続取得して movement / CLV を評価できる状態にする。
- feature ablation で market 以外の情報価値を確認する。
- out-of-fold prediction store の schema を決める。

当面は収益最大化より、calibration、CLV、odds band / venue / surface / distance 別の歪み検出を優先する。
