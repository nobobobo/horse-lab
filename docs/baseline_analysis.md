# Baseline 分析

## 対象データ

2026-05-10 時点の baseline は、JRA-VAN `RACE` 日次 backfill と `0B41` realtime odds backfill から作った replay dataset を使う。

- Raw run: `backfill_daily_RACE_20250509_20260509_v1`
- 取得期間: 2025-05-09 から 2026-05-09
- RACE staging: `races=3564`, `entries=48173`, `results=47660`, `odds=365501`, `payouts=6913`
- Replay dataset v2: `races=3283`, `feature_rows=45287`, `results=45287`, `odds=45287`, `odds_timeseries=7054152`, `payouts=48582`
- QA artifact: `artifacts/data_quality/daily_backfill_RACE_20250509_20260509_with_payouts_v1/report.json`

`v2` は `0B41` の win odds time series、`HR/H1` 由来の official payout/pool、market movement、expanded past-performance、jockey/trainer stats、categorical-safe person IDs を含む。

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

## LightGBM Ablation

2026-05-10 に official payout/pool と `0B41` odds time series を合流した dataset で ablation を実行した。

- Artifact: `artifacts/lightgbm_ablation/daily_backfill_RACE_20250509_20260509_with_payouts_v1/ablation_summary.json`
- Train: 2025-05-10 から 2026-02-28
- Validation: 2026-03-01 から 2026-05-03
- Train rows: 36788
- Validation rows: 8499

| Scenario | Log loss | Brier | ECE | 読み方 |
| --- | ---: | ---: | ---: | --- |
| full | 0.20793 | 0.05796 | 0.00976 | best。market + form/person/history を全部使う |
| no_market | 0.22621 | 0.06144 | 0.00499 | market 系を抜くと大きく劣化。市場情報の寄与が非常に大きい |
| no_movement | 0.20831 | 0.05805 | 0.01041 | odds movement を抜いても full と小差。現時点では entry/closing に近い odds が主信号 |

`full` の top gain は `entry_win_odds` が 50.8%。`no_market` では `jockey_past_win_rate`、`last_finish_position`、`top3_rate_last5` が上位に来る。つまり market 非依存の signal は存在するが、単体では market-implied を上回るほど強くない。Phase 4 では OOF prediction と calibration で、market と非 market model の残差を重ねる方向がよい。

## 馬連 Simulation

`0B42` のローカル smoke data と `HR` official payout を使い、馬連 favorite strategy の settlement を確認した。

- Odds: `data/interim/jravan/backfill_0B41_0B42_20250510_20250511_o2_v1/odds.csv`
- Payouts: `data/processed/jravan/daily_backfill_RACE_20250509_20260509_with_payouts_v1/replay/payouts.csv`
- Artifact: `artifacts/quinella_sim/backfill_0B42_20250510_20250511_v1/`
- Races considered: 72
- Bets: 72
- Wins: 11
- Stake: 7200 JPY
- Payout: 5230 JPY
- ROI: -27.36%

これは収益戦略ではなく settlement smoke test。馬連の本格評価には `0B42` を過去1年分 staging 化し、favorite ではなく model probability / pair probability を出す必要がある。

`jravan-build-quinella-replay-dataset` で O2 staging から pair-level replay dataset も生成できるようにした。

- Dataset: `data/processed/jravan/quinella_backfill_0B42_20250510_20250511_v1/replay`
- Races: 72
- Latest pair odds: 6559
- Odds time series: 954372
- Official payout rows: 68

この dataset の `odds.csv` と `payouts.csv` を `quinella-sim` に渡すと、上記 favorite simulation と同じ settlement 結果になる。

## 馬連 Full Evaluation

2026-05-11 に、S3 の日次 backfill から `0B42_jvgets.txt` だけを同期し、2025-05-10 から 2026-05-03 の settlement 可能期間で full evaluation を実行した。2026-05-09 以降は result/payout が未確定の race が混ざるため除外。

- Raw files: 3528
- O2 records: 558040
- Pair odds read: 50567252
- Compact replay dataset: `data/processed/jravan/quinella_0B42_20250510_20260503_full_v1/replay`
- Latest pair odds: 323904
- Official quinella payout rows: 3295
- Settleable races: 3283

Favorite pair strategy は、各 race で最新 odds が最も低い馬連ペアに 100 JPY 固定で賭ける基準線。

| Strategy | Races | Bets | Wins | Hit rate | Stake | Payout | Profit | ROI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Favorite, settleable only | 3283 | 3283 | 583 | 17.76% | 328300 | 273530 | -54770 | -16.68% |
| Positive-edge diagnostic, settleable only | 3283 | 3283 | 146 | 4.45% | 328300 | 247350 | -80950 | -24.66% |

`positive-edge` は model probability ではなく、利用可能な pair odds だけを正規化した診断用なので、収益戦略としては扱わない。馬連 market では欠損ペアや発売停止/取消の扱いがあり、単純な正規化 edge は過信できない。

Favorite strategy の odds band 別 ROI:

| Odds band | Bets | Wins | Hit rate | Profit | ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10-15 | 21 | 14 | 66.67% | -200 | -9.52% |
| 15-20 | 75 | 37 | 49.33% | -680 | -9.07% |
| 20-30 | 291 | 95 | 32.65% | -4740 | -16.29% |
| 30-50 | 1011 | 225 | 22.26% | -11240 | -11.12% |
| 50+ | 1885 | 212 | 11.25% | -37910 | -20.11% |

この full evaluation は、馬連の settlement accounting が機能すること、favorite benchmark が控除率に負けること、pair probability model が必要なことを確認する基準線。

## Phase 3.5 が必要な理由

Phase 4 の ensemble に入る前に、各 Level 0 が同じ market signal を再学習するだけの状態を避ける必要がある。

優先して追加するデータ:

- Odds time series: opening/latest/min/max、snapshot count、implied probability movement、CLV。
- Payout / pool: settlement proxy、将来の official payout mapper、控除率・市場効率の検証。
- Expanded past performance: top3 rate、平均賞金、平均走破時計、同距離成績、距離変化、前走条件、馬体重。
- Person stats: 騎手/調教師の target race 前 run count / win rate。
- Categorical-safe IDs: `jockey:01020` / `trainer:04050` のように ID を順序数として扱わせない。

`RACE` daily backfill 単体では O1 odds が runner あたり 1 snapshot のため、movement 系特徴量は実質ゼロ情報になる。`0B41` O1 staging を `build_replay_dataset_from_staging(..., odds_staging_dirs=...)` で合流すると、runner あたり odds snapshot 中央値は 152 になり、opening/latest/min/max、pool movement、CLV に進める。

## Phase Status

- Phase 2: 完了。market baseline と LightGBM baseline を同じ time-series split で比較できる。
- Phase 3: 完了。Kelly/backtest/paper trading artifact を出力できる。
- Phase 3.5: 完了。`jravan-replay-v2` の実データ build、official payout/pool ingest、LightGBM ablation が通る。
- Phase 3.6: 完了。Data QA と馬連 settlement simulation が通る。

Phase 4 に入る条件:

- `0B42` の過去1年 backfill を staging/replay 可能にする。
- market / form / person-history model の OOF prediction を生成して store に保存する。

当面は収益最大化より、calibration、CLV、odds band / venue / surface / distance 別の歪み検出を優先する。
