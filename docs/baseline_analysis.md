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

### Identity Feature Rebuild

2026-05-13 に `entries.csv` を正本にした identity-aware replay を再 build し、同一馬の same-venue / same-distance / same-grade 履歴特徴量を追加した。

- Dataset: `data/processed/jravan/daily_backfill_RACE_20250509_20260509_identity_features_v1/replay`
- Feature version: `jravan-replay-v4`
- QA artifact: `artifacts/data_quality/daily_backfill_RACE_20250509_20260509_identity_features_v1/report.json`
- Races: `3283`
- Feature rows: `45287`
- Odds snapshots: `7054152`
- Unique horses: `11631`
- Missing `horse_id`: `0`
- Feature count: `63`

同じ split で LightGBM ablation を再実行した。

| Scenario | Log loss | Brier | ECE | 読み方 |
| --- | ---: | ---: | ---: | --- |
| full | 0.20894 | 0.05811 | 0.01033 | market + v4 feature。旧 v2 full より小さく悪化 |
| no_market | 0.22551 | 0.06118 | 0.00475 | 旧 v2 no-market より改善。identity-derived feature は market 抜きで少し効く |
| no_movement | 0.21064 | 0.05866 | 0.01147 | movement を抜くと full より悪化 |

新規 feature は `same_distance_top3_rate`、`same_venue_top3_rate`、`same_grade_top3_rate` が no-market で中位に入り、素材としては有効。一方で full model は market 周辺 feature の寄与が大きく、追加 feature をそのまま全部入れると calibration が悪化した。次は feature selection、regularization、segment-specific calibration、residual overlay の中で制御して使う。

### Profile / Race Title Feature Rebuild

2026-05-13 に、既存 `RACE` raw から取り出せる馬プロフィール・レースタイトル系 feature を追加して `jravan-replay-v5` を build した。

- Dataset: `data/processed/jravan/daily_backfill_RACE_20250509_20260509_profile_features_v2/replay`
- Feature version: `jravan-replay-v5`
- QA artifact: `artifacts/data_quality/daily_backfill_RACE_20250509_20260509_profile_features_v2/report.json`
- Races: `3283`
- Feature rows: `45287`
- Odds snapshots: `7054152`
- Unique horses: `11631`
- Feature count: `70`

追加 feature:

- Entry profile: `horse_symbol_code`、`breed_code`、`coat_color_code`、`trainer_affiliation_code`
- Race title/class hints: `race_grade_group`、`race_title_type`、`race_has_title`

実データの値分布を見ると、`horse_symbol_code`、`coat_color_code`、`trainer_affiliation_code` は分散があり、race title/grade 系も `ordinary`、`grade_code:E`、`graded`、`listed` などに分かれた。一方で `breed_code` は全行 `breed:1` だった。つまり、v5 は「今の raw から取れるプロフィール強化」としては有効だが、ユーザーが想定している sire/dam/damsire/grandparents の血統情報にはまだ届いていない。

同じ split で LightGBM ablation を再実行した。

| Scenario | Log loss | Brier | ECE | 読み方 |
| --- | ---: | ---: | ---: | --- |
| full | 0.20900 | 0.05814 | 0.00858 | market + v5 feature。旧 v2/v4 full より悪化 |
| no_market | 0.22502 | 0.06118 | 0.00419 | profile feature により no-market は v2/v4 より改善 |
| no_movement | 0.20856 | 0.05786 | 0.01076 | v5 では movement を抜いた方が最良 |

Feature importance では `coat_color_code` が full で rank 33、no-market で rank 26 に入り、`horse_symbol_code`、`trainer_affiliation_code`、`race_title_type`、`race_grade_group` も小さいながら gain を持った。`breed_code` は gain 0。採用判断としては、v5 feature は dataset には残すが、full model の default 採用 feature にはせず、feature selection、residual overlay、pedigree master ingest 後の specialist model で使う。

### Feature-set Study

2026-05-22 に、`jravan-replay-v5` で LightGBM feature-set study を追加し、同じ split で `full`、`market_only`、`no_market_selected`、`profile_pedigree_rating` を比較した。

- Dataset: `data/processed/jravan/daily_backfill_RACE_20250509_20260509_profile_features_v2/replay`
- Artifact: `artifacts/lightgbm_feature_sets/daily_backfill_RACE_20250509_20260509_profile_features_v2/feature_set_study_summary.json`
- Validation: 2026-03-01 から 2026-05-03

| Scenario | Features | Log loss | Brier | ECE | 読み方 |
| --- | ---: | ---: | ---: | ---: | --- |
| full | 70 | 0.20900 | 0.05814 | 0.00858 | 全 feature。v5 では market 周辺の歪みを拾い切れていない |
| market_only | 12 | 0.20827 | 0.05784 | 0.00953 | 今回の最良。market feature だけで full より良い |
| no_market_selected | 57 | 0.22502 | 0.06118 | 0.00419 | 非 market signal は calibration は静かだが予測力不足 |
| profile_pedigree_rating | 57 | 0.22502 | 0.06118 | 0.00419 | v5 では外部 pedigree/rating が未 join のため no-market と同等 |

結論として、現時点では `market_only` が probability baseline として最も強い。`profile_pedigree_rating` が改善しなかったのは、sire/dam/damsire や point-in-time rating の実データがまだこの v5 dataset に入っていないため。次の採用ゲートは、外部 horse master / rating history を合流した v6 dataset で `profile_pedigree_rating` が `no_market_selected` を上回るか、または residual overlay で market のごく小さい補正として効くかを見る。

### 2026-05-25 Replay Refresh

2026-05-25 に、追加の `RACE` staging と 2026-05-23/24 の realtime odds raw を取り込み、settlement 可能な replay dataset を更新した。

- Dataset: `data/processed/jravan/daily_RACE_20250509_20260525_with_rt_v1/replay`
- QA: `artifacts/data_quality/daily_RACE_20250509_20260525_with_rt_v1/report.json`
- Feature version: `jravan-replay-v5`
- Race dates in QA: 2025-05-10 から 2026-05-17
- QA warnings: none

| Item | Count |
| --- | ---: |
| Races | 3424 |
| Entries / features / results | 47218 |
| Latest odds rows | 47218 |
| Odds time series rows | 7122030 |
| Payout rows | 50654 |
| Unique horses | 11714 |
| Missing horse ID rows | 0 |
| Median odds snapshots per runner | 150 |

2026-05-23/24 の `0B30/0B41/0B42` raw は S3 から local に同期し、staging では 72 races、odds `1,248,505` rows を確認した。ただしこの期間は結果・払戻がまだ揃っていないため、settleable replay からは除外されている。次回 `RACE` settlement を取得した時点で、同じ raw odds を再合流して評価に入れる。

同じ updated dataset で feature-set study を再実行した。

- Artifact: `artifacts/lightgbm_feature_sets/daily_RACE_20250509_20260525_with_rt_v1/feature_set_study_summary.json`
- Validation: 2026-03-01 から 2026-05-17

| Scenario | Features | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| full | 79 | 0.21116 | 0.05872 | 0.00938 |
| market_only | 12 | 0.21045 | 0.05839 | 0.00807 |
| no_market_selected | 57 | 0.22651 | 0.06168 | 0.00477 |
| profile_pedigree_rating | 66 | 0.22651 | 0.06168 | 0.00477 |

更新後も `market_only` が最良。追加 feature は情報としては増えているが、現状の one-year dataset では market-implied probability を上回るほどではない。引き続き、外部 pedigree / rating / pace-bias 系の非 market signal を増やして residual に効くかを見る。

## Phase 4 OOF / Stacking 準備

OOF prediction は、各 validation fold の予測を、その fold を学習に使っていない Level 0 model だけで作る予測。meta learner が in-fold prediction を見て過学習するのを避けるため、stacking では必須の学習素材になる。

2026-05-11 に、runner-level win probability の Phase 4 初回 OOF artifact を生成した。

- Dataset: `data/processed/jravan/daily_backfill_RACE_20250509_20260509_with_payouts_v1/replay`
- Feature version: `jravan-replay-v2`
- Validation: 2026-03-01 から 2026-05-09
- Folds: `202603`, `202604`, `202605`
- Artifact: `artifacts/oof/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20260301_20260509/oof_report.json`
- Stored predictions: `25497`
- Meta dataset: `artifacts/stacking/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20260301_20260509/meta_features.csv`
- Meta rows: `8499`

OOF metrics:

| Model | Predictions | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| market | 8499 | 0.20398 | 0.05713 | 0.00523 |
| lightgbm_full | 8499 | 0.20817 | 0.05802 | 0.01066 |
| lightgbm_no_market | 8499 | 0.22670 | 0.06142 | 0.00550 |

単体性能は market がまだ最良。Phase 4 の狙いは、LightGBM が market を単純に上回ることではなく、market が外している race/runner で非 market signal が残差を補えるかを meta learner と holdout / paper trading で検証すること。

### Logistic Meta Learner MVP

OOF meta dataset から、標準ライブラリだけで動く logistic meta learner を追加した。入力予測は logit 変換し、出力は race 内で win probability が合計 1 になるよう正規化する。初回評価では `202603` / `202604` fold で学習し、最新の `202605` fold を temporal holdout にした。

- Artifact: `artifacts/stacking_meta/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20260301_20260509/meta_evaluation.json`
- Train rows: `7511`
- Holdout rows: `988`
- Holdout winners: `71`

Holdout metrics:

| Model | Log loss | Brier |
| --- | ---: | ---: |
| Logistic meta | 0.20270 | 0.05625 |
| LightGBM full | 0.20188 | 0.05585 |
| Market-implied | 0.20223 | 0.05589 |
| LightGBM no-market | 0.22288 | 0.05973 |

結論として、初回の logistic stacking は holdout で最良ではない。係数は market と LightGBM full を強く見ており、no-market signal は小さい補助に留まった。現時点では Phase 4 の採用条件は「meta learner が holdout で market / best Level 0 を上回ること」。この条件を満たすまでは market / LightGBM full を基準線として維持する。

### Convex Blend Search

Logistic meta learner は自由度が高めなので、より解釈しやすい比較として convex blend も追加した。各 Level 0 予測列に非負重みを付け、重み合計を 1 に固定する。重みは train folds の log loss を最小化する grid search で選び、同じ `202605` fold で holdout 評価する。

- Artifact: `artifacts/stacking_blend/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20260301_20260509/blend_evaluation.json`
- Grid step: `0.05`
- Candidates: `231`
- Selected weights:
  - LightGBM full: `0.10`
  - LightGBM no-market: `0.00`
  - Market-implied: `0.90`

Holdout metrics:

| Model | Log loss | Brier |
| --- | ---: | ---: |
| Convex blend | 0.20189 | 0.05581 |
| LightGBM full | 0.20188 | 0.05585 |
| Market-implied | 0.20223 | 0.05589 |
| LightGBM no-market | 0.22288 | 0.05973 |

Blend は market より改善したが、log loss では LightGBM full にごく僅差で負けた。一方で Brier は blend が最良なので、確率の平均二乗誤差ではわずかに改善している。現時点では「有望だが採用ゲート未通過」。次は holdout fold を増やす、venue/surface/distance/odds band 別に blend が効く subset を探す、calibration を入れる、という順がよい。

### Phase 4 Walk-Forward Study

Phase 4 完走判定として、OOF 期間を 2025-10-01 から 2026-05-09 まで拡張し、月次 walk-forward study を実行した。

- OOF artifact: `artifacts/oof/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/oof_report.json`
- Meta dataset: `artifacts/stacking/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/meta_features.csv`
- Phase 4 study: `artifacts/phase4_study/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/phase4_study_report.json`
- OOF folds: `202510` から `202605`
- Stored predictions: `82410`
- Meta rows: `27470`
- Walk-forward holdout folds: `202601` から `202605`
- Walk-forward observations: `16173`

Walk-forward overall:

| Method | Log loss | Brier |
| --- | ---: | ---: |
| Convex blend | 0.20136 | 0.05643 |
| Market-implied | 0.20139 | 0.05643 |
| Logistic meta | 0.20204 | 0.05658 |
| LightGBM full | 0.20624 | 0.05741 |
| LightGBM no-market | 0.22539 | 0.06111 |

採用判定は `promote_ensemble_candidate`。ただし改善幅は market に対して log loss `0.000026` と非常に小さい。つまり「Phase 4 は成功したが、強い収益シグナルを発見した」というより、「market を壊さず、ごく小さく LightGBM signal を足す restrained ensemble の候補を作れた」という評価。

Fold 別の blend weight:

| Holdout | LightGBM full | LightGBM no-market | Market |
| --- | ---: | ---: | ---: |
| 202601 | 0.00 | 0.00 | 1.00 |
| 202602 | 0.00 | 0.00 | 1.00 |
| 202603 | 0.05 | 0.00 | 0.95 |
| 202604 | 0.05 | 0.00 | 0.95 |
| 202605 | 0.05 | 0.00 | 0.95 |

Segment study では、blend は `Niigata`、`Chukyo`、`1201_1600m`、`15頭以上`、market probability `0.05-0.10` などで market より良かった。一方で `10頭以下`、`2000m超`、障害、`Hanshin` では悪化した。現時点では segment-specific model を採用するほど改善幅は大きくないため、Phase 5 では全体 restrained blend を paper trading で監視し、segment は alert / diagnostics として使う。

## Phase 5 Paper Trading

Phase 4 の `convex_blend` を paper trading candidate として model registry に登録し、walk-forward predictions を paper trading replay に流した。

- Registry: `artifacts/model_registry/phase5_convex_blend_candidate/model_registry.json`
- Conservative paper trading: `artifacts/paper_trading/phase5_convex_blend_202601_202605/paper_trading_report.json`
- Diagnostic paper trading: `artifacts/paper_trading/phase5_convex_blend_202601_202605_diagnostic_min_edge_0/paper_trading_report.json`
- Candidate stage: `paper_trading_candidate`
- Serving weights: LightGBM full `0.05`, LightGBM no-market `0.00`, market `0.95`

Conservative replay は minimum edge 2%、closing odds の条件で実行した。

| Metric | Value |
| --- | ---: |
| Races | 1138 |
| Predictions | 16173 |
| Bet records | 1 |
| Positive edge decisions | 17 |
| Stake | 100 JPY |
| Profit | -100 JPY |
| ROI | -100.0% |

bet が 1 件しか出ないのは異常ではない。candidate は market に非常に近い restrained blend なので、同じ closing odds に対して大きな positive edge はほぼ出ない。Phase 5 の主目的は収益確認ではなく、daily pipeline、artifact、監視指標、採用ゲートの整備。実運用判断には、締切前 snapshot での live-like prediction と CLV 監視が必要。

## Phase 6 Daily Paper Trading

Phase 6 では、registry candidate を日次運用に近い形で再学習し、当日分を paper trading replay する `daily-paper-trading-run` を追加した。

- Daily artifact: `artifacts/daily_paper_trading/phase6_20260503/daily_paper_trading_report.json`
- Monitoring artifact: `artifacts/paper_trading_monitoring/phase6_summary.json`
- Target date: 2026-05-03
- Train end: 2026-05-02
- Target races: 35
- Target runners: 496
- Serving weights: LightGBM full `0.05`, LightGBM no-market `0.00`, market `0.95`

Daily result:

| Metric | Value |
| --- | ---: |
| Log loss | 0.20367 |
| Brier | 0.05585 |
| ECE | 0.01272 |
| Bet records | 0 |
| Positive edge decisions | 0 |
| Mean edge | -0.21054 |

Phase 5 + Phase 6 を aggregate した monitoring summary は `report_count=2`、`observations=16669`、`log_loss=0.20143`、`bet_records=1`。まだ資金投入判断ではなく、日次で candidate を回し続けて calibration drift、CLV、segment 別の悪化を監視する段階。

### 2026-05-25 Daily Paper Trading Refresh

updated replay dataset で 2026-05-10 から 2026-05-17 を paper trading した。

- Artifact: `artifacts/paper_trading/daily_RACE_20250509_20260525_with_rt_v1_20260510_20260517/daily_paper_trading_report.json`
- Dataset: `data/processed/jravan/daily_RACE_20250509_20260525_with_rt_v1/replay`
- Train end: 2026-05-09
- Serving weights: LightGBM full `0.05`、LightGBM no-market `0.00`、market `0.95`

| Metric | Value |
| --- | ---: |
| Train races | 3318 |
| Target races | 106 |
| Target runners | 1452 |
| Log loss | 0.21108 |
| Brier | 0.05895 |
| ECE | 0.00644 |
| Bet records | 3 |
| Positive edge decisions | 15 |
| Stake | 700 JPY |
| Profit | -700 JPY |
| ROI | -100.0% |

これは model candidate の棄却というより、現行 `convex_blend` が market に非常に近く、minimum edge 2% を満たす場面が少ないことを改めて確認した結果。CLV は latest-available setup では 0 件なので、次は締切前 snapshot を固定した live-like inference で CLV を測る。

## Market Calibration Study

追加モデル第一弾として、market-implied probability に Platt-style calibration をかける walk-forward study を追加した。

- Artifact: `artifacts/market_calibration/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/market_calibration_report.json`
- Input: Phase 4 meta dataset
- Market column: `pred__market_implied_probability__market_implied_oof_v1`
- Holdout folds: `202601` から `202605`
- Observations: `16173`

| Method | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: |
| Market baseline | 0.201385 | 0.056434 | 0.003510 |
| Market calibrated | 0.201386 | 0.056432 | 0.002798 |

Calibration は ECE を改善したが、log loss は `0.000001` だけ悪化した。採用判定は `keep_market_baseline`。これは悪い結果ではなく、market がすでにかなり well-calibrated であることを確認した形。次に calibration を使うなら、全体一律ではなく odds band / venue / surface / field size など segment-specific にする価値がある。

### Residual Overlay / Segment Calibration

2026-05-13 に market を anchor とし、非 market model を residual として小さく足す `stacking-residual-overlay-study` を追加した。

- Artifact: `artifacts/residual_overlay/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/residual_overlay_report.json`
- Market: `pred__market_implied_probability__market_implied_oof_v1`
- Overlay: `pred__lightgbm_win_probability__lightgbm_no_market_oof_v1`
- Holdout folds: `202601` から `202605`
- Observations: `16173`

| Method | Log loss | Brier |
| --- | ---: | ---: |
| Market baseline | 0.201385 | 0.056434 |
| Residual overlay | 0.201385 | 0.056434 |
| No-market overlay baseline | 0.225390 | 0.061106 |

全 fold で selected alpha は `0.0`。つまり、現時点の no-market signal は全体へ一律に足すより、market をそのまま使う方がよい。

Segment-specific calibration も追加し、market probability band と venue 別に検証した。

| Calibration | Log loss | Brier | ECE | 判定 |
| --- | ---: | ---: | ---: | --- |
| Market baseline | 0.201385 | 0.056434 | 0.003510 | baseline |
| Odds band calibration | 0.201412 | 0.056452 | 0.002727 | ECE は改善、log loss は悪化 |
| Venue calibration | 0.201452 | 0.056414 | 0.003204 | Brier/ECE は改善、log loss は悪化 |

結論として、calibration 系は monitoring/diagnostics として有用だが、現時点では prediction candidate として market baseline を置き換えない。次の改善余地は、market と相関しにくい追加データ、具体的には pedigree、official rating、詳細過去走、pace/track-bias にある。

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

### 0B42 / 0B30 Recent Capture

2026-05-23/24 の realtime capture を S3 から同期し、0B42 馬連 dataset と 0B30 catalog を更新した。

- Raw run: `rt_20260523_20260524_0B30_0B41_0B42_v3`
- 0B42 dataset: `data/processed/jravan/quinella_0B42_20260523_20260524_v1/replay`
- 0B30 catalog: `artifacts/data_catalog/rt_20260523_20260524_0B30_catalog.json`

| Item | Count |
| --- | ---: |
| Realtime races | 72 |
| 0B42 latest pair odds | 7596 |
| 0B42 odds time series | 1092349 |
| 0B42 payout rows | 0 |
| 0B30 files | 72 |
| 0B30 records | 432 |
| 0B30 O1/O2/O3/O4/O5/O6 records | 72 each |

0B42 payout rows が 0 なのは、対象日の result/payout がまだ replay 側に入っていないため。raw odds は有効で、settlement 可能になった後に再 build すれば評価対象になる。0B30 は O1/O2/O3/O4/O5/O6 が取れており、三連複・三連単などの mapper を追加するための raw catalog として使える。

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
- Phase 3.7: 完了。OOF prediction store、`level0-oof`、meta dataset builder が通る。
- Phase 4: 完了。OOF、meta dataset、logistic meta learner、convex blend search、walk-forward/segment study を生成/評価済み。convex blend は market を小幅に上回り、paper trading 候補。
- Phase 5: 完了。model registry、paper trading replay、CLV/backtest artifacts を生成済み。
- Phase 6: 完了。日次 paper trading run と monitoring summary を実装し、実データ smoke を通過。次は日次蓄積と追加 specialist model の投入。
- Phase 7: 進行中。Data catalog、identity map、market calibration study、Windows automated fetch wrapper、pre-fetch deploy gate、2026-05-23/24 realtime capture の取り込みを追加。次は settlement 後の再 build と 0B30 O3-O6 mapper。

当面は収益最大化より、calibration、CLV、odds band / venue / surface / distance 別の歪み検出を優先する。
