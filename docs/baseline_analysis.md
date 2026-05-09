# Baseline 分析

## 対象データ

2026-05-09 時点の最初の実データ baseline は、JRA-VAN `RACE` の日次 backfill から作った replay dataset を使う。

- Raw run: `backfill_daily_RACE_20250509_20260509_v1`
- 取得期間: 2025-05-09 から 2026-05-09
- JVOpen 日次試行: 366 days
- 成功 run: 164
- 失敗 run: 202
- ローカル raw: 164 files / 約 834MB
- Staging: `races=3564`, `entries=48173`, `results=47660`, `odds=47487`
- Replay dataset: `races=3283`, `feature_rows=45287`, `results=45287`, `odds=45287`

失敗 run は `raw/jravan/failed/` に log/manifest を退避している。現時点では、失敗の多くは日次 range に対する `JVOpen` の no-data/failure として扱う。成功 run だけでも Phase 2 の初回 baseline には十分な規模がある。

## Market-Implied Baseline

実行コマンド:

```bash
PYTHONPATH=src python3 -m horse_lab.cli market-replay \
  data/processed/jravan/daily_backfill_RACE_20250509_20260509_v1/replay \
  --start-date 2025-05-09 \
  --end-date 2026-05-09 \
  --as-of 2026-05-09T00:00:00
```

結果:

- Races: 3283
- Observations: 45287 runner rows
- Winners: 3286
- Mean predicted probability: 0.07249
- Empirical win rate: 0.07256
- Log loss: 0.20508
- Brier score: 0.05783
- Expected calibration error: 0.00138
- Bet records: 0

## LightGBM Baseline

実行コマンド:

```bash
PYTHONPATH=src python3 -m horse_lab.cli lightgbm-train \
  data/processed/jravan/daily_backfill_RACE_20250509_20260509_v1/replay \
  artifacts/lightgbm/daily_backfill_RACE_20250509_20260509_v1 \
  --train-end-date 2026-02-28 \
  --valid-start-date 2026-03-01 \
  --valid-end-date 2026-05-08 \
  --as-of 2026-05-09T00:00:00
```

split:

- Train: 2025-05-09 から 2026-02-28
- Validation: 2026-03-01 から 2026-05-08
- Train races: 2680
- Train rows: 36788
- Validation races: 603
- Validation rows: 8499

特徴量:

- Numeric features: 24
- Categorical features: 7
- 主な numeric: `entry_win_odds`, `entry_popularity_rank`, `race_distance_m`, `race_field_size`, `age`, `carried_weight_kg`, `body_weight_kg`, `days_since_last_run`, `avg_finish_position_last3`, `win_rate_last5`, `same_surface_win_rate`, `last_odds`
- 主な categorical: `race_venue`, `race_surface`, `race_track_condition`, `race_weather`, `race_grade`, `race_direction`, `sex`

validation 結果:

- Observations: 8499 runner rows
- Winners: 603
- Mean predicted probability: 0.07095
- Empirical win rate: 0.07095
- Log loss: 0.20706
- Brier score: 0.05753
- Expected calibration error: 0.00901
- Feature importance: `artifacts/lightgbm/daily_backfill_RACE_20250509_20260509_v1/feature_importance.csv`

feature importance 上位:

| Rank | Feature | Gain fraction | Note |
| ---: | --- | ---: | --- |
| 1 | `entry_win_odds` | 57.8% | market 情報が支配的 |
| 2 | `trainer_id` | 5.3% | ID feature の扱いは要改善 |
| 3 | `jockey_id` | 5.0% | ID feature の扱いは要改善 |
| 4 | `body_weight_kg` | 4.4% | entry 詳細が効いている |
| 5 | `days_since_last_run` | 3.1% | 過去走由来 feature が効いている |
| 6 | `body_weight_diff_kg` | 2.4% | entry 詳細が効いている |
| 7 | `horse_number` | 2.1% | 枠/馬番バイアスの候補 |
| 8 | `avg_distance_m_last3` | 2.1% | 過去走由来 feature が効いている |
| 9 | `avg_finish_position_last3` | 1.8% | 過去走由来 feature が効いている |

`entry_win_odds` が強すぎるため、現モデルはかなり market-following。これは初回 baseline としては自然だが、期待値モデルに進むには odds movement、closing-line value、コース/距離/馬場適性、クラス変化など、market が過小評価しやすい説明変数を足す必要がある。

また、`trainer_id` / `jockey_id` は現在 numeric feature として扱われている。ID の大小に順序的意味はないため、次の改善では categorical encoding または target/OOF encoding の候補にする。

同じ validation window の market-implied baseline:

```bash
PYTHONPATH=src python3 -m horse_lab.cli market-replay \
  data/processed/jravan/daily_backfill_RACE_20250509_20260509_v1/replay \
  --start-date 2026-03-01 \
  --end-date 2026-05-08 \
  --as-of 2026-05-09T00:00:00
```

- Observations: 8499 runner rows
- Winners: 603
- Mean predicted probability: 0.07095
- Empirical win rate: 0.07095
- Log loss: 0.20398
- Brier score: 0.05713
- Expected calibration error: 0.00523
- Bet records: 0

## Market vs LightGBM

現時点では、market-implied baseline の方が LightGBM baseline より良い。

| Model | Log loss | Brier | ECE |
| --- | ---: | ---: | ---: |
| Market-implied | 0.20398 | 0.05713 | 0.00523 |
| LightGBM | 0.20706 | 0.05753 | 0.00901 |

これは悪い結果ではなく、むしろ自然な初回 baseline。単勝 market は既に騎手、馬場、過去走、調教、直前気配、資金流入などの集合知を含んでいる。現在の LightGBM は初期 feature と初期 hyperparameter のみで、market feature も `entry_win_odds` / `entry_popularity_rank` に強く依存しているため、まだ市場確率を上回るほどの独自情報を持っていない。

この段階での LightGBM の役割は、いきなり収益を出すことではなく、以下を検証すること。

- replay dataset から point-in-time safe に学習できるか
- race 内で勝率が 1 に正規化されるか
- market baseline と同じ validation window で比較できるか
- 過去走 feature が学習 pipeline に接続できるか
- 今後の feature ablation / calibration / stacking の土台になるか

次に market を超えるためには、単勝オッズの再現ではなく、market が過小評価しやすい局面を拾う feature が必要になる。優先度が高いのは、odds movement、closing-line value、距離/馬場適性、クラス変化、休み明け、斤量変化、騎手・厩舎・コース相性、取消/除外を含む field integrity の改善。

## 解釈

Market-implied baseline は、単勝オッズの逆数を race 内で正規化して勝率に変換する。これは市場が持つ集合知を確率に変換した基準線であり、非 market model が最初に超えるべき benchmark になる。

`bet_records=0` は正常な結果。予測確率を同じ単勝オッズから作っているため、控除率込みの市場を正規化しただけでは positive edge が基本的に発生しない。したがってこの baseline は、今の段階では「賭けて利益を出すモデル」ではなく、以下を検証するための基準線として使う。

- raw to staging to replay の data integrity
- point-in-time safe な odds selection
- race 内 probability normalization
- calibration quality
- backtest engine の settlement/accounting
- 今後の LightGBM や ensemble が超えるべき log loss / Brier / calibration 基準

Calibration はかなり良い。平均予測確率と実勝率が近く、ECE も低い。一方で、0.4 以上の高確率帯は sample が少なく、bin ごとの誤差は大きくなりやすい。今後はオッズ帯、人気帯、競馬場、距離、surface ごとに分解して、market が強い領域と歪みが出る領域を探す。

## Phase 2 Status

Phase 2 の定義は「market-implied probability baseline と LightGBM/sklearn baseline を作り、time-series split で評価する」こと。

現状:

- Market-implied baseline: 完了
- Replay-ready dataset: 完了
- Kelly/backtest/calibration loop: 完了
- 過去走 feature: 初期版完了
- 大規模 dataset での性能ボトルネック解消: 完了
- LightGBM pipeline scaffold: 実装済み
- LightGBM 実学習: 完了
- Market vs LightGBM 同一 window 比較: 完了
- LightGBM feature importance 保存: 完了

結論として、Phase 2 は完了。現 dataset で market-implied baseline と LightGBM baseline を同じ validation window で比較できる状態になった。

推奨する最初の split:

- Train: 2025-05-09 から 2026-02-28
- Validation: 2026-03-01 から 2026-05-08
- 2026-05-09 以降の未確定/当日 race は validation から除外

Phase 2 の完了条件:

- LightGBM validation log loss / Brier / calibration が保存されている: 完了
- market baseline と同じ validation window で比較できる: 完了
- positive edge が出る場合も、まず paper backtest と calibration を優先して確認する: 継続方針

未完了だが Phase 2 blocker ではないもの:

- hyperparameter tuning
- probability calibration の追加
- feature ablation
- out-of-fold prediction store
- ID feature の categorical/target encoding

## 次の実装候補

1. `jockey_id` / `trainer_id` の categorical/OOF encoding を設計し、ID の大小を数値として読ませない。
2. odds time series を `0B31/0B41` から追加し、closing-line value と market movement feature を作る。
3. payout/pool を ingest し、backtest settlement と控除率検証をより厳密にする。
4. field_size mismatch の skipped race を調査し、取消/除外 runner の扱いを改善する。
5. LightGBM の calibration と out-of-fold prediction store を作り、stacking/ensemble の Level 0 出力として使える形にする。
