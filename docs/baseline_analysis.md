# Baseline Analysis

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
- LightGBM 実学習: 未実行

結論として、Phase 2 は完走可能。追加データなしでも、現 dataset で初回 LightGBM baseline を train/validation できる。残タスクは `ml` optional dependency を入れて、時系列 split で `horse-lab lightgbm-train` を実行し、market baseline と比較すること。

推奨する最初の split:

- Train: 2025-05-09 から 2026-02-28
- Validation: 2026-03-01 から 2026-05-08
- 2026-05-09 以降の未確定/当日 race は validation から除外

Phase 2 の完了条件:

- LightGBM validation log loss / Brier / calibration が保存されている
- market baseline と同じ validation window で比較できる
- LightGBM の特徴量重要度または feature contribution を確認できる
- positive edge が出る場合も、まず paper backtest と calibration を優先して確認する

## 次の実装候補

1. `lightgbm` optional dependency を導入し、現 replay dataset で初回 training を実行する。
2. market baseline と LightGBM baseline を同じ validation window で比較する summary doc を作る。
3. odds time series を `0B31/0B41` から追加し、closing-line value と market movement feature を作る。
4. payout/pool を ingest し、backtest settlement と控除率検証をより厳密にする。
5. field_size mismatch の skipped race を調査し、取消/除外 runner の扱いを改善する。
