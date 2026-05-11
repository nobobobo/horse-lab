# Horse Lab アーキテクチャ

## 目的

Horse Lab は、日本競馬の予想を単一モデルではなく複数の専門モデルの集合知として扱う研究・シミュレーション基盤。最終出力は「勝ちそうな馬」ではなく、calibrated probability、market odds、資金制約を合わせた期待値ベースの bet decision とする。

## システム構成

1. **Data layer**
   - JRA-VAN raw dump を immutable に保持し、staging CSV、replay dataset、model artifact を versioned にする。
   - race、entry、result、odds snapshot/time series、payout/pool、horse/person metadata を runner 単位で join できる形に正規化する。
   - すべての学習特徴量は `as_of` と `feature_version` を持つ。

2. **Feature layer**
   - runner-level feature を基本粒度にする。
   - 近走、距離/馬場/クラス適性、斤量、枠、馬体重、騎手/調教師 stats、market movement、CLV、pool を point-in-time safe に作る。
   - target race の result、締切後 odds、確定後人気を混入させない。

3. **Level 0 specialist models**
   - Market model: 単勝 odds から market-implied probability を作る基準線。
   - Tabular form model: LightGBM/CatBoost。entry、過去走、person stats、market movement の非線形相互作用を担当。
   - Track/pace/bias model: 開催・馬場・脚質バイアスを担当。
   - Pedigree/suitability model: 血統、距離、馬場、成長曲線を担当。
   - Text/Transformer model: 調教コメント等の非構造データが揃ってから追加。

4. **Level 1 meta learner**
   - Level 0 の out-of-fold prediction のみを入力にする。
   - MVP は logistic regression / isotonic calibration から始める。
   - 将来は LightGBM ranker、Dirichlet calibration、Plackett-Luce 系へ拡張する。

5. **Backtest / paper trading**
   - bankroll、fractional Kelly、minimum edge、max stake、daily stop loss、odds timing を明示する。
   - `bet_decisions.csv` に賭けた理由/見送った理由を runner 単位で残す。
   - 的中率ではなく log loss、Brier、ECE、ROI、drawdown、turnover、CLV、odds band 別の期待値を見る。

6. **MLOps**
   - Windows worker は JV-Link extraction 専用。raw dump は S3-first で保存し、Windows local disk には溜めない。
   - macOS/Linux 側は S3 pull、ingest、replay build、training、backtest、report を CLI/DAG 化する。
   - model registry には model version、feature version、training window、calibration/backtest metrics を保存する。

## ロードマップ

### Phase 0: 契約定義 完了

- `schemas.py` と model interface を定義。
- `race_id`、`runner_id`、`as_of`、`feature_version`、`model_version` を全レイヤーに通した。

### Phase 1: 履歴 replay 完了

- CSV repository と replay pipeline を実装。
- JRA-VAN RA/SE/O1 の minimal mapper を実装。
- `RACE` daily backfill から replay-ready dataset を作成可能。

### Phase 2: Baseline 完了

- market-implied baseline と LightGBM baseline を実装。
- time-series split で market vs LightGBM を比較。
- 初回結果は market が LightGBM より良く、LightGBM は `entry_win_odds` 依存が強い。詳細は `docs/baseline_analysis.md`。

### Phase 3: Backtest / Paper Trading 完了

- `BacktestConfig` に Kelly、odds filter、race stake cap、daily stop loss、odds timing を追加。
- `BetRecord` と `BetDecision` を分離。
- `backtest_config.json` と `bet_decisions.csv` を出力。
- `market-replay` と `jravan-daily-market-replay` が backtest artifact 出力に対応。

### Phase 3.5: Data Enrichment 実装完了

Phase 4 の ensemble に入る前に、各 Level 0 が同じ market signal をなぞらないようデータを厚くする。

- `odds_timeseries.csv`: selected complete races の全 win odds snapshot を保存。
- `payouts.csv`: 単勝は result と latest odds の proxy を残しつつ、JRA-VAN `HR` 公式払戻と `H1` 票数/プールを取り込む。馬連は `HR` の公式 payout を simulation settlement に使う。
- market movement features: opening/latest/min/max odds、snapshot count、implied probability movement、latest pool。
- expanded past-performance features: top3 rate、平均賞金、平均走破時計、同距離成績、距離変化、前走条件、馬体重。
- jockey/trainer historical stats: target race より前の run count / win rate。
- categorical-safe person IDs: `jockey:01020` / `trainer:04050` のように ID を順序数として読ませない。
- `0B41` / `0B42` historical backfill: 単勝/複勝/枠連と馬連の realtime odds raw を S3 に日付単位で保存する。単勝モデルの replay には `0B41` の win snapshot だけを合流し、`0B42` は馬連/多点式 simulation 用の raw/staging として分離する。
- `RACE` staging と realtime odds staging は `jravan-build-replay-dataset --odds-staging-dir ...` で合流する。race/entry/result は `RACE` を正本、odds は追加 staging から補強する。
- `jravan-replay-v2` 実データ build と LightGBM training は完了。`0B41` 合流後は runner あたり中央値 152 snapshot まで増え、movement 系特徴量の検証が可能になった。

### Phase 3.6: Data QA / 馬連 Simulation 実装完了

- replay dataset QA CLI を追加し、file size、row count、race coverage、odds snapshot 分布、payout source 分布を JSON 化する。
- 2025-05-10 から 2026-05-03 の complete replay dataset は `races=3283`、`feature_rows=45287`、`odds_timeseries=7054152`、`payouts=48582`。
- `HR/H1` 由来の official payout/pool は `official_rows=6581`、うち馬連 `quinella=3295`。
- LightGBM ablation は `full`、`no_market`、`no_movement` を同一 split で比較する。
- `0B42` 馬連 odds と official payout を使う `quinella-sim` を追加。現在は favorite / positive-edge strategy の settlement 検証が目的。

### Phase 3.7: Phase 4 足場 完了

- `0B42` O2 staging から pair-level 馬連 replay dataset を作る CLI を追加。
- 2日分 smoke では `races=72`、latest pair odds `6559`、odds time series `954372`、official payout `68` を生成できた。
- 全量評価向けに、`0B42` raw から latest pair odds だけを直接作る compact builder を追加。full time series CSV を省略して disk / memory を抑える。
- Stacking 用の OOF prediction store schema を追加。`prediction_role`、`fold_id`、train/validation window、`feature_version`、model version、target、probability、metadata を CSV に保存する。
- `level0-oof` CLI で monthly expanding window の out-of-fold prediction を生成する。
- `stacking-build-meta-dataset` CLI で OOF prediction を runner-level の meta learner 入力テーブルへ pivot する。

### Phase 4: Ensemble 完了

Phase 4 の runner-level win probability stacking は一通り完了。採用判断は単発 holdout ではなく walk-forward gate に寄せる。

- 2025-10-01 から 2026-05-09 の OOF artifact を生成済み。
- Folds: `202510` から `202605`
- Models: `market`, `lightgbm_full`, `lightgbm_no_market`
- Stored predictions: `82410`
- Meta dataset rows: `27470`
- Artifact:
  - `artifacts/oof/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/oof_predictions.csv`
  - `artifacts/stacking/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/meta_features.csv`
  - `artifacts/phase4_study/daily_backfill_RACE_20250509_20260509_with_payouts_v1_20251001_20260509/phase4_study_report.json`

実装済み study:

- `stacking-train-meta`: dependency-light logistic meta learner。
- `stacking-search-blend`: Level 0 予測列の convex blend weight search。
- `stacking-phase4-study`: walk-forward 評価、fold 別 weight/coefficient、segment 別 metrics を出力。

Walk-forward 結果は、`202510` から `202512` を最小 meta train folds とし、`202601` から `202605` を順次 holdout にしたもの。

| Method | Log loss | Brier |
| --- | ---: | ---: |
| Convex blend | 0.20136 | 0.05643 |
| Market-implied | 0.20139 | 0.05643 |
| Logistic meta | 0.20204 | 0.05658 |
| LightGBM full | 0.20624 | 0.05741 |
| LightGBM no-market | 0.22539 | 0.06111 |

Convex blend が market をわずかに上回ったため、Phase 4 の候補としては `promote_ensemble_candidate`。ただし改善幅は log loss で `0.000026` と小さいため、即資金投入ではなく paper trading gate へ進める。実運用では market を主軸に、LightGBM full を 0-5% 程度混ぜる restrained blend を候補にする。no-market model は単体では弱く、blend weight も 0 になったため、現時点では診断用に留める。

### Phase 5: 自動化

- S3 raw pull から ingest、replay build、feature generation、training、inference、paper trading report までを DAG 化する。
- 本番前に paper trading 期間を置き、ROI より calibration と CLV を優先監視する。

## JRA-VAN / S3 運用方針

- Primary source は JRA-VAN Data Lab.。
- JV-Link は Windows worker で実行する。
- raw dump、stdout/stderr、manifest は S3 に upload し、成功後に Windows local file を削除する。
- `RACE` など蓄積系は `JVOpen`、`0B31/0B41` など realtime odds は `JVRTOpen` を使う。
- 過去1年の MVP backfill は `0B41` と `0B42` を日次粒度で S3 に保存する。Windows local disk の肥大化を避けるため、各日付を upload したら local output を削除する。
- 将来の三連単/三連複 simulation 向けに `0B30` は保存期間内に日次または時間帯別で自動蓄積する。推奨は EventBridge Scheduler + SSM Run Command + EC2 start/stop で、worker は収集時だけ起動する。
- `HR` 払戻と `H1` 票数は `RACE` raw に含まれるため、settlement と pool 検証は race backfill から復元する。過去の odds movement は realtime 系を保存していない期間は復元できない。
- Mac 開発環境は必要な run だけを S3 から local `data/raw/` に sync する。
- `data/raw/`、`data/interim/`、`data/processed/`、`artifacts/` は git 管理しない。

## 主要 CLI

```bash
horse-lab jravan-s3-pull-raw <run_id> data/raw/jravan
horse-lab jravan-ingest-dir data/raw/jravan/<run_id> data/interim/jravan/<run_id>
horse-lab jravan-build-replay-dataset data/interim/jravan/<run_id> data/processed/jravan/<run_id>/replay
horse-lab jravan-build-replay-dataset data/interim/jravan/<race_run_id> data/processed/jravan/<run_id>/replay --odds-staging-dir data/interim/jravan/<odds_run_id>
horse-lab market-replay data/processed/jravan/<run_id>/replay --start-date YYYY-MM-DD --end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS
horse-lab lightgbm-train data/processed/jravan/<run_id>/replay artifacts/lightgbm/<run_id> --train-end-date YYYY-MM-DD --valid-start-date YYYY-MM-DD --valid-end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS
horse-lab lightgbm-ablation data/processed/jravan/<run_id>/replay artifacts/lightgbm_ablation/<run_id> --train-end-date YYYY-MM-DD --valid-start-date YYYY-MM-DD --valid-end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS
horse-lab level0-oof data/processed/jravan/<run_id>/replay artifacts/oof/<run_id> --validation-start-date YYYY-MM-DD --validation-end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS
horse-lab stacking-build-meta-dataset artifacts/oof/<run_id>/oof_predictions.csv data/processed/jravan/<run_id>/replay/results.csv artifacts/stacking/<run_id>
horse-lab stacking-train-meta artifacts/stacking/<run_id>/meta_features.csv artifacts/stacking_meta/<run_id>
horse-lab stacking-search-blend artifacts/stacking/<run_id>/meta_features.csv artifacts/stacking_blend/<run_id>
horse-lab stacking-phase4-study artifacts/stacking/<run_id>/meta_features.csv data/processed/jravan/<run_id>/replay/races.csv artifacts/phase4_study/<run_id>
horse-lab jravan-data-qa data/processed/jravan/<run_id>/replay artifacts/data_quality/<run_id>/report.json
horse-lab jravan-build-quinella-replay-dataset data/interim/jravan/<o2_run_id> data/processed/jravan/<run_id>/replay/payouts.csv data/processed/jravan/<quinella_run_id>/replay --start-date YYYY-MM-DD --end-date YYYY-MM-DD
horse-lab jravan-build-quinella-replay-dataset-raw data/raw/jravan data/processed/jravan/<run_id>/replay/payouts.csv data/processed/jravan/<quinella_run_id>/replay --start-date YYYY-MM-DD --end-date YYYY-MM-DD --pattern 0B42_jvgets.txt
horse-lab quinella-sim data/interim/jravan/<o2_run_id>/odds.csv data/processed/jravan/<run_id>/replay/payouts.csv artifacts/quinella_sim/<run_id> --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

## 避けるべき落とし穴

- random split で評価する。
- 締切後 odds や確定人気を training feature に混ぜる。
- market odds 由来 baseline の `bet_records=0` を異常と誤解する。
- ID を numeric として学習させる。
- payout/pool を無視して theoretical edge だけを見る。
- full Kelly をそのまま使う。
