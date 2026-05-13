# Horse Lab アーキテクチャ

## 目的

Horse Lab は、日本競馬の予想を単一モデルではなく複数の専門モデルの集合知として扱う研究・シミュレーション基盤。最終出力は「勝ちそうな馬」ではなく、calibrated probability、market odds、資金制約を合わせた期待値ベースの bet decision とする。

## システム構成

1. **Data layer**
   - JRA-VAN raw dump を immutable に保持し、staging CSV、replay dataset、model artifact を versioned にする。
   - race、entry、result、odds snapshot/time series、payout/pool、horse/person metadata を runner 単位で join できる形に正規化する。
   - すべての学習特徴量は `as_of` と `feature_version` を持つ。
   - `runner_id` は race-local ID、`horse_id` は同一馬を日付横断で結ぶ ID。processed replay には `entries.csv` を出し、`runner_id -> horse_id` を追跡する。

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

### Phase 5: Paper Trading / 自動化 完了

Phase 5 は、実賭け前の paper trading gate と model registry を実装済み。

- `model-registry-register-phase4` で Phase 4 の convex blend candidate を registry JSON に登録する。
- `paper-trading-run` で walk-forward predictions を paper trading replay し、`paper_predictions.csv`、`paper_trading_report.json`、`clv_report.csv`、`backtest/bet_decisions.csv` を出力する。
- Registry artifact: `artifacts/model_registry/phase5_convex_blend_candidate/model_registry.json`
- Paper trading artifact: `artifacts/paper_trading/phase5_convex_blend_202601_202605/paper_trading_report.json`

Phase 5 の conservative paper trading replay では、minimum edge 2% の条件で bet は 1 件のみ。これは convex blend が market に非常に近く、positive edge が薄いことを示す。したがって Phase 5 の結論は「live bet へ進む」ではなく、「paper trading candidate として監視を始める」。次は締切前 odds snapshot を使う live-like daily inference を回し、CLV と calibration を週次で見る。

### Phase 6: Daily Paper Trading / Monitoring 完了

Phase 6 は、承認済み registry candidate を日次運用に近い形で再学習・推論・紙トレードする足場。

- `daily-paper-trading-run` を追加。指定日の前日までを training window とし、当日 races に対して Level 0 を生成し、registry の serving weights で candidate prediction を作る。
- 生成物は `level0_predictions.csv`、`candidate_predictions.csv`、`daily_paper_trading_report.json`、および通常の `paper/` artifacts。
- `paper-trading-monitoring-summary` を追加。Phase 5 の paper report と Phase 6 の daily report をまとめ、probability/backtest/decision/CLV を aggregate する。
- 実データ smoke: `2026-05-03` の 35 races / 496 runners で daily run が成功。
- Monitoring artifact: `artifacts/paper_trading_monitoring/phase6_summary.json`

2026-05-03 の daily paper trading は market に近い restrained blend のため、minimum edge 2% では bet 0 件。これは Phase 5 と整合的で、直ちに live bet するシグナルではない。今後は毎開催日の paper trading を積み、CLV、calibration drift、segment drift を週次で見る。

追加モデルの計画は `docs/model_roadmap.md` に分離した。次の主戦場は「market を壊さずに残差を拾う model」と「馬連/三連系の pair/tuple probability model」。

### Phase 7: Data / Model Expansion 進行中

Phase 7 は、データの説明可能性、追加モデル、fetch 自動化を並行して進める段階。

- `docs/data_catalog.md` を追加し、source、feature provenance、identity、data gaps を整理。
- replay dataset に `entries.csv` を追加。これにより processed 側でも `runner_id -> horse_id` を復元できる。
- `jravan-data-qa` に `dataset_manifest`、feature provenance、identity coverage を追加。
- `stacking-market-calibration-study` を追加。market-implied probability に Platt-style calibration をかけ、walk-forward で market baseline と比較する。
- 初回 study では ECE は改善したが log loss はほぼ同等で微悪化したため、採用判断は `keep_market_baseline`。
- Windows 側に `Invoke-JvLinkAutomatedFetchToS3.ps1` を追加。`RACE` と `0B30/0B41/0B42` を S3-first で取る scheduled job の入口にする。
- Windows Task Scheduler 用に `Register-JvLinkAutomatedFetchTask.ps1` を追加。短期 smoke はこれで回せるが、本番寄り運用は EventBridge Scheduler + SSM + EC2 start/stop を正本にする。
- `jravan-replay-v4` を build。same-venue、same-distance、same-grade など、`horse_id` を grouping key にした identity-derived segment history features を追加した。
- v4 QA は `races=3283`、`feature_rows=45287`、`odds_timeseries=7054152`、`unique_horses=11631`、`missing_horse_id=0`、`feature_count=63`。
- v4 LightGBM ablation では `no_market` が旧 v2 よりわずかに改善したが、`full` は旧 v2 より悪化。追加 feature は素材として残し、採用は feature selection / residual overlay / segment calibration 側で制御する。
- 追加データ方針として、競馬場/回り/天候/grade/年齢/体重/性別/近走は既存 dataset から強化し、rating、pedigree、parents/grandparents、詳細通過順/上がり/着差は追加 ingest 対象にする。
- Residual overlay study と segment-specific calibration study を追加。market を置き換えるのではなく、market の歪みを小さく補正できるかを walk-forward で検証する。

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
- Data catalog と identity 方針は `docs/data_catalog.md` を正本にする。

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
horse-lab stacking-market-calibration-study artifacts/stacking/<run_id>/meta_features.csv artifacts/market_calibration/<run_id>
horse-lab stacking-residual-overlay-study artifacts/stacking/<run_id>/meta_features.csv artifacts/residual_overlay/<run_id>
horse-lab stacking-segment-calibration-study artifacts/stacking/<run_id>/meta_features.csv artifacts/segment_calibration/<run_id> --segment-name market_probability_band
horse-lab stacking-phase4-study artifacts/stacking/<run_id>/meta_features.csv data/processed/jravan/<run_id>/replay/races.csv artifacts/phase4_study/<run_id>
horse-lab model-registry-register-phase4 artifacts/phase4_study/<run_id>/phase4_study_report.json artifacts/model_registry/<candidate>/model_registry.json
horse-lab paper-trading-run artifacts/phase4_study/<run_id>/walkforward_predictions.csv data/processed/jravan/<run_id>/replay artifacts/paper_trading/<candidate> --method convex_blend --as-of YYYY-MM-DDTHH:MM:SS
horse-lab daily-paper-trading-run data/processed/jravan/<run_id>/replay artifacts/model_registry/<candidate>/model_registry.json artifacts/daily_paper_trading/<run_id> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS
horse-lab paper-trading-monitoring-summary artifacts/paper_trading_monitoring/<run_id>/summary.json artifacts/paper_trading/<candidate>/paper_trading_report.json artifacts/daily_paper_trading/<run_id>/daily_paper_trading_report.json
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
