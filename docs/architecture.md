# Horse Lab Architecture

## 目的

日本競馬の予想を単一モデルに閉じず、複数の専門モデルの集合知として扱う。最終アウトプットは「勝ちそうな馬」ではなく、確率、オッズ、資金制約を合わせた期待値ベースのベッティング判断とする。

## 全体アーキテクチャ

1. Data Layer
   - race、entry、result、odds_snapshot、payout、horse、jockey、trainer、pedigree、weather、track condition を正規化して保存する。
   - すべての特徴量は `as_of` を持ち、レース前に取得できた情報だけで生成する。
   - 生データは immutable、加工済み特徴量は versioned にする。

2. Feature Layer
   - runner 単位の特徴量を基本粒度にする。
   - 例: 近走指数、距離適性、馬場適性、競馬場/コース適性、枠順、斤量、休養間隔、騎手/調教師、血統、直前オッズ変化、当日馬場バイアス。
   - point-in-time join を標準化し、データリークを防ぐ。

3. Level 0 Specialist Models
   - Tabular form model: LightGBM/CatBoost。近走、枠、斤量、騎手、調教師、オッズなどの非線形相互作用を担当。
   - Pace/track-bias model: 階層ベイズまたは状態空間モデル。当日/開催内の馬場傾向や脚質バイアスを担当。
   - Pedigree/suitability model: ベイズ推定、graph embedding、または factorization。血統、距離、馬場、成長曲線を担当。
   - Market model: オッズから market-implied probability を作り、過剰人気/過小人気の補正を担当。
   - Text/Transformer model: 調教コメント、厩舎コメント、レース映像/ラップ文脈など非構造データが揃ってから追加。

4. Level 1 Meta Learner
   - Level 0 の out-of-fold prediction だけを学習入力にする。
   - 出力は runner ごとの calibrated win/place probability。
   - MVP は logistic regression / isotonic calibration から始め、後で LightGBM ranker、Dirichlet calibration、Plackett-Luce 系へ拡張する。

5. Betting & Backtest Layer
   - 過去オッズの時点別 snapshot を使い、締切直前、一定分前、直前変動ありなどを再現する。
   - 期待値: `p * odds - 1` を基本にし、控除率とプール方式の影響を別途補正する。
   - Kelly fraction: `f = (p * odds - 1) / (odds - 1)`。実運用では fractional Kelly、上限 stake、日次損失制限、レース分散制御を必須にする。

6. MLOps Layer
   - Batch training: Airflow/Prefect/GitHub Actions などで日次または開催単位に実行。
   - Model registry: model version、feature version、training window、calibration metrics、backtest metrics を保存。
   - Live inference: レース前データ取得、特徴量生成、Level 0 推論、stacking、bet sizing、通知/出力を DAG 化。
   - Monitoring: 的中率ではなく calibration、Brier score、log loss、ROI、max drawdown、turnover、オッズ帯別の期待値を監視する。

## MVP ロードマップ

### Phase 0: Contract First

- schema と model interface を固定する。
- race_id、runner_id、as_of、feature_version、model_version を全レイヤーに通す。
- MVP の対象券種は単勝に限定し、複勝は schema だけ先に用意する。

### Phase 1: Historical Dataset

- 過去レース、出走表、結果、単勝オッズ snapshot を取り込む。
- runner 単位の学習テーブルを作る。
- 初期特徴量は 20-50 個に抑える。

### Phase 2: First Baseline

- market-implied probability baseline と LightGBM/sklearn baseline を作る。
- time-series split で評価し、random split は使わない。
- 評価指標は log loss、Brier score、calibration curve、オッズ帯別 ROI。

### Phase 3: Backtest Engine

- bankroll、stake sizing、odds timing、bet filters を明示したシミュレーターを作る。
- fractional Kelly、minimum edge、maximum stake per race、daily stop loss を実装する。
- closing odds と pre-race odds の差分を比較する。

### Phase 4: Ensemble

- Level 0 を 2-4 モデルに増やす。
- out-of-fold prediction store を作る。
- Level 1 meta learner と calibration を導入する。

### Phase 5: Automation

- データ取得から推論までを CLI/DAG 化する。
- 推論結果を race card、probability、fair odds、market odds、edge、stake として出力する。
- 本番前に paper trading 期間を置く。

## 最初に避けるべき落とし穴

- 着順ラベルを作る時点で未来情報が混入すること。
- 締切後オッズや確定人気を学習時に使うこと。
- 的中率だけでモデルを評価すること。
- Kelly を full Kelly で運用すること。
- オッズデータの取得時刻を無視すること。

## Implemented MVP Baseline

The first executable baseline is dependency-light and supports win bets only.

- `horse_lab.models.market.MarketImpliedProbabilityModel` converts latest pre-race win odds into normalized race-level probabilities.
- `horse_lab.betting.kelly` calculates edge, full Kelly fraction, fractional Kelly stake fraction, and yen stake size.
- `horse_lab.backtesting.simulator.BacktestSimulator` simulates runner-level win bets over historical predictions, odds, and results.
- `horse_lab.evaluation.metrics` summarizes ROI, hit rate, turnover, and max drawdown.
- `horse_lab.data` and `horse_lab.features` define protocols for storage and point-in-time feature generation.

## Implemented Historical Replay

The first historical replay layer uses local CSV fixtures and read-only CSV repositories.

- `horse_lab.data.csv_parsing` converts CSV rows into domain schemas.
- `horse_lab.data.csv_repositories` implements race, odds, result, and feature repositories over local files.
- `sample_data/` provides deterministic fixtures for two races, four runners, multiple odds timestamps, and point-in-time feature rows.
- `horse_lab.pipelines.replay.run_market_replay` coordinates repositories, the market-implied baseline, and the Kelly backtest simulator.

Replay uses the latest per-runner win odds at or before `as_of`, which is point-in-time safe. It is not yet a coherent same-timestamp tote snapshot selector.

JRA-VAN Data Lab. remains the intended production source for JRA data. Future JRA-VAN adapters should implement the same repository protocols so modeling and backtesting logic do not change when the data source changes.
