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

## Planned JRA-VAN Ingest Boundary

JRA-VAN JV-Link should run in a separate Windows environment and export raw JV-Data dumps or normalized staging files. The macOS/Linux Python code owns validation, parsing, and conversion into the canonical `Race`, `Entry`, `OddsQuote`, `Result`, and `FeatureRow` schemas.

- `horse_lab.data.jravan` contains platform-independent raw record helpers for JV-Data text dumps.
- `data/raw/`, `data/interim/`, and `data/processed/` are ignored by git because real vendor data should stay local.
- Web scraping remains a fallback, not the primary ingestion path, because stable point-in-time odds snapshots matter more than quick page extraction.

## Implemented JRA-VAN Staging Pipeline

The first real-data ingest path is now a local staging pipeline for JV-Data text dumps.

- `horse_lab.data.jravan.ingest_jvdata_file_to_staging` reads a CP932 JV-Data dump, maps supported records, and writes canonical `races.csv`, `entries.csv`, `results.csv`, and `odds.csv`.
- Raw JV-Data stays in CP932/Shift-JIS. Human-readable inspection copies can be converted to UTF-8, but parsing should always read the raw bytes with `encoding="cp932"` so fixed-width byte offsets remain valid.
- The RA/SE mapper uses official JV-Data 4.9.0.1 byte positions for the minimal fields needed by the canonical schema. The O1 mapper also uses the official single/place/bracket odds record layout and currently expands the win-odds block into per-runner `OddsQuote` rows.
- `horse_lab.data.jravan.map_jvdata_records` supports the RA/SE/O1 slice and records unsupported record types as skipped records by default.
- Duplicate RA/SE keys and duplicate O1 quote keys use last-record-wins semantics, which gives deterministic behavior for dumps containing later updates.
- Mapping failures include source path and line number so bad vendor rows can be quarantined without guessing.

Example:

```python
from horse_lab.data.jravan import ingest_jvdata_file_to_staging

export = ingest_jvdata_file_to_staging(
    "data/raw/jravan/20260508/jvdata.txt",
    "data/interim/jravan/20260508",
)
print(export.csv_paths)
```

CLI:

```bash
horse-lab jravan-ingest \
  data/raw/jravan/20260508/jvdata.txt \
  data/interim/jravan/20260508

horse-lab jravan-preview \
  data/raw/jravan/20260508/jvdata.txt \
  data/raw/jravan/20260508/jvdata.utf8.txt
```

## JRA-VAN Data Acquisition Plan

Primary path:

1. Use JRA-VAN Data Lab. and the official SDK/JV-Link on a Windows environment.
2. The Windows worker downloads JV-Data and exports raw text dumps by date and data kind, for example `data/raw/jravan/YYYYMMDD/*.txt`.
3. Sync those dumps to the Mac development environment.
4. Run `ingest_jvdata_file_to_staging` to produce repository-compatible staging CSVs.
5. Use `JVOpen` for accumulated data such as `RACE`, and `JVRTOpen` for realtime data such as `0B30`, `0B31`, and `0B41`.
6. Add scheduled O1 extraction so the staging pipeline can populate `odds.csv` from real tote snapshots.

Current Windows worker probes:

- `JvLinkDump.exe`: accumulated JV-Data via `JVOpen`, currently used for `RACE`.
- `JvLinkRtDump.exe`: realtime JV-Data via `JVRTOpen`, currently verified for NHK Mile Cup race key `2026051005020611`.
- Verified realtime outputs: `0B31` latest single/place/bracket odds, `0B30` all-bet odds, and `0B41` time-series single/place/bracket odds.
- `horse-lab jravan-ingest-dir data/raw/jravan data/interim/jravan/YYYYMMDD` combines multiple raw dumps into one staging dataset.
- `Invoke-JvLinkRtRaceList.ps1` accepts race keys and collects realtime odds for multiple races into per-race raw dump directories.

Operational options for the Windows worker:

- Short term: borrow any Windows machine or use a Windows VM/cloud instance only for JV-Link extraction.
- Medium term: keep a small scheduled Windows task that writes raw dumps to a shared folder.
- Later: wrap the worker behind a thin service or artifact handoff, but keep vendor authentication and JV-Link calls outside the modeling code.

Web scraping remains a fallback only for exploratory checks. It is weaker for this project because it is more brittle, may not preserve historical point-in-time odds snapshots, and can create legal/terms-of-use risk.
