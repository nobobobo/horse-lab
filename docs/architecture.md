# Horse Lab アーキテクチャ

## 目的

日本競馬の予想を単一モデルに閉じず、複数の専門モデルの集合知として扱う。最終アウトプットは「勝ちそうな馬」ではなく、確率、オッズ、資金制約を合わせた期待値ベースのベッティング判断とする。

## 全体アーキテクチャ

1. データ層
   - race、entry、result、odds snapshot、payout、horse、jockey、trainer、pedigree、weather、track condition を正規化して保存する。
   - すべての特徴量は `as_of` を持ち、レース前に取得できた情報だけで生成する。
   - 生データは immutable、加工済み特徴量は versioned にする。

2. 特徴量層
   - runner 単位の特徴量を基本粒度にする。
   - 例: 近走指数、距離適性、馬場適性、競馬場/コース適性、枠順、斤量、休養間隔、騎手/調教師、血統、直前オッズ変化、当日馬場バイアス。
   - point-in-time join を標準化し、データリークを防ぐ。

3. Level 0 専門モデル
   - Tabular form model: LightGBM/CatBoost。近走、枠、斤量、騎手、調教師、オッズなどの非線形相互作用を担当。
   - Pace/track-bias model: 階層ベイズまたは状態空間モデル。当日/開催内の馬場傾向や脚質バイアスを担当。
   - Pedigree/suitability model: ベイズ推定、graph embedding、または factorization。血統、距離、馬場、成長曲線を担当。
   - Market model: オッズから market-implied probability を作り、過剰人気/過小人気の補正を担当。
   - Text/Transformer model: 調教コメント、厩舎コメント、レース映像/ラップ文脈など非構造データが揃ってから追加。

4. Level 1 メタ学習器
   - Level 0 の out-of-fold prediction だけを学習入力にする。
   - 出力は runner ごとの calibrated win/place probability。
   - MVP は logistic regression / isotonic calibration から始め、後で LightGBM ranker、Dirichlet calibration、Plackett-Luce 系へ拡張する。

5. ベッティング・バックテスト層
   - 過去オッズの時点別 snapshot を使い、締切直前、一定分前、直前変動ありなどを再現する。
   - 期待値は `p * odds - 1` を基本にし、控除率とプール方式の影響を別途補正する。
   - Kelly fraction は `f = (p * odds - 1) / (odds - 1)`。実運用では fractional Kelly、上限 stake、日次損失制限、レース分散制御を必須にする。

6. MLOps 層
   - Batch training: Airflow/Prefect/GitHub Actions などで日次または開催単位に実行。
   - Model registry: model version、feature version、training window、calibration metrics、backtest metrics を保存。
   - Live inference: レース前データ取得、特徴量生成、Level 0 推論、stacking、bet sizing、通知/出力を DAG 化。
   - Monitoring: 的中率ではなく calibration、Brier score、log loss、ROI、max drawdown、turnover、オッズ帯別の期待値を監視する。

## MVP ロードマップ

### Phase 0: 契約定義

- schema と model interface を固定する。
- race_id、runner_id、as_of、feature_version、model_version を全レイヤーに通す。
- MVP の対象券種は単勝に限定し、複勝は schema だけ先に用意する。

### Phase 1: 履歴データセット

- 過去レース、出走表、結果、単勝オッズ snapshot を取り込む。
- runner 単位の学習テーブルを作る。
- 初期特徴量は 20-50 個に抑える。

### Phase 2: 最初の baseline

- market-implied probability baseline と LightGBM/sklearn baseline を作る。
- time-series split で評価し、random split は使わない。
- 評価指標は log loss、Brier score、calibration curve、オッズ帯別 ROI。

### Phase 3: バックテストエンジン

- bankroll、stake sizing、odds timing、bet filters を明示したシミュレーターを作る。
- fractional Kelly、minimum edge、maximum stake per race、daily stop loss を実装する。
- closing odds と pre-race odds の差分を比較する。

### Phase 4: Ensemble

- Level 0 を 2-4 モデルに増やす。
- out-of-fold prediction store を作る。
- Level 1 meta learner と calibration を導入する。

### Phase 5: 自動化

- データ取得から推論までを CLI/DAG 化する。
- 推論結果を race card、probability、fair odds、market odds、edge、stake として出力する。
- 本番前に paper trading 期間を置く。

## 最初に避けるべき落とし穴

- 着順ラベルを作る時点で未来情報が混入すること。
- 締切後オッズや確定人気を学習時に使うこと。
- 的中率だけでモデルを評価すること。
- Kelly を full Kelly で運用すること。
- オッズデータの取得時刻を無視すること。

## 実装済み MVP baseline

最初の実行可能な baseline は依存を軽くし、単勝のみを対象にしている。

- `horse_lab.models.market.MarketImpliedProbabilityModel` は締切前の最新単勝オッズを race 内で正規化し、runner ごとの market-implied probability に変換する。
- `horse_lab.betting.kelly` は edge、full Kelly fraction、fractional Kelly stake fraction、円建て stake を計算する。
- `horse_lab.backtesting.simulator.BacktestSimulator` は過去の prediction、odds、result を使って runner-level の単勝バックテストを行う。
- `horse_lab.evaluation.metrics` は ROI、hit rate、turnover、max drawdown に加えて、log loss、Brier score、calibration bins を集計する。
- `horse_lab.data` と `horse_lab.features` は storage と point-in-time feature generation の protocol を定義する。

## 実装済み履歴 replay

最初の履歴 replay 層は、ローカル CSV fixture と読み取り専用 CSV repository で構成している。

- `horse_lab.data.csv_parsing` は CSV row を domain schema に変換する。
- `horse_lab.data.csv_repositories` は race、odds、result、feature repository をローカルファイル上に実装する。
- `sample_data/` は 2 レース、4 runners、複数の odds timestamp、point-in-time feature row を含む deterministic fixture を提供する。
- `horse_lab.pipelines.replay.run_market_replay` は repository、market-implied baseline、Kelly backtest、probability calibration を接続する。

replay は `as_of` 以前の runner ごとの最新単勝オッズを使う。これは point-in-time safe だが、現時点では「全 runner が同一 timestamp の tote snapshot」を選ぶ実装ではない。

JRA-VAN Data Lab. は JRA データの本番向け primary source とする。今後の JRA-VAN adapter は同じ repository protocol を実装し、データソースが変わっても modeling/backtesting logic は変えない。

## JRA-VAN ingest 境界

JRA-VAN JV-Link は Windows 環境で実行し、raw JV-Data dump または normalized staging file を出力する。macOS/Linux 側の Python code は validation、parsing、canonical schema への変換を担当する。

- `horse_lab.data.jravan` は JV-Data text dump を扱う platform-independent な raw record helper を持つ。
- `data/raw/`、`data/interim/`、`data/processed/` は vendor data を含み得るため git 管理しない。
- Web scraping は primary ingest path ではなく exploratory fallback とする。安定した point-in-time odds snapshot が必要なため、ページ抽出より JV-Link/S3 artifact を優先する。

## 実装済み JRA-VAN staging pipeline

最初の real-data ingest path として、JV-Data text dump をローカル staging CSV に変換する pipeline を実装済み。

- `horse_lab.data.jravan.ingest_jvdata_file_to_staging` は CP932 の JV-Data dump を読み、対応 record を map し、canonical な `races.csv`、`entries.csv`、`results.csv`、`odds.csv` を書き出す。
- raw JV-Data は CP932/Shift-JIS のまま保持する。人間確認用の UTF-8 copy は作ってよいが、fixed-width byte offset を保つため parsing は raw bytes を `encoding="cp932"` で読む。
- RA/SE mapper は canonical schema に必要な最小 field について公式 JV-Data 4.9.0.1 の byte position を使う。O1 mapper も公式の単複枠 odds layout を使い、単勝 odds block を runner ごとの `OddsQuote` に展開する。
- `horse_lab.data.jravan.map_jvdata_records` は RA/SE/O1 slice を扱い、未対応 record type は default で skipped record として記録する。
- duplicate RA/SE key と duplicate O1 quote key は last-record-wins とし、後続 update を含む dump でも deterministic に処理する。
- mapping failure には source path と line number を含め、問題のある vendor row を特定できるようにする。

Python API 例:

```python
from horse_lab.data.jravan import ingest_jvdata_file_to_staging

export = ingest_jvdata_file_to_staging(
    "data/raw/jravan/20260508/jvdata.txt",
    "data/interim/jravan/20260508",
)
print(export.csv_paths)
```

CLI 例:

```bash
horse-lab jravan-ingest \
  data/raw/jravan/20260508/jvdata.txt \
  data/interim/jravan/20260508

horse-lab jravan-preview \
  data/raw/jravan/20260508/jvdata.txt \
  data/raw/jravan/20260508/jvdata.utf8.txt
```

## JRA-VAN データ取得計画

Primary path:

1. Windows 環境で JRA-VAN Data Lab. と公式 SDK/JV-Link を使う。
2. Windows worker は JV-Data を短命の working directory に download する。
3. raw text dump、log、manifest をただちに S3 へ upload する。
4. S3 upload 成功後にのみ Windows local file を削除する。Windows volume は data lake ではなく一時 cache とする。
5. 必要な dump だけを S3 から Mac 開発環境へ sync する。
6. `ingest_jvdata_file_to_staging` または `jravan-ingest-dir` で repository-compatible な staging CSV を作る。
7. `jravan-build-replay-dataset` で complete race のみを抽出し、`races.csv`、`features.csv`、`results.csv`、`odds.csv`、`replay_dataset_report.json` を出力する。
8. `RACE` のような蓄積系データは `JVOpen`、`0B30`、`0B31`、`0B41` のようなリアルタイム系データは `JVRTOpen` を使う。
9. staging pipeline が実 tote snapshot から `odds.csv` を作れるよう、O1 extraction を schedule 化する。

現在の Windows worker probe:

- `JvLinkDump.exe`: `JVOpen` 経由の蓄積 JV-Data dump。現在は `RACE` に使用。
- `JvLinkRtDump.exe`: `JVRTOpen` 経由の realtime JV-Data dump。NHK マイルカップ race key `2026051005020611` で検証済み。
- 検証済み realtime output: `0B31` latest single/place/bracket odds、`0B30` all-bet odds、`0B41` time-series single/place/bracket odds。
- `Invoke-JvLinkDumpToS3.ps1`: 蓄積 JV-Link extraction を実行し、run directory を S3 へ upload し、成功後に local file を削除する。
- `Invoke-JvLinkHistoricalRangeToS3.ps1`: 3-6か月などの履歴範囲を指定して `RACE` などの蓄積 DataSpec を S3-first で収集する。
- `Invoke-JvLinkRtRaceListToS3.ps1`: race key list の realtime odds extraction を実行し、S3 upload 後に local file を削除する。
- `Invoke-S3RawUpload.ps1`: local raw directory を manifest 付きで S3 upload し、必要に応じて upload 後に削除する。
- `horse-lab jravan-s3-pull-raw <run_id> data/raw/jravan`: S3 から Mac workspace へ raw run を sync する。まず `--dry-run` で実行される `aws s3 sync` を確認する。
- `horse-lab jravan-ingest-dir data/raw/jravan data/interim/jravan/YYYYMMDD`: 複数 raw dump を 1 つの staging dataset に結合する。
- `horse-lab jravan-build-replay-dataset data/interim/jravan/YYYYMMDD data/processed/jravan/YYYYMMDD/replay`: staging CSV から complete-race replay dataset を作る。
- `horse-lab market-replay data/processed/jravan/YYYYMMDD/replay --start-date YYYY-MM-DD --end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS`: market baseline、Kelly backtest、probability calibration diagnostics を実行する。
- `horse-lab jravan-daily-market-replay <run_id> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --as-of YYYY-MM-DDTHH:MM:SS`: S3 raw sync、staging ingest、replay dataset build、market replay、report 出力を 1 つの日次 smoke workflow として実行する。ローカルに raw がある場合は `--skip-s3-pull` を使う。
- `Invoke-JvLinkRtRaceList.ps1`: 複数 race key の realtime odds を per-race raw dump directory に収集する。

Windows worker の運用方針:

- 短期: Windows VM/cloud instance を JV-Link extraction 専用に使う。
- 中期: Windows scheduled task で raw dump を S3 に書き出す。
- 後期: worker を薄い service または artifact handoff として包む。ただし vendor authentication と JV-Link call は modeling code の外に置く。

6か月分の履歴蓄積を開始する例:

```powershell
& 'C:\horse-lab\scripts\Invoke-JvLinkHistoricalRangeToS3.ps1' `
  -DataSpecs RACE `
  -FromDate 20251108000000 `
  -RunId historical_RACE_20251108_20260508
```

この wrapper は内部で `Invoke-JvLinkDumpToS3.ps1` を呼ぶため、upload 成功後は既定で Windows local file を削除する。Windows volume を data lake にしない方針は維持する。

S3 raw artifact lake:

- Bucket: `s3://horse-lab-jravan-244306245597-apne1/`
- Canonical raw prefix: `raw/jravan/<dataset_or_run_id>/...`
- bucket は private に保ち、public access block、server-side encryption、versioning を有効にする。
- Windows EC2 は instance role で S3 write する。Windows local raw file は temporary とし、upload manifest が S3 に保存された後に削除する。
- Mac 側の開発 path は S3 to local: `aws s3 cp s3://horse-lab-jravan-244306245597-apne1/raw/jravan/<run_id>/... data/raw/jravan/<run_id>/...`。
- replay-ready local path:

```bash
horse-lab jravan-s3-pull-raw \
  <run_id> \
  data/raw/jravan

horse-lab jravan-ingest-dir \
  data/raw/jravan/<run_id> \
  data/interim/jravan/<run_id>

horse-lab jravan-build-replay-dataset \
  data/interim/jravan/<run_id> \
  data/processed/jravan/<run_id>/replay

horse-lab market-replay \
  data/processed/jravan/<run_id>/replay \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --as-of YYYY-MM-DDTHH:MM:SS
```

日次 smoke は上記の分解コマンドを 1 つにまとめた以下のコマンドでも実行できる。

```bash
horse-lab jravan-daily-market-replay \
  <run_id> \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --as-of YYYY-MM-DDTHH:MM:SS
```

出力 report は `data/processed/jravan/<run_id>/market_replay_report.json` に保存する。Windows 側は raw を S3 に即 upload して削除し、Mac 側はこの日次 command で S3 から必要な run だけを取得して検証する。

market baseline は同じ market odds から作った確率を同じ market odds に対して評価するため、positive edge が出ず `bet_records=0` になることがある。この場合でも `market-replay` は `log_loss`、`brier_score`、calibration bins を返す。これにより、非 market alpha model を追加する前に、replay loop の data integrity と probability quality を検証できる。

Web scraping は exploratory check 用の fallback に留める。壊れやすく、historical point-in-time odds snapshot を保存しにくく、規約・法務リスクもあるため、本番 ingest では JV-Link/S3 artifact path を優先する。

## 今後の学習データと特徴量

market baseline の次に学習モデルを作るには、単勝 odds と結果だけでは足りない。まずは JRA-VAN から以下を優先して ingest する。

1. 出走時点で確定している horse/entry 情報
   - 馬齢、性別、斤量、枠番、馬番、騎手、調教師、所属、馬体重、馬体重増減。
   - MVP の tabular baseline に直結する。

2. 過去走履歴
   - 各馬の過去 N 走の着順、距離、馬場、上がり、通過順、タイム、着差、クラス、人気、オッズ。
   - 近走能力、距離適性、馬場適性、ローテーション、クラス変化を作るために必須。

3. race 条件
   - 競馬場、コース、距離、芝/ダート/障害、右左、内外、天候、馬場状態、クラス、頭数。
   - race-level context と course bias の基礎になる。

4. odds time series
   - 締切 60/30/10/5/1 分前、または取得できる範囲の O1/0B31/0B41。
   - 市場の歪み、直前人気変動、closing-line value を見るために必要。

5. payout と pool
   - 払戻、売上 pool、控除率推定。
   - backtest の settlement と market efficiency analysis に必要。

6. pedigree と調教/コメント
   - 血統は距離・馬場適性モデルに有効。調教/コメントは後段の text/Transformer model 用で、MVP 後でよい。

大規模データは最初から 2020 年以降を全部 ingest する必要はない。まず 2-3 か月分で pipeline と feature correctness を固め、その後 1 年、3 年、5 年へ広げる。モデル学習に入る目安は、最低でも数千レース、できれば数万 runner row。LightGBM baseline では 1-2 年分から始め、validation は必ず時系列 split にする。

## 実装済み学習特徴量の第一段

LightGBM baseline に向けた最初の feature engineering として、entry detail と過去走履歴を追加した。

- `entries.csv` は `horse_name`、`sex`、`breed_code`、`coat_color_code`、`trainer_affiliation_code`、`entry_win_odds`、`entry_popularity_rank` を保持する。
- `jravan-build-replay-dataset` の `features.csv` は entry detail、race condition、past performance を同じ runner-level feature row にまとめる。
- `horse_lab.features.PastPerformanceFeatureBuilder` は `entries + races + results + odds` を `horse_id` で結合し、対象 race より前の走歴だけを使って point-in-time safe な rolling features を作る。
- 最初の過去走特徴量は `past_run_count`、`days_since_last_run`、`avg_finish_position_last3`、`best_finish_position_last3`、`win_rate_last5`、`avg_distance_m_last3`、`same_surface_run_count`、`same_surface_win_rate`、`avg_odds_last3`、`last_finish_position`、`last_odds`。
- odds は各過去走の発走前に取得された単勝 odds の最新値だけを使う。対象 race 前に存在していても、過去走の発走後に取得された odds は使わない。
