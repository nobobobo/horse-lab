# Phase 3.5 Data Enrichment Design

## Purpose

Phase 4 の ensemble に入る前に、各 Level 0 モデルが同じ薄い market signal をなぞらないよう、replay dataset に必要な情報量を増やす。

この phase は新しい本番データ取得そのものではなく、JRA-VAN raw/staging から以下の enriched data を保持・特徴量化できる pipeline を作る。

- odds time series
- payout / pool settlement proxy
- market movement features
- expanded past-performance features
- jockey / trainer historical stats
- categorical-safe ID features

## Scope

In scope:

- `jravan-build-replay-dataset` が latest odds だけでなく、selected complete races の win odds time series を追加出力する。
- replay dataset が `payouts.csv` を追加出力する。初期版は latest win odds と result から derived payout を作る。
- runner feature に market movement と pool size features を追加する。
- past-performance feature を増やし、距離/馬場/クラス/人気/過去 odds の情報を厚くする。
- jockey/trainer historical run count and win rate を point-in-time safe に作る。
- `jockey_id` / `trainer_id` を numeric として学習させないよう categorical-safe string にする。
- docs に Phase 3.5 の位置づけと Phase 4 前の data requirements を反映する。

Out of scope:

- JV-Link から新しい DataSpec を実際に取得する AWS/Windows 実行。
- 公式払戻 record layout の完全 mapper。初期版は odds/result 由来の settlement proxy と明記する。
- place/exacta/trifecta など単勝以外の本格 backtest。
- bloodline/text/pace model の実装。

## Architecture

`odds.csv` は既存互換のため latest win quote per runner を保持する。新しく `odds_timeseries.csv` を追加し、complete races に属する全 win quotes を保存する。これにより既存 `market-replay` と LightGBM training は壊さず、Phase 3 backtest の `closing` / `minutes_before_start` policy に必要な材料を保持できる。

`payouts.csv` は初期版では official payout record ではなく、winner result と selected latest win odds から `payout_jpy_per_100` を算出する settlement proxy とする。将来 JRA-VAN の払戻 record を mapper に追加したら同じ CSV contract を official source に置き換える。

Feature generation は runner-level の `features.csv` に集約する。過去走・騎手・調教師の集計は target race の start time より前の race だけを使い、time leakage を避ける。

## Acceptance Criteria

- `jravan-build-replay-dataset` output に `odds_timeseries.csv` と `payouts.csv` が追加される。
- replay report に odds time series と payout rows の counts が入る。
- existing tests and CLI summaries remain backward-compatible where possible.
- feature rows include market movement, pool, expanded past-performance, jockey/trainer stats.
- LightGBM feature inference treats jockey/trainer IDs as categorical strings, not numeric IDs.
- docs/architecture.md reflects Phase 3.5 and removes or compresses outdated repetition.
- Full test suite passes.
