# 追加モデル計画

## 現在地

現在の best candidate は market-implied probability を 95%、LightGBM full を 5% 混ぜる restrained convex blend。calibration は market を壊していないが、edge はほとんど出ていない。

したがって次の開発は「市場そのものを当てる」ではなく、以下の二つを狙う。

- market が過小評価している runner / race segment を見つける。
- 単勝以外、特に馬連・三連系の組み合わせ確率を作る。

## 優先順位

### 1. Market Calibration Model

目的: odds band、venue、surface、distance、field size 別に market probability を補正する。

実装:

- 入力: market-implied probability、odds movement、pool、race condition、field size。
- モデル: isotonic / Platt / beta calibration。
- 評価: log loss、ECE、odds band 別 expected value。

採用条件:

- Walk-forward で market より ECE が改善。
- Log loss を悪化させない。
- Paper trading で CLV が悪化しない。

初回結果:

- 全体一律の Platt-style calibration は ECE を改善したが、log loss は market baseline とほぼ同等で微悪化。
- 現時点では採用せず、segment-specific calibration の候補として継続。
- 次の改善方向は odds band、venue、surface、field size を分けた calibration。

### 2. CatBoost / Categorical Tabular Model

目的: jockey/trainer/venue/grade/surface など categorical interaction を LightGBM より自然に扱う。

実装:

- Level 0 名: `catboost_full`
- 入力: `jravan-replay-v2` の runner features。
- 比較: `lightgbm_full`、`lightgbm_no_market`、`market`。

採用条件:

- no-market でも LightGBM no-market を上回る。
- blend weight が安定して 0 より大きい。

### 3. Residual / Overlay Model

目的: market probability を直接置き換えず、market と実績の residual を予測する。

実装:

- target: `label - market_probability` または market logit residual。
- 出力: residual を market logit に小さく足して race 内正規化。
- feature: form/person/history/bias。market odds は強制的に除外または低重みにする。

採用条件:

- Restrained blend より positive edge decision の質が改善。
- Segment 別の過剰補正が出ない。

2026-05-13 の `jravan-replay-v4` では identity-derived segment history feature を追加した。`no_market` LightGBM は旧 v2 より改善したが、`full` は悪化したため、これらの feature は market と混ぜる前に residual / overlay model 側で小さく使うのが第一候補。

同日の `jravan-replay-v5` では `horse_symbol_code`、`coat_color_code`、`trainer_affiliation_code`、`race_grade_group`、`race_title_type` など既存 `RACE` raw から取れる profile/title feature を追加した。これらは一部 gain を持つが、full model の log loss は改善しなかったため、default model に無条件投入せず、feature selection と segment/residual model 側で扱う。

### 4. Track Bias / Pace Specialist

目的: 開催日・競馬場・馬場・距離の一時的な偏りを捉える。

必要データ:

- 過去同日/同開催の race result。
- 通過順、脚質 proxy、上がり、馬場状態。
- 枠番・馬番別の成績。

実装:

- target race より前の同日 race だけで bias feature を作る。
- feature: inside/outside bias、front/closer bias、track speed index。

採用条件:

- 当日後半 race の calibration / hit ranking が改善。
- 同日 leakage がない。

### 5. Pedigree / Suitability Specialist

目的: market odds や近走成績だけでは拾いにくい距離・馬場・成長曲線の適性を、血統と馬プロファイルから推定する。

必要データ:

- `horse_id` に紐づく sire、dam、damsire、可能なら grandparents。
- 品種、性別、生年月日、馬齢、馬体重推移。
- 父系/母系ごとの距離、馬場、成長時期、競馬場別成績。
- G1/G2/G3 や条件戦など class/title 正規化。

実装:

- 初期は categorical / target-safe aggregate: sire surface win rate、damsire distance bucket top3 rate など。
- raw `horse_id` や sire ID の直接暗記は避け、OOF stats または十分な regularization を使う。
- 将来は sire/dam graph embedding を作り、tabular model の Level 0 に入れる。

採用条件:

- no-market model の log loss / Brier が改善。
- market blend の alpha/weight が 0 より安定して大きい。
- 新馬/若駒/距離延長/馬場替わりの segment で改善する。

現状メモ:

- `RACE/SE` の `breed_code` は現在の one-year dataset では全て `breed:1` で、pedigree signal としては使えない。
- `horse_symbol_code` は分散があり、小さいながら model gain も出た。ただし血統そのものではない。
- 血統 specialist を作るには、`horse_id` をキーにした馬 master / 血統 master の ingest を別途追加する必要がある。

### 6. Rating / Class Specialist

目的: 公式/外部レーティング、斤量、クラス、レースタイトルを使って能力差の prior を作る。

必要データ:

- 公式 rating または独自 Elo / speed figure。
- Race title normalization: G1/G2/G3、Listed、OP、条件戦、未勝利、新馬など。
- 過去走の class transition と着差、時計、上がり。

実装:

- rating delta within race、class up/down、斤量補正、speed figure trend を feature 化。
- market odds を抜いた no-market / residual model の主特徴量にする。

### 7. Pair Probability Model for 馬連

目的: `0B42` と official payout を使い、馬連の pair-level probability を推定する。

実装:

- 入力: runner-level win/place-like probability、pair odds、人気差、同脚質/同枠/騎手厩舎 stats、market pair probability。
- モデル: pair-level LightGBM / logistic。
- settlement: `quinella-sim` と official `HR` payout。

採用条件:

- favorite baseline ROI を上回る。
- market pair-implied baseline より log loss / Brier が改善。
- 欠損 pair / 取消 / 発売停止の扱いが明示される。

### 8. Trifecta / Trio Extension

目的: 三連複・三連単 simulation の基盤を作る。

必要データ:

- `0B30` 全賭式 odds snapshot の日次または時間帯別蓄積。
- official payout/pool。
- runner-level probability と pair model output。

実装順:

1. `0B30` raw を S3-first で毎開催日保存。
2. trio/trifecta odds mapper を追加。
3. 組み合わせ数を抑える candidate generator を作る。
4. simulation は stake cap と max combinations per race を強制する。

## 次に実装する順番

1. **Calibration study**
   - Market calibration の OOF artifact を追加。
   - ECE と log loss の改善を確認する。

2. **CatBoost または categorical LightGBM 改善**
   - 依存追加を許容するなら CatBoost。
   - 依存を増やさないなら jockey/trainer OOF target encoding を先に作る。

3. **Residual model**
   - Market を anchor にして overlay だけを学習する。
   - Phase 4 の Level 0 に `residual_overlay` を追加する。

4. **馬連 pair model**
   - `quinella_0B42_20250510_20260503_full_v1` を学習用 dataset に拡張。
   - favorite baseline ではなく pair probability baseline を作る。

5. **日次運用**
   - Windows worker は `RACE` と `0B30` を S3 に upload。
   - Mac/Linux 側は S3 pull、ingest、daily-paper-trading-run、monitoring summary を実行。

6. **Identity-aware feature rebuild**
   - replay dataset に出す `entries.csv` を正本にする。
   - `horse_id` は直接 feature にせず、OOF horse stats、horse embedding、過去走集約のキーとして使う。
   - 既存 processed dataset は `entries.csv` 追加後に再 build する。
   - 初回 v4 rebuild は完了。same-venue / same-distance / same-grade 履歴は no-market signal としては有効だが、full model では calibration を悪化させたため、次は residual overlay と feature selection で扱う。

7. **Profile/title feature selection**
   - v5 rebuild は完了。`horse_symbol_code`、`coat_color_code`、`trainer_affiliation_code`、race title/grade 系は弱い signal を持つ。
   - `breed_code` は今回 dataset では情報量がない。
   - 次は full model に全部入れるのではなく、no-market/residual model での selected feature subset と regularization を比較する。

## 当面の採用ゲート

- Data integrity: target race より後の情報を特徴量に混ぜない。
- Probability: market baseline より log loss / ECE が悪化しない。
- Simulation: positive edge が増えても ROI/CLV が悪化するなら不採用。
- Stability: fold ごとの weight / coefficient が極端に反転しない。
- Operability: 日次で再現でき、artifact が registry と monitoring に残る。
