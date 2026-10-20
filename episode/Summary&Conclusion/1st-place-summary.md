18-⑤Kaggle実践3 Biohub細胞トラッキング：1位解法のまとめ

<img src="image.png" width="800" />

## Abstract
- biohub 1位解法のまとめ(Public 0.976 / Private 0.977)
- 「幾何学的追跡」から「ビジュアルトラッキング(視覚的追跡)」へのパラダイムシフトにより、CV・LB共に圧倒的スコアで優勝した1位チーム(sersasj)の完全解説。

## 概要
先日、Biohub細胞トラッキングコンペが完了しました。
で、惨敗だった僕の結果(1353位/3800人)はさておき、Kaggleの素晴らしいところは、勝者たちの解法が惜しみなく公開されること。
この学びの宝庫を徹底的に分析し、確実に自身のスキルへと昇華させるべく自分の血肉にしてやりますばい。

## biohub 1位の解法まとめ
1位の方だけでなく、上位陣の人たちはモデルを複数駆使してて、また学習にも工夫が随所にみられました。そして公開されるのは学習後モデルそのものでして、学習に使用したコードやデータ群までは公開されないんですね。ただ上位陣はそれぞれ自身の提出コードの解説ページを公開してくれてて、そこで結構情報を得られたりします。

- 解説のURL
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/1st-place-solution
- ソースコード
https://www.kaggle.com/code/sersasj/biohub-1st-place-inference

---

## プロジェクト構成

1位解法のリポジトリ(`biohub-1st-place-reproduction`)および推論処理で使用されている主要スクリプト・重みファイル群の役割・構成一覧です。

```text
biohub-1st-place-reproduction/
├── biohub-1st-place-inference.py       # Kaggle推論エントリーポイント
├── cell_tracking/                      # メインパッケージ
│   ├── inference.py                    # 推論処理全体の統括(main/cli)
│   ├── common/                         # 共通処理(tracks, I/O, 正規化, 幾何計算)
│   ├── detector/                       # 検出器モジュール
│   │   ├── model.py                    # 3D U-Net + Cellposeヘッド定義
│   │   ├── inference.py                # 検出推論・TTA・ピーク抽出
│   │   └── resnet.py                   # 3D ResNetバックボーン
│   ├── soon/                           # ビジュアルトラッカー(Soon Net)
│   │   ├── model.py                    # SoonNet(CNN Tokenizer + EVA Transformer + FPN)
│   │   ├── crops.py                    # 局所3Dボリューム(3フレーム)切り出し
│   │   ├── scoring.py                  # 状態分類・占有マップ推論
│   │   └── checkpoints.py              # 重みロード
│   └── linker/                         # 学習型リンカー(Learned Linker)
│       ├── ensemble.py                 # EdgeEnsemble(5-fold平均)
│       ├── link.py                     # キャッシュからのエッジ推論・貪欲デコード
│       ├── graph.py                    # TrackGraph構築・後処理
│       ├── postprocess.py              # 線形フィッティング平滑化・短トラック除去
│       └── tiered_gate.py              # 高信頼度ゲーティング
└── weights/                            # 5-fold学習済みモデル重み群
    ├── det_fold0.ckpt ~ det_fold4.ckpt # 3D U-Net検出器重み(5-fold)
    ├── soon_fold0.pt ~ soon_fold4.pt   # Soon Net重み(5-fold)
    └── edge_fold0/ ~ edge_fold4/       # 学習型リンカー重み(edge.pt, config.json)
```

### 主要モデル重みファイル一覧

| モジュール | 重みファイル | パラメータ規模・入力 | 役割・機能 |
| :--- | :--- | :--- | :--- |
| **Detector** | `det_fold0.ckpt` 〜 `det_fold4.ckpt` | 約3.1M params<br>入力: 128×128×64 crop | **3D U-Net検出器(5-fold):** 3Dフロー(dz, dy, dx)と中心確率マップを予測。xy反転4x TTAと組み合わせて各フレーム20パスで高精度検出。 |
| **Soon Net** | `soon_fold0.pt` 〜 `soon_fold4.pt` | 約744K params (Tokenizer:236K, Transformer:508K)<br>入力: 3フレーム局所3D (16×64×64) | **ビジュアルトラッカー(5-fold):** 細胞の「もうすぐ分裂する」外見変化の3クラス分類と、FPNによる移動先・娘細胞の占有マップ(Occupancy map)を直接予測。 |
| **Linker** | `edge_fold0/edge.pt` 〜 `edge_fold4/edge.pt` | 軽量MLP/線形モデル<br>特徴量: ノード42次元, ペア23次元 | **学習型リンカー(5-fold):** 検出点ペアとSoon Net占有マップ・幾何特徴から接続スコア・分裂スコアを算出。貪欲デコードでトラックを構築。 |

---

## 1位解法の全体像：なぜ勝てたのか？

多くの参加者(2位を含む上位陣も含む)は、検出後の細胞トラッキングにおいて「座標間のユークリッド距離」「カルマンフィルタによる移動速度・方向」「輝度の差」といった**幾何学的特徴量(幾何ヒューリスティクス)**に大きく依存していました。

しかし1位チームは、このアプローチの本質的な弱点を見抜いていました。

> **1位の洞察 (Key Insight):**
> 1. **幾何学的特徴量は細胞そのものを見ていない(blind):** 距離や速度だけでは、細胞が密集する領域での交差や、分裂直後の急激な挙動変化に追従できない。
> 2. **細胞は分裂する直前に見た目が変化する:** 生物学的に、細胞は分裂する前に丸まり、光の反射やテクスチャ、形状が明確に変化する。人間が見れば「もうすぐ分裂する」と容易に分かる。
> 3. **「点と点の距離」ではなく「画像そのものに基づく視覚的追跡(Visual Tracking)」へ:** 候補点同士の距離計算をメインにするのではなく、ニューラルネットワークに「細胞の次のフレームでの存在領域(占有マップ)」を画像から直接予測させる。

この発想の転換により、1位チームは **「3D U-Net検出器」→「Soon Net(視覚的追跡・分裂予測)」→「Learned Linker(学習型グラフ接続)」** という、高度に洗練された視覚的追跡システムを構築しました。

---

## 4大処理フェーズの詳細解説

1位解法の一連の処理は、以下の4つの処理段階(フェーズ)で構成されています。

```mermaid
flowchart TD
    subgraph Phase1["フェーズ1: 検出 (Detection)"]
        A[3Dタイムラプス画像] --> B["3D U-Net (3.1M params)<br/>MAE事前学習 + 疑似ラベル"]
        B --> C["5-fold × 4x TTA (xy反転)<br/>20パス平均化"]
        C --> D["細胞中心座標 & 3Dフロー抽出"]
    end

    subgraph Phase2["フェーズ2: 視覚追跡 (Soon Net)"]
        D --> E["各細胞の周囲 16×64×64 クロップ<br/>(t-1, t, t+1 の3フレーム)"]
        E --> F["Small CNN Tokenizer + EVA Transformer"]
        F --> G1["3クラス分類ヘッド<br/>(間期 / 分裂直前 / 分裂直後)"]
        F --> G2["FPNヘッド: 占有マップ<br/>(継続マップ & 分裂マップ)"]
    end

    subgraph Phase3["フェーズ3: 学習型リンカー (Learned Linker)"]
        D --> H["候補生成 (半径30μm, 近傍k=16)"]
        G1 --> I["特徴量結合<br/>ノード42次元 + ペア23次元<br/>(占有マップ適合度・幾何・強度)"]
        G2 --> I
        H --> I
        I --> J["EdgeEnsemble (5-fold MLP)<br/>Fork Configuration Loss"]
        J --> K["貪欲デコード (Greedy Decode)<br/>log p(opt) - log p(no child) > 0"]
    end

    subgraph Phase4["フェーズ4: 後処理 (Post Processing)"]
        K --> L["Tiered Temporal Gating<br/>(高信頼度トラックによる選別)"]
        L --> M["1フレーム ギャップ補完 (Gap Closing)"]
        M --> N["孤立ノード枝刈り & 3フレーム未満削除"]
        N --> O["分裂保護付き 線形フィッティング平滑化"]
        O --> P["最終提出データ (submission.csv)"]
    end
```

---

### フェーズ1: 検出 (Detection)

#### 1. アーキテクチャと特徴
- **vanilla 3D U-Net + Cellposeスタイルヘッド:** 128×128×64のクロップサイズで学習。
- **異方性プーリング (Anisotropic Pooling):** 顕微鏡画像のZ軸方向の解像度がXY平面と異なるため、ダウンサンプリングを `(1,2,2), (1,2,2), (2,2,2)` と異方的に設計。
- **4チャンネル出力:** 各ボクセルに対して「中心確率(1ch)」と「3D方向オフセット・フロー(dz, dy, dxの3ch)」を出力。
- **Deep Supervision:** 1/2スケール、1/4スケールの中間層でもフローと確率マップの損失を計算し、勾配消失を防ぎながら多重解像度の特徴を学習。

#### 2. スパースな正解データへの対策 (MAE事前学習 & 疑似ラベリング)
- **Masked Autoencoder (MAE) 事前学習:** ドイツがん研究センター(MIC-DKFZ)の `nnssl` フレームワークを活用。大量の未ラベル3D顕微鏡画像から自己教師あり学習で堅牢な3D視覚表現を獲得。
- **2段階ファインチューニング & 疑似ラベル (Pseudo Labeling):**
  - Round 1学習後、5-fold平均 + 4x flip TTAを用いて全195本の動画と150個の外部クロップに対して高信頼度な検出結果を擬似ラベル化。
  - 公開のtwoPassトラッカーで長さ3以上のトラックのみを抽出・精緻化し、スパースだった正解データを大幅に増強。
- **推論時TTA:** 本番推論では、5フォールド × 4倍反転TTA (xy反転) = **1フレームあたり計20回のフォワードパス**を実行して平均化。

#### フェーズ1 処理シーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Entry as biohub-1st-place-inference.py
    participant Main as cell_tracking/<br/>inference.py<br/>(main / run_worker / build_caches)
    participant DetMod as cell_tracking/detector<br/>inference.py
    participant UNet as cell_tracking/detector<br/>model.py
    participant LinkCache as cell_tracking/linker<br/>cache.py
    participant IntSidecar as cell_tracking/linker<br/>intensity.py

    Entry->>Main: cli() -> main()
    Main->>Main: run_worker(GPU数)
    Main->>Main: build_caches(detector, soon, raw, dirs, movie, device)
    Note over Main: 【フェーズ1: 検出 (Detection)】
    loop 各フレーム t = 0 .. n_t-1 (3Dボリューム vol)
        Main->>DetMod: extract_frame_cache(detector, vol, device, frame_at)
        loop 4方向反転 TTA (xy反転)
            DetMod->>UNet: forward(crops) [5-foldアンサンブル]
            UNet-->>DetMod: 3Dフロー (dz, dy, dx) + 中心確率 (prob)
        end
        Note over DetMod: 20パス平均化 & ピーク検出 (極大値抽出)
        DetMod-->>Main: coords (細胞座標), feats (幾何特徴), vols (体積)
        Main->>LinkCache: write_frame_cache(dirs["cache"], coords, feats, vols, t)
        Note over LinkCache: frame_000x キャッシュ保存
        Main->>IntSidecar: write_intensity_sidecar(dirs["intensity"], vol, coords, vols, t)
        Note over IntSidecar: frame_000x_intensity.npz キャッシュ保存
    end
```

---

### フェーズ2: 分裂直前 / ビジュアルトラッキング (Soon Net)

1位解法の最大のブレイクスルーとなったのが、この独自設計モデル **`Soon Net`** です。

#### 1. モチベーション
従来の幾何学的追跡では、分裂直前の親細胞と2つの娘細胞の対応付けで誤判定が頻発していました。しかし細胞は分裂直前になると、球状化し、細胞質の輝度分布や境界のコントラストが劇的に変化します。この「見た目の予兆」を捉えるために開発されました。

#### 2. アーキテクチャ
- **入力:** 時点 $t-1, t, t+1$ における対象細胞周辺の局所3Dボリューム (サイズ: 16×64×64)。
- **小型CNNトークナイザー (Small CNN Tokenizer, 236K params):** 局所ボリュームを過学習なくコンパクトな特徴トークン列へと変換。
- **Transformerバックボーン (EVA Blocks, 508K params):** 時間軸方向 ($t-1 \leftrightarrow t \leftrightarrow t+1$) のコンテキストを統合。
- **2系統の出力ヘッド:**
  1. **状態分類ヘッド (Classification Head):**
     - 「間期 (interphase)」「もうすぐ分裂 (soon-to-divide)」「分裂直後 (just divided)」の3クラス確率を出力。
  2. **FPNヘッド (Occupancy Map Head):**
     - $t-1, t, t+1$ の各フレームにおける「継続マップ(continuation map)」と「分裂マップ(division map)」の2種類の3D占有マップを予測。
     - **継続マップ:** 通常細胞について「次のフレームで細胞がどの座標に移動するか」を球体ガウス分布で点灯。
     - **分裂マップ:** 母細胞について「次のフレームで2つの娘細胞がそれぞれどこに出現するか」、娘細胞について「前フレームで母細胞がどこにいたか」を点灯。

```text
【Soon Netの予測する占有マップの挙動】
通常継続時:
  フレーム t-1 [  ●  ] ──→ フレーム t [   ●   ] ──→ フレーム t+1 [    ●   ] (移動先を点灯)

細胞分裂時:
  フレーム t (母細胞) [   ● (soon)   ] ──→ フレーム t+1 (分裂マップ) [  ● (娘A)   ● (娘B)  ] (両娘の位置を点灯)
```

この占有マップにより、「リンカーが次にどこへ線を結べばよいか」の答えそのものが画像から直接得られるようになりました。

#### フェーズ2 処理シーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Main as cell_tracking/inference.py
    participant SoonScore as cell_tracking/soon/scoring.py
    participant SoonModel as cell_tracking/soon/model.py
    participant SoonCache as cell_tracking/linker/soon_cache.py
    participant OccMod as cell_tracking/linker/occupancy.py

    Note over Main: build_caches() 内のフレームループ (フェーズ1の直後)
    loop 各フレーム t = 0 .. n_t-1
        Main->>Main: window(t) [前・現・次フレーム (nvol, prev, nxt) 取得]
        Main->>Main: score_soon(soon, nvol, coords, prev, nxt, device)
        Main->>SoonScore: score_points_per_net(coords, 16×64×64 crops)
        SoonScore->>SoonModel: forward(crops) [5-fold Soon Netアンサンブル]
        Note over SoonModel: CNN Tokenizer + Transformer + FPNヘッド推論
        SoonModel-->>SoonScore: 3クラス状態確率, 継続占有マップ, 分裂占有マップ
        SoonScore-->>Main: scores (5-fold平均), occ, div
        Main->>SoonCache: write_soon_frame(dirs["soon"], scores)
        Note over SoonCache: soon_frame_000x キャッシュ保存
        opt t >= 1 (1フレーム遅延 flush)
            Main->>Main: flush(t - 1, coords)
            Main->>OccMod: write_occupancy(dirs["occ"], occ) [継続マップ]
            Main->>OccMod: write_occupancy(dirs["occ_div"], div) [分裂マップ]
            Note over OccMod: occupancy_000x.npz キャッシュ保存
        end
    end
```

---

### フェーズ3: 学習型リンカー (Learned Linker)

#### 1. 候補抽出と特徴量設計
- 探索半径 `support_radius_um = 30.0μm`、最近傍 `k = 16` の候補細胞ペアを列挙。
- 以下の豊富な特徴量を統合：
  - **ノード特徴量 (42次元):** 検出スコア、細胞体積・主軸比率、Soon Netの3クラス確率(soon scores)など。
  - **ペア特徴量 (23次元):** 幾何学的変位ベクトル、局所強度(intensity)の一致度、**Soon Netが予測した占有マップとの一致度スコア(occupancy score)**、局所オプティカルフロー適合度。

#### 2. 損失関数と学習の工夫
- **アノテーション済みエッジのみのNLL損失 (NLL on annotated edges only):**
  - データセットの正解ラベルはスパースであり、正解がついていないエッジが大量に存在します。通常通りに学習させると「未アノテーションの真のエッジ」を「陰性例(繋いではいけない)」として誤学習してしまいます。1位解法では、アノテーションが存在するエッジのみで負の対数尤度(NLL)を計算し、未アノテーション部をペナルティから除外しました。
- **分岐構成損失 (Fork Configuration Loss):**
  - 親ノードに対して「分裂して子A, Bを持つ」「継続して子Aのみを持つ」「子なし」という排他的な構造全体に対してソフトマックス損失を適用。
  - 分裂イベントはデータ全体の数%と稀少なため、**学習時に分裂を8倍オーバーサンプリング**。
- **タイムステップ拡張 (Time-step Augmentation):**
  - 学習データの25%で1フレーム飛ばし $(t, t+2)$ のペアを意図的に提示し、検出漏れによるギャップ耐性をモデルに学習させました。

#### 3. 貪欲デコード (Greedy Decode)
- 複雑なILP(整数線形計画法)や外部ソルバーは使わず、高速で破綻のない貪欲デコードを採用。
- 各親細胞と各選択肢について、**「子を持たない場合との対数オッズ比」** を算出：
  $$\Delta = \log p(	ext{option}) - \log p(	ext{no child})$$
- $\Delta > 0$ となる候補のみを残し、スコア降順にソート。
- 「1つの親は最大1つの選択肢(継続または分裂)」「1つの子は最大1つの親」という制約を満たすように上から順に確定。
- 親がつかなかった子細胞は新規トラックとして開始。

#### フェーズ3 処理シーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Main as cell_tracking/inference.py
    participant LinkMovie as cell_tracking/linker/link.py
    participant LinkCache as cell_tracking/linker/cache.py
    participant OccMod as cell_tracking/linker/occupancy.py
    participant EdgeEns as cell_tracking/linker/ensemble.py

    Note over Main: run_worker() から呼び出し
    Main->>Main: link_and_post(edge, cfg, dirs, movie, device)
    Note over Main: 【フェーズ3: グラフ接続 (Learned Linker)】
    Main->>LinkMovie: link_movie_from_cache(edge, dirs["cache"], movie, ...)
    LinkMovie->>LinkCache: 全フレームの座標・幾何特徴・輝度キャッシュ読込
    LinkMovie->>OccMod: Soonスコア & 継続・分裂占有マップ読込
    Note over LinkMovie: 半径30μm・近傍k=16で候補ペア抽出<br/>特徴量生成 (ノード42次元, ペア23次元)
    LinkMovie->>EdgeEns: forward(features) [5-fold MLPリンカー]
    EdgeEns-->>LinkMovie: 接続確率 & 分裂確率
    Note over LinkMovie: 貪欲デコード (Greedy Decode)<br/>log p(option) - log p(no child) > 0 を降順ソート<br/>各親最大1選択肢、各子最大1親でエッジ確定
    LinkMovie-->>Main: frames, links (接続リスト), link_probs
```

---

### フェーズ4: 後処理 (Post Processing)

複雑なヒューリスティクスを極力排除し、シンプルかつ確実なルールのみで構成：

1. **Tiered Temporal Gate (段階的信頼度ゲート):**
   - 信頼度スコア `TIER_CONF = 0.3` 未満かつ長さ `TIER_MIN_LEN = 3` 未満の不確実な短エッジを排除。
2. **1フレーム ギャップ補完 (Gap Closing):**
   - 1フレームの一時的な検出漏れを距離閾値(4.0μm)・ドリフト閾値(30.0μm)内で補完。
3. **孤立ノードの枝刈り & 長さフィルタ:**
   - どこにも繋がっていない孤立ノードを削除。全体の長さが3フレーム未満のトラックを除去。
4. **分裂保護付き 線形フィッティング平滑化 (Line-fit Smoothing):**
   - 細胞のトラッキング座標の揺らぎを局所線形回帰で滑らかに補正。
   - ただし、分裂分岐点(`stop_at_forks=True`)では娘細胞の座標が母細胞側へと引きずられて潰れないよう平滑化を停止。
5. **分裂不応期ルール (Division Refractory Rule):**
   - 生まれたばかりの娘細胞が、直後のフレームで再び分裂することを生物学的に禁止。
   - 競合が発生した場合は、予測信頼度が高い分裂を優先して採用。

#### フェーズ4 処理シーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Entry as biohub-1st-place-inference.py
    participant Main as cell_tracking/inference.py
    participant GraphMod as cell_tracking/linker/graph.py
    participant ScoreMod as cell_tracking/linker/scoring.py
    participant TierGate as cell_tracking/linker/tiered_gate.py
    participant PostMod as cell_tracking/linker/postprocess.py
    participant Tracks as cell_tracking/common/tracks.py

    Note over Main: link_and_post() 内でグラフ構築・後処理
    Main->>GraphMod: graph_from_pair_links(B, frames, links)
    GraphMod-->>Main: g (TrackGraph初期グラフ)

    Main->>ScoreMod: link_confidence(frames, links, link_probs)
    ScoreMod-->>Main: conf (リンク確信度)

    Main->>TierGate: tiered_temporal_gate(g, conf, conf_thresh=0.3, min_len=3)
    Note over TierGate: 信頼度0.3未満かつ長さ3未満の低信頼エッジを排除
    TierGate-->>Main: g (選別済みグラフ)

    Main->>GraphMod: postprocess_graph(B, frames, g, linefit=True, max_gap=1, ...)
    GraphMod->>Tracks: close_gaps(frames, g, max_gap=1, gap_dist_um=4.0)
    Note over Tracks: 1フレームの一時的検出漏れを補間
    GraphMod->>Tracks: prune_isolated(g)
    Note over Tracks: 孤立ノード削除
    GraphMod->>PostMod: filter_short_tracks(g, min_track_len=3)
    Note over PostMod: 3フレーム未満の短トラック削除
    GraphMod->>PostMod: linefit_smooth(stop_at_forks=True)
    Note over PostMod: 局所線形回帰平滑化 (分裂ノードは保護)
    GraphMod-->>Main: g (後処理完了TrackGraph)

    Note over Main: run_worker() へ戻り行データ変換
    Main->>Tracks: graph_to_rows(movie_name, g)
    Tracks-->>Main: rows
    Note over Main: main() で全動画完了後に提出ファイル出力
    Main->>Tracks: write_submission(all_rows, OUTPUT)
    Tracks-->>Entry: submission.csv 書き出し完了
```

---

## ソースコード (`inference.py`) の推論フロー詳解

推論コード `biohub-1st-place-inference.py` から呼び出される `cell_tracking/inference.py` を起点とした、フェーズ1〜フェーズ4のファイルベースの処理シーケンスおよび各ファイル(オブジェクト)の役割分担です。

### フェーズ1〜フェーズ4 全体シーケンス図

```mermaid
sequenceDiagram
    autonumber
    participant Entry as biohub-1st-place-inference.py
    participant Main as cell_tracking/inference.py
    participant DetMod as cell_tracking/detector/inference.py
    participant SoonScore as cell_tracking/soon/scoring.py
    participant LinkMovie as cell_tracking/linker/link.py
    participant GraphMod as cell_tracking/linker/graph.py
    participant Tracks as cell_tracking/common/tracks.py

    Entry->>Main: cli() -> main()
    Main->>Main: load_stack(device) [各モデル重みをロード (5-fold)]
    Note over Main: テスト動画をキューに登録しワーカースレッド起動
    loop 各ワーカースレッド run_worker(k) [動画ごとの処理]
        Main->>Main: run_worker(k)
        Main->>Main: build_caches(detector, soon, raw, dirs, movie, device)
        Note over Main: 【フェーズ1: 検出】 & 【フェーズ2: 視覚追跡】
        loop 全フレーム t = 0 .. n_t-1
            Main->>DetMod: extract_frame_cache() [3D U-Net + 4x TTA]
            DetMod-->>Main: 座標・幾何特徴・輝度キャッシュ保存
            Main->>Main: score_soon()
            Main->>SoonScore: score_points_per_net() [Soon Net 5-fold]
            SoonScore-->>Main: soonスコア・占有マップキャッシュ保存
        end

        Main->>Main: link_and_post(edge, cfg, dirs, movie, device)
        Note over Main: 【フェーズ3: 接続】 & 【フェーズ4: 後処理】
        Main->>LinkMovie: link_movie_from_cache() [EdgeEnsemble + 貪欲デコード]
        LinkMovie-->>Main: frames, links, link_probs
        Main->>GraphMod: graph_from_pair_links() & postprocess_graph()
        Note over GraphMod: ギャップ補完・孤立点削除・短トラック削除・線形平滑化
        GraphMod-->>Main: g (完成したTrackGraph)
        Main->>Tracks: graph_to_rows(name, g)
    end
    Note over Main: main() で全動画集約後
    Main->>Tracks: write_submission(all_rows, OUTPUT)
    Tracks-->>Entry: submission.csv 書き出し完了
```

### 各ファイル (オブジェクト) の役割と分担

| フェーズ | ファイルパス (オブジェクト) | 主な関数 / クラス | 担当する具体的な処理 |
| :--- | :--- | :--- | :--- |
| **全体統括** | `cell_tracking/inference.py` | `cli()`, `main()`, `run_worker()`, `build_caches()`, `link_and_post()` | 各種重みのロード、動画ごとのワーカースレッド管理、フェーズ1〜4の実行制御。 |
| **フェーズ1** | `cell_tracking/detector/inference.py` | `load_unetcellpose_ensemble()`, `extract_frame_cache()` | 3D U-Netによる立体画像推論、4方向反転TTA(計20パス)、細胞の中心座標・3Dフロー・体積の抽出。 |
| **フェーズ1〜2** | `cell_tracking/linker/cache.py`<br>`cell_tracking/linker/intensity.py`<br>`cell_tracking/linker/occupancy.py` | `write_frame_cache()`, `write_intensity_sidecar()`, `write_occupancy_from_maps()` | 検出座標、細胞輝度、Soon Netが予測した占有マップをフレームごとの中間キャッシュとして保存。 |
| **フェーズ2** | `cell_tracking/soon/scoring.py`<br>`cell_tracking/soon/model.py` | `score_points_per_net()`, `SoonNet.forward()` | 細胞周囲の3フレーム局所画像から、「分裂直前か」の3クラス分類と、次フレームの移動先・分裂先の占有マップを生成。 |
| **フェーズ3** | `cell_tracking/linker/link.py`<br>`cell_tracking/linker/ensemble.py` | `link_movie_from_cache()`, `EdgeEnsemble` | 近傍候補ペアの探索、Soon Net占有マップ適合度の結合、5-fold平均スコアの算出、貪欲デコードによるエッジ確定。 |
| **フェーズ4** | `cell_tracking/linker/tiered_gate.py` | `tiered_temporal_gate()` | 信頼度の低い孤立した短い接続エッジを選別して排除。 |
| **フェーズ4** | `cell_tracking/linker/graph.py`<br>`cell_tracking/linker/postprocess.py` | `postprocess_graph()`, `linefit_smooth()`, `filter_short_tracks()` | ギャップ補完、孤立ノード削除、3フレーム未満トラックの除去、分裂保護付き線形フィッティング平滑化。 |
| **出力** | `cell_tracking/common/tracks.py` | `graph_to_rows()`, `write_submission()` | 完成したトラックグラフから提出用フォーマットの行データを生成し、`submission.csv` へ書き出し。 |

### コード内の重要パラメータ一覧

| 変数名 | 設定値 | 意味と役割 |
| :--- | :--- | :--- |
| `DET_TTA` | `"xy"` | 検出時のテスト時データ拡張(XY平面の反転4通り)。 |
| `DET_AUTOCAST` / `SOON_AUTOCAST` | `"fp16"` | 推論時の半精度高速化。 |
| `SOON_BATCH` | `64` | Soon Net推論時のバッチサイズ。 |
| `EDGE_SHORTLIST` | `"consensus"` | エッジ候補選別の合意形成戦略。 |
| `THRESHOLD` | `0.1` | エッジ採用の最小閾値。 |
| `MAX_CHILDREN` | `2` | 1つの親ノードが持てる最大子ノード数(分裂時は2)。 |
| `DIV_REFRACTORY` | `True` | 分裂直後の再分裂を禁止する不応期ルールの有効化。 |
| `DIV_REFRACTORY_ORDER` | `"prob"` | 分裂不応期の競合解決を確率スコア順で実施。 |
| `LINEFIT` | `True` | 線形フィッティング平滑化の適用。 |
| `MAX_GAP` | `1` | ギャップ補完で許容する最大フレーム間隔(1フレーム欠損を救済)。 |
| `GAP_UM` / `GAP_DRIFT_UM` | `4.0` / `30.0` | ギャップ補完時の距離閾値(μm)およびドリフト許容半径。 |

---

## 検証結果とスコアの推移

1位解法のCV(交差検証)アブレーション結果は、各コンポーネントの圧倒的な有効性を証明しています。

### 表: 全処理統合後の最終 5-fold CV スコア

| 評価GT | Soon Net単体 | 全処理統合後 (Full) | 通常エッジJaccard (単体 / フル) | 分裂Jaccard (単体 / フル) | 分裂 TP / FP / FN (フル) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original GT** (元正解) | 0.9288 | **0.9575** | 0.8914 / 0.9089 | 0.3738 / **0.4857** | TP: 102 / FP: 62 / FN: 46 |
| **Augmented GT** (拡張正解) | 0.9464 | **0.9856** | 0.8903 / 0.9090 | 0.5617 / **0.7654** | TP: 434 / FP: 62 / FN: 71 |

- **Soon Netの占有マップ単体でも 0.9288〜0.9464** という極めて高い初期スコアを達成。
- **全処理統合後(リンカー統合後)** では、元正解で **0.9575**、拡張正解で **0.9856** という驚異的な一致率に到達。
- 特に分裂Jaccardにおいて、0.3738 → **0.4857** (拡張GTでは0.5617 → **0.7654**) と劇的な精度向上を記録。
- **最終リーダーボード:** Public LB: **0.976** / Private LB: **0.977** (見事単独1位を獲得)。

---

## 2位解法との比較：何が勝敗を分けたのか？

| 項目 | 1位解法 (sersasj) | 2位解法 (soheilayati) |
| :--- | :--- | :--- |
| **追跡の基本思想** | **ビジュアルトラッキング (視覚的追跡)**<br>細胞の見た目変化・FPN占有マップから直接追跡 | **幾何ヒューリスティクス + 特徴量工学**<br>座標間距離・速度・カルマンフィルタに基づく |
| **分裂(Division)の捉え方** | **Soon Netによる事前検知**<br>「間期・直前・直後」の3クラス分類 + 娘細胞占有マップ | **ペア特徴量 + 分裂スタッキングモデル**<br>距離・幾何特徴からLightGBM/ペアモデルで判定 |
| **アンサンブル手法** | **厳密なOOF 5-fold平均**<br>(検出器 5-fold × Soon Net 5-fold × Linker 5-fold) | **多層メタスタッキング & 多数決合意**<br>(4系統トラッカー + 多数のGBDT/線形分類器) |
| **デコード方式** | **シンプルな貪欲デコード (Greedy Decode)**<br>対数オッズ比による高速・堅牢な逐次決定 | **多段制約ソルバー & 競合調停**<br>(Stage-6リランキング、エッジ合意、フォーク投票) |
| **システム複雑度** | **コンパクトかつ洗練されたDNN設計**<br>(総パラメータ数は数M程度でコードも簡潔) | **極めて複雑な処理構成**<br>(多数の重みファイルと何百行もの特徴量抽出) |

### 勝因のまとめ
1. **問題の本質を突いたモデル設計:**
   「細胞追跡において最も難しいのは分裂である」「細胞は分裂前に見た目が変わる」という生物学的真実に着目し、幾何計算ではなく専用の深層学習モデル(`Soon Net`)で解かせたこと。
2. **占有マップ(Occupancy Map)による視覚的リンク候補の提示:**
   単に状態を分類するだけでなく、FPNによって「次のフレームで細胞が存在する領域」「娘細胞が出現する領域」を画像ベースのヒートマップとして予測させたため、リンカーが迷わずに正しいエッジを選択できたこと。
3. **データ不完全性への完璧な対処:**
   スパースな正解アノテーションに対して、MAE自己教師あり学習、高精度な疑似ラベル生成、そして「アノテーション済みエッジのみで損失を計算する」損失関数設計を徹底したこと。
4. **過学習のない強固な交差検証(CV)の仕組み:**
   検出器・Soon Net・リンカーの全段階で厳密にOOF(Out-Of-Fold)を徹底し、テスト時も5-fold平均とTTAを貫いたため、Private LBでも全くシェイクせず(Public 0.976 → Private 0.977)圧倒的な精度を維持できたこと。
