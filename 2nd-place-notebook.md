18-⑤Kaggle実践3 Biohub細胞トラッキング：2位の解法まとめ

<img src="image.png" width="800" />

## Abstract
- biohub 2位解法のまとめ

## 概要
先日、Biohub細胞トラッキングコンペが完了しました。
Private LB 1353位/3800人という大変残念な結果となってしまいました。初参加ということで雰囲気を味わうということを目的にしてたんですが、実際がっつり頑張ったのに、銅メダルにも届かないという結果は、すごく悔しいです。
しかし、
Kaggleは勝者の解法は公開されるという素晴らしい決まり事がありまして、もうこれは徹底的に解析して自分のスキルにするしかないなと。

## biohub 2位の解法まとめ
2位の方だけでなく、上位陣の人たちはモデルを複数駆使してて、また学習にも工夫が随所にみられるました。そして公開されるのは学習後モデルそのものでして、学習に使用したコードやデータ群までは公開されないんですね。ただ上位陣はそれぞれ自身の提出コードの解説ページを公開してくれてて、そこで結構情報を得られたりします。

- 解説のURL
https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution
- ソースコード
https://www.kaggle.com/code/soheilayati/biohub-2nd-place-notebook

## プロジェクト構成

このプロジェクトで使用ファイル（JSON, py, 重みファイル `.pt` / `.npz` / `.txt` 等）群とディレクトリの役割を機能カテゴリ別に整理。
**input**: /kaggle/input/datasets/soheilayati

---

### 1. 共通・データ・環境 (`.py` / `.zarr` / `.whl`)

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-v343-hard-dim-code/` | `biohub/` パッケージ全般<br>(`pipeline/`, `tracker/`, `detector/` 等) | **メインパイプライン実装:** 検出、グラフ解法(Solver)、コンセンサス、モーション推定などの全コード本体。 |
| `input/test/` | `*.zarr` (4 データセット) | **入力テストボリューム:** 推論対象の 3D 顕微鏡タイムラプス画像データセット。 |
| `biohub-cell-tracking-wheels/`<br>`input/support_pack/wheels/` | `zarr-*.whl`<br>`tracksdata-*.whl` 等 | **オフライン配布ホイール:** インターネット遮断環境下で必要なライブラリ群をインストールするための wheel 群。 |

---

### 2. 検出器 (Detector) 重みファイル (`.pt`)

3D画像ボリュームから細胞の存在ヒートマップやオフセット・フローを抽出する深層学習モデル群。
- train/ の 3D 画像と正解の細胞中心座標を入力とし、3D U-Net などのニューラルネットワークを PyTorch で、GPU 学習 させて生成。
- 過学習を防ぎ汎化性能を高めるため、データを 5 つに分けた交差検証 (5-Fold CV) や、フレーム数を変えたモデル (3フレーム/5フレーム) をそれぞれ学習させて保存。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-cell-tracking-weights-v25-consensus/` | `fold0.ep05.pt` 〜 `fold4.ep05.pt`<br>`v24.fold0.ep05.pt` 〜 `v24.fold4.ep05.pt` | **主幹検出器 (V21 & V24):** 5分割交差検証(5-fold)による主検出モデル計10個。ヒートマップアンサンブルの基盤。 |
| `biohub-seed1-detector/` | `full.seed1.ep02.pt` | **独立シード検出器:** アンサンブルのヒートマップ多様性を向上させるための別シードモデル。 |
| `biohub-v35-strong-detectors/` | `strong.fold0.ep05.pt` 〜 `strong.fold4.ep05.pt` | **Stage-6 強検出器:** 高感度・高精度な第6ステージ用の5-fold検出モデル。 |
| `biohub-v41-probe-weights/` | `v39.fold0.ep05.pt` 〜 `v39.fold4.ep05.pt` | **V39 検出器 (5-frame入力):** 5フレーム時系列を入力として捉える高精度検出モデル。 |
| `biohub-v57-deployment-weights/` | `v38.fold0.ep05.pt` 〜 `v38.fold4.ep05.pt` | **V38 検出器 (残差検出):** V39 と組み合わせてリカバリ・補完を行う検出モデル。 |
| `biohub-v389-structured-head/` | `residual.pt`<br>`v389_structured_manifest.json` | **残差補正モデル:** V230木構造対応付けと併用し、画像残差から座標精度を高めるモデル。 |

---

### 3. トラッカー・グラフ判定モデル (`.npz` / `.json` / `.txt`)

検出された細胞候補点同士を時系列で結び、細胞分裂や枝刈りを判定する機械学習モデル群。
- 検出器が見つけた細胞候補点から、「細胞間のユークリッド距離」「動きの滑らかさ」「輝度の一致度」など何百種類の特徴量を計算して表データ (DataFrame) を作る。
- 正解のトラッキング線（エッジ）を教師データとして、LightGBM やロジスティック回帰 を学習させてツリー構造や重みをシリアライズする。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-cell-tracking-weights-v25-consensus/` | `count_model_v1.json` | **細胞数推定モデル:** 画像統計量からフレーム内の細胞数を推定し、探索予算(budget)を決定する。 |
| 同上 | `edge_model_b125.npz` | **エッジ結合モデル (GBM):** 時間軸方向の細胞間の接続確率を予測する。 |
| 同上 | `pair_model_v0.npz`<br>`pair_stack_faithful.npz` | **ペアモデル / ペアスタック:** 2点間の同一細胞判定および分裂判定。 |
| 同上 | `pair_allimg_lr.npz`<br>`pair_full_lr.npz`<br>`pair_hardneg_lr.npz`<br>`pair_highres_lr.npz` | **多様なペア判定器 (LR):** 全画像版、難例(Hard Negative)版、高解像度版などのロジスティック回帰スタック。 |
| 同上 | `fork_confirmation_lr.npz`<br>`edge_tracklet_lr_c0p003.npz` | **分岐確定 / トラックレット結合:** 細胞分裂(fork)の最終確認および小トラック(tracklet)の結合判定。 |
| 同上 | `component_current_lgb_l15_fold*.txt`<br>`edge_tracklet_lgb_l7_fold*.txt`<br>`edge_postselect_lgb_l31_fold*.txt`<br>(各 `.json` メタデータ) | **LightGBM ブースター群:** 連結成分評価、トラックレット評価、最終エッジ後処理選択用の勾配ブースティング決定木群。 |
| `biohub-v57-deployment-weights/` | `fold0.txt` 〜 `fold4.txt`<br>`v57_deployment_manifest.json` | **V57 リスク評価モデル (GBM):** V39 との置き換え時にノード除去のリスク度合いをスコアリングする。 |
| `biohub-v343-hard-dim-code/v343_models/` | `full.dim.txt`<br>`full.cell.txt`<br>`v343_dim_scaffold_manifest.json` | **特化足場モデル (Dim Specialist):** 暗い細胞や微弱な領域の検出足場を補強する LightGBM モデル。 |
| `biohub-v343-hard-dim-code/v230_models/` | `tree.txt` | **木構造対応付けモデル (V230):** 過去フレームと未来フレームの細胞対応付けツリー定義。 |

---

### 4. 後処理・保護モジュール (動的読み込み Python + マニフェスト JSON)

各バージョンごとの保護・改善ロジックを担う独立したモジュール群。`importlib` を通じて動的に `.py` がロードされ、同時に `.json` で設定・検証が実行される。
- ここは AI の自動学習というよりも、開発者の職人芸（エラー分析と最適化） 。
 - 途中段階の予測結果を可視化し、「暗い細胞が途切れやすい」「端っこにノイズの枝が残る」といった課題を見つけ、それを救うアルゴリズム（V394, V399, V454, V507, V533）を設計している。
- そのパラメータ（hole_um=4.5, min_contrast=0.1 など）をグリッドサーチ等で探索し、最も CV スコアが高くなった数値をマニフェスト (.json) に固定。

| 相対パス | ファイル | 役割・機能 |
| :--- | :--- | :--- |
| `biohub-v394-component-protector/` | `v394_component_protector.py`<br>`v394_component_manifest.json` | **連結成分保護:** 短すぎるトラックやノイズ成分をクラスタ別特性に応じてプルーニング(枝刈り)・保護する。 |
| `biohub-v399-tail-protector/` | `v399_tail_protector.py`<br>`v399_tail_manifest.json` | **終端テイル保護:** トラック終端の不確実な余分な枝や偽陽性テイルをトリミングする。 |
| `biohub-v454-current-graph-gap-admission/` | `v454_gap_admission.py`<br>`v454_manifest.json` | **ギャップ補完許可判定:** 途切れた細胞トラック間の接続(ギャップ)を最終グラフに許可・追加する。 |
| `biohub-v507-localization-fork/` | `v507_candidate.py`<br>`v507_manifest.json` | **局所化・分岐修正:** 細胞分裂(fork)判定の修正と、細胞重心座標の高精度化を行う。 |
| `biohub-v533-addback-edge/` | `v533_addback_edge.py`<br>`v533_manifest.json` | **最終エッジ補完 & 境界トリム:** コントラスト閾値に基づくエッジの再追加(add-back)および画像端(xy edge)の不要ノード除去。 |

---

# Biohub V533 シーケンス図 (セル単位)

ノートブックのコードセル(`# %%`)ごとに分割したシーケンス図。5～7が処理の本体。ここを理解するとほぼ網羅できたといっても過言ではない。

## 1. (cell 1: line 7〜133) パス探索とマニフェスト読み込み
要は、ローカル環境とKaggle環境とではパスが違うので、それを吸収している処理。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant Walk as walk()
    participant CodeRoot as find_code_root()
    participant TestRoot as find_test_root()
    participant WeightsRoot as find_weights_root()
    participant UniqueRoot as find_unique_root()
    participant JsonLoad as json.loads()

    Script->>CodeRoot: find_code_root()
    CodeRoot->>Walk: walk(ROOTS)
    Walk-->>CodeRoot: ディレクトリ巡回
    CodeRoot-->>Script: CODE_ROOT

    Script->>TestRoot: find_test_root()
    TestRoot->>Walk: walk(ROOTS)
    Walk-->>TestRoot: ディレクトリ巡回
    TestRoot-->>Script: TEST_ROOT

    Script->>WeightsRoot: find_weights_root()
    WeightsRoot->>Walk: walk(ROOTS)
    Walk-->>WeightsRoot: ディレクトリ巡回
    WeightsRoot-->>Script: WEIGHTS_ROOT

    Script->>UniqueRoot: find_unique_root(filename)
    UniqueRoot->>Walk: walk(ROOTS)
    Walk-->>UniqueRoot: ディレクトリ巡回
    UniqueRoot-->>Script: 各バージョンRoot (V39, V57, V343, ...)

    Script->>JsonLoad: json.loads(manifest_path.read_text())
    JsonLoad-->>Script: 各種マニフェスト辞書
```

---

## 2. (cell 2: line 134〜164) 依存関係 (zarr) のオフライン導入
入力データ(zarr)の読み込み準備(zarr操作ライブラリのインストールとか)。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ImportMod as importlib.import_module()
    participant Walk as walk()
    participant PipInstall as subprocess.run()

    Script->>ImportMod: importlib.import_module('zarr')
    alt すでにインストール済み
        ImportMod-->>Script: モジュール読み込み成功
    else 未インストール (ModuleNotFoundError)
        Script->>Walk: walk(ROOTS)
        Walk-->>Script: wheel_dir (zarr-*.whl 所在)
        Script->>PipInstall: subprocess.run(pip install --no-index --find-links=...)
        PipInstall-->>Script: インストール完了
    end
```

---

## 3. (cell 3: line 165〜185) 環境記録とデバイス選定
実行環境のチェックとか(GPU載ってないならエラーにしてる)。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ModImport as __import__()
    participant SelectDevice as biohub.device.select_device()

    loop 各主要ライブラリ ('numpy', 'scipy', 'torch', etc.)
        Script->>ModImport: __import__(m)
        ModImport-->>Script: バージョン確認
    end

    Script->>SelectDevice: select_device(allow_cpu=False)
    SelectDevice-->>Script: SPEC (device='cuda', capability, vram_gb, amp_dtype)
```

---

## 4. (cell 4: line 186〜535) モデルアンサンブルのロードと検証
cell 4 では、30 個以上 のモデルを同時にロードしてアンサンブルを構築する。

- このセルで実際にロードしている「アンサンブル」の内訳
  - **主幹検出器チーム(10個)**: 
    データを 5 分割して学習させた 5 人のモデル (fold0 〜 fold4) × 2 世代分 (V21 と V24) = 計 10 個
  - **強力な専門検出器チーム(5個)**:
    別の設定で極限まで鍛えた Stage-6 強力モデル 5 個
  - **長時間の動きを見るチーム(5個)**:
    5 フレームの時系列を捉える V39 モデル 5 個
  - **決定木・トラッカーチーム(数十個)**:
    細胞同士を繋ぐか判定する LightGBM のツリー群 (BoosterEnsemble)

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant CountModelLoad as CountModel.load()
    participant EdgeModelLoad as EdgeModel.load()
    participant LoadModels as load_models()
    participant ShaCheck as hashlib.sha256()
    participant PairModelLoad as PairModel.load() / PairStack.load()
    participant BoosterLoad as BoosterEnsemble.load() / Booster()

    Script->>CountModelLoad: CountModel.load('count_model_v1.json')
    CountModelLoad-->>Script: COUNT_MODEL

    Script->>EdgeModelLoad: EdgeModel.load('edge_model_b125.npz')
    EdgeModelLoad-->>Script: EDGE_MODEL

    Script->>ShaCheck: sha256(extra_path.read_bytes())
    ShaCheck-->>Script: ハッシュ一致検証

    Script->>LoadModels: load_models(paths, device)
    LoadModels-->>Script: MODELS (V21/V24 consensus), SEED_MODELS

    Script->>LoadModels: load_models(strong_paths, device)
    LoadModels-->>Script: STRONG_MODELS (Stage-6)

    Script->>LoadModels: load_models(V39/V38 paths, device)
    LoadModels-->>Script: V39_MODELS, V38_MODELS

    Script->>PairModelLoad: PairModel.load(), PairStack.load()
    PairModelLoad-->>Script: PAIR_MODEL, PAIR_STACK, ALLIMG_MODEL, etc.

    Script->>BoosterLoad: BoosterEnsemble.load(), Booster()
    BoosterLoad-->>Script: COMP7_MODEL, V343_DIM_MODEL, V343_CELL_MODEL
```

---

## 5. (cell 5: line 541〜705) 処理A: ボリューム取得・検出・並列追跡

- **volumes_from_zarr()** ...処理の一番最初で、ロバスト正規化、等方化ダウンサンプリング(1, 4, 4)、float16 化を実施。
  (このコンペはz,y,xで[1.625μm, 0.40625μm, 0.40625μm]の解像度の違いがある)。
- **image_quantiles()** ... Dataset毎の統計情報::四分位数を取得している。取得したデータは、予測見積もりnode数に使用している。
- **v394.acquisition_cluster()** ... "acq_00:通常コントラストの標準的な撮影群" or "acq_01:低コントラスト・暗めの撮影群"を判別している。
- **detect_original_and_rotation()** ... メインの細胞検出器(V21モデル群)を実行する際に、「通常画像」と「180度回転させた画像（TTA）」の2通りで推論を行い、2種類の候補テーブルを同時に生成する関数。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant VolZarr as volumes_from_zarr()
    participant Quantiles as image_quantiles()
    participant Cluster as v394.acquisition_cluster()
    participant DetectRot as detect_original_and_rotation()
    participant PeakRet as retain_original_peaks()
    participant Scaffold as specialist_scaffold()
    participant DetectBank as detect() (detect_bank)
    participant RunChain as run_chain() (track_dataset_learned_v8)

    Script->>VolZarr: volumes_from_zarr(path)
    VolZarr-->>Script: shared_volumes

    Script->>Quantiles: image_quantiles(path)
    Quantiles-->>Script: shared_quantiles

    Script->>Cluster: v394.acquisition_cluster(path, router)
    Cluster-->>Script: v394_cluster

    Script->>DetectRot: detect_original_and_rotation(V21_MODELS, shared_volumes)
    DetectRot-->>Script: c21_count, c21, original_flow

    Script->>PeakRet: retain_original_peaks(c21, c21_count)
    PeakRet-->>Script: c21 (ピーク保持)

    Script->>Scaffold: specialist_scaffold(shared_volumes, c21, ...)
    Scaffold-->>Script: c21, v343_union

    Script->>DetectBank: detect(V24 / Stage6 / V39 / V38, shared_volumes)
    DetectBank-->>Script: c24, c_stage6, c39, c38

    par 5系統の並列トラッキング (ThreadPoolExecutor: 3 workers)
        Script->>RunChain: run_chain('v21_track', candidate_override=c21, ...)
        RunChain-->>Script: v21
    and
        Script->>RunChain: run_chain('v24_track', candidate_override=c24)
        RunChain-->>Script: v24
    and
        Script->>RunChain: run_chain('stage6_track', candidate_override=c_stage6)
        RunChain-->>Script: stage6
    and
        Script->>RunChain: run_chain('v39_track', candidate_override=c39)
        RunChain-->>Script: v39
    and
        Script->>RunChain: run_chain('v38_track', candidate_override=c38)
        RunChain-->>Script: v38
    end
```

### 5.1 v394.acquisition_cluster()のシーケンス

- メタデータ抽出（ステップ 1〜5）: 生ボリューム全体をスキャンせずに、Zarr の属性メタデータ（.attrs）から寸法・解像度スケール・パーセンタイル輝度を読み込む。
- 特徴量作成（ステップ 6）: コントラスト比やダイナミックレンジなど、撮影条件を特徴づける数値を計算して辞書にまとめます。
- 標準化（ステップ 7〜11）: 学習時に保存された平均・標準偏差(scaler_mean, scaler_scale)を使って標準化する。
- 最近傍分類（ステップ 12〜14）: あらかじめ学習・決定されている 2 つのクラスタ中心(canonical_centers)との距離を測り、近い方のインデックス(0 または 1)を選ぶ。
- 返却（ステップ 15〜16）: "acq_00" または "acq_01" の文字列として呼び出し元へ戻します。この情報は、後段の4つの後処理フェーズで使用される。

#### &emsp;&emsp; **あらかじめ学習・決定されている 2 つのクラスタ中心(canonical_centers)** とは ... 
&emsp;&emsp;&emsp;[右記](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution)に記載があり。``I grouped movies by image measurements, including brightness and contrast, using standardized features and K-means clustering.``
&emsp;&emsp;GTデータは使わず、trainデータから20特徴量を抜き出して、教師なし学習の「K-means クラスタリング（k=2）」で生成したもの。
&emsp;&emsp;trainデータ(3D動画輝度情報)を使わなかったのは、「情報量が多すぎる」のとDataset毎のコントラストや輝度情報などの 「撮影環境(カメラの写り具合)の違い」を学習させたかったためと思われ。
&emsp;&emsp;もし、trainデータを直接使用すると、「この動画は細胞が左にある」「こっちは右にある」という 細胞の位置のズレばかり を見てしまいかねない。
&emsp;&emsp;K-means によって求まった 2つのグループの重心(中心点)を、そのまま canonical_centers として .npz ファイルに保存している。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(Main Script)
    participant Func as acquisition_cluster()
    participant Zarr as Zarr Storage<br/>(root.attrs & array)
    participant Router as Router Dict<br/>(V394_ROUTER)

    Caller->>Func: acquisition_cluster(path, router)
    activate Func

    Note over Func,Zarr: 1. メタデータと配列情報の取得
    Func->>Zarr: zarr.open_group(path, mode="r")
    Zarr-->>Func: root (Zarr Group)
    Func->>Zarr: root.attrs["multiscales"] から<br/>ボクセルスケール[1.625 μm,0.40625μm,0.40625μm]取得
    Func->>Zarr: array.shape から画像寸法<br/>(frames=100, depth=64, height=256, width=256) 取得
    Func->>Zarr: root.attrs["image_statistics"]["quantiles"] から<br/>各パーセンタイル輝度値(q) 取得

    Note over Func: 2. 特徴量辞書 values の構築<br/>・画像寸法、スケール、ビット深度<br/>・log1p(各パーセンタイル)<br/>・レンジ幅 (q99 - q01, q999 - q001)<br/>・コントラスト比 (q90/q10, q999/q99)

    Note over Func,Router: 3. 特徴量ベクトル化と標準化
    Func->>Router: router["feature_names"] を参照
    Router-->>Func: names = 特徴量名のリスト
    Note over Func: 特徴量名のリスト={<br/>--- 幾何・解像度情報 (8個) ---<br/>'frames', 'depth', 'height', 'width', 'scale_z', 'scale_y', 'scale_x', 'dtype_bits',<br/>--- 輝度パーセンタイル (8個) ---<br/>'log1p_q0p0', 'log1p_q0p001', 'log1p_q0p01', 'log1p_q0p1','log1p_q0p9', 'log1p_q0p99', 'log1p_q0p999', 'log1p_q1p0',<br/>--- レンジ・コントラスト比 (4個) ---<br/>'log1p_range_q99_q01', 'log1p_range_q999_q001','contrast_q90_q10', 'tail_q999_q99'" }
    Func->>Func: 辞書から特徴ベクトルを作成 (vector)
    Note over Func: vectorに、"2."で構築した特徴量辞書 valuesから、names の順番通りに数値を抜き出して、長さ20の1次元配列を格納。
    Func->>Router: router["scaler_mean"], router["scaler_scale"] を参照
    Router-->>Func: 平均値・スケール値
    Func->>Func: scaled = (vector - mean) / scale
    Note over Func: 標準化(Standardization): scaled = (値 - 平均) / 標準偏差
    Note over Func,Router: 4. 代表クラスタ中心との照合 (最近傍探索)
    Func->>Router: router["canonical_centers"] を参照
    Router-->>Func: 代表クラスタ中心 (2点)
    Func->>Func: ユークリッド距離の計算<br/>norm(canonical_centers - scaled, axis=1)
    Func->>Func: 最も近い中心のインデックス取得<br/>label = argmin(...)  # 0 または 1

    Note over Func: 5. クラスタID文字列の生成<br/>f"acq_{label:02d}" -> "acq_00" または "acq_01"

    Func-->>Caller: "acq_00" または "acq_01" を返却
    deactivate Func
```

### 5.2 detect_original_and_rotation()のシーケンス
&emsp;&emsp;・引数1 models: fold0.ep05.pt ～ fold4.ep05.pt
    &emsp;&emsp;&emsp;&emsp;・ **fold[0-4].ep05.pt** ... input:前後3フレームの3D画像、output:1.中心ヒートマップ(heatmap)、2.サブボクセル・オフセット(offset)、3.後退運動ベクトルと
    &emsp;&emsp;&emsp;&emsp;&emsp;不確実性(flow, flow_logvar)、4.画像特徴記述子(features)のモデル。一言で言うと、「前後3フレームの立体画像を入力として受け取り、細胞の中心位
    &emsp;&emsp;&emsp;&emsp;&emsp;置・サブピクセル補正量・前フレームからの移動量・細胞の特徴ベクトルを一括予測する、弱シグナル細胞に強い3D U-Net 検出器」 を、5分割交差検
    &emsp;&emsp;&emsp;&emsp;&emsp;証（5-Fold）で学習させている。
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力1.**中心ヒートマップ(heatmap)** ... 細胞の中心が存在する確率(0〜1）
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力2.**サブボクセル・オフセット(offset)** ... 粗いグリッド(1.625μm)の中での微小な位置ズレ量($\Delta z, \Delta y, \Delta x$)。元の高解像度座標に復元するために使用。
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力3.**後退運動ベクトルと不確実性(flow, flow_logvar)** ... 当該細胞が「前フレーム($t-1$)でどこから動いてきたか」という移動ベクトル(オプティカルフロー)と、
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;その予測の自信のなさ(分散)
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;出力4.**画像特徴記述子(features)** ... 細胞の見た目の特徴を表す埋め込みベクトル。フレーム間で「同じ細胞同士」を結びつける照合に使用。
&emsp;&emsp;・引数2 vols: shared_volumes ... zarrの本体。形状:(T, Z, Y, X) = (100, 64, 64, 64)float16型。すでに「正規化」と「等方化ダウンサンプリング(1.625 μm 立方体グリッド)」済。
&emsp;&emsp;・引数3 device: SPEC.device ... torch.device("cuda")。要はGPU指定。
&emsp;&emsp;・引数4 amp_dtype: SPEC.amp_dtype ... torch.float16。
&emsp;&emsp;・引数5 batch: バッチサイズ(整数値 8)。100フレームある動画を、「1回に8フレームずつまとめて」GPU に送って推論する。
&emsp;&emsp;・引数6 return_flow=True: 真偽値フラグ。モデルが出力する 「細胞の移動ベクトル(flow: 前フレームへの動きの向きと大きさ)」を戻り値に含めるかどうか。True に設定することで、
&emsp;&emsp;&emsp;関数の3番目の戻り値として original_flowを受け取る。このベクトルは、後段で「途切れた細胞の軌跡を繋ぐ(Gap closure)」際に、細胞がどこから移動してきたかを辿るために再利用する。

- 時系列スタックの作成（ステップ2〜3）:
指定したバッチの時刻 ts に対して、前後1フレームを含む時系列ボリューム（3フレーム分）を GPU に転送、入力テンソル x を構築。
- 通常推論と回転推論の累積加算（ステップ4〜12）:
同じバッチ x に対して、通常画像と 180度回転画像(torch.flip)の両方をモデルに入力。
得られた出力から、通常ヒートマップ(heat)、元画像と回転画像をTTA平均化したヒートマップ(rotated_heat)、移動ベクトル(flow)、不確実性(flow_logvar)、サブボクセルオフセット(offset)、および細胞識別特徴量(features)をそれぞれ取り出してモデル間で累積加算(+=)。
- モデルアンサンブルの平均化(ステップ 13〜18):
累積加算した各テンソルをモデル数 $k$ で除算し、モデル間のアンサンブル平均値を算出。
- 2系統の候補テーブル化(ステップ 19〜28):
zip(rows, (heat, rotated_heat)) により _append_rows() を2回呼び出す。
内部ではピーク検出・座標補正・細胞識別特徴量(48次元)のサンプリング等(ステップ 20〜25)を行い、通常ヒートマップ由来の候補(c21_count 相当: ステップ 26)と、回転TTAヒートマップ由来の候補(c21 相当: ステップ 27〜28)の2系統の DataFrame を同時に生成する。
- 一括返却(ステップ 29):
c21_count、c21、および途切れた軌跡の接続に使う運動ベクトル original_flow の3点をまとめて呼び出し元に返却。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(2nd-place-notebook.py)
    participant Func as detect_original_and_rotation()
    participant Stacks as _stacks()<br/>(コンテキスト構築)
    participant GPU as PyTorch / GPU<br/>(V21 Models)
    participant Append as _append_rows()<br/>(ピーク抽出・テーブル化)

    Caller->>Func: detect_original_and_rotation(models, vols, device, return_flow=True)
    activate Func

    loop バッチ単位の処理 (batch=8)
        Func->>Stacks: _stacks(vols, ts, context=3)
        Stacks-->>Func: 前後フレームを重ねた3D入力テンソル(x[8, 3, 64, 64, 64])

        loop 各モデルごとの推論 (for model in models)
            Note over Func,GPU: 1. 通常と180度回転の2通りで推論<br/>with torch.autocast("cuda", dtype=amp_dtype,enabled=amp_dtype is not None and device.type == "cuda"):の効果(自動混合精度を有効)範囲内。
            Func->>GPU: original = model(x)
            Func->>GPU: rotated = model(torch.flip(x, dims=(-2, -1)))
            GPU-->>Func: 各モデルの出力辞書 (heatmap, flow, offset, features)

            Note over Func: 2. 戻り値整形
            Func->>Func: 戻り値1(heat)の累積加算: 通常ヒートマップの正規化(0-1: from sigmoid())
            Func->>Func: 戻り値2(rotated_heat)※の累積加算:回転ヒートマップではなく、「元画像と回転画像をTTAアンサンブル(平均化)したヒートマップ」 が格納される。
            Func->>Func: 戻り値3(flow)の累積加算
            Func->>Func: 戻り値4(flow_logvar)の累積加算
            Func->>Func: 戻り値5(offset)の累積加算
            Func->>Func: 戻り値6(features)の累積加算
            Note over Func: heat:(8, Z, Y, X):通常ヒートマップ。<br/> rotated_heat:(8, Z, Y, X):元画像と回転画像をTTAアンサンブル(平均化)したヒートマップ。<br/>flow:(8, 3, Z, Y, X):前後のフレーム間で細胞が移動する「3次元移動量ベクトル(dz, dy, dx)」単位は[μm]。<br/>flow_logvar:(8, Z, Y, X):動きの不確実性(分散の対数)<br/>offset:(8, 3, Z, Y, X):3次元ベクトル(Z, Y, X)方向の移動量・微小ズレ<br/>features:(8, n_emb, Z, Y, X):細胞識別特徴量(n_embは48チャンネル)
        end

        Note over Func: 3. モデル数 k=5 で除算してアンサンブル平均化
        Func->>Func: 戻り値1(heat)の累積加算 / k
        Func->>Func: 戻り値2(rotated_heat)の累積加算 / k
        Func->>Func: 戻り値3(flow)の累積加算 / k
        Func->>Func: 戻り値4(flow_logvar)の累積加算 / k
        Func->>Func: 戻り値5(offset)の累積加算 / k
        Func->>Func: 戻り値6(features)の累積加算 / k

        opt return_flow == True
            Note over Func: flow_batches に flow を退避
        end

        Note over Func,Append: 4. ピーク検出と候補抽出 (2系列)
        Func->>Append: _append_rows(heat, ...)<br/> 「1行＝1個の細胞」とした検出テーブル(DataFrame)を生成
        Append->>Append: ステップ1：細胞中心のピーク検出(L16〜23)
        Append->>Append: ステップ2：座標の精密補正(サブボクセル化)(29〜32)
        Append->>Append: ステップ3：動きベクトルと不確実性(標準偏差)の取得(L26〜27)
        Append->>Append: ステップ4：周囲の環境・幾何特徴の算出(L34〜38)
        Append->>Append: ステップ5：ピーク位置のボクセルから、48次元の 細胞識別特徴量(Embedding) を抜き出し(L43〜45)
        Append->>Append: ステップ6：DataFrame化(L47〜64)
        Note over Append: t, z, y, x ... 時間と精密化された3次元空間座標<br/>score, rank_in_frame	... 確信度スコアと、そのフレーム内での順位<br/>flow_z, flow_y, flow_x ... 次のフレームへの移動量予測ベクトル（$\mu\text{m}$）<br/>flow_sigma_um ... 動き予測の不確実性（標準偏差）<br/>near_um, density_10um ... 周囲の混雑度・最寄りの細胞距離<br/>peak_width_um, boundary_um	... 細胞サイズ・視野境界までの距離<br/>emb_0 〜 emb_47 ... 細胞識別特徴量(48次元ベクトル)
        Append-->>Func: rows[0] に候補追加(後の c21_count)
        Func->>Append: _append_rows(rotated_heat, ...)
        Append-->>Func: rows[1] に候補追加(後の c21)
    end
    Func-->>Caller: (c21_count, c21, original_flow) を返却
    deactivate Func

```

※**sigmoid** ... シグモイド関数(活性化関数)は、勾配消失問題があるが、それは学習時かつ中間層で使用する時に表面化する。今回は推論時かつ最終層で使用するだけなので勾配計算が発生せず問題ない。
※**戻り値2(rotated_heat)の算出** ...
  &emsp;・probability = 0.5 * (torch.sigmoid(original["heatmap"].float()) + torch.flip(torch.sigmoid(rotated["heatmap"].float()), dims=(-2, -1)))
  &emsp;&emsp;→ 元画像と回転画像をシグモイド化 → TTAアンサンブル(平均化)する。
  &emsp;・averaged_logit = torch.logit(probability.clamp(1e-6, 1.-1e-6))
  &emsp;&emsp;→ 確率($0 \sim 1$)を、元のモデル出力と同じ形式である Logit(未正規化スコア：$-\infty \sim +\infty$)に逆変換。Logit(ロジット)関数は、シグモイド関数の逆関数で、数式は以下の通り：
    $$\text{logit}(p) = \log\left(\frac{p}{1 - p}\right)$$
  &emsp;&emsp;もし確率 $p$ が完全に 0.0 や 1.0 になると：
  &emsp;&emsp;&emsp;$p = 0 \implies \log(0) \to -\infty$
  &emsp;&emsp;&emsp;$p = 1 \implies \frac{1}{0} \to +\infty$（ゼロ除算）
  &emsp;&emsp;となり、オーバーフローや数値計算エラー(NaN や Inf)が発生するため、確率値を $0.000001 \sim 0.999999$ の範囲に強制的に収める(クリッピングする)処理をしている。
  &emsp;・rh = torch.sigmoid(averaged_logit[:, 0].float()).cpu().numpy()
  &emsp;&emsp;→ averaged_logitをシグモイド化。sigmoid化 → logitで戻し → またsigmoid化してこの流れは無駄になってる。

### 5.3 retain_original_peaks()のシーケンス
**概要**: TTA平均化で取りこぼしてしまった細胞を、元画像の候補から3次元空間上で探し出して“救済・合流”させる。3次元の空間幾何計算(KD-Tree)や重複排除(NMS)を使用。

**引数**:
| **引数名** | **データ型** | **中身（中身の正体）** | 
|---|---|---|
| c21 | DataFrame | 「TTA回転アンサンブル」で平均化したヒートマップから検出した細胞候補テーブル(ノイズが少なしメインとなるベース) | 
| c21_count | DataFrame | 「元画像単体(TTAなし)」のヒートマップから検出した細胞候補テーブル(TTAで消えてしまった細胞をここから探して救済する) | 

**処理概要**:
- 空き枠の確認(ステップ 4): 1フレームあたりの上限枠(デフォルト3000件)に対して、TTA版(c21)の検出数に余裕があるか確認します。
- KD-Tree による重複判定(ステップ 5〜8): 元画像側(c21_count)の候補のうち、TTA側の既存細胞から $3.0\ \mu\text{m}$ より離れているものだけ を「見逃し細胞」として抽出します。
- NMS 間引き(ステップ 9〜11): 救済候補同士が近すぎる場合に共倒れしないよう NMS をかけ、空き枠数(slots)の分だけスコア上位を採用します。
- コンテキスト再計算(ステップ 14〜16): 細胞が追加されて周囲の配置が変わるため、最寄り細胞距離(near_um)や混雑度(density_10um)を再計算して整合性を保ちます。
- 返却(ステップ 20): 合成・救済された最終テーブル(result $\to$ 次の c21)と、何件救済されたかを記録した監査辞書(peak_retention_audit)を返します。

```mermaid
sequenceDiagram
    autonumber
    actor Caller as 呼び出し元<br/>(2nd-place-notebook.py)
    participant Func as retain_original_peaks()<br/>(heat_peak_retention.py)
    participant KDTree as cKDTree<br/>(3次元空間インデックス)
    participant NMS as select_frame()<br/>(NMS重複排除)
    participant NMSSub as NMS重複排除本体
    participant Feat as _neighbour_features()<br/>(近傍特徴量の再計算)

    Caller->>Func: retain_original_peaks(c21, c21_count)
    activate Func

    Note over Func: 入力バリデーション(カラム一致・非空チェック)

    loop 各フレームごとの処理 (for t in sorted_times)
        Note over Func: 当該時刻 t の候補インデックス(s)を取得<br/>hi: c21側(TTA), oi: c21_count側(Original)
        Func->>Func: 空き枠の確認: slots = TOP_K_PER_FRAME - len(hi)

        alt slots > 0 かつ len(oi) > 0 (空き枠があり、元画像候補が存在)
            Note over Func,KDTree: 1. 空間的重複の判定(3.0μmチェック)
            Func->>KDTree: cKDTree(c21の3D実座標 * SCALE_ARR[1.625, 0.40625, 0.40625])
            Func->>KDTree: query(c21_countの3D実座標 * SCALE_ARR)
            KDTree-->>Func: 最寄りのc21細胞までの距離 (distance)
            Func->>Func: distance > 3.0μm の候補のみ抽出 (available)

            opt available な候補が存在する場合
                Note over Func,NMS: 2. 救済候補内のNMSとスロット割当
                Func->>NMS: select_frame(座標, スコア, MATCH_UM=3.0)
                NMS->>NMS: idx = np.arange(len(zyx), スコア, 3μm)
                NMS->>NMSSub: nms(zyx[idx]*SCALE_ARR,scores[idx],3μm)
                NMSSub->>NMSSub: aaa
                NMS-->>Func: 近接重複を除去した上位インデックス (keep)
                Func->>Func: 空き枠分を採用: selected = oi[available[keep]][:slots]
            end
        end

        alt selected が存在する場合 (救済セルがある)
            Note over Func: 3. テーブル結合とソート
            Func->>Func: pd.concat([c21の行, 救済されたc21_countの行])
            Func->>Func: score の降順で安定ソート (sort_values)

            Note over Func,Feat: 4. 周囲環境特徴量の再計算
            Func->>Feat: _neighbour_features(全セルの3D座標, スコア)
            Feat-->>Func: 更新された near_um, density_10um
            Func->>Func: rank_in_frame を 0..N-1 で再採番

            Note over Func: 5. 座標・特徴量フィールドの非改変チェック
            Func->>Func: assert(元の座標やemb_0..47が破壊されていないこと)
            Func->>Func: pieces に合成フレームを追加
        else selected が空の場合 (救済セルなし)
            Func->>Func: pieces に c21 の行をそのまま追加
        end
    end

    Note over Func: 6. 全フレームの結合と最終監査
    Func->>Func: result = pd.concat(pieces, ignore_index=True)
    Func->>Func: 総行数の整合性検証 (len(result) == len(c21) + len(additions))

    Func-->>Caller: (result, peak_retention_audit) を返却
    deactivate Func
```

- **配列とスカラーの比較(NumPyのブロードキャスト)** ... NumPyは、1次元配列とスカラーを == で比較すると、ブロードキャスト機能によって配列の全要素に対してスカラー t との一致判定が自動的に行われます。
    ```python
    heat_times = np.array([0, 0, 1, 2, 1])
    t = 1  # スカラー値
    print(heat_times == t)
    # 出力: array([False, False, True, False, True])
    ```

---

## 6. (cell 5: line 706〜789) 処理B: グラフ統合・コンセンサス・ギャップ補完

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant AddSeg as add_full_length_segments()
    participant AddStage6 as add_stage6_v24_consensus_edges()
    participant Exchange as apply_exchange()
    participant AddConsensus as add_consensus_edges()
    participant ForkVotes as fork_votes()
    participant Materialize as materialize()
    participant TempMotion as sample_temporal_motion()
    participant ProposeGaps as propose_gaps()
    participant MatGaps as materialize_gaps()
    participant Corresp as apply_correspondence()
    participant MotionSub as apply_motion_substitution()

    Script->>AddSeg: add_full_length_segments(v21.graph, v24.graph)
    AddSeg-->>Script: phase19_raw

    Script->>AddSeg: add_full_length_segments(phase19_raw, stage6.graph)
    AddSeg-->>Script: stage_raw

    Script->>AddStage6: add_stage6_v24_consensus_edges(stage_raw, stage6.graph, v24.graph)
    AddStage6-->>Script: v37_raw

    Script->>Exchange: apply_exchange(v37_raw, v39.graph, v39_actions, removed)
    Exchange-->>Script: v51_raw

    Script->>AddConsensus: add_consensus_edges(v51_raw, witnesses)
    AddConsensus-->>Script: v52_raw

    Script->>AddSeg: add_full_length_segments(v52_raw, v38.graph)
    AddSeg-->>Script: after_v38_raw

    Script->>ForkVotes: fork_votes(after_v38_raw, division_witnesses)
    ForkVotes-->>Script: division_votes
    Script->>Materialize: materialize(after_v38_raw, division_actions)
    Materialize-->>Script: final_raw (分裂コンセンサス後)

    Script->>TempMotion: sample_temporal_motion(V21_MODELS, shared_volumes, ...)
    TempMotion-->>Script: gap_backward, gap_forward

    Script->>ProposeGaps: propose_gaps(final_raw, ...)
    ProposeGaps-->>Script: gap_actions
    Script->>MatGaps: materialize_gaps(final_raw, gap_actions)
    MatGaps-->>Script: final_raw (ギャップ補完後)

    Script->>Corresp: apply_correspondence(final_raw, final_out, ..., V230_MODELS)
    Corresp-->>Script: final_raw, final_out

    Script->>MotionSub: apply_motion_substitution(final_raw, final_out, ...)
    MotionSub-->>Script: final_raw, final_out
```

---

## 7. (cell 5: line 790〜887) 処理C: グラフ確定と多段プロテクター後処理

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant ValidGraph as validate_graph()
    participant FinalizeCV as finalize_cv_graph()
    participant PruneComp as v394.prune_components()
    participant TrimTails as v399.trim_terminal_tails()
    participant GapAdmit as v454.apply_gap_admission()
    participant Prep507 as v507.prepare_actions()
    participant Apply507 as v507.apply_prepared()
    participant Apply533 as v533.apply()

    Script->>ValidGraph: validate_graph(final_raw / final_out)
    ValidGraph-->>Script: 検証通過

    Script->>FinalizeCV: finalize_cv_graph(final_raw, final_out)
    FinalizeCV-->>Script: v389_graph

    Script->>PruneComp: v394.prune_components(v389_graph, shared_volumes, ...)
    PruneComp-->>Script: v394_graph

    Script->>TrimTails: v399.trim_terminal_tails(v394_graph, shared_volumes, ...)
    TrimTails-->>Script: v399_graph

    Script->>GapAdmit: v454.apply_gap_admission(v399_graph, ...)
    GapAdmit-->>Script: v454_graph

    Script->>Prep507: v507.prepare_actions(v454_graph, c21, ...)
    Prep507-->>Script: v507_prepared
    Script->>Apply507: v507.apply_prepared(v454_graph, v507_prepared)
    Apply507-->>Script: v507_graph

    Script->>Apply533: v533.apply(v507_graph, c21, v394_cluster, path)
    Apply533-->>Script: v533_graph (最終グラフを graphs リストへ追加)
```

---

## 8. (cell 6: line 891〜911) サブミッションの検証・出力
submission.csvを生成する処理。大したことはやってない。

```mermaid
sequenceDiagram
    autonumber
    actor Script as Main Script
    participant BuildSub as build_submission()
    participant WriteSub as write_submission()
    participant FileSys as Path.replace() / write_text()

    Script->>BuildSub: build_submission(graphs)
    BuildSub-->>Script: df (DataFrame), summary

    Script->>WriteSub: write_submission(df, temporary_output)
    WriteSub-->>Script: 一時ファイル (.tmp.csv) 書き込み完了

    Script->>FileSys: temporary_output.replace(OUT)
    FileSys-->>Script: submission.csv 確定置換
    Script->>FileSys: audit_path.write_text(audit_json)
    FileSys-->>Script: v533_deployment_audit.json 保存完了
```
