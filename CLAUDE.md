# Weather Forecast Project

## プロジェクト概要

過去の気象データから「翌日の天気（晴れ/曇り/雨）」と「翌日の最高気温」を予測する機械学習モデル。
Python初心者からスタートし、段階的にモデルを高度化していく学習プロジェクト。

## ロードマップ

### Stage 1: scikit-learn ランダムフォレスト（現在）
- 天気分類（3クラス: 晴れ/曇り/雨）+ 気温回帰（翌日最高気温）
- ランダムフォレスト（決定木100本のアンサンブル）
- 精度目標: 分類 accuracy 60%以上、回帰 MAE 3℃以内

### Stage 2: PyTorch LSTM / Transformer（未着手）
- Stage 1の結果をベースラインとして、ディープラーニングモデルに置き換え
- 時系列の依存関係を捉える
- データパイプライン（前処理・評価）はStage 1から流用

### Stage 3: 物理モデル / ハイブリッド（未着手）
- ML + 物理シミュレーションの組み合わせ

## 現在の状態

- **Stage 1のコードは完成**: 学習・予測・評価・可視化の全パイプラインが動作する
- **サンプルデータでの結果**: 分類accuracy 49%（目標未達）、気温MAE 2.39℃（目標達成）
- **実データ未取得**: 気象庁の東京データをダウンロードすれば精度改善が見込める
- **テスト未作成**: TDDで追加する予定

## 次のステップ

1. 気象庁の実データ（東京、過去5年分）を取得して学習→精度確認
2. テストを TDD で追加
3. 実データで目標精度を達成したら Stage 2 へ

## アーキテクチャ

オニオンアーキテクチャ。依存は内向きのみ。

```
controller → usecase → domain ← infrastructure
```

| 層 | 責務 | 外部依存 |
|----|------|---------|
| domain | 型定義（WeatherCategory enum等）、評価指標の純粋計算 | なし |
| usecase | 前処理、学習、予測、評価のビジネスロジック | scikit-learn |
| infrastructure | 気象庁CSV読み込み、モデル保存(pickle)、グラフ描画(matplotlib) | pandas, matplotlib |
| controller | 各層を組み合わせてフローを実行 | なし（usecase+infraを呼ぶだけ） |

## 技術スタック

- Python 3.12 + venv
- pandas, numpy, scikit-learn, matplotlib
- 日本語フォント: Noto Sans CJK JP（`sudo apt install fonts-noto-cjk`）

## データ

- **入力**: 気象庁の日別気象データCSV（Shift_JIS, 3行ヘッダー）
- **対象地点**: 東京（デフォルト）
- **8特徴量**: 平均気温, 最高気温, 最低気温, 降水量, 平均湿度, 平均気圧, 平均風速, 日照時間
- **天気分類ルール**: 天気概況テキストから3カテゴリに変換（優先順位: 雨 > 曇り > 晴れ）
- **サンプルデータ生成**: `infrastructure/jma_data_loader.py` の `create_sample_data()` でsin波季節変動の擬似データを生成可能

## CLI

```bash
python main.py train --sample        # サンプルデータで学習
python main.py train --csv data/tokyo.csv  # 実データで学習
python main.py predict --temp 22 --humidity 60 --pressure 1013 ...  # 予測
```

## ドキュメント

- `README.md` — セットアップ・使い方
- `docs/architecture.md` — オニオンアーキテクチャの詳細・データフロー図
- `docs/models.md` — MLモデルの仕組み・前処理・評価指標

コード変更時はドキュメントの更新も必要（`.claude/settings.json` のフックでリマインド）。

## ユーザーについて

- Python・機械学習はほぼ初めて。専門用語は「なぜその名前か」も添えて説明する
- 設計判断は自分で下したい（選択肢と推奨を提示する）
- オニオンアーキテクチャによるレイヤー分離を重視
