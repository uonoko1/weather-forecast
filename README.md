# weather-forecast

過去の気象データから「明日の天気」と「明日の最高気温」を予測する機械学習モデル。

## 機能

| 機能 | 内容 | 例 |
|------|------|-----|
| 天気分類 | 当日の気象データから翌日が晴れ/曇り/雨のどれかを予測 | 湿度70%, 気圧1008hPa → 「雨」 |
| 気温予測 | 当日の気象データから翌日の最高気温を予測 | 平均気温22℃, 日照3h → 「30.6℃」 |

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

日本語フォント（グラフ表示用）:

```bash
sudo apt install -y fonts-noto-cjk
```

## 使い方

### 学習

```bash
# サンプルデータで学習（動作確認用）
python main.py train --sample

# 気象庁CSVで学習
python main.py train --csv data/tokyo.csv
```

学習が完了すると以下が生成される:
- `models/` — 学習済みモデル（.pkl）
- `output/` — 評価グラフ（混同行列、予測vs実測、特徴量重要度）

### 予測

```bash
python main.py predict --temp 22 --max-temp 28 --min-temp 15 \
  --precip 0 --humidity 60 --pressure 1013 --wind 3 --sunshine 6
```

全オプション:

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--temp` | 平均気温 (℃) | 20.0 |
| `--max-temp` | 最高気温 (℃) | 25.0 |
| `--min-temp` | 最低気温 (℃) | 15.0 |
| `--precip` | 降水量 (mm) | 0.0 |
| `--humidity` | 平均湿度 (%) | 60.0 |
| `--pressure` | 平均気圧 (hPa) | 1013.0 |
| `--wind` | 平均風速 (m/s) | 3.0 |
| `--sunshine` | 日照時間 (h) | 6.0 |

## 気象庁データの取得方法

1. [気象庁 過去の気象データ・ダウンロード](https://www.data.jma.go.jp/gmd/risk/obsdl/index.php) にアクセス
2. 地点: 東京（または任意の地点）
3. 項目: 平均気温、最高気温、最低気温、降水量、平均湿度、平均海面気圧、平均風速、日照時間、天気概況（昼）
4. 期間: 5年分程度
5. CSVダウンロード → `data/` ディレクトリに配置

## プロジェクト構成

```
weather-forecast/
├── src/
│   ├── domain/           # 型定義・評価指標（外部依存ゼロ）
│   ├── usecase/          # 前処理・学習・予測・評価のロジック
│   ├── infrastructure/   # CSV読み込み・モデル保存・グラフ描画
│   └── controller/       # フロー制御（各層を繋ぐ）
├── docs/
│   ├── architecture.md   # アーキテクチャ詳細
│   └── models.md         # 機械学習モデルの解説
├── main.py               # CLIエントリポイント
├── data/                 # 気象庁CSV
├── models/               # 学習済みモデル（.gitignore）
└── output/               # 評価グラフ（.gitignore）
```

詳細は [docs/architecture.md](docs/architecture.md) を参照。
