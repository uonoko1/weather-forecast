"""天気予測モデル - CLIエントリポイント。

使い方:
    # サンプルデータで学習（まず動作確認用）
    python main.py train --sample

    # 気象庁CSVで学習
    python main.py train --csv data/tokyo.csv

    # 予測（学習済みモデルが必要）
    python main.py predict
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="天気予測モデル")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # train コマンド
    train_parser = subparsers.add_parser("train", help="モデルを学習する")
    train_parser.add_argument("--csv", type=str, help="気象庁CSVファイルのパス")
    train_parser.add_argument(
        "--sample", action="store_true", help="サンプルデータで学習する（動作確認用）"
    )

    # predict コマンド
    predict_parser = subparsers.add_parser("predict", help="天気を予測する")
    predict_parser.add_argument("--temp", type=float, default=20.0, help="平均気温")
    predict_parser.add_argument("--max-temp", type=float, default=25.0, help="最高気温")
    predict_parser.add_argument("--min-temp", type=float, default=15.0, help="最低気温")
    predict_parser.add_argument("--precip", type=float, default=0.0, help="降水量(mm)")
    predict_parser.add_argument("--humidity", type=float, default=60.0, help="平均湿度(%)")
    predict_parser.add_argument("--pressure", type=float, default=1013.0, help="平均気圧(hPa)")
    predict_parser.add_argument("--wind", type=float, default=3.0, help="平均風速(m/s)")
    predict_parser.add_argument("--sunshine", type=float, default=6.0, help="日照時間(h)")

    args = parser.parse_args()

    if args.command == "train":
        from src.controller.train_controller import run_training

        if not args.csv and not args.sample:
            print("エラー: --csv か --sample のどちらかを指定してください")
            print("例: python main.py train --sample")
            sys.exit(1)

        run_training(csv_path=args.csv, use_sample=args.sample)

    elif args.command == "predict":
        from src.controller.predict_controller import run_prediction

        data = {
            "avg_temperature": args.temp,
            "max_temperature": args.max_temp,
            "min_temperature": args.min_temp,
            "precipitation": args.precip,
            "avg_humidity": args.humidity,
            "avg_pressure": args.pressure,
            "avg_wind_speed": args.wind,
            "sunshine_hours": args.sunshine,
        }

        print("📊 入力データ:")
        for k, v in data.items():
            print(f"  {k}: {v}")
        print()

        result = run_prediction(data)
        print("🔮 予測結果:")
        print(f"  明日の天気: {result['weather']}")
        print(f"  明日の最高気温: {result['max_temperature']}℃")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
