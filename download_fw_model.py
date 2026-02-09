import argparse
from faster_whisper import download_model

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description="faster-whisper モデルダウンローダー")
    parser.add_argument(
        "model_name", 
        type=str, 
        help="ダウンロードするモデル名 (例: kotoba-tech/kotoba-whisper-v2.2-faster)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="保存先ディレクトリ (指定しない場合はデフォルトのキャッシュディレクトリ)"
    )

    args = parser.parse_args()

    print(f"Downloading model: {args.model_name} ...")
    
    try:
        # download_model関数を使用することで、メモリ/VRAMへのロードを行わずにダウンロードだけ実行します
        model_path = download_model(args.model_name, output_dir=args.output_dir)
        
        print("-" * 30)
        print("Download Complete!")
        print(f"Model saved at: {model_path}")
        print("-" * 30)

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()