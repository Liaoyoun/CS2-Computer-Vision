"""訓練流程的進入點：訓練 YOLO、把結果上傳 MLflow、註冊成一個模型版本。

由 Jenkinsfile.train 在 GPU 執行機的 container 裡呼叫，也可以在本機直接執行：

    python train/train.py --epochs 3 --device cpu
"""

import argparse
import json
from pathlib import Path

RESULT_FILE = Path("train_result.json")


def build_parser():
    parser = argparse.ArgumentParser(description="訓練 YOLO 並上傳到 MLflow")
    parser.add_argument("--data", default="coco8.yaml", help="資料集設定檔")
    parser.add_argument("--base-model", default="yolo11n.pt", help="基礎模型")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU 編號，或 cpu")
    parser.add_argument("--experiment", default="yolo-detection")
    parser.add_argument("--model-name", default="yolo-detector", help="MLflow 上註冊的模型名稱")
    return parser


def collect_metrics(result, names):
    """把 Ultralytics 的驗證結果整理成 MLflow 可以直接記錄的 {名稱: 數值}。"""
    metrics = {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
    }
    for class_id, value in enumerate(result.box.maps):
        class_name = str(names[class_id]).replace(" ", "_")
        metrics[f"mAP50_95/{class_name}"] = float(value)
    return metrics


def log_training_outputs(mlflow, run_dir, best_pt):
    """上傳權重檔與 Ultralytics 產生的圖表，方便在 MLflow 介面上直接看。"""
    mlflow.log_artifact(str(best_pt), artifact_path="weights")
    for pattern in ("*.png", "*.jpg", "results.csv", "args.yaml"):
        for file in sorted(run_dir.glob(pattern)):
            mlflow.log_artifact(str(file), artifact_path="training")


def main():
    args = build_parser().parse_args()

    # 這些套件很大，放在這裡才匯入，CI 測試 collect_metrics 時就不需要安裝它們。
    import mlflow
    import mlflow.onnx
    import onnx
    from ultralytics import YOLO, settings

    # 由這支程式自己記錄 MLflow，關掉 Ultralytics 內建的那一份，避免同一次訓練出現兩個 Run。
    settings.update({"mlflow": False, "datasets_dir": str(Path.cwd() / "datasets")})

    # 位址來自環境變數 MLFLOW_TRACKING_URI，程式裡不寫死。
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data": args.data,
                "base_model": args.base_model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
            }
        )

        model = YOLO(args.base_model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(Path.cwd() / "runs"),
            name=run.info.run_id,
        )

        best_pt = Path(model.trainer.best)
        run_dir = best_pt.parent.parent

        best_model = YOLO(str(best_pt))
        result = best_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
        metrics = collect_metrics(result, best_model.names)
        mlflow.log_metrics(metrics)

        log_training_outputs(mlflow, run_dir, best_pt)

        # 偵測服務用 ONNX：它不綁定顯示卡型號，訓練機匯出的檔案搬到正式環境也能用。
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz)

        # 每次訓練成功都註冊成一個新的模型版本；要不要部署由人在部署關卡決定。
        model_info = mlflow.onnx.log_model(
            onnx.load(onnx_path),
            name="model",
            registered_model_name=args.model_name,
        )

    summary = {
        "run_id": run.info.run_id,
        "model_name": args.model_name,
        "model_version": str(model_info.registered_model_version),
        "metrics": {key: value for key, value in metrics.items() if "/" not in key},
    }
    RESULT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模型版本：{summary['model_name']} v{summary['model_version']}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
