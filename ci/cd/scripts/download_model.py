"""部署流程第一步：印出指定模型版本與現任正式版的指標，再把模型下載下來。

    python scripts/download_model.py --name yolo-detector --version 3 --out model
"""

import argparse
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ALIAS = "champion"
SUMMARY_METRICS = ("precision", "recall", "mAP50", "mAP50_95")


def get_metrics(client, model_version):
    if not model_version.run_id:
        return {}
    return client.get_run(model_version.run_id).data.metrics


def get_champion(client, name):
    try:
        return client.get_model_version_by_alias(name, ALIAS)
    except MlflowException:
        return None  # 第一次部署，還沒有正式版


def print_comparison(client, name, candidate):
    champion = get_champion(client, name)
    candidate_metrics = get_metrics(client, candidate)
    champion_metrics = get_metrics(client, champion) if champion else {}
    champion_label = f"正式版 v{champion.version}" if champion else "正式版（尚無）"

    print(f"{'指標':<12}{'要部署的 v' + str(candidate.version):>18}{champion_label:>18}")
    for key in SUMMARY_METRICS:
        new = candidate_metrics.get(key)
        old = champion_metrics.get(key)
        new_text = f"{new:.4f}" if new is not None else "-"
        old_text = f"{old:.4f}" if old is not None else "-"
        print(f"{key:<12}{new_text:>18}{old_text:>18}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", default="model")
    args = parser.parse_args()

    client = MlflowClient()
    candidate = client.get_model_version(args.name, args.version)  # 版本不存在會在這裡失敗
    print_comparison(client, args.name, candidate)

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{args.name}/{args.version}",
        dst_path=str(out_dir),
    )

    onnx_file = out_dir / "model.onnx"
    if not onnx_file.is_file():
        raise SystemExit(f"下載的模型版本裡找不到 {onnx_file}")
    print(f"已下載：{onnx_file}")


if __name__ == "__main__":
    main()
