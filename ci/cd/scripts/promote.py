"""部署流程最後一步：rollout 成功後，把 champion 標記移到剛上線的模型版本。

只在偵測服務確定更新成功之後才執行，所以 champion 永遠代表真的在正式環境跑的版本。

    python scripts/promote.py --name yolo-detector --version 3 --image registry/…:v3-abc1234
"""

import argparse

from mlflow import MlflowClient

ALIAS = "champion"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--image", required=True, help="這個版本對應的 image，記下來方便追查")
    args = parser.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(args.name, ALIAS, args.version)
    client.set_model_version_tag(args.name, args.version, "deployed_image", args.image)

    print(f"{args.name}@{ALIAS} 現在指向 v{args.version}")


if __name__ == "__main__":
    main()
