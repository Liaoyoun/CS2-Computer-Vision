import torch
import json
import shutil

from ultralytics import YOLO
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_PATH / "dataset"
YAML_PATH = DATASET_PATH / "data.yaml"
BASE_MODEL_PATH = PROJECT_PATH / "yolo11s.pt"

IMAGE_SIZE = 640
EPOCHS = 150
BATCH_SIZE = -1
PATIENCE = 30
DEVICE = 0


# 檢查資料集是否完整
def checkData():

    missing = []

    for folder in ["train", "valid"]:

        imagesPath = DATASET_PATH / folder / "images"
        labelsPath = DATASET_PATH / folder / "labels"

        if not imagesPath.is_dir():

            missing.append(
                f"{folder}/images 不存在"
            )

        elif not any(
            path.is_file()
            for path in imagesPath.iterdir()
        ):

            missing.append(
                f"{folder}/images 沒有圖片"
            )

        if not labelsPath.is_dir():

            missing.append(
                f"{folder}/labels 不存在"
            )

        elif not any(
            labelsPath.glob("*.txt")
        ):

            missing.append(
                f"{folder}/labels 沒有標註檔"
            )

    if missing:

        print(
            f"資料集有問題："
            f"{', '.join(missing)}"
        )

        return False

    print("資料集檢查完成")

    return True


# 檢查並建立 data.yaml
def checkYaml():

    DATASET_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    content = f"""path: {DATASET_PATH.as_posix()}

train: train/images
val: valid/images

nc: 4

names:
  0: CTBody
  1: CTHead
  2: TBody
  3: THead
"""

    if (
        not YAML_PATH.exists()
        or YAML_PATH.read_text(
            encoding="utf-8"
        ).strip() != content.strip()
    ):

        YAML_PATH.write_text(
            content,
            encoding="utf-8"
        )

        print(
            "已自動建立或修正 dataset/data.yaml"
        )

    print(
        f"找到 data.yaml：{YAML_PATH}"
    )

    return True


# 檢查 GPU 與 CUDA 環境
def checkEnv():

    if not torch.cuda.is_available():

        print("CUDA 無法使用")

        return False

    gpu = torch.cuda.get_device_name(
        DEVICE
    )

    print(
        f"GPU：{gpu}"
    )

    return True


# 檢查基礎模型是否存在
def checkBaseModel():

    if not BASE_MODEL_PATH.is_file():

        print(
            f"找不到模型：{BASE_MODEL_PATH}"
        )

        return False

    if BASE_MODEL_PATH.stat().st_size == 0:

        print(
            "yolo11s.pt 是空檔案"
        )

        return False

    try:

        YOLO(
            str(BASE_MODEL_PATH)
        )

    except Exception as error:

        print(
            f"yolo11s.pt 無法載入：{error}"
        )

        return False

    print(
        f"基礎模型：{BASE_MODEL_PATH}"
    )

    return True


# 建立 history 與 best 資料夾
def checkModel():

    historyPath = PROJECT_PATH / "history"
    bestPath = PROJECT_PATH / "best"

    historyPath.mkdir(
        parents=True,
        exist_ok=True
    )

    bestPath.mkdir(
        parents=True,
        exist_ok=True
    )

    return historyPath, bestPath

# 訓練 YOLO 模型
def trainModel(
    modelPath,
    historyPath
):

    print(
        f"使用模型：{modelPath}"
    )

    print(
        f"使用資料集：{YAML_PATH}"
    )

    print(
        f"imgsz：{IMAGE_SIZE}"
    )

    print(
        f"epochs：{EPOCHS}"
    )

    print(
        f"batch：{BATCH_SIZE}"
    )

    model = YOLO(
        str(modelPath)
    )

    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=PATIENCE,
        project=str(historyPath),
        name="run",
        exist_ok=False
    )

    bestModel = Path(
        model.trainer.best
    )

    if not bestModel.is_file():

        raise FileNotFoundError(
            f"找不到訓練完成的最佳模型："
            f"{bestModel}"
        )

    return bestModel


# 儲存本次訓練模型資訊
def saveInfo(bestModel):

    runPath = (
        bestModel.parent.parent
    )

    model = YOLO(
        str(bestModel)
    )

    result = model.val(
        data=str(YAML_PATH),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    classMaps = [
        float(value)
        for value in result.box.maps
    ]

    info = {
        "run": runPath.name,
        "precision": float(
            result.box.mp
        ),
        "recall": float(
            result.box.mr
        ),
        "mAP50": float(
            result.box.map50
        ),
        "mAP50-95": float(
            result.box.map
        ),
        "CTBodyMap50-95": classMaps[0],
        "CTHeadMap50-95": classMaps[1],
        "TBodyMap50-95": classMaps[2],
        "THeadMap50-95": classMaps[3]
    }

    infoFile = (
        runPath
        / "info.json"
    )

    infoFile.write_text(
        json.dumps(
            info,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print()
    print(
        f"已建立模型資訊：{infoFile}"
    )

    print(
        f"Precision："
        f"{info['precision']:.4f}"
    )

    print(
        f"Recall："
        f"{info['recall']:.4f}"
    )

    print(
        f"mAP50："
        f"{info['mAP50']:.4f}"
    )

    print(
        f"mAP50-95："
        f"{info['mAP50-95']:.4f}"
    )

    print(
        f"CTBody mAP50-95："
        f"{info['CTBodyMap50-95']:.4f}"
    )

    print(
        f"CTHead mAP50-95："
        f"{info['CTHeadMap50-95']:.4f}"
    )

    print(
        f"TBody mAP50-95："
        f"{info['TBodyMap50-95']:.4f}"
    )

    print(
        f"THead mAP50-95："
        f"{info['THeadMap50-95']:.4f}"
    )

    return info


# 將模型轉換為 TensorRT
def TensorRT(bestModel):

    model = YOLO(
        str(bestModel)
    )

    enginePath = model.export(
        format="engine",
        imgsz=IMAGE_SIZE,
        batch=1,
        device=DEVICE
    )

    enginePath = Path(
        enginePath
    )

    print(
        f"TensorRT 匯出完成："
        f"{enginePath}"
    )

    return enginePath


# 更新最佳訓練結果
def updateBest(
    newModel,
    enginePath,
    newInfo,
    bestPath
):

    bestInfoPath = (
        bestPath
        / "info.json"
    )

    if not bestInfoPath.exists():

        better = True

    else:

        try:

            bestInfo = json.loads(
                bestInfoPath.read_text(
                    encoding="utf-8"
                )
            )

            newScore = (
                newInfo["mAP50-95"],
                newInfo["mAP50"]
            )

            oldScore = (
                bestInfo["mAP50-95"],
                bestInfo["mAP50"]
            )

            better = (
                newScore
                > oldScore
            )

        except Exception:

            better = True

    if not better:

        print(
            "本輪模型沒有超越目前最佳模型"
        )

        return False

    for file in bestPath.iterdir():

        if file.is_file():

            file.unlink()

    shutil.copy2(
        newModel,
        bestPath / "best.pt"
    )

    shutil.copy2(
        enginePath,
        bestPath / "best.engine"
    )

    infoFile = (
        newModel.parent.parent
        / "info.json"
    )

    shutil.copy2(
        infoFile,
        bestPath / "info.json"
    )

    print(
        f"新的最佳模型："
        f"{newInfo['run']}"
    )

    print(
        f"best.pt 已更新至："
        f"{bestPath / 'best.pt'}"
    )

    print(
        f"best.engine 已更新至："
        f"{bestPath / 'best.engine'}"
    )

    return True


# 主程式
def main():

    print(
        "========== "
        "CS2 dataset YOLO 訓練開始 "
        "=========="
    )

    if not checkYaml():

        return

    if not checkData():

        return

    if not checkEnv():

        return

    if not checkBaseModel():

        return

    historyPath, bestPath = (
        checkModel()
    )

    modelPath = BASE_MODEL_PATH

    print()
    print(
        "本輪為 dataset baseline"
    )

    print(
        "不使用舊 best.pt"
    )

    print(
        f"本輪使用模型：{modelPath}"
    )

    newModel = trainModel(
        modelPath,
        historyPath
    )

    newInfo = saveInfo(
        newModel
    )

    try:

        enginePath = TensorRT(
            newModel
        )

    except Exception as error:

        print()
        print(
            f"TensorRT 匯出失敗：{error}"
        )

        print(
            f"訓練模型仍保留在：{newModel}"
        )

        return

    updateBest(
        newModel,
        enginePath,
        newInfo,
        bestPath
    )

    print()
    print(
        "========== "
        "本輪訓練完成 "
        "=========="
    )

    print(
        f"本輪模型：{newModel}"
    )

    print(
        f"mAP50："
        f"{newInfo['mAP50']:.4f}"
    )

    print(
        f"mAP50-95："
        f"{newInfo['mAP50-95']:.4f}"
    )

    print(
        f"CTHead："
        f"{newInfo['CTHeadMap50-95']:.4f}"
    )

    print(
        f"THead："
        f"{newInfo['THeadMap50-95']:.4f}"
    )


if __name__ == "__main__":

    main()