from pathlib import Path

from datasets import load_dataset


PROJECT_PATH = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_PATH / "dataset2026"

CLASS_NAMES = {
    0: "CTBody",
    1: "CTHead",
    2: "TBody",
    3: "THead"
}

CLASS_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3
}

SPLIT_MAP = {
    "train": "train",
    "validation": "valid"
}


def createFolders():

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} 已存在"
        )

    for split in SPLIT_MAP.values():

        (OUTPUT_PATH / split / "images").mkdir(
            parents=True,
            exist_ok=False
        )

        (OUTPUT_PATH / split / "labels").mkdir(
            parents=True,
            exist_ok=False
        )


def convertBox(
    box,
    width,
    height
):

    x, y, boxWidth, boxHeight = box

    x1 = max(
        0.0,
        float(x)
    )

    y1 = max(
        0.0,
        float(y)
    )

    x2 = min(
        float(width),
        float(x + boxWidth)
    )

    y2 = min(
        float(height),
        float(y + boxHeight)
    )

    newWidth = x2 - x1
    newHeight = y2 - y1

    if newWidth <= 0 or newHeight <= 0:
        return None

    centerX = (
        x1 + x2
    ) / 2 / width

    centerY = (
        y1 + y2
    ) / 2 / height

    newWidth /= width
    newHeight /= height

    return (
        centerX,
        centerY,
        newWidth,
        newHeight
    )


def processSplit(
    dataset,
    sourceSplit,
    targetSplit
):

    imagesPath = (
        OUTPUT_PATH
        / targetSplit
        / "images"
    )

    labelsPath = (
        OUTPUT_PATH
        / targetSplit
        / "labels"
    )

    stats = {
        "images": 0,
        "background": 0,
        "boxes": 0,
        "classes": {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
    }

    splitData = dataset[sourceSplit]

    for index, item in enumerate(
        splitData
    ):

        image = item["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        if width != 640 or height != 640:

            raise ValueError(
                f"{sourceSplit} 第 {index} 張"
                f"不是 640x640："
                f"{width}x{height}"
            )

        imageName = (
            f"{sourceSplit}_{index:06d}.jpg"
        )

        imagePath = (
            imagesPath
            / imageName
        )

        labelPath = (
            labelsPath
            / f"{Path(imageName).stem}.txt"
        )

        image.save(
            imagePath,
            quality=95
        )

        objects = item["objects"]

        boxes = objects["bbox"]
        categories = objects["categories"]

        if len(boxes) != len(categories):

            raise ValueError(
                f"{sourceSplit} 第 {index} 張"
                " bbox 與 category 數量不同"
            )

        lines = []

        for box, sourceClass in zip(
            boxes,
            categories
        ):

            sourceClass = int(
                sourceClass
            )

            if sourceClass == 0:
                continue

            if sourceClass not in CLASS_MAP:

                raise ValueError(
                    f"非法 Class ID："
                    f"{sourceClass}"
                )

            classId = (
                CLASS_MAP[sourceClass]
            )

            newBox = convertBox(
                box,
                width,
                height
            )

            if newBox is None:
                continue

            (
                centerX,
                centerY,
                boxWidth,
                boxHeight
            ) = newBox

            lines.append(
                f"{classId} "
                f"{centerX:.8f} "
                f"{centerY:.8f} "
                f"{boxWidth:.8f} "
                f"{boxHeight:.8f}"
            )

            stats["boxes"] += 1

            stats[
                "classes"
            ][classId] += 1

        if not lines:

            stats[
                "background"
            ] += 1

        content = "\n".join(
            lines
        )

        if content:
            content += "\n"

        labelPath.write_text(
            content,
            encoding="utf-8"
        )

        stats["images"] += 1

        if stats["images"] % 250 == 0:

            print(
                f"{targetSplit}: "
                f"{stats['images']} / "
                f"{len(splitData)}"
            )

    return stats


def createYaml():

    content = """path: .

train: train/images
val: valid/images

nc: 4

names:
  0: CTBody
  1: CTHead
  2: TBody
  3: THead
"""

    (
        OUTPUT_PATH
        / "data.yaml"
    ).write_text(
        content,
        encoding="utf-8"
    )


def printStats(
    split,
    stats
):

    print()
    print(
        f"========== {split} =========="
    )

    print(
        f"圖片數：{stats['images']}"
    )

    print(
        f"Background："
        f"{stats['background']}"
    )

    print(
        f"標註總數：{stats['boxes']}"
    )

    for classId in range(4):

        print(
            f"{CLASS_NAMES[classId]}："
            f"{stats['classes'][classId]}"
        )


def main():

    print(
        "下載 CS2 2026 dataset..."
    )

    dataset = load_dataset(
        "fvossel/csgo-player-detection"
    )

    createFolders()

    allStats = {}

    for sourceSplit, targetSplit in (
        SPLIT_MAP.items()
    ):

        allStats[targetSplit] = (
            processSplit(
                dataset,
                sourceSplit,
                targetSplit
            )
        )

    createYaml()

    for split, stats in (
        allStats.items()
    ):

        printStats(
            split,
            stats
        )

    print()
    print(
        "dataset2026 建立完成："
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()