from types import SimpleNamespace

from train.train import build_parser, collect_metrics


def test_collect_metrics_includes_overall_and_per_class_values():
    result = SimpleNamespace(
        box=SimpleNamespace(mp=0.9, mr=0.8, map50=0.85, map=0.6, maps=[0.7, 0.5])
    )

    metrics = collect_metrics(result, {0: "person", 1: "traffic light"})

    assert metrics == {
        "precision": 0.9,
        "recall": 0.8,
        "mAP50": 0.85,
        "mAP50_95": 0.6,
        "mAP50_95/person": 0.7,
        "mAP50_95/traffic_light": 0.5,
    }


def test_parser_defaults_are_small_enough_for_a_demo():
    args = build_parser().parse_args([])

    assert args.data == "coco8.yaml"
    assert args.base_model == "yolo11n.pt"
    assert args.epochs == 3


def test_parser_accepts_jenkins_parameters():
    args = build_parser().parse_args(
        ["--epochs", "150", "--imgsz", "640", "--batch", "32", "--device", "0"]
    )

    assert (args.epochs, args.imgsz, args.batch, args.device) == (150, 640, 32, "0")
