"""Short live pipeline check, with mouse input disabled. Does not save frames."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ultralytics import YOLO
from yolo.detection import get_area, TargetSelector
from runtime.realtime import LatestCapture, LatestValue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=8)
    args = parser.parse_args()
    model = YOLO(str(Path(__file__).resolve().parents[1] / 'best/best.engine'), task='detect')
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(blank, imgsz=640, device=0, verbose=False)
    # Fixed short throughput check; not a hyperparameter search.
    samples = []
    for _ in range(35):
        started = time.perf_counter()
        model.predict(blank, imgsz=640, device=0, conf=0.35,
                    verbose=False)[0].cpu()
        samples.append(time.perf_counter() - started)
    print(json.dumps({'synthetic_inference_fps': 1 / np.mean(samples[5:])}), flush=True)
    capture = LatestCapture(args.camera).start()
    mailbox, stop = LatestValue(), threading.Event()
    render_errors = []

    def render():
        while not stop.is_set():
            results = mailbox.wait_take(stop)
            if results is not None:
                try:
                    results[0].plot(conf=False, line_width=2)
                except Exception as error:
                    render_errors.append(str(error))
                    return

    renderer = threading.Thread(target=render, daemon=True)
    renderer.start()
    sequence, count, skipped = 0, 0, 0
    # Exclude camera opening/format negotiation from steady-state FPS.
    first = capture.read_latest(0, timeout=10)
    if first is None:
        stop.set()
        capture.stop()
        renderer.join(timeout=2)
        raise RuntimeError(capture.error or 'No camera frames received')
    sequence = first[0]
    model.predict(get_area(first[2]), imgsz=640, device=0, conf=0.35,
                verbose=False)[0].cpu()
    started = time.perf_counter()
    ages, inference = [], []
    selector = TargetSelector()
    try:
        while time.perf_counter() - started < args.seconds:
            packet = capture.read_latest(sequence)
            if packet is None:
                if capture.error:
                    raise RuntimeError(capture.error)
                continue
            seq, captured, frame = packet
            skipped += max(0, seq - sequence - 1)
            sequence = seq
            t0 = time.perf_counter()
            results = [model.predict(get_area(frame), imgsz=640, device=0, conf=0.35,
                                  verbose=False)[0].cpu()]
            selector.pick(results, '死鬥', 180)
            now = time.perf_counter()
            ages.append((now - captured) * 1000)
            inference.append((now - t0) * 1000)
            count += 1
            mailbox.put(results)
        elapsed = time.perf_counter() - started
        print(json.dumps({
            'frames': count, 'seconds': elapsed, 'capture_fps': capture.fps,
            'reported_source_fps': capture.reported_fps,
            'live_inference_fps': count / elapsed, 'skipped_frames': skipped,
            'buffer_size_supported': capture.buffer_supported,
            'capture_to_result_ms_median': float(np.median(ages)) if ages else None,
            'capture_to_result_ms_p95': float(np.percentile(ages, 95)) if ages else None,
            'inference_ms_median': float(np.median(inference)) if inference else None,
            'preview_errors': render_errors,
        }), flush=True)
        if not count:
            raise RuntimeError('No camera frames received')
    finally:
        stop.set()
        capture.stop()
        renderer.join(timeout=2)


if __name__ == '__main__':
    main()
