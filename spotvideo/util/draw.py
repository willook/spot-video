from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_similarity_histogram(
    log_dir: Path,
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    name: str,
):
    plt.figure()
    true_mask = np.array(labels) == 1
    range = (min(similarities), max(similarities))
    # plt.plot(sorted_scores)
    plt.hist(
        similarities[~true_mask],
        bins=50,
        range=range,
        alpha=0.6,
        color="r",
        label="Noise videos",
    )
    plt.hist(
        similarities[true_mask],
        bins=50,
        range=range,
        alpha=0.6,
        color="g",
        label="Distorted videos",
    )
    plt.axvline(
        threshold,
        color="b",
        linestyle="--",
        label=f"Threshold: {threshold:.2f}",
    )
    plt.legend()
    plt.savefig(log_dir / f"{name}.png")


def save_plot(log_dir: Path, data: np.ndarray, name: str):
    plt.figure()
    plt.plot(data, label=name)
    plt.savefig(log_dir / f"{name}.png")
    plt.close()


def compare_frame_by_frame(
    origin_video_path: str, distorted_video_path: str, log_dir: Path
):
    cap_origin = cv2.VideoCapture(origin_video_path)
    cap_distorted = cv2.VideoCapture(distorted_video_path)
    assert cap_origin.isOpened(), f"Cannot open video file: {origin_video_path}"
    assert cap_distorted.isOpened(), f"Cannot open video file: {distorted_video_path}"
    fps = cap_origin.get(cv2.CAP_PROP_FPS)

    name1 = Path(origin_video_path).stem
    name2 = Path(distorted_video_path).stem
    filename = f"{name1}_{name2}.mp4"

    writer = cv2.VideoWriter(
        log_dir / filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(cap_origin.get(3) + cap_distorted.get(3)), int(cap_origin.get(4))),
    )

    while True:
        ret_origin, frame_origin = cap_origin.read()
        ret_distorted, frame_distorted = cap_distorted.read()
        if not ret_origin:
            frame_origin = np.zeros_like(frame_distorted)
        if not ret_distorted:
            frame_distorted = np.zeros_like(frame_origin)
        if not ret_origin and not ret_distorted:
            break
        merged = np.hstack((frame_origin, frame_distorted))
        writer.write(merged)
    writer.release()
    print(f"{filename} saved")
