import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def compare_frame_by_frame(origin_video_path, distorted_video_path, log_dir):
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


def compare_feature(
    origin_feature, distorted_feature, origin_mask=None, distorted_mask=None
):
    # TODO: cross-correlation shift 대처
    # find shift between two features
    # best_shift = np.argmax(correlation) - (len(x) - 1)
    # print("Best shift:", best_shift)
    video_length = min(len(origin_feature), len(distorted_feature))
    origin_feature = origin_feature[:video_length]
    distorted_feature = distorted_feature[:video_length]
    origin_mask = origin_mask[:video_length] if origin_mask is not None else None
    distorted_mask = (
        distorted_mask[:video_length] if distorted_mask is not None else None
    )

    if origin_mask is not None and distorted_mask is not None:
        mask = origin_mask & distorted_mask
    elif origin_mask is not None:
        mask = origin_mask
    elif distorted_mask is not None:
        mask = distorted_mask
    else:
        mask = np.ones(video_length, dtype=bool)

    origin_masked = origin_feature * mask
    distorted_masked = distorted_feature * mask
    similarity = np.dot(origin_masked, distorted_masked) / (
        np.linalg.norm(origin_masked) * np.linalg.norm(distorted_masked)
    )
    return similarity


def predict_score(origin_video_dir, distorted_video_dir):
    origin_video_feature = None
    for video_name in os.listdir(origin_video_dir):
        origin_video_path = os.path.join(origin_video_dir, video_name)
        origin_video_feature, origin_mask = extract_feature(origin_video_path)
        plt.figure()
        plt.plot(origin_video_feature, label="origin")
        plt.savefig(log_dir / "origin_feature.png")
        plt.close()

    scores = []
    video_names = []
    # sort by int(stem)
    distorted_video_names = sorted(
        os.listdir(distorted_video_dir), key=lambda x: int(Path(x).stem)
    )
    # for video_name in tqdm(os.listdir(distorted_video_dir)):
    for video_name in distorted_video_names:
        distorted_video_path = os.path.join(distorted_video_dir, video_name)
        distorted_video_feature, distorted_mask = extract_feature(distorted_video_path)
        plt.figure()
        file_name = Path(distorted_video_path).stem
        plt.plot(distorted_video_feature, label=f"distorted_{file_name}")
        plt.savefig(log_dir / f"distorted_feature_{file_name}.png")
        plt.close()

        score = compare_feature(
            origin_video_feature, distorted_video_feature, origin_mask, distorted_mask
        )
        print(f"{video_name}: {score}")
        scores.append(score)
        video_names.append(video_name)

    return np.array(scores)


def f1_score(y_true, y_pred):
    tp = sum([1 for x, y in zip(y_true, y_pred) if x == y and x == 1])
    fp = sum([1 for x, y in zip(y_true, y_pred) if x != y and x == 0])
    fn = sum([1 for x, y in zip(y_true, y_pred) if x != y and x == 1])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def accuracy(y_true, y_pred):
    return sum([1 for x, y in zip(y_true, y_pred) if x == y]) / len(y_true)


origin_video_dir = "data/A_track/original/"
distorted_video_dir = "data/A_track/distorted/"
label_file = "data/A_track/label.txt"
log_dir = Path("log_dir")
log_dir.mkdir(exist_ok=True)
threshold = 0.9

compare_frame_by_frame(
    "data/A_track/original/Falling Hare (1943).mpeg",
    "data/A_track/distorted/15.mpeg",
    log_dir,
)
use_cache = False

start = time.time()
try:
    cache_file = log_dir / "scores.json"
    assert use_cache and cache_file.exists()
    scores_json = json.load(open(cache_file))
    scores = np.array(scores_json)
except:
    scores = predict_score(origin_video_dir, distorted_video_dir)
    json.dump(scores.tolist(), open(cache_file, "w"))


# 히스토그램과 임계값 표시
threshold = threshold_otsu(scores)

# label and predicted label
with open(label_file, "r") as f:
    labels = [int(line.strip()) for line in f]
predicted_labels = [1 if scores > threshold else 0 for scores in scores]

end = time.time()

# print result
for i, (label, predicted_label, score) in enumerate(
    zip(labels, predicted_labels, scores)
):
    if label == predicted_label:
        print(f"video {i+1}: label {label}, predicted {predicted_label}, score {score}")
    else:
        print(
            f"    video {i+1}: label {label}, predicted {predicted_label}, score {score}"
        )

acc = accuracy(labels, predicted_labels)
f1 = f1_score(labels, predicted_labels)

print(f"Accuracy: {acc}, F1: {f1}")
print(f"Threshold: {threshold}")
print(f"Time: {end-start:.2f}s")

elapsed = end - start

plt.figure()
true_mask = np.array(labels) == 1
range = (min(scores), max(scores))
# plt.plot(sorted_scores)
plt.hist(scores[~true_mask], bins=50, range=range, alpha=0.6, color="r")
plt.hist(scores[true_mask], bins=50, range=range, alpha=0.6, color="g")
plt.axvline(
    threshold,
    color="b",
    linestyle="--",
    label=f"Threshold: {threshold:.2f}",
)
plt.savefig(log_dir / "scores.png")

with open(log_dir / "result.txt", "w") as f:
    f.write(f"Accuracy: {acc}, F1: {f1}\n")
    f.write(f"Threshold: {threshold}\n")
    f.write(f"Time: {elapsed:.2f}s\n")
    f.write(f"Scores: {scores}\n")
    f.write(f"Threshold: {threshold}\n")
    f.write(f"Labels: {labels}\n")
    f.write(f"Predicted Labels: {predicted_labels}\n")

# main.py를 log_dir에 복사
os.system(f"cp main.py {log_dir}")
