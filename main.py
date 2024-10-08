import os
import time
import argparse
from pathlib import Path

import numpy as np

from spotvideo.preprocess.signal import FeatureExtractor
from spotvideo.util.draw import save_plot, save_similarity_histogram
from spotvideo.model.similarity import SimilarityClassifier
from spotvideo.util.metric import accuracy, f1_score
from spotvideo.type.dataset import DatasetConstructor
from spotvideo.type.result import Result


def evaluate(
    origin_video: str,
    distorted_videos: list[str],
    labels: np.ndarray,
    log_dir: Path,
    args: argparse.Namespace,
):
    threshold: float = args.threshold
    same_length: bool = args.same_length
    use_cache: bool = args.use_cache

    start = time.time()

    extractor = FeatureExtractor()
    origin_feature, origin_mask = extractor.extract(origin_video, use_cache=use_cache)
    distorted_features, distorted_masks = extractor.batch_extract(
        distorted_videos, use_cache=use_cache
    )

    # predict labels based on similarity
    classifier = SimilarityClassifier()
    predicted_labels, info = classifier.predict(
        origin_feature,
        distorted_features,
        origin_mask,
        distorted_masks,
        threshold=threshold,
        same_length=same_length,
    )
    threshold = info["threshold"]
    similarities = info["similarities"]

    end = time.time()
    elapsed = end - start

    acc = accuracy(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)

    result = Result(labels, predicted_labels, similarities, threshold, elapsed, acc, f1)

    # print and save results
    for i, (label, predicted_label, similarity) in enumerate(
        zip(labels, predicted_labels, similarities)
    ):
        # if label == predicted_label:
        #     continue
        print(
            f"[{i+1}] label {label}, predicted {predicted_label}, similarity {similarity}"
        )

    print(f"Accuracy: {acc}, F1: {f1}")
    print(f"Threshold: {threshold}")
    print(f"Time: {end-start:.2f}s")

    os.makedirs(log_dir, exist_ok=True)
    save_plot(log_dir, origin_feature, "origin_feature")
    for i, distorted_feature in enumerate(distorted_features):
        save_plot(log_dir, distorted_feature, f"distorted_feature_{i+1}")

    save_similarity_histogram(
        log_dir, similarities, labels, threshold, "similarity_histogram"
    )
    with open(log_dir / "result.txt", "w") as f:
        f.write(f"Accuracy: {acc}, F1: {f1}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Time: {elapsed:.2f}s\n")
        f.write(f"Similarities: {similarities}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Labels: {labels}\n")
        f.write(f"Predicted Labels: {predicted_labels}\n")
    print(f"Results saved in {log_dir}")
    return result


def main(args):
    dataset_constructor = DatasetConstructor(args.data_dir)
    for key in dataset_constructor.keys():
        log_dir = Path(args.log_dir) / key
        origin_video, distorted_videos, labels = dataset_constructor.get_dataset(key)
        if origin_video is None:
            print(f"Category {key} does not have original video")
            continue
        assert (
            len(distorted_videos) != 0
        ), f"Category {key} does not have distorted videos"
        evaluate(origin_video, distorted_videos, labels, log_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/markcloud/")
    parser.add_argument("--log_dir", type=str, default="log_dir/markcloud/")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument(
        "--threshold", type=float, default=None, help="If None, use otsu thresholding"
    )
    parser.add_argument(
        "--same_length", action="store_true", help="If true, not use DTW"
    )
    args = parser.parse_args()
    main(args)
