import os
import pickle
import time
from pathlib import Path
import argparse

from spotvideo.preprocess.signal import FeatureExtractor
from spotvideo.util.draw import save_plot, save_similarity_histogram
from spotvideo.model.similarity import SimilarityClassifier
from spotvideo.util.metric import accuracy, f1_score


def get_distorted_videos(distorted_dir):
    return [
        os.path.join(distorted_dir, video_name)
        for video_name in sorted(
            os.listdir(distorted_dir), key=lambda x: int(Path(x).stem)
        )
    ]


def main(args):
    origin_dir = args.origin_dir
    distorted_dir = args.distorted_dir
    label_file = args.label_file
    log_dir = Path(args.log_dir)

    assert os.path.isdir(origin_dir), f"Cannot find directory: {origin_dir}"
    assert os.path.isdir(distorted_dir), f"Cannot find directory: {distorted_dir}"
    assert os.path.isfile(label_file), f"Cannot find file: {label_file}"

    start = time.time()

    # find target videos
    origin_video = os.path.join(origin_dir, os.listdir(origin_dir)[0])
    distorted_videos = get_distorted_videos(distorted_dir)

    # extract features or load from cache if use_cache is True
    try:
        cache_file = log_dir / "cache_feature.pickle"
        assert args.use_cache and cache_file.exists()
        with open(cache_file, "rb") as f:
            origin_feature, origin_mask, distorted_features, distorted_masks = (
                pickle.load(f)
            )
    except:
        extractor = FeatureExtractor()
        origin_feature, origin_mask = extractor.extract(origin_video)
        distorted_features, distorted_masks = extractor.batch_extract(distorted_videos)
        with open(cache_file, "wb") as f:
            pickle.dump(
                (origin_feature, origin_mask, distorted_features, distorted_masks), f
            )

    # predict labels based on similarity
    classifier = SimilarityClassifier()
    predicted_labels, info = classifier.predict(
        origin_feature, distorted_features, origin_mask, distorted_masks
    )
    threshold = info["threshold"]
    similarities = info["similarities"]

    end = time.time()
    elapsed = end - start

    # evaluate
    labels = [int(line.strip()) for line in open(label_file)]
    acc = accuracy(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels)

    # print and save results
    for i, (label, predicted_label, similarity) in enumerate(
        zip(labels, predicted_labels, similarities)
    ):
        if label == predicted_label:
            continue
        print(
            f"[{i+1}] label {label}, predicted {predicted_label}, similarity {similarity}"
        )

    print(f"Accuracy: {acc}, F1: {f1}")
    print(f"Threshold: {threshold}")
    print(f"Time: {end-start:.2f}s")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_dir", type=str, default="data/A_track/original/")
    parser.add_argument("--distorted_dir", type=str, default="data/A_track/distorted/")
    parser.add_argument("--label_file", type=str, default="data/A_track/label.txt")
    parser.add_argument("--log_dir", type=str, default="log_dir")
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    main(args)
