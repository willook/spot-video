import time
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from spotvideo.util.video import preprocess_frame


class FeatureExtractor:
    def batch_extract(self, video_path: list[str], key_frame_interval=11):
        features = []
        masks = []
        for path in tqdm(video_path, desc="Extracting features"):
            feature, mask = self.extract(path, key_frame_interval)
            features.append(feature)
            masks.append(mask)

        return features, masks

    def check_cache(self, key: str):
        return Path(key).exists()

    def load_cache(self, key: str):
        with open(key, "rb") as f:
            feature, mask = pickle.load(f)
        return feature, mask

    def save_cache(self, key: str, value: tuple):
        with open(key, "wb") as f:
            pickle.dump(value, f)

    def caching_extract(self, video_path: str):
        try:
            cache_file = video_path + ".cache"
            assert self.check_cache(cache_file)
            feature, mask = self.load_cache(cache_file)
        except AssertionError:
            feature, mask = self._extract(feature)
            self.save_cache(cache_file, (feature, mask))
        return feature, mask

    def extract(self, video_path: str, key_frame_interval=11):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video file: {video_path}"
        feature = []
        _, frame = cap.read()
        prev_frame = preprocess_frame(frame)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i_frame in range(1, total_frame):
            if i_frame % key_frame_interval == 0:
                _, frame = cap.read()
            else:
                cap.grab()
                continue
            current_frame = preprocess_frame(frame)
            diff_img = cv2.absdiff(prev_frame, current_frame)
            diff = np.mean(diff_img)
            feature.append(diff)
            prev_frame = current_frame

        feature_np = np.array(feature)
        return self.preprocess_feature(feature_np)

    def preprocess_feature(
        self, feature: np.ndarray, p_value: float = 0.05, window_size: int = 15
    ):
        # Remove high-value outlier (low bound is 0) by p_value
        index = np.argsort(feature)
        pass_index = sorted(index[: len(feature) - int(len(index) * p_value)])
        mask = np.zeros(len(feature), dtype=bool)
        mask[pass_index] = True
        passed = feature[pass_index]
        # smoothing with moving average to remove noise
        smoothed = np.convolve(passed, np.ones(window_size) / window_size, mode="same")
        # normalize feature mean to 0, std to 1
        normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)

        # recover the original length
        recovered = np.zeros(len(feature))
        recovered[pass_index] = normalized
        return recovered, mask
