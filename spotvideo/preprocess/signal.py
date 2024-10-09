import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from spotvideo.util.video import preprocess_frame


class FeatureExtractor:
    def batch_extract(
        self, video_path: list[str], key_frame_interval=11, use_cache=False
    ):
        features = []
        masks = []
        for path in tqdm(video_path, desc="Extracting features"):
            feature, mask = self.extract(path, key_frame_interval, use_cache)
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

    def extract(self, video_path: str, key_frame_interval=11, use_cache=False):
        try:
            cache_file = video_path + ".cache"
            assert use_cache and self.check_cache(cache_file)
            feature, mask = self.load_cache(cache_file)
        except AssertionError:
            feature, mask = self._extract(video_path, key_frame_interval)
            self.save_cache(cache_file, (feature, mask))
        return feature, mask

    def _extract(self, video_path: str, key_frame_interval=11):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video file: {video_path}"
        feature = []
        i_frame = 1
        ret, frame = cap.read()
        prev_frame = preprocess_frame(frame)
        while True:
            ret, frame = cap.read()
            i_frame += 1
            if not ret:
                break
            if i_frame % key_frame_interval != 0:
                continue
            current_frame = preprocess_frame(frame)
            diff_img = cv2.absdiff(prev_frame, current_frame)
            # 선형 배수가 아닐수도 있다
            # -> DTW로 해결할 수 있을 것 같다
            # 요부분을 유사도로 사용해도 되겠다
            diff = np.mean(diff_img)
            feature.append(diff)
            prev_frame = current_frame

        # normalize feature mean to 0, std to 1
        feature_np = np.array(feature)
        return self.preprocess_feature(feature_np)

    def preprocess_feature(
        self, feature: np.ndarray, p_value: float = 0.05, window_size: int = 11
    ):
        # remove p_value from high
        index = np.argsort(feature)
        # band_index = sorted(
        #     index[int(len(index) * p_value / 2) : len(feature) - int(len(index) * p_value / 2)]
        # )
        low_index = sorted(index[: len(feature) - int(len(index) * p_value)])
        mask = np.zeros(len(feature), dtype=bool)
        mask[low_index] = True
        low_passed = feature[low_index]
        # band_passed = feature[low_index]

        # smoothing with moving average
        smoothed = np.convolve(
            low_passed, np.ones(window_size) / window_size, mode="same"
        )

        # normalize feature mean to 0, std to 1
        normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)

        # recover the original length
        recovered = np.zeros(len(feature))
        recovered[low_index] = normalized
        return recovered, mask
