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

    def extract(self, video_path: str, key_frame_interval=11):
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
        # remove p_value/2 from both ends
        index = np.argsort(feature)
        band_index = sorted(
            index[int(len(index) * p_value / 2) : -int(len(index) * p_value / 2)]
        )
        mask = np.zeros(len(feature), dtype=bool)
        mask[band_index] = True
        band_passed = feature[band_index]

        # smoothing with moving average
        smoothed = np.convolve(
            band_passed, np.ones(window_size) / window_size, mode="same"
        )
        
        # normalize feature mean to 0, std to 1
        normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)

        # recover the original length
        recovered = np.zeros(len(feature))
        recovered[band_index] = normalized
        return recovered, mask
