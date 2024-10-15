import numpy as np
from skimage.filters import threshold_otsu
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from spotvideo.util.metric import cosine_similarity


class SimilarityClassifier:
    def __init__(self):
        pass

    def predict(
        self,
        origin_feature: np.ndarray,
        distorted_features: np.ndarray,
        origin_mask: np.ndarray = None,
        distorted_masks: np.ndarray = None,
        threshold: float = None,
        same_length: bool = True,
    ):
        """
        Predict similarity score between origin and distorted features
        Args:
            origin_feature (np.ndarray): 1D array of origin feature
            distorted_features (List[np.ndarray]): List of 1D array of distorted features
            origin_mask (np.ndarray): 1D array of origin mask
            distorted_masks (List[np.ndarray]): List of 1D array of distorted masks
            threshold (float): threshold to classify, if None, use otsu thresholding
            same_length (bool): whether to compare features of same length
        Returns:
            list: label of distorted features, 1 for similar, 0 for dissimilar
        """
        # TODO: cross-correlation shift 대처
        # find shift between two features
        # best_shift = np.argmax(correlation) - (len(x) - 1)
        # print("Best shift:", best_shift)
        similarities = []
        pbar = tqdm(distorted_features, desc="Predicting")
        for distorted_feature, distorted_mask in zip(
            distorted_features, distorted_masks
        ):
            similarity = (
                self.compute_similarity(
                    origin_feature, distorted_feature, origin_mask, distorted_mask
                )
                if same_length
                else self.compute_dtw_similarity(
                    origin_feature, distorted_feature, origin_mask, distorted_mask
                )
            )
            similarities.append(similarity)
            pbar.update(1)
        pbar.close()

        threshold = (
            threshold_otsu(np.array(similarities)) if threshold is None else threshold
        )
        predicted_labels = [
            1 if similarity > threshold else 0 for similarity in similarities
        ]
        info = {
            "threshold": threshold,
            "similarities": np.array(similarities),
        }
        return predicted_labels, info

    def compute_dtw_similarity(
        self, origin_feature, distorted_feature, origin_mask, distorted_mask
    ):
        feature1 = origin_feature[origin_mask]
        feature2 = distorted_feature[distorted_mask]
        distance, path = fastdtw(feature1, feature2)
        # breakpoint()
        matched_feature1 = np.array([feature1[i] for i, j in path])
        matched_feature2 = np.array([feature2[j] for i, j in path])
        similarity = cosine_similarity(matched_feature1, matched_feature2)
        return similarity

    def compute_similarity(
        self, origin_feature, distorted_feature, origin_mask=None, distorted_mask=None
    ):
        # # check cross correlation
        # correlation = np.correlate(origin_feature, distorted_feature, mode="full")
        # # argsort = np.argsort(correlation)
        # sorted_correlation = np.argsort(correlation)

        # shift1 = sorted_correlation[-1] - (len(origin_feature) - 1)
        # shift2 = sorted_correlation[-2] - (len(origin_feature) - 1)

        # print("Shift:", shift1, shift2)
        # from matplotlib import pyplot as plt

        # plt.plot(correlation)
        # plt.savefig("correlation.png")
        # breakpoint()
        signal_length = min(len(origin_feature), len(distorted_feature))
        origin_feature = origin_feature[:signal_length]
        distorted_feature = distorted_feature[:signal_length]
        origin_mask = origin_mask[:signal_length] if origin_mask is not None else None
        distorted_mask = (
            distorted_mask[:signal_length] if distorted_mask is not None else None
        )

        if origin_mask is not None and distorted_mask is not None:
            mask = origin_mask & distorted_mask
        elif origin_mask is not None:
            mask = origin_mask
        elif distorted_mask is not None:
            mask = distorted_mask
        else:
            mask = np.ones(signal_length, dtype=bool)

        origin_masked = origin_feature * mask
        distorted_masked = distorted_feature * mask
        similarity = cosine_similarity(origin_masked, distorted_masked)
        return similarity
