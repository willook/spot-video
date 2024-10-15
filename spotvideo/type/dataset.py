import os
from typing import Literal
from pathlib import Path


class Category:
    original: str
    distorted: list[str]

    def __init__(self):
        self.original = None
        self.distorted = []


class DatasetConstructor:
    categories: dict[str, Category] = {}
    n_videos: int = 0

    def __init__(
        self, data_dir: str, dataset_name: Literal["markcloud", "youtube"] = "markcloud"
    ):
        if dataset_name == "markcloud":
            self.init_markcloud_dataset(data_dir)
        elif dataset_name == "youtube":
            self.init_youtube_dataset(data_dir)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    def init_markcloud_dataset(self, data_dir: str):
        data_dir = Path(data_dir)
        video_meta_filename = data_dir / "A_track" / "video_meta.txt"
        assert os.path.exists(
            video_meta_filename
        ), f"{video_meta_filename} does not exist"

        labels = self.generate_markcloud_label(video_meta_filename)

        origin_dir = data_dir / "A_track" / "original"
        origin_video_filename = os.listdir(origin_dir)[0]
        category_name = Path(origin_video_filename).stem
        self.categories[category_name] = Category()
        self.categories[category_name].original = (
            origin_dir / origin_video_filename
        ).as_posix()
        self.categories["other"] = Category()
        self.n_videos = 1

        distorted_dir = data_dir / "A_track" / "distorted"
        distorted_videos = os.listdir(distorted_dir)
        for distorted_video in distorted_videos:
            if not distorted_video.endswith(".mpeg"):
                continue
            index = int(Path(distorted_video).stem) - 1
            filename = (distorted_dir / distorted_video).as_posix()
            if labels[index] == 1:
                self.categories[category_name].distorted.append(filename)
            else:
                self.categories["other"].distorted.append(filename)
            self.n_videos += 1
        print(f"Found {self.n_videos} videos in 1 categories")

    def generate_markcloud_label(self, video_meta_filename):
        labels = []
        with open(video_meta_filename, "r", encoding="utf8") as f:
            for line in f:
                words = line.strip().split()
                if len(words) == 0:
                    continue
                label = 0 if words[1] == "노이즈" and words[2] == "데이터" else 1
                labels.append(label)
        return labels

    def init_youtube_dataset(self, data_dir: str):
        self.n_videos = 0
        # find all video located in the */original/ or */distorted/
        for root, _, files in os.walk(data_dir):
            for file in files:
                # check if the file is a video file
                if not (file.endswith(".mp4") or file.endswith(".mpeg")):
                    continue
                head, split = os.path.split(root)
                _, category_name = os.path.split(head)
                # check if the dir of the file is original or distorted
                if split not in ["original", "distorted"]:
                    continue
                # add the file
                if category_name not in self.categories:
                    self.categories[category_name] = Category()
                if split == "original":
                    self.categories[category_name].original = os.path.join(root, file)
                else:
                    self.categories[category_name].distorted.append(
                        os.path.join(root, file)
                    )
                self.n_videos += 1
        print(f"Found {self.n_videos} videos in {len(self.categories)} categories")

    def get_dataset(
        self, category_name: str, max_distorted_videos: int = 10, seed: int = 1234
    ):
        origin_video_name = self.categories[category_name].original
        distorted_video_names = []
        labels = []
        for key in self.categories:
            label = 1 if key == category_name else 0
            for video in self.categories[key].distorted:
                distorted_video_names.append(video)
                labels.append(label)
        return origin_video_name, distorted_video_names, labels

    def keys(self):
        return self.categories.keys()

    def __len__(self):
        return len(self.categories)

    def __next__(self):
        for key in self.keys():
            yield self.get_dataset(key)

    def __iter__(self):
        return self
