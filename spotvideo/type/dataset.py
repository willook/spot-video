import os


class Category:
    original: str
    distorted: list[str]

    def __init__(self):
        self.original = None
        self.distorted = []


class DatasetConstructor:
    categories: dict[str, Category] = {}

    def __init__(self, data_dir: str):
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
