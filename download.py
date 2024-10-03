import os
import sys
import zipfile

import gdown


def download(file_id, data_dir="data"):
    data_zip_file = "data.zip"
    if not os.path.exists(data_zip_file):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", data_zip_file, quiet=False
        )
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        with zipfile.ZipFile(data_zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)


def generate_label(data_dir="data"):
    labels = []
    video_meta_file = os.path.join(data_dir, "A_track/video_meta.txt")
    with open(video_meta_file, "r", encoding='utf8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) == 0:
                continue
            label = 0 if words[1] == "노이즈" and words[2] == "데이터" else 1
            labels.append(label)
    assert len(labels) == 90, f"labels length should be 90, but {len(labels)}"
    assert sum(labels) == 72, f"labels sum should be 72, but {sum(labels)}"
    label_file = os.path.join(data_dir, "A_track/label.txt")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label}\n")


if __name__ == "__main__":
    data_dir = "data"
    download(sys.argv[1], data_dir=data_dir)
    generate_label(data_dir=data_dir)
