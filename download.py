import os
import sys
import json
import zipfile
from pathlib import Path

import gdown
import ffmpeg
from tqdm import tqdm
from pytubefix import YouTube


def ffmpeg_extract_clip(input_filename, output_filename, start_time, end_time):
    input_filename = str(input_filename)
    output_filename = str(output_filename)
    start_time = str(start_time)
    end_time = str(end_time)
    ffmpeg.input(input_filename, ss=start_time, to=end_time, loglevel="quiet").output(
        output_filename
    ).run()


def progress_func(stream, chunk, bytes_remaining):
    current = stream.filesize - bytes_remaining
    done = int(50 * current / stream.filesize)

    sys.stdout.write(
        "\r Downloading: [{}{}] {} MB / {} MB".format(
            "=" * done,
            " " * (50 - done),
            "{:.2f}".format(bytes_to_megabytes(current)),
            "{:.2f}".format(bytes_to_megabytes(stream.filesize)),
        )
    )
    sys.stdout.flush()


def complete_func(stream, file_handle):
    sys.stdout.write("\n")
    sys.stdout.flush()


def bytes_to_megabytes(bytes_size):
    megabytes_size = bytes_size / (1024**2)
    return megabytes_size


def download_youtube_dataset(metadata, root_dir="data"):
    data_dir = Path(root_dir) / "youtube"
    os.makedirs(data_dir, exist_ok=True)

    video_filename = data_dir / (metadata["title"] + ".mp4")
    url = metadata["url"]
    uid = metadata["uid"]

    path_to_split = data_dir / uid
    os.makedirs(path_to_split, exist_ok=True)

    # download youtube video
    yt = YouTube(
        url, on_progress_callback=progress_func, on_complete_callback=complete_func
    )

    if not os.path.exists(video_filename):
        yt.streams.get_highest_resolution().download(
            str(data_dir),
        )
    assert os.path.exists(video_filename), f"{video_filename} does not exist"

    # split video by chapters
    chapters = yt.chapters
    k = 1
    for chapter in tqdm(chapters, desc="Splitting", unit="chapters"):
        start_time = chapter.start_seconds + 1
        end_time = chapter.start_seconds + chapter.duration - 1
        if end_time - start_time < 10:
            continue
        output_filename = path_to_split / f"{uid}c{str(k).zfill(3)}.mp4"
        k += 1
        if not os.path.exists(output_filename):
            ffmpeg_extract_clip(video_filename, output_filename, start_time, end_time)


def download_markcloud_dataset(metadata, root_dir="data"):
    file_id = metadata["file_id"]
    data_dir = Path(root_dir) / "markcloud"
    zip_filename = data_dir / "data.zip"

    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(zip_filename):
        gdown.download(
            url=f"https://drive.google.com/uc?id={file_id}",
            output=str(zip_filename),
            quiet=False,
        )

    with zipfile.ZipFile(zip_filename, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Unzipping", unit="files"):
            if not os.path.exists(data_dir / member.filename):
                zf.extract(member, data_dir)

    if not os.path.exists(data_dir / "label.txt"):
        generate_label(data_dir)


def generate_label(data_dir):
    labels = []
    video_meta_filename = data_dir / "A_track/video_meta.txt"
    label_filename = data_dir / "label.txt"
    with open(video_meta_filename, "r", encoding="utf8") as f:
        for line in f:
            words = line.strip().split()
            if len(words) == 0:
                continue
            label = 0 if words[1] == "노이즈" and words[2] == "데이터" else 1
            labels.append(label)

    assert len(labels) == 90, f"labels length should be 90, but {len(labels)}"
    assert sum(labels) == 72, f"labels sum should be 72, but {sum(labels)}"

    with open(label_filename, "w", encoding="utf8") as f:
        for label in labels:
            f.write(f"{label}\n")


if __name__ == "__main__":
    json_file = "metadata.json"
    with open(json_file, "r", encoding="utf8") as f:
        metadata = json.load(f)
    data_dir = "data"
    download_markcloud_dataset(metadata["markcloud"], data_dir)
    for metadata in metadata["youtube"]:
        download_youtube_dataset(metadata, data_dir)
