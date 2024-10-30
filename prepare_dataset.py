import os
import sys
import zipfile
from pathlib import Path

import ffmpeg
import argparse
from tqdm import tqdm
from pytubefix import YouTube


def prepare_markcloud_dataset(metadata, root_dir="data"):

    download_dir = Path(root_dir) / "downloads"
    zip_filename = download_dir / "data.zip"
    data_dir = Path(root_dir) / "markcloud"

    assert os.path.exists(zip_filename), f"{zip_filename} does not exist"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    with zipfile.ZipFile(zip_filename, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Unzipping", unit="files"):
            # if not os.path.exists(data_dir / member.filename): # cache
            zf.extract(member, data_dir)


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
        "\rDownloading: [{}{}] {} MB / {} MB".format(
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


def prepare_youtube_dataset(root_dir="data"):
    params = [
        {
            "uid": "v001",
            "title": "THE BEST NEW ANIMATION MOVIES & SERIES 2024 (Trailers)",
            "url": "https://youtu.be/xfrAN3nZuko",
        },
        {
            "uid": "v002",
            "title": "THE BEST UPCOMING MOVIES 2024 (Trailers)",
            "url": "https://youtu.be/1lUlmpoQsOA",
        },
    ]
    for param in params:
        download_and_split_video(param, root_dir)


def download_and_split_video(metadata, root_dir="data"):
    download_dir = Path(root_dir) / "downloads"
    os.makedirs(download_dir, exist_ok=True)

    video_filename = download_dir / (metadata["title"] + ".mp4")
    url = metadata["url"]
    uid = metadata["uid"]

    path_to_split = Path(root_dir) / "youtube" / uid
    os.makedirs(path_to_split, exist_ok=True)

    # download youtube video
    yt = YouTube(
        url, on_progress_callback=progress_func, on_complete_callback=complete_func
    )

    if not os.path.exists(video_filename):
        yt.streams.get_highest_resolution().download(
            str(download_dir),
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
        filename = f"{uid}c{str(k).zfill(3)}"
        os.makedirs(path_to_split / filename / "original", exist_ok=True)
        output_filename = path_to_split / filename / "original" / f"{filename}.mp4"
        k += 1
        if not os.path.exists(output_filename):
            ffmpeg_extract_clip(video_filename, output_filename, start_time, end_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="markcloud")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    dataset_name = args.name
    data_dir = args.data_dir

    if dataset_name == "markcloud":
        prepare_markcloud_dataset(data_dir)
    elif dataset_name == "youtube":
        prepare_youtube_dataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
