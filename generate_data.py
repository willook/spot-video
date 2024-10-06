import os
import pickle
import time
from pathlib import Path
import argparse
import cv2
import numpy as np
import random
from tqdm import tqdm

from spotvideo import augment


class Augmentor:
    def __init__(self, number_of_augmentation=1, identity_only=False):
        self.augmentation_dict = dict(
            [
                (name, cls)
                for name, cls in augment.__dict__.items()
                if isinstance(cls, type)
            ]
        )

        self.augmentation_class_name = ""
        self.augmentation_names_with_params = ""
        self.augmentations = list()

        if identity_only:
            self.augmentations.append(augment.Identity())
            self.augmentation_names_with_params += (
                f" {self.augmentations[-1].getName()}"
            )
            self.augmentation_class_name += " Identity"
        else:
            count = 0
            class_list = list()
            while True:
                aug_name, aug_class = random.choice(
                    list(self.augmentation_dict.items())
                )
                if "identity" in aug_name.lower():
                    continue
                if aug_class in class_list:
                    continue

                class_list.append(aug_class)
                self.augmentations.append(aug_class())

                self.augmentation_names_with_params += (
                    f" {self.augmentations[-1].getName()}"
                )

                self.augmentation_class_name += f" {aug_name}"

                count += 1

                if count >= number_of_augmentation:
                    break

    def __call__(self, frame):
        for aug in self.augmentations:
            frame = aug(frame)

        return frame

    def getName(self, with_params=True):
        if with_params:
            return self.augmentation_names_with_params
        else:
            return self.augmentation_class_name


def augment_video(
    video_path: str, clip_dir: str, number_of_output: int, number_of_augmentation: int
):
    output_dir = clip_dir / "distorted"
    os.makedirs(output_dir, exist_ok=True)

    meta_file = open(clip_dir / "video_meta.txt", "w", encoding="utf8")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open video file: {video_path}"

    # Get the width, height, and frames per second (fps) of the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in range(1, number_of_output + 1):
        print(f"Processing {idx}.mp4 of {video_path.name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

        aug = Augmentor(number_of_augmentation)

        meta_file.write(f"{idx}{aug.getName()}\n")

        # Define the codec and create a VideoWriter object
        writer = cv2.VideoWriter(
            output_dir / f"{idx}.mp4", fourcc, fps, (width, height)
        )

        # Process frames with tqdm progress bar
        with tqdm(
            total=total_frames,
            desc=f"Processing frames ({aug.getName(False)})",
            unit="frames",
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert frame to float and normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0

                frame = aug(frame)

                frame = (frame * 255).astype(np.uint8)

                writer.write(frame)

                pbar.update(1)

            writer.release()
    cap.release()
    meta_file.close()


def main(args):
    clip_dir = Path(args.clip_dir)
    assert os.path.isdir(clip_dir), f"Cannot find directory: {clip_dir}"

    number_of_output = args.number_of_output
    number_of_augmentation = args.number_of_augmentation

    clip_list = list(clip_dir.glob("*"))
    clip_file_list = list(clip_dir.glob("*/original/*.mp4"))

    for fp in clip_list:
        assert os.path.isdir(fp), f"Cannot find directory: {fp}"
    for fp in clip_file_list:
        assert os.path.isfile(fp), f"Cannot find file: {fp}"

    assert len(clip_list) == len(clip_file_list), "Only one video per directory!"

    for video_input, video_path in zip(clip_file_list, clip_list):
        print(f"Process {video_input.name}")
        augment_video(video_input, video_path, number_of_output, number_of_augmentation)
        print(f"Process {video_input.name} Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_dir",
        type=str,
        default="data/youtube/v001/",
        help="clip_dir/{clip_name}/original/{clip_name}.mp4",
    )
    parser.add_argument("--number_of_output", type=int, default=2)
    parser.add_argument("--number_of_augmentation", type=int, default=2)
    args = parser.parse_args()
    main(args)
