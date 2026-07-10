import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import overlay_frame, tqdm

# Constants
N = 100
VARIANCE = 100
NUM_VIDEOS = 1

# Setup
input_video = cv2.VideoCapture("videos/input.mp4")
fps = input_video.get(cv2.CAP_PROP_FPS)
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
all_frames = np.array(
    [
        cv2.cvtColor(input_video.read()[1], cv2.COLOR_BGR2GRAY)
        for _ in range(frame_count)
    ]
)
input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
first_N_frames = np.array(
    [cv2.cvtColor(input_video.read()[1], cv2.COLOR_BGR2GRAY) for _ in range(N)]
)
video_writer = cv2.VideoWriter(
    filename=os.path.join("videos", "kernel_density_estimation" + ".mp4"),
    fourcc=cv2.VideoWriter_fourcc(*"avc1"),
    fps=fps,
    frameSize=(NUM_VIDEOS * width, height),
    isColor=True,
)


def kernel_density_estimate(inference_frame, variance):
    inference_frame = inference_frame.astype(np.float32)  # (H,W)
    diff = training_frames - inference_frame[None, ...]  # (N,H,W)

    kernels = factor * np.exp(-(diff**2) / (2 * variance))  # (N,H,W)
    return np.mean(kernels, axis=0)


input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
training_frames = first_N_frames.astype(np.float32)  # (N,H,W)
factor = 1 / np.sqrt(2 * np.pi * VARIANCE)

for frame in tqdm(all_frames):
    probability_density = kernel_density_estimate(frame, VARIANCE)
    vis = cv2.normalize(probability_density, None, 0, 255, cv2.NORM_MINMAX)
    log_vis = -np.log(probability_density + 1e-8)
    vis = cv2.normalize(log_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)
    # cv2.imshow("test", vis)
    # if cv2.waitKey(1) == ord("q"): break
    video_writer.write(vis)
