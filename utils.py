import numpy as np
import cv2
from functools import partial
from tqdm import tqdm as _tqdm

def overlay_frame(frame, detections, color, alpha):
    mask = detections == 255

    overlay = frame.copy()
    overlay[mask] = color

    frame_f = frame.astype(np.float32)
    overlay_f = overlay.astype(np.float32)

    frame_f[mask] = (1 - alpha) * frame_f[mask] + alpha * overlay_f[mask]
    return frame_f.astype(np.uint8)

tqdm = partial(
    _tqdm,
    desc="Processing Frames",
    bar_format="{desc}: {bar}  {n}/{total}",
    dynamic_ncols=True,
    mininterval=0.1,
    smoothing=1,
    leave=False,
    colour="yellow"
)

def get_video_properties(cap):
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = number_of_frames / fps if fps else 0

    print("\nVideo Properties")
    print("=" * 26)
    print(f"{'Frame width:':<20}{frame_width:>1.0f}px")
    print(f"{'Frame height:':<20}{frame_height:>1.0f}px")
    print(f"{'Number of frames:':<20}{number_of_frames:>1.0f}")
    print(f"{'FPS:':<20}{fps:>1.2f}")
    print(f"{'Video Duration:':<20}{video_duration:>1.2f}s")
    print("=" * 26)