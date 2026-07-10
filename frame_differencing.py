import cv2
import numpy as np
import os
from utils import overlay_frame, tqdm

def frame_differencing(background_frame, current_frame, threshold):
    diff = current_frame.astype(np.float32) - background_frame.astype(np.float32) # [-255,255]^3
    diff = np.abs(diff) / 255.0                                                   # [0,1]^3
    diff_intensity = np.sum(diff, axis=2)                                         # [0,3]
    diff_intensity = diff_intensity > threshold                                   # {0,1}
    return diff_intensity.astype(np.uint8) * 255  

def frame_differencing_videos(input_video_path, folder, output_name):
  print("Single Object Detection: Frame Differencing")

  input_video = cv2.VideoCapture(input_video_path)
  width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = float(input_video.get(cv2.CAP_PROP_FPS))

  video_writer = cv2.VideoWriter(filename=os.path.join(folder, output_name + ".mp4"),
                                 fourcc=cv2.VideoWriter_fourcc(*"avc1"),
                                 fps=fps,
                                 frameSize=(3 * width, height),
                                 isColor=True)
  
  frame_ok, background_frame = input_video.read()
  stepsize_difference = 3

  for frame_idx in tqdm(range(frame_count)):
    frame_ok, current_frame = input_video.read()
    if not frame_ok:
      break
    
    if frame_idx % stepsize_difference == 0:
      background_frame = current_frame.copy()

    detections = frame_differencing(background_frame, current_frame, 0.7)
    stacked = np.hstack((current_frame,
                         np.stack([detections]*3, axis=-1),
                         overlay_frame(current_frame, detections, [0, 0, 255], 0.6)))
    video_writer.write(stacked)

  cv2.destroyAllWindows()
  input_video.release()
  video_writer.release()