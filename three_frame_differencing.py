import cv2
import numpy as np
import os
from utils import overlay_frame, tqdm

def three_frame_differencing(background_frame, current_frame, threshold):
  diff = current_frame.astype(np.float32) - background_frame.astype(np.float32) # [-255,255]^3
  diff = np.abs(diff) / 255.0                                                   # [0,1]^3
  diff_intensity = np.sum(diff, axis=2)                                         # [0,3]
  diff_intensity = diff_intensity > threshold                                   # {0,1}
  return diff_intensity.astype(np.uint8) * 255  

def three_frame_differencing_videos(input_video_path, folder, output_name):
  print("Single Object Detection: Three Frame Differencing")

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
  

  frames = []
  delay = 15

  while True:
    frame_ok, frame = input_video.read()
    if not frame_ok:
      break
    frames.append(frame)

  input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

  for frame_idx in tqdm(range(frame_count)):
    frame_ok, current_frame = input_video.read()

    past_frame = frames[max(frame_idx - delay, 0)]
    future_frame = frames[min(frame_idx + delay, frame_count - 1)]

    detections_past_frame = three_frame_differencing(past_frame, current_frame, 0.7)
    detections_future_frame = three_frame_differencing(future_frame, current_frame, 0.7)
    detections = np.bitwise_and(detections_past_frame, detections_future_frame)

    stacked = np.hstack((current_frame, np.stack([detections] * 3, axis=-1), overlay_frame(current_frame, detections, [0, 0, 255], 0.6)))
    video_writer.write(stacked)

  cv2.destroyAllWindows()
  input_video.release()
  video_writer.release()