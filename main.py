import os
import cv2
import matplotlib.pyplot as plt
from background_subtraction import background_subtraction_videos
from frame_differencing import frame_differencing_videos
from three_frame_differencing import three_frame_differencing_videos
from single_gaussian import single_gaussian_videos
from single_gaussian_emaf import single_gaussian_emaf_videos


def get_t_th_frame(videopath, frame_t):
    cap = cv2.VideoCapture(videopath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame in range(frame_count):
        frame_ok, frame = cap.read()
        if frame_ok:
            frames.append(frame)

    ret = frames[frame_t]
    cv2.imwrite(f"{videopath[:-4]}_{frame_t}_frame.png", ret)


def main():
    input_video_path = os.path.join("videos", "input.mp4")
    # background_subtraction_videos(input_video_path=input_video_path, folder="videos", output_name="background_subtraction")
    # frame_differencing_videos(input_video_path=input_video_path, folder="videos", output_name="frame_differencing")
    # three_frame_differencing_videos(input_video_path=input_video_path, folder="videos", output_name="three_frame_differencing")
    # single_gaussian_videos(input_video_path=input_video_path, folder="videos", output_name="single_gaussian")
    single_gaussian_emaf_videos(input_video_path=input_video_path, folder="videos", output_name="single_gaussian_emaf")


if __name__ == "__main__":
    main()
