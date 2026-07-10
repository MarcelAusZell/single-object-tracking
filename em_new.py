import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import tqdm, overlay_frame
import time
import os


# Constants
K = 3 
N = 90
EPS = 1e-6
MAX_VAR = 100
NUM_VIDEOS = 2
EM_ITER = 20

# Setup
input_video = cv2.VideoCapture("videos/input.mp4")
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int( input_video.get(cv2.CAP_PROP_FPS))
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
all_frames = np.array([cv2.cvtColor(input_video.read()[1], cv2.COLOR_BGR2GRAY) for _ in range(frame_count)])
first_N_frames = all_frames[:N]
video_writer = cv2.VideoWriter(
    filename=os.path.join("videos", "em.mp4"),
    fourcc=cv2.VideoWriter_fourcc(*"avc1"),
    fps=fps,
    frameSize=(width, NUM_VIDEOS * height),
    isColor=True
)

def gaussian(x, means, vars):
  # x.shape     = (H,W)
  # means.shape = (K,H,W)
  # vars.shape  = (K,H,W)
  diff = x[None, :] - means
  maha = -0.5 * diff**2 / vars
  factor = 1 / np.sqrt(2 * np.pi * vars)
  # gaussian.shape = (K,H,W)
  return factor * np.exp(maha)

def e_step(frames, pis, means, vars):
    # frames: (N,H,W)
    # means/vars/pis: (K,H,W)
    diff = frames[:, None, :, :] - means[None, :, :, :]          # (N,K,H,W)
    maha = -0.5 * diff**2 / vars[None, :, :, :]                  # (N,K,H,W)
    factor = 1 / np.sqrt(2 * np.pi * vars)[None, :, :, :]        # (1,K,H,W)
    g = factor * np.exp(maha)                                    # (N,K,H,W)
    nom = pis[None, :, :, :] * g                                 # (N,K,H,W)
    den = np.sum(nom, axis=1)                                    # (N,H,W)
    return nom / (den[:, None, :, :] + EPS)                      # (N,K,H,W)

def m_step(frames, gammas):
  # frames.shape    = (N,H,W)
  # gammas.shape    = (N,K,H,W)
  N = frames.shape[0] 
  N_js = np.sum(gammas, axis=0) # (K,H,W)
  pis_new = N_js / N            # (K,H,W)
  means_new = np.sum(gammas * frames[:, None, ...], axis=0) / (N_js + EPS)
  vars_new  = np.sum(gammas * (frames[:, None, ...] - means_new[None, ...])**2, axis=0) / (N_js + EPS)
  vars_new = np.maximum(vars_new, MAX_VAR)
  return pis_new, means_new, vars_new

# EM Initilization
mu0 = np.mean(first_N_frames, axis=0)
sigma2 = np.var(first_N_frames, axis=0) + EPS
means = np.stack([mu0 for k in range(K)])
vars  = np.stack([sigma2 for _ in range(K)])
pis   = np.full((K, height, width), 1.0/K, dtype=np.float32)

# EM Loop
for _ in tqdm(range(EM_ITER)):
  gammas = e_step(first_N_frames, pis, means , vars)
  pis, means, vars,  = m_step(first_N_frames, gammas)


input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)


# Visualization
for _ in tqdm(range(frame_count)):
  # gaussian.shape = (K,H,W)
  # pis.shape      = (K,H,W)

  # pi_k    => how frequently does Gaussian k occur (high pi_k     = appears often                | small pi_k    = rare event)
  # sigma_k => how stable is the intensity          (high sigma_k  = pixel intensity varies a lot | small sigma_k = pixel intensity barely changes)
  # ==> Background pixels typically appear most of the time (=> hight pi_k) are stable over time (=> low sigma_k)
  # ==> pi_k / sigma_k good indicator

  frame_ok, current_frame = input_video.read()

  intensity = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

  g = gaussian(intensity, means, vars)
  score = pis / np.sqrt(vars)
  background_k = np.argmax(score, axis=0)

  g_bg = g[
      background_k,
      np.arange(height)[:,None],
      np.arange(width)[None,:]
  ]

  pi_bg = pis[
      background_k,
      np.arange(height)[:,None],
      np.arange(width)[None,:]
  ]

  p_bg = pi_bg * g_bg

  threshold = 1e-8
  mask = (p_bg < threshold).astype(np.uint8) * 255
  log_bg = 1 - np.log(p_bg + EPS)
  log_bg = cv2.normalize(log_bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

  vis0 = current_frame
  vis1 = cv2.applyColorMap(log_bg, cv2.COLORMAP_HOT)
  vis2 = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
  vis3 = overlay_frame(current_frame, mask, [0, 0, 255], 0.6)


  out = np.vstack([vis0, vis1])
  
  video_writer.write(out)


cv2.destroyAllWindows()
video_writer.release()