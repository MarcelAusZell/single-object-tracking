import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils import tqdm
import time


input_video = cv2.VideoCapture("videos/input.mp4")
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int( input_video.get(cv2.CAP_PROP_FPS))
frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

N = 100         # first N frames of video
K = 2           # number of gaussian components
first_n_frames = []

for frame_idx in range(N):
    _, frame = input_video.read()
    first_n_frames.append(frame)

input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
first_n_frames = np.array(first_n_frames)

# EM Initilization
means = np.array([first_n_frames[int(np.random.uniform(0,N))] for _ in range(K)])
vars = np.full_like(means,[100,100,100])
pis = np.full((K, height, width), 1.0 / K) 

def gaussian(x, means, vars):
    # x.shape           = (H,W,3)
    # means.shape       = (K,H,W,3)
    # vars.shape = (K,H,W,3)
    D = 3
    x = np.stack([x] * K, axis=0)                       # (H,W,3) ==> (K,H,W,3)
    diff = x - means                                    # (K,H,W,3)
    maha = np.sum((diff * diff) / vars, axis=-1)        # (K,H,W)                
    det = np.prod(vars, axis=-1)                        # (K,H,W)
    factor = 1.0 / np.sqrt(((2.0 * np.pi) ** D) * det)  # (K,H,W)
    return factor * np.exp(-0.5 * maha)                 # (K,H,W)   

def responsibilities(x, pis, means, vars):
    # x.shape           = (H,W,3)
    # pis.shape         = (K,)
    # means.shape       = (K,H,W,3)
    # vars.shape = (K,H,W,3)
    # gaussian.shape    = (K,H,W)
    nominator = pis * gaussian(x, means, vars)                # (K,H,W)
    denominator = np.sum(nominator, axis=0)                   # (H,W)
    return nominator / (denominator[None, ...] + 1e-12)       # (K,H,W)


def e_step(xs, pis, means, vars_):
    # returns gammas with shape (N,K,H,W)
    gammas = np.empty((xs.shape[0], K, height, width))
    for n in range(xs.shape[0]):
        gammas[n] = responsibilities(xs[n], pis, means, vars_)

    return gammas

def m_step(xs, gammas):
    # xs:     (N,H,W,3)
    # gammas: (N,K,H,W)

    eps = 1e-12
    N_j = np.sum(gammas, axis=0)                                     # (K,H,W)
    new_pis = N_j / xs.shape[0]                                      # (K,H,W)

    new_means = np.sum(gammas[..., None] * xs[:, None, ...], axis=0) # (N,K,H,W,1) * (N,1,H,W,3) = (N,K,H,W,3) =sum=> (K,H,W,3)
    new_means = new_means / (N_j[..., None] + eps)                   # (K,H,W,3) / (K,H,W,1)     = (K,H,W,3)

    diff = xs[:, None, ...] - new_means[None, ...]                   # (N,1,H,W,3) - (1,K,H,W,3) = (N,K,H,W,3)
    new_vars = np.sum(gammas[..., None] * (diff * diff), axis=0)     # (N,K,H,W,1) * (N,K,H,W,3) = (N,K,H,W,3) =sum=> (K,H,W,3)
    new_vars = new_vars / (N_j[..., None] + eps)                     # (K,H,W,3) / (K,H,W,1)     = (K,H,W,3)
    new_vars = np.maximum(new_vars,25)

    return new_pis, new_means, new_vars

# EM Iterations
max_iter = 4
for _ in tqdm(range(max_iter)):
    gammas = e_step(first_n_frames, pis, means, vars)
    pis, means, vars = m_step(first_n_frames, gammas)

pixel_likelihoods = np.zeros((width, height,3))
for frame in tqdm(range(frame_count)):
    frame_ok, frame = input_video.read()


    gamma0 = responsibilities(frame, pis, means, vars)[0]
    mask = (gamma0 > 0.5).astype(np.uint8) * 255
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) == ord("q"): break

