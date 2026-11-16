# Loss-resilient Gaussian-splat telepresence

Concealing network packet loss in a multi-camera 3D video call with masked discrete diffusion.

Status: work in progress. The capture and rendering stages are implemented; the codec, the loss simulation and the
concealment model are being built. Code and results are added stage by stage (see [Status](#status)).

## The problem

In a 3D video call, a few cameras film a person and the other side sees them as a 3D object that can be viewed from
any angle, for example in a headset. The 3D representation has to be compressed and sent over a real network many
times per second. Real networks lose packets, and a live call cannot wait for a retransmission. In ordinary 2D video
a lost packet gives a frozen or damaged block. In 3D it removes part of the person: a hole in the body, a missing
arm, or a surface that jumps between frames. These errors are visible from every viewpoint and they are what a
viewer notices first.

The receiver therefore has to fill in what was lost, quickly, and in a way that stays consistent between the
cameras and over time. It should also know which parts of the picture were measured and which were made up.

## Objective

Build one complete, measurable chain from cameras to rendered novel views, and answer four questions with it:

1. **Capture.** How does the quality of the 3D reconstruction depend on the camera rig: number of cameras,
   synchronisation error, calibration error, exposure mismatch and depth noise?
2. **Compression.** How many bits does a frame need when each camera view is turned into a short sequence of
   discrete tokens, compared with a standard video codec (HEVC) at the same quality?
3. **Loss.** How fast does quality at novel views fall when token packets are lost, with the simple receivers used
   today (leave a hole, or copy the previous frame)?
4. **Concealment.** How much of that loss does a masked discrete diffusion model recover when it predicts the
   missing tokens from the received tokens, the other cameras and the previous frame? What does each sampling step
   cost in latency, and is the model's confidence a usable signal of where it is wrong?

A fifth, smaller question uses the same chain: can a screen-space network that receives a cheap splat render from
few cameras (colour, coverage, depth) predict the full-quality image in one pass, and is the result stable over
time?

## Data

[UVG-VPC](https://ultravideo.fi/UVG-VPC/) (Gautier et al., QoMEX 2023): twelve voxelised point-cloud sequences of
people in motion, 250 frames each at 25 frames per second. The 9-bit version is used. The point clouds are the
ground truth: they are filmed by virtual cameras, so camera count, pose, timing, exposure and depth noise are known
exactly and can be changed one at a time.

The dataset is licensed for non-commercial research and is not redistributed here. A script downloads it from the
official source and packs it losslessly.

## Method

```mermaid
flowchart LR
    A[Point-cloud sequence] --> B[Virtual RGB-D camera ring]
    B --> C[Tokeniser: one grid of discrete tokens per view]
    C --> D[Packets over a lossy channel]
    D --> E[Masked discrete diffusion fills the lost tokens]
    E --> F[Decode views and lift to pixel-aligned Gaussians]
    F --> G[Render held-out viewpoints]
    G --> H[Compare with renders of the full point cloud]
```

- **Capture.** A ring of virtual cameras around the person gives colour, depth and a coverage (alpha) map per view.
- **Representation.** Every foreground pixel becomes one Gaussian at its 3D position, shaped as a thin disc on the
  local surface (pixel-aligned Gaussians). Views are merged and rendered with a Gaussian-splat rasteriser written
  in PyTorch.
- **Tokens.** A convolutional autoencoder with finite scalar quantisation turns each view (colour, alpha, depth)
  into a small grid of discrete tokens. A frame is the token grids of all cameras.
- **Channel.** Tokens are grouped into packets. Packets are dropped independently or in bursts (Gilbert-Elliott
  model).
- **Concealment.** A transformer trained as a masked (absorbing-state) discrete diffusion model. Lost tokens are
  the masked positions; received tokens are kept exactly. The model attends to the tokens of the other cameras and
  of the previous frame, and fills the gaps in a small number of sampling steps.
- **Provenance.** Every Gaussian carries whether it comes from received or generated tokens, so a rendered image
  has a map of measured versus generated pixels.

## Evaluation protocol

- **Viewpoints.** Quality is always measured at six viewpoints that no capture camera occupies: four on the ring
  between cameras and two raised by 20 degrees. The reference is a render of the full point cloud.
- **Metrics.** PSNR (full frame and person region), SSIM, LPIPS, fraction of the person left uncovered, a
  frame-to-frame flicker measure, and the worst 10 % of frames, since a call is judged by its bad moments.
  Bitrate in bits per frame; latency in milliseconds per frame on one GPU.
- **Splits.** Sequences are never split by frame. Seven sequences are used for training and for tuning, one for
  validation, and four are held out for testing: one with an outfit not seen in training and three of a person
  not seen in training.
- **Baselines.** No concealment; copy of the previous frame; HEVC-coded views at matched bitrate.
- **Seeds.** Learned models are trained with several seeds and gains are reported with their spread.

## Status

| Stage | Question | State |
|---|---|---|
| `01_capture_rig` | Capture | implemented, results being written up |
| `02_token_codec` | Compression | in progress |
| `03_packet_loss` | Loss | planned |
| `04_diffusion_concealment` | Concealment | planned |
| `05_screen_space_refinement` | Refinement of cheap renders | planned |

Results, analysis, limitations and reproduction commands are added to this file as each stage is finished.

## Limitations known in advance

- The cameras are simulated. Real sensors add rolling shutter, lens distortion, lighting changes and depth errors
  that a noise model only approximates.
- The dataset has few people and few outfits, so results on unseen people are a small-sample estimate.
- Masked-token recovery of lost packets has been studied for 2D video (for example Reparo, arXiv:2305.14135).
  This project studies it for multi-view human splats judged at novel viewpoints; it does not claim the idea of
  token-level concealment itself.

## Repository layout

```
lossysplat/      library: data access, cameras, renderers, lifting to Gaussians, metrics
experiments/     one folder per stage, numbered as in the table above
results/         small result files and figures, one folder per stage
scripts/         dataset download and packing
tests/           unit tests
```

## References

- Gautier et al., "UVG-VPC: Voxelized Point Cloud Dataset for Visual Volumetric Video-based Coding", QoMEX 2023.
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023.
- Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505.
- Sahoo et al., "Simple and Effective Masked Diffusion Language Models", arXiv:2406.07524.
- Shi et al., "Simplified and Generalized Masked Diffusion for Discrete Data", arXiv:2406.04329.
- Li et al., "Reparo: Loss-Resilient Generative Codec for Video Conferencing", arXiv:2305.14135.
