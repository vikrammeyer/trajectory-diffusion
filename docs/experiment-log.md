# Conditional Generation of Trajectories
- **Goal:** Generate the trajectory of the car conditioned on the parameters to the trajectory optimization problem ($x_0$ and obstacles)

## Fully Connected Denoiser
- Just used the state trajectory dataset, didn't mess with controls data
- Tried to get a simple baseline architecture implemented that did not require fancy UNet with convolutions and self-attention
    - Kind of mimicked Unet architecture with downsampling then upsamling in layer sizes
- `scripts/diffusion_ffnet.py --epochs 100 --results <sub-folder in results>`
- Initially the dataset was unnormalized and the resulting sampled trajectories just looked like gaussian noise around the origin
    - `results/state-unnormalized-fc`
- Even after normalizing the dataset upon loading (all $x,y,v \in [-1, 1]$ and $\theta$ encoded as $\cos \theta \in [-1, 1]$) the sampled trajectories were bad (1-2 orders of magnitude too large to even see on the plot)
    - Probably the architecture b/c even with lots of additional training, no changes were seen
    - `results/state-normalized-fc`
- Not too confident in bug free implementation since it was kind of hacked together from multiple sources
    - But I would not be suprised if the fully connected architecture is just not powerful enough for the denoising task

## Unet1D Denoiser
- Modified the UNet1D implementation from Phil Wang so that it can be conditioned on something for classifier free guidance (also implemented in same repo but only for 2D case)
- **Hypothesis:** Inductive biases introduced by convolutions, self-attention, and up/down sampling will help the denoiser perform much better (at least it should since each pass requires so much more compute)

#  Data Generation
- 8 cores generating 500k instances overnight
- `python3 scripts/gen_data.py -n 500000 -f 2000 -o data/nov28-overnight`
