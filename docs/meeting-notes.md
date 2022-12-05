# Project Check-in w/ Dr. Moyer
- Don't use graph neural network for agent interaction, just use basic feed forward encoder since the agent graph structure is fully connected and static over the timesteps
- first experiment:
  - original: to generate all the agents' trajectories conditioned on the history trajectory for all the agents and the encoded road representation
  - new: generate ego agents' trajectory conditioned on the entire trajectory trajectory of all the other agents (then the diffusion model can generate different valid possible trajectories given others' motion)
    - this will not be a direct motion prediction application but will be using the dataset to perform something cool
- Idea: superresolution/inpainting of trajectory (finegrain the timestep resolution given a larger trajectory)

## next steps
- write up pytorch dataset
- get VM w/ GPU working and an easy test locally -> deploy remotely & train process setup
- train the first experiment as suggested


# Today

  
## Baseline Comparisons:
- Fully Connected Network (obstacles -> trajectory)
  - will probably perform poorly but 
- CVAE (already setup, just need to train for equivalent amount of time)

### metrics
- % collsion free trajectories (this is what was used in the Neurips workshop paper)
- dynamics violations ?? (this could be interesting to see the "quality" of the generated trajectories)



