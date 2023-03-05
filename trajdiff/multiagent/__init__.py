from trajdiff.multiagent.multiagent import *
from trajdiff.multiagent.dataset import MultiAgentDataset

from types import SimpleNamespace

cfg = SimpleNamespace(
    xmin=0,
    xmax=800,
    ymin=0,
    ymax=800,
    max_radius=20,
    max_speed=2,
    start_buffer=10, # to make sure no collisions in the initial states for all agents
)

# 2 is slow enough to prevent collision penetration most of the time
# 5 is too fast and sometimes penetration collisions occur