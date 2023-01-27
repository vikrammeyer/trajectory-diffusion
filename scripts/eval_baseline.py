import statistics

from diff_traj.cfg import cfg
from diff_traj.dataset.dataset import StateDataset
from diff_traj.utils.eval import dynamics_violation, n_collision_states
from diff_traj.viz import Visualizations

viz = Visualizations(cfg)
dataset = StateDataset(cfg, './data/test-set')
di = iter(dataset)

metrics_checkpoints = {'violations': [], 'energys': [], 'collision_states': []}

collision_free = 0
n_total = 0

violations = []
energys = []
collision_states = []

viz = Visualizations(cfg)

for i, (traj, param) in enumerate(di):
    traj, param = dataset.un_normalize(traj.squeeze(), param)
    violation, energy = dynamics_violation(cfg, traj)
    violations.append(violation)
    energys.append(energy)

    n_collisions, state_obst_idxs = n_collision_states(cfg, traj, param)
    if n_collisions == 0:
        collision_free += 1
    n_total += 1

    viz.save_trajectory(traj, param, f'results/test-set-pics/{i}.png',f'{n_collisions}: {state_obst_idxs}', state_obst_idxs)
    collision_states.append(n_collisions)


print('-----------------------------------')
print(f'Results on Test Set')
print('-----------------------------------')

print('dynamics violations per trajectory:')
print(f'mean: {statistics.mean(violations):4f}')
print(f'std: {statistics.stdev(violations):4f}')

print('associated energy per trajectory:')
print(f'mean: {statistics.mean(energys):4f}')
print(f'std: {statistics.stdev(energys):4f}')

print('# collision states per trajectory')
print(f'mean: {statistics.mean(collision_states):4f}')
print(f'std: {statistics.stdev(collision_states):4f}')

print(f'Collision Free Trajectories: {collision_free} / {n_total}')

# -----------------------------------
# Results on Test Set
# -----------------------------------
# dynamics violations per trajectory:
# mean: 0.000000
# std: 0.000000
# associated energy per trajectory:
# mean: 42.406661
# std: 2.954642
# # collision states per trajectory
# mean: 5.468000
# std: 2.032020
# Collision Free Trajectories: 0 / 1000