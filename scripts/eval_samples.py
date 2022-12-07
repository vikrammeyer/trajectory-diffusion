import statistics
from diff_traj.utils.io import read_file
from diff_traj.dataset.dataset import StateDataset
from diff_traj.cfg import cfg
from pathlib import Path
from diff_traj.viz import Visualizations
from diff_traj.utils.eval import dynamics_violation, n_collision_states
from diff_traj.utils.io import write_obj

viz = Visualizations(cfg)
dataset = StateDataset(cfg, '')
sample_folder = Path('/Users/vikram/research/trajectory-diffusion/data/diff-samples-v2')

metrics_checkpoints = {'violations': [], 'energys': [], 'collision_states': []}

for checkpoint in [1, 10, 20 ,30 , 40, 50, 60, 70, 80, 90, 99]:
    files =  list(sample_folder.glob(f'{checkpoint}-sampled-*.pkl'))
    files.sort()

    violations = []
    energys = []
    collision_states = []
    for sample_file in files:
        (gt_trajs, params, sampled_trajs)  = read_file(sample_file)

        sampled_trajs = sampled_trajs.squeeze()
        for i in range(0, sampled_trajs.shape[0]):
            traj, param = dataset.un_normalize(sampled_trajs[i], params[i])
            violation, energy = dynamics_violation(cfg, traj)
            violations.append(violation)
            energys.append(energy)
            collision_states.append(n_collision_states(cfg, traj, param))

    print('-----------------------------------')
    print(f'Checkpoint {checkpoint}')
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


    metrics_checkpoints['violations'].append(violations)
    metrics_checkpoints['energys'].append(energys)
    metrics_checkpoints['collision_states'].append(collision_states)

print('done')
write_obj(metrics_checkpoints, 'results/diff-metrics.pkl')
