import argparse
import statistics
from diff_traj.utils.io import read_file
from diff_traj.dataset.dataset import StateDataset
from diff_traj.cfg import cfg
from pathlib import Path
from diff_traj.viz import Visualizations
from diff_traj.utils.eval import dynamics_violation, n_collision_states
from diff_traj.utils.io import write_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--sample_folder', default='/Users/vikram/research/trajectory-diffusion/data/diff-samples-v2/')
    args = parser.parse_args()

    sample_folder = Path(args.sample_folder)
    viz = Visualizations(cfg)
    dataset = StateDataset(cfg, '')

    metrics_checkpoints = {'violations': [], 'energys': [], 'collision_states': []}

    collision_free = 0
    n_total = 0

    for checkpoint in [99]: #[1, 10, 20 ,30 , 40, 50, 60, 70, 80, 90, 99]:
        files =  list(sample_folder.glob(f'{checkpoint}-sampled-*.pkl'))
        files.sort()

        violations = []
        energys = []
        collision_states = []
        mse = []
        for sample_file in files:
            (gt_trajs, params, sampled_trajs)  = read_file(sample_file)

            gt_trajs = gt_trajs.squeeze()
            sampled_trajs = sampled_trajs.squeeze()
            for i in range(0, sampled_trajs.shape[0]):
                gt_traj, param = dataset.un_normalize(gt_trajs[i], params[i])
                traj, _ = dataset.un_normalize(sampled_trajs[i], params[i])
                # violation, energy = dynamics_violation(cfg, traj)
                # violations.append(violation)
                # energys.append(energy)
                if n_collision_states(cfg, traj, param) == 0: collision_free += 1
                n_total += 1
                # collision_states.append(n_collision_states(cfg, traj, param))

        print('-----------------------------------')
        print(f'Checkpoint {checkpoint}')
        print('-----------------------------------')

        print(collision_free)
        print(n_total)

    #     print('dynamics violations per trajectory:')
    #     print(f'mean: {statistics.mean(violations):4f}')
    #     print(f'std: {statistics.stdev(violations):4f}')

    #     print('associated energy per trajectory:')
    #     print(f'mean: {statistics.mean(energys):4f}')
    #     print(f'std: {statistics.stdev(energys):4f}')

    #     print('# collision states per trajectory')
    #     print(f'mean: {statistics.mean(collision_states):4f}')
    #     print(f'std: {statistics.stdev(collision_states):4f}')

    #     metrics_checkpoints['violations'].append(violations)
    #     metrics_checkpoints['energys'].append(energys)
    #     metrics_checkpoints['collision_states'].append(collision_states)

    # print('done')
    # write_obj(metrics_checkpoints, 'results/baseline-metrics.pkl')

if __name__ == '__main__':
    main()