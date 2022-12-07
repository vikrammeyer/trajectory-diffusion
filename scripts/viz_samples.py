from diff_traj.utils.io import read_file
from diff_traj.dataset.dataset import StateDataset
from diff_traj.cfg import cfg
from pathlib import Path
from diff_traj.viz import Visualizations

viz = Visualizations(cfg)
dataset = StateDataset(cfg, '')
sample_folder = Path('/Users/vikram/research/trajectory-diffusion/data/diff-samples-v2')
output_folder = Path("results/sample-diff-pics")

if not output_folder.exists():
    output_folder.mkdir()


for checkpoint in [1, 10, 20 ,30 , 40, 50, 60, 70, 80, 90, 99]:
    files =  list(sample_folder.glob(f'{checkpoint}-sampled-*.pkl'))
    files.sort()
    chckpt_dir = output_folder/f"checkpt-{checkpoint}/"

    if not chckpt_dir.exists(): chckpt_dir.mkdir()

    sample = 1
    for sample_file in files:
        (gt_trajs, params, sampled_trajs)  = read_file(sample_file)

        sampled_trajs = sampled_trajs.squeeze()
        for i in range(0, sampled_trajs.shape[0]):
            traj, param = dataset.un_normalize(sampled_trajs[i], params[i])
            viz.save_trajectory(traj, param, chckpt_dir/f'{sample}.png')
            print('vized ', sample)
            sample += 1
    print('finished a checkpoint')

    print('done')
