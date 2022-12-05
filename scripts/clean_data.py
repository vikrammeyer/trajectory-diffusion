from diff_traj.viz import Visualizations
from diff_traj.dataset.dataset import StateDataset
from diff_traj.cfg import cfg
import matplotlib.pyplot as plt

import lovely_tensors as lt
lt.monkey_patch()

in_file = 'data/nov29-night'

viz = Visualizations(cfg)

plt.axis([0,120, -10, 10])
plt.ion()
plt.show()

# with open(in_file, 'r') as in_f, open(out_file, 'w') as out_f:
#     reader, writer = csv.reader(in_f), csv.writer(out_f)

#     for i, row in enumerate(reader, 1):
#         x0 = [float(i) for i in row[:4]]
#         obstacles = [float(i) for i in row[4:13]]
#         traj = [float(i) for i in row[13:13+160]]
#         plt.clf()
#         plt.axis([0,120, 0, cfg.lane_width + 20])
#         viz.show_trajectory(traj, x0, obstacles)
#         in_  = input('press enter to continue (d to delete)')
#         if in_.strip().lower() == 'd':
#             print(f'deleting sample {i}')
#         else:
#             writer.writerow(row)

data = StateDataset(cfg, in_file)
di = iter(data)

for traj, params in di:
    traj = traj.squeeze()
    plt.clf()
    plt.axis([0,120, 0, cfg.lane_width + 20])
    traj, params = data.un_normalize(traj, params)

    viz.show_trajectory(traj, params)
    input('press to continue')