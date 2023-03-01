from trajdiff.utils import read_file
import csv

# output_file = 'small.csv'

# i = 0
# with open(output_file, 'w') as f:
#     writer = csv.writer(f, delimiter=',')

#     for n in range(125,151):

#         data = read_file(f'data/subset/chunk{n}.pkl')

#         for row in data:
#             w = list(row['obsts']) + list(row['states']) + list(row['controls']) + list(row['duals']) + [row['iters'], row['t_proc'], row['t_wall']]
#             writer.writerow(w)
#             i += 1

# print('samples: ', i)

import glob


sample_batches = glob.glob("data/diff-samples-v2/99-sampled-*")

output_file = 'samples.csv'
with open(output_file, 'w') as f:
    writer = csv.writer(f, delimiter=',')

    for file in sample_batches:
        _, params, trajs = read_file(file)
        trajs = trajs.squeeze()
        for r in range(params.shape[0]):
            p = params[r].tolist()
            t = trajs[r].tolist()
            writer.writerow(p + t)
        print('finished', file)

