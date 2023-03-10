import matplotlib.pyplot as plt
import matplotlib.animation as animation

from einops import rearrange
from trajdiff.utils import read_file
from trajdiff.multiagent import cfg
from trajdiff.multiagent.dataset import unnormalize_0_to_1

def gif(trajectories, sprites, cfg, filename='bouncing_sprites.gif'):
    n = len(sprites)
    total_frames = len(trajectories[0])
    frame_rate = 10

    def plot_sprites(frame):
        plt.clf() # clear the previous frame

        plt.xlim([cfg.xmin, cfg.xmax])
        plt.ylim([cfg.ymin, cfg.ymax])

        for i in range(n):
            x, y = trajectories[i][frame]
            plt.gca().add_artist(plt.Circle((x, y), sprites[i].radius, color='blue'))

    fig = plt.figure()
    plt.axis('equal')

    ani = animation.FuncAnimation(fig, plot_sprites, frames=total_frames, interval=1000/frame_rate, blit=False)

    ani.save(filename, writer='pillow')

def predictions_gif(history_trajs, future_gt_trajs, future_pred_trajs, cfg, filename='bouncing_sprites.gif'):
    r = cfg.max_radius

    n_agents, history_len, _ = history_trajs.shape
    future_len = future_gt_trajs.shape[1]

    total_frames = history_len + future_len
    frame_rate = 10

    def plot_sprites(frame):
        plt.clf() # clear the previous frame

        plt.xlim([cfg.xmin, cfg.xmax])
        plt.ylim([cfg.ymin, cfg.ymax])

        for agent_idx in range(n_agents):
            if frame < history_len:
                x,y = history_trajs[agent_idx, frame, :]
                plt.gca().add_artist(plt.Circle((x, y), r, color='green'))
            else:
                # print(future_pred_trajs[agent_idx].shape)
                xp,yp = future_pred_trajs[agent_idx,frame-history_len, :]
                # print(future_gt_trajs[agent_idx].shape)
                x,y = future_gt_trajs[agent_idx,frame-history_len, :]
                plt.gca().add_artist(plt.Circle((xp, yp), r, color='red'))
                plt.gca().add_artist(plt.Circle((x, y), r, color='blue'))
                # connect GT and prediction to visualize error
                plt.gca().add_artist(plt.Line2D((x, xp), (y, yp), color='gray', linestyle= '--'))

    fig = plt.figure()
    plt.axis('equal')

    ani = animation.FuncAnimation(fig, plot_sprites, frames=total_frames, interval=1000/frame_rate, blit=False)

    ani.save(filename, writer='pillow')

def show_predictions(n, samples_file, output_folder, cfg):
    data = read_file(samples_file)

    history_trajs = data['history']
    future_gt_trajs = data['future_gt']
    future_pred_trajs = data['future_pred']

    for i in range(len(history_trajs)):
        if i >= n: break
        # BUG: oops predictions are saved as [agents, channels, length]
        pred = rearrange(future_pred_trajs[i], 'a c t -> a t c')
        pred = pred.cpu()
        history_trajs_i = unnormalize_0_to_1(history_trajs[i].cpu(), cfg)
        future_gt_trajs_i = unnormalize_0_to_1(future_gt_trajs[i].cpu(), cfg)
        future_pred_trajs_i = unnormalize_0_to_1(pred, cfg)
        predictions_gif(history_trajs_i, future_gt_trajs_i, future_pred_trajs_i, cfg, output_folder/f'pred{i}.gif')

if __name__ == '__main__':
    from pathlib import Path

    samples = Path('results/t1000n100000/samples0.pkl')

    output_folder = Path('results/t1000n100000/gifs/')
    if not output_folder.exists():
        output_folder.mkdir()

    n = 5

    show_predictions(n,samples,output_folder,cfg)