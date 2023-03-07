import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def predictions_gif(history_trajs, future_gt_trajs, future_pred_trajs, sprites, cfg, filename='bouncing_sprites.gif'):
    n = len(sprites)
    total_frames = len(history_trajs[0])
    frame_rate = 10

    def plot_sprites(frame):
        plt.clf() # clear the previous frame

        plt.xlim([cfg.xmin, cfg.xmax])
        plt.ylim([cfg.ymin, cfg.ymax])

        for i in range(n):
            x, y = history_trajs[i][frame]
            plt.gca().add_artist(plt.Circle((x, y), sprites[i].radius, color='blue'))

    fig = plt.figure()
    plt.axis('equal')

    ani = animation.FuncAnimation(fig, plot_sprites, frames=total_frames, interval=1000/frame_rate, blit=False)

    ani.save(filename, writer='pillow')