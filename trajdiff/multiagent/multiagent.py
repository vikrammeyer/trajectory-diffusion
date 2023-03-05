import matplotlib.pyplot as plt
import matplotlib.animation as animation

import random
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class Sprite:
    def __init__(self, x, y, radius, x_speed, y_speed, color, cfg):
        self.x = x
        self.y = y
        self.radius = radius
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.color = color
        self.cfg = cfg

    def update(self, sprites):
        self.x += self.x_speed
        self.y += self.y_speed

        # Bounce off the walls
        if self.x + self.radius > self.cfg.xmax or self.x - self.radius < 0:
            self.x_speed = -self.x_speed
        if self.y + self.radius > self.cfg.ymax or self.y - self.radius < 0:
            self.y_speed = -self.y_speed

        # Apply repulsion forces
        for sprite in sprites:
            if sprite != self:
                dx = sprite.x - self.x
                dy = sprite.y - self.y
                distance = math.sqrt(dx**2 + dy**2)

                # check if distance == 0
                if distance == 0.0:
                    distance = 1e-4

                if distance < 50:
                    force = math.exp(distance / 1000)
                    self.x_speed -= force * dx / distance
                    self.y_speed -= force * dy / distance

                    if self.x_speed < 0:
                        self.x_speed = max(-self.cfg.max_speed, self.x_speed)
                    else:
                        self.x_speed = min(self.cfg.max_speed, self.x_speed)


def generate_multiple_trajs(seed, n_trajs, n_agents, traj_len, cfg):
    random.seed(seed)

    samples = []
    for _ in range(n_trajs):
        trajectories, sprites = generate_trajectory(n_agents, traj_len, cfg)
        data = {'trajectories': trajectories, 'radii': [sprite.radius for sprite in sprites]}
        samples.append(data)

    return samples

def generate_trajectory(n_agents, traj_len, cfg):
    sprites = []
    for i in range(n_agents):
        in_collision = True
        while in_collision:
            x = random.randint(0 + cfg.max_radius, cfg.xmax - cfg.max_radius)
            y = random.randint(0 + cfg.max_radius, cfg.ymax - cfg.max_radius)

            for sprite in sprites:
                in_collision = sprite.x == x or sprite.y == y

            if len(sprites) == 0:
                in_collision = False

        radius = cfg.max_radius #random.randint(10, MAX_RADIUS)

        zero_speed = True
        while zero_speed:
            x_speed = random.randint(-cfg.max_speed, cfg.max_speed)
            y_speed = random.randint(-cfg.max_speed, cfg.max_speed)
            zero_speed = x_speed == 0 or y_speed == 0

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        sprite = Sprite(x, y, radius, x_speed, y_speed, color)
        sprites.append(sprite)

    trajectories = [[] for _ in range(n_agents)]

    for i in range(traj_len + cfg.start_buffer):
        for j, sprite in enumerate(sprites):
            if i >= cfg.start_buffer:
                trajectories[j].append((sprite.x, sprite.y))
            sprite.update(sprites)

    return trajectories, sprites

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


def in_collision(x1, y1, x2, y2, r1, r2):
    return (x1 - x2)**2 + (y1 - y2)**2 < (r1 + r2)**2


def calc_collisions(trajectories, radii):
    n_collisions = 0

    traj_steps = len(trajectories[0])
    n_agents = len(radii)
    for step in range(traj_steps):
        for agent in range(n_agents):
            x1,y1 = trajectories[agent][step]
            r1 = radii[agent]
            for agent2 in range(agent+1, n_agents):
                x2, y2 = trajectories[agent2][step]
                r2 = radii[agent2]

                if in_collision(x1, y1, x2, y2, r1, r2):
                    n_collisions += 1

    return n_collisions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n_agents', type=int, default=10)
    parser.add_argument('-l','--traj_len', type=int, default=100)
    args  = parser.parse_args()

    traj, sprites = generate_trajectory(args.n_agents, args.traj_len)

    radii = [sprite.radius for sprite in sprites]
    print('# collisions:', calc_collisions(traj, radii))

    gif(traj, sprites)