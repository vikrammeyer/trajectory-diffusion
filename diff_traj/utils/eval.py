from diff_traj.cfg import cfg
from diff_traj.utils.geometry import *
from casadi import *
from diff_traj.dataset.opt import dynamics
import numpy as np

def n_collision_states(state_traj, obstacles):
    obsts = []
    for i in range(0, len(obstacles), 3):
        x, y, r = obstacles[i:i+3]
        obsts.append(Circle(x, y, r))

    collisions = 0
    for i in range(0, len(state_traj), 4):
        x, y, theta = state_traj[i:i+3]

        car = Rect(x, y, cfg.car_length, cfg.car_width)
        for obst in obsts:
            if collision(obst, car): collisions += 1

    return collisions

def dynamics_violation(cfg, state_traj):
    problem = Opti()
    u = problem.variable(2)
    x0 = problem.parameter(4)
    x1 = problem.parameter(4)

    nxt_state = dynamics(x0, u, cfg.interval_dur)
    cost = (x1 - nxt_state).T @ (x1 - nxt_state)
    problem.minimize(cost)

    problem.subject_to(
        problem.bounded(
            -cfg.max_accel, u[0], cfg.max_accel
        )
    )
    problem.subject_to(
        problem.bounded(
            -cfg.max_ang_vel, u[1], cfg.max_ang_vel
        )
    )

    problem.solver('ipopt', {}, {})

    sm = 0.0
    prev = np.array([0.0, 0.0, 0.0, 0.0])
    for i in range(0, len(state_traj), 4):
        cur = state_traj[i:i+4]
        sm += calc_step_violation(problem, prev, cur)
        prev = cur
    return sm

def calc_step_violation(problem, prev_x, cur_x):
    problem.set_value(problem.p[:4], prev_x)
    problem.set_value(problem.p[4:], cur_x)
    sol = problem.solve()
    return sol.value(problem.f)

if __name__ == '__main__':
    from diff_traj.utils.io import read_file
    data_chunk = '/Users/vikram/research/trajectory-diffusion/data/subset/chunk125.pkl'
    data = read_file(data_chunk)
    violations = []
    for sample in data:
        violations.append(dynamics_violation(cfg, sample['states']))

    import statistics

    print(f'mean: {statistics.mean(violations):4f}')
    print(f'std: {statistics.stdev(violations):4f}')
