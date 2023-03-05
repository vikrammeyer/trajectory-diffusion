import logging
import random

from casadi import *

from trajdiff.static_obst.obstacles import generate_obstacles, generate_obstacles_w_constraints
from trajdiff.static_obst.opt import setup_problem


def gen_samples(cfg, n, seed, constrain_obsts):
    random.seed(seed)

    samples = []

    problem = setup_problem(cfg)

    while len(samples) < n:
        try:
            (obst, x, u, duals, iters, t_proc, t_wall) = gen_and_solve_problem(
                cfg, problem, constrain_obsts
            )

            samples.append(
                {
                    "obsts": obst,
                    "states": x,
                    "controls": u,
                    "duals": duals,
                    "iters": iters,
                    "t_proc": t_proc,
                    "t_wall": t_wall,
                }
            )

            logging.info(
                f"Problem {len(samples)} solved in {iters} iters in {t_proc} seconds"
            )

        except RuntimeError as e:
            # Casadi throws runtime errors for issues with solving problems
            logging.error(f"{e}")

    return samples


def gen_and_solve_problem(cfg, problem, constrain_obsts=False):
    if constrain_obsts:
        obsts = generate_obstacles_w_constraints(cfg)
    else:
        obsts = generate_obstacles(cfg)

    problem.set_value(problem.p[4 : 4 + 3 * cfg.n_obstacles], obsts)

    solution = problem.solve()

    primals = solution.value(problem.x)
    states = primals[: cfg.traj_length]
    controls = primals[cfg.traj_length :]

    duals = solution.value(problem.lam_g)
    iters = solution.stats()["iter_count"]
    t_proc = solution.stats()["t_proc_total"]
    t_wall = solution.stats()["t_wall_total"]

    return (obsts, states, controls, duals, iters, t_proc, t_wall)


if __name__ == "__main__":
    from trajdiff.static_obst.cfg import cfg

    data = gen_samples(cfg, 5, 42)
    print(data)
