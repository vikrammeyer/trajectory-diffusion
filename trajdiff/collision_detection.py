import numpy as np
from casadi import Opti, cos, sin


def in_collision(obstacle, car_state, cfg):
    """Detect a collision between a circle obstacle [x,y,r] and a
    rotated rectanglular car [x,y,v,theta]
    """
    # SETUP
    p = np.array([car_state[0], car_state[1]])
    angle = car_state[3]
    hl = cfg.car_length / 2
    hw = cfg.car_width / 2

    R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    R_inv = np.linalg.inv(R)

    print(p.shape)
    print(R.shape)
    print(R_inv.shape)

    l = np.array([-hl, -hw]) + R_inv * p
    u = np.array([hl, hw]) + R_inv * p

    c = np.array([obstacle[0], obstacle[1]])
    r = obstacle[2]

    # SOLVE
    opti = Opti()

    z1 = opti.variable(2)
    z2 = opti.variable(2)

    # might need to do the weird thing with [[I -I], [-I I]]
    # NVM, thats just the matrix equivalent to the below thing
    # opti.minimize((z1 - z2).T @ (z1 - z2))
    opti.minimize((z1[0] - z2[0]) ** 2 + (z1[1] - z2[1]) ** 2)

    # z1 must stay in/on the rectangle for the car
    # ok = z1.T @ R_inv
    # # opti.subject_to(opti.bounded(l[0],R_inv[0,0]*z1[0] + R_inv[0][1]*z1[1] ,u[0]))
    # # opti.subject_to(opti.bounded(l[1],R_inv[1,0]*z1[0] + R_inv[1,1]*z1[1] ,u[1]))

    # opti.subject_to(opti.bounded(l[0],ok[0],u[0]))
    # opti.subject_to(opti.bounded(l[1],ok[1],u[1]))

    # problem with the vector constraint
    opti.subject_to(opti.bounded(l, R_inv @ z1, u))
    # z2 must stay in/on the circle
    opti.subject_to(opti.bounded(0, (z2 - c).T @ (z2 - c), r**2))

    # NLP b/c of the quadratic constraint due to ensuring z2 in the circle

    opti.solver("ipopt")

    # opti.set_initial(z1, p)
    # opti.set_initial(z2, c)

    try:
        sol = opti.solve()
        return sol.value(opti.f) < 1e-7
    except RuntimeError:
        print(opti.debug.value(opti.f))
        print(opti.debug.value(z1))
        print(opti.debug.value(z2))

        print(opti.debug.show_infeasibilities())

    # print(sol.stats())
    # print(sol)


if __name__ == "__main__":
    from trajdiff.cfg import cfg

    # assert in_collision([0,0,1], [0,0,0,0], cfg)

    assert not in_collision([0, 0, 1], [10, 0, 0, 0], cfg)
