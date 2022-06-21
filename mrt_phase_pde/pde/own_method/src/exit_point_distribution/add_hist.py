import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution



if __name__ == '__main__':
    prob = ExitPointDistribution.load('..\\..\\data\\exit_points_distr_matrix.pickle')

    a_bins = np.insert(prob.a, len(prob.a), prob.a[-1] + np.diff(prob.a)[0])

    for i, _v in enumerate(prob.v):
        print(i)
        print(f'v: {_v}')
        for j, _a in enumerate(prob.a):
            print(f'a: {_a}')
            hist = plt.hist(prob[i][j][:, 0], bins=a_bins, density=True)
            p = hist[0]

            prob.p[i][j] = p

    prob.save('..\\..\\data\\exit_points_distr_matrix.pickle')