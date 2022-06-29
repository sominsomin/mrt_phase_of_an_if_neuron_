from mrt_phase_pde.pde.own_method.get_T_N.get_T_N_single import get_T_N

D = .1

if __name__=='__main__':
    for i in range(0, 7):
        get_T_N(i, D=D)
