from mrt_phase_pde.pde.own_method.get_T_N.get_T_N_single import get_T_N

D_list = [0.0, 0.1, 0.25, 0.5, 1.0, 5.0]

if __name__=='__main__':
    for D in D_list:
        print(D)
        for i in range(0, 7):
            get_T_N(i, D=D)
