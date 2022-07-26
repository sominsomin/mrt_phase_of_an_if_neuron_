from mrt_phase_own_method.pde.own_method.get_T_N.get_T_N import get_T_N

D_list = [0.0, 0.1, 0.25, 0.5, 1.0]

if __name__=='__main__':
    for D in D_list:
        print(f'D = {D}')
        for i in range(1, 7):
            get_T_N(i, D=D)
