import ray
import itertools
import numpy as np

all_poss = list(itertools.product(np.arange(1), np.arange(67)))
ray.init(num_gpus=7)
@ray.remote(num_gpus=1)
def doit(ID):

    chosen_realz, chosen_pidx = ID

    import sys
    sys.path.append('/data/taylor_group/Nima/ATLAS')
    sys.path.append('../')

    import pickle
    import glob
    import os
    
    import numpy as np
    import jax.numpy as jnp
    import jax
    jax.config.update("jax_enable_x64", True)

    from ATLAS.data import PTA_Data
    from ATLAS.signals.factorized.base import GaussianTiming
    from ATLAS.signals.timing.base import build_multi_psr_timing_model

    # Loading data

    pnames = [
    'B1855+09', 'B1937+21','B1953+29', 'J0023+0923', 'J0030+0451', 'J0340+4130', 'J0406+3039', 
    'J0437-4715', 'J0509+0856', 'J0557+1551', 'J0605+3757', 'J0610-2100', 'J0613-0200', 'J0636+5128', 
    'J0645+5158', 'J0709+0458', 'J0740+6620', 'J0931-1902', 'J1012+5307', 'J1012-4235', 'J1022+1001', 
    'J1024-0719', 'J1125+7819', 'J1312+0051', 'J1453+1902', 'J1455-3330', 'J1600-3053', 'J1614-2230', 
    'J1630+3734', 'J1640+2224', 'J1643-1224', 'J1705-1903', 
    'J1713+0747', 
    'J1719-1438', 'J1730-2304', 
    'J1738+0333', 'J1741+1351', 'J1744-1134', 'J1745+1017', 'J1747-4036', 'J1751-2857', 'J1802-2124', 
    'J1811-2405', 'J1832-0836', 'J1843-1113', 'J1853+1303', 'J1903+0327', 'J1909-3744', 'J1910+1256', 
    'J1911+1347', 'J1918-0642', 'J1923+2515', 'J1944+0907', 'J1946+3417', 'J2010-1323', 'J2017+0603', 
    'J2033+1734', 'J2043+1711', 'J2124-3358', 'J2145-0750', 'J2214+3000', 'J2229+2643', 'J2234+0611', 
    'J2234+0944', 'J2302+4442', 'J2317+1439', 'J2322+2057'
    ]

    parfiles_ref = sorted(glob.glob('/data/taylor_group/Nima/par/*.par'))
    timfiles_ref = sorted(glob.glob('/data/taylor_group/Nima/tim/*.tim'))
    ### Filter the par and tim files to remove "ao-only" files
    parfiles = []
    for pname in pnames:
        for p in parfiles_ref:
            if pname in p and not 'ao' in p and not 'gbt' in p:
                parfiles.append(p)
    timfiles = []
    for pname in pnames:
        for p in timfiles_ref:
            if pname in p and not 'ao' in p and not 'gbt' in p:
                timfiles.append(p)
    assert len(parfiles) == len(pnames)

    with open('/data/taylor_group/Nima/NG15_v1p1_final_pint_psrs.pkl', 'rb') as fin:
        psrs = pickle.load(fin)[chosen_pidx : chosen_pidx + 1]
    # with open(f'/data/taylor_group/Nima/ATLAS/Data/Pickle/Set1/{chosen_realz}_{chosen_pidx}.pkl', 'rb') as fin:
    #     psrs = [pickle.load(fin)]
        
    if psrs[0].name!='J1713+0747':
    # if psrs[0].name in ['J1903+0327', 'J1909-3744', 'J2322+2057']:
        
        data = PTA_Data(psrs, 
                        fixed_res = False,
                        marg_timing = False, 
                        diag_white_cov = False,
                        linear_timing = True,
                        )

        ## Timing Model
        SAMPLE = [psrs[pidx].fitpars[1:] for pidx in range(data.npsrs)]
        for i, sublist in enumerate(SAMPLE):
            for j, s in enumerate(sublist):
                SAMPLE[i][j] = s.replace('A1DOT', 'XDOT')

        tm_model = build_multi_psr_timing_model(parfiles[chosen_pidx : chosen_pidx + 1], 
                                                timfiles[chosen_pidx : chosen_pidx + 1], 
                                                SAMPLE, 
                                                load_how_many_in_parallel = 1,
                                                data = data)
        data.add_timing_design_matrix(tm_model.Mmats)

        ## Red Noise Model
        Udim  = 500 #the number of PCA components per pulsar
        sig_gtm = GaussianTiming(name='dm',
                    nmodes=Udim,
                    timing_model = tm_model,
                    lower_bound_psd = jnp.ones(int(Udim/2)) * -9,
                    upper_bound_psd = jnp.ones(int(Udim/2)) * -1,
                    data = data,
                    basis = None)

        # sdir = f'/data/taylor_group/Nima/ATLAS/Set3/{chosen_realz}' + f'/{psrs[0].name}'
        sdir = f'/data/taylor_group/Nima/ATLAS/NG15_AdaptusBasis' + f'/{psrs[0].name}'
        os.makedirs(sdir, exist_ok=True)
        np.save(sdir + '/U.npy', sig_gtm.U)
        np.save(sdir + '/explained_variance_ratio.npy', sig_gtm.explained_variance_ratio)
        np.save(sdir + '/last_z_scale.npy', sig_gtm.z_scale)
        np.save(sdir + '/Mmat.npy', data.Mmat)

lazy_values = [doit.remote(_) for _ in all_poss]
values = ray.get(lazy_values)
ray.shutdown()
