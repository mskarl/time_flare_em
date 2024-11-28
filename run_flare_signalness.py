import numpy as np
from run_flare_fit import start_run
import time

sob = np.load("bg_sob_spatial_energy_signalness_cuts.npy", allow_pickle=True).item()

for key in list(sob.keys()):
    t0 = time.time()
    llh = start_run(sob, "SIGNAL", key, 0, 1000, start_ns=10)
    print("total time = ", time.time() - t0)
    dic = {key: llh}

    np.save("bg_llh_with_signalness_cuts_{}.npy".format(key), dic, allow_pickle=True)

    print(len(llh), len(llh[0]))