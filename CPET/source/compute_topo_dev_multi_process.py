import numpy as np
from CPET.source.calculator import calculator
import warnings

warnings.filterwarnings(action="ignore")


def main():
    options = {
        "path_to_pqr": "../../tests/test_files/test_large.pqr",
        "center": [104.785, 113.388, 117.966],
        "x": [105.785, 113.388, 117.966],
        "y": [104.785, 114.388, 117.966],
        "n_samples": 100,
        "dimensions": [1.5, 1.5, 1.5],
        "step_size": 0.01,
        "concur_slip": 16,
        "filter_radius": 100.0,
        "filter_in_box": True,
        # "filter_resids": ["HEM"]
    }

    topo = calculator(options)
    hist = topo.compute_topo()
    # print(hist)
    np.savetxt("hist_cpet.txt", hist)


main()
