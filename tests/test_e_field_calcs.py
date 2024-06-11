import numpy as np
from CPET.source.calculator import calculator
from CPET.utils.calculator import make_histograms, construct_distance_matrix_alt2, construct_distance_matrix
import warnings 
warnings.filterwarnings(action='ignore')
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from CPET.utils.calculator import (
    compute_field_on_grid,
    calculate_field_at_point, 
    calculate_electric_field_dev_c_shared,
    calculate_electric_field_base,
    calculate_electric_field_dev_c_shared,
    calculate_electric_field_c_shared_full_alt,
    calculate_electric_field_c_shared_full,
    calculate_electric_field_gpu_torch, 
    )


class test_efield_calcs:
    def __init__(self):
        self.test_file = "./test_files/test_large.pdb"
        self.options = {
            "dtype": "float32", 
            "center": {
                "method": "first", 
                "atoms": {
                    "CD": 2
                }
            }, 
            "x":[1, 0, 0],
            "y": [0, 1, 0],
            "filter_radius": 10.0,
            "initializer": "uniform", 
            "dimensions":  [
                    2.0,
                    2.0,
                    2.0
            ], 
            "filter_in_box": True, 
            "CPET_method": "woohoo",
            "batch_size": 100

        }
        # use existing machinery to generate mesh of starting points
        self.topo = calculator(self.options, path_to_pdb = self.test_file)
        self.gather_reference_from_point_utility() # this is ground 
        self.gather_reference_simplest_implementation() # this is ground 
        self.gather_reference_from_utility() # this yields inf values

    # ground truth can be our compute_field_on_grid or the simplest cpu implementation we have

    def gather_reference_from_utility(self):
        #field = self.topo.compute_box()
        field = compute_field_on_grid(
            grid_coords=self.topo.random_start_points, 
            x=self.topo.x, 
            Q=self.topo.Q
        )
        field = np.round(field, 3)
        print("efield shape: {}".format(field.shape))
        print(field[1000:1001])
        # round field to 3 decimal places
        print("inf values: {}".format(np.isinf(field).any()))
        
        self.utility_field = field


    def gather_reference_from_point_utility(self):
        field_list = []
        for i in self.topo.random_start_points:
            field = calculate_field_at_point(
                Q=self.topo.Q, 
                x=self.topo.x, 
                x_0=i
            )
            field_list.append(field)

        field = np.array(field_list)
        # convert convert self.topo.random_start_points(N, 3) and fields(N, 3) to (N, 6)
        field_formatted = np.hstack((self.topo.random_start_points, field))
        field_formatted = np.round(field_formatted, 3)
        print("efield shape: {}".format(field_formatted.shape))
        print(field_formatted[1000:1001])
        print("inf values: {}".format(np.isinf(field_formatted).any()))
        self.point_utility_field = field_formatted

        
    def gather_reference_simplest_implementation(self): 
        field_list = []
        for i in self.topo.random_start_points:
            field = calculate_electric_field_base(
                Q=self.topo.Q, 
                x=self.topo.x, 
                x_0=i
            )
            field_list.append(field)

        field = np.array(field_list)
        # convert convert self.topo.random_start_points(N, 3) and fields(N, 3) to (N, 6)
        field_formatted = np.hstack((self.topo.random_start_points, field))
        field_formatted = np.round(field_formatted, 3)
        print("efield shape: {}".format(field_formatted.shape))
        print("inf values: {}".format(np.isinf(field_formatted).any()))
        print(field_formatted[1000:1001])
        # round field to 3 decimal places
        
        self.reference_field = field_formatted
        
    
    def test_two_reference_fields(self):
        np.testing.assert_allclose(self.point_utility_field, self.reference_field, rtol=1e-2, atol=1e-2)


    def field_equality(self, test_field):
        # print max difference between the two fields
        #print("max diff: {}".format(np.max(np.abs(self.utility_field - self.reference_field))))
        #np.testing.assert_allclose(self.reference_field, self.point_utility_field, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.reference_field, test_field, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(self.point_utility_field, test_field, rtol=1e-2, atol=1e-2)


    def test_calculators(self):
        calculator_function_list = [
            calculate_electric_field_dev_c_shared, # works
            calculate_electric_field_c_shared_full, # works
            calculate_electric_field_c_shared_full_alt, # works
            calculate_electric_field_gpu_torch,
            # add your functions here
        ]

        for calculator_function in calculator_function_list:
            #print name of calculator
            print("----"*15)
            print(calculator_function.__name__)
            field_list = []
            for i in self.topo.random_start_points:
                field = calculator_function(
                    Q=self.topo.Q, 
                    x=self.topo.x, 
                    x_0=i
                )
                field_list.append(field)

            field = np.array(field_list)
            # convert convert self.topo.random_start_points(N, 3) and fields(N, 3) to (N, 6)
            field_formatted = np.hstack((self.topo.random_start_points, field))
            field_formatted = np.round(field_formatted, 3)
            # round field to 3 decimal places
            self.field_equality(field_formatted)

        print("All tests passed!")
    

    # Note: if you create a new calculator, you should add a test here. 
    # TODO: pujan - add compatability with your gpu calcs
    # TODO: pujan - make your GPU field calcs either (a) a function (b) reimplement them here 
    # TODO: the latter is not ideal cause if you ever change those implementations then you have
    # TODO: to change them here too.
     
