"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""
# Dev script to benchmark the performance of various topo calc methods
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""

import numpy as np
import time
from multiprocessing import Pool
from torch.profiler import profile, ProfilerActivity
import torch


from CPET.utils.io import parse_pdb
from CPET.utils.parallel import task, task_batch, task_base, task_complete_thread
from CPET.utils.calculator import (
    initialize_box_points_random,
    initialize_box_points_uniform,
    compute_field_on_grid,
    calculate_electric_field_dev_c_shared,
    compute_ESP_on_grid,
)
from CPET.utils.io import (
    parse_pdb,
    parse_pqr,
    filter_radius,
    filter_radius_whole_residue,
    filter_residue,
    filter_in_box,
    calculate_center,
    filter_resnum,
    filter_resnum_andname,
    default_options_initializer
)
from CPET.utils.gpu import (
    propagate_topo_matrix_gpu,
    compute_curv_and_dist_mat_gpu,
    batched_filter_gpu,
    generate_path_filter_gpu,
)


class calculator:
    def __init__(self, options, path_to_pdb=None):
        # self.efield_calc = calculator(math_loc=math_loc)
        self.options = default_options_initializer(options)
        
        self.profile = options["profile"]
        self.path_to_pdb = path_to_pdb
        self.step_size = options["step_size"]
        self.n_samples = options["n_samples"]
        self.dimensions = np.array(options["dimensions"])
        self.concur_slip = options["concur_slip"]
        self.GPU_batch_freq = options["GPU_batch_freq"]
        self.dtype = options["dtype"]
        self.max_streamline_init = options["max_streamline_init"] if "max_streamline_init" in options.keys() else "true_rand"

        #Be very careful with the box_shift option. The box needs to be centered at the origin and therefore, the code will shift protein in the opposite direction of the provided box vector 
        self.box_shift = options["box_shift"] if "box_shift" in options.keys() else [0,0,0]
        
        if ".pqr" in self.path_to_pdb:
            (
                self.x,
                self.Q,
                self.atom_number,
                self.resids,
                self.residue_number,
                self.atom_type
            ) = parse_pqr(self.path_to_pdb)
        else:
            (
                self.x,
                self.Q,
                self.atom_number,
                self.resids,
                self.residue_number,
                self.atom_type,
            ) = parse_pdb(self.path_to_pdb, get_charges=True)

        ##################### define center axis

        if type(options["center"]) == list:
            self.center = np.array(options["center"])

        elif type(options["center"]) == dict:
            method = options["center"]["method"]
            centering_atoms = [
                (element, options["center"]["atoms"][element])
                for element in options["center"]["atoms"]
            ]
            pos_considered = [
                pos
                for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.center = calculate_center(pos_considered, method=method)
        else:
            raise ValueError("center must be a list or dict")

        ##################### define x axis

        if type(options["x"]) == list:
            self.x_vec_pt = np.array(options["x"])

        elif type(options["x"]) == dict:
            method = options["x"]["method"]
            centering_atoms = [
                (element, options["x"]["atoms"][element])
                for element in options["x"]["atoms"]
            ]
            pos_considered = [
                pos
                for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.x_vec_pt = calculate_center(pos_considered, method=method)

        else:
            raise ValueError("x must be a list or dict")

        ##################### define y axis

        if type(options["y"]) == list:
            self.y_vec_pt = np.array(options["y"])

        elif type(options["y"]) == dict:
            method = options["y"]["method"]
            centering_atoms = [
                (element, options["y"]["atoms"][element])
                for element in options["y"]["atoms"]
            ]
            pos_considered = [
                pos
                for atom_res in centering_atoms
                for idx, pos in enumerate(self.x)
                if (self.atom_type[idx], self.residue_number[idx]) == atom_res
            ]
            self.y_vec_pt = calculate_center(pos_considered, method=method)

        else:
            raise ValueError("y must be a list or dict")

        
        if "filter_resids" in options.keys():
            # print("filtering residues: {}".format(options["filter_resids"]))
            self.x, self.Q, self.residue_number, self.resids = filter_residue(
                self.x, 
                self.Q,
                self.residue_number, 
                self.resids, 
                filter_list=options["filter_resids"]
            )

        if "filter_resnum" in options.keys():
            # print("filtering residues: {}".format(options["filter_resids"]))
            self.x, self.Q, self.residue_number, self.resids = filter_resnum(
                self.x,
                self.Q,
                self.residue_number,
                self.resids,
                filter_list=options["filter_resnum"],
            )

        if "filter_resnum_andname" in options.keys():
            # print("filtering residues: {}".format(options["filter_resids"]))
            self.x, self.Q, self.residue_number, self.resids, self.atom_number, self.atom_type = filter_resnum_andname(
                self.x,
                self.Q,
                self.residue_number,
                self.resids,
                self.atom_number,
                self.atom_type,
                filter_list=options["filter_resnum_andname"],
            )

        if "filter_radius" in options.keys():
            print("filtering by radius: {} Ang".format(options["filter_radius"]))

            r = np.linalg.norm(self.x, axis=1)
            # print("r {}".format(r))

            #Default is whole residue-inclusive filtering to ensure proper topology convergence
            self.x, self.Q = filter_radius_whole_residue(
                x=self.x,
                Q=self.Q,
                resids=self.resids,
                resnums=self.residue_number,
                center=self.center,
                radius=float(options["filter_radius"]),
            )

            # print("center {}".format(self.center))
            r = np.linalg.norm(self.x, axis=1)
            # print("r {}".format(r))

        if "filter_in_box" in options.keys():
            if bool(options["filter_in_box"]):
                # print("filtering charges in sampling box")
                self.x, self.Q = filter_in_box(
                    x=self.x, Q=self.Q, center=self.center, dimensions=self.dimensions
                )

        assert "CPET_method" in options.keys(), "CPET_method must be specified"

        if (
            options["CPET_method"] == "volume" or options["CPET_method"] == "volume_ESP"
        ):
            N_cr = 2 * self.dimensions / self.step_size
            N_cr = [int(N_cr[0]),int(N_cr[1]),int(N_cr[2])]
            (self.mesh, self.transformation_matrix) = initialize_box_points_uniform(
                center=self.center,
                x=self.x_vec_pt,
                y=self.y_vec_pt,
                N_cr = N_cr,
                dimensions=self.dimensions,
                dtype=self.dtype,
                inclusive=True
            )

        # self.transformation_matrix and self.uniform_transformation_matrix are the same
        self.max_steps = round(2 * np.linalg.norm(self.dimensions) / self.step_size)
        #print("max steps: ", max_steps)
        if (
            options["initializer"] == "random"
        ):            
                (
                    self.random_start_points,
                    self.random_max_samples,
                    self.transformation_matrix,
                    self.max_streamline_len,
                ) = initialize_box_points_random(
                    self.center,
                    self.x_vec_pt,
                    self.y_vec_pt,
                    self.dimensions,
                    self.n_samples,
                    dtype=self.dtype,
                    max_steps=self.max_steps,
                )
        elif (options["CPET_method"] != "volume" and options["CPET_method"] != "volume_ESP") and options["initializer"] == "uniform":
            num_per_dim = round(self.n_samples ** (1 / 3))
            if num_per_dim**3 < self.n_samples:
                num_per_dim += 1
            self.n_samples = num_per_dim**3
            #print("num_per_dim: ", num_per_dim)
            grid_density = 2 * self.dimensions / (num_per_dim + 1)
            print("grid_density: ", grid_density)
            seed=None
            if self.max_streamline_init == "fixed_rand":
                print("Fixing max steps with Random seed 42")
                seed=42
            (
                self.random_start_points, 
                self.random_max_samples, 
                self.transformation_matrix,
            ) = initialize_box_points_uniform(
                center=self.center,
                x=self.x_vec_pt,
                y=self.y_vec_pt,
                dimensions=self.dimensions,
                N_cr = [num_per_dim, num_per_dim, num_per_dim],
                dtype=self.dtype,
                max_steps=self.max_steps, 
                ret_rand_max=True, 
                inclusive=False,
                seed=seed
            ) 
            # convert mesh to list of x, y, z points
            #print(self.random_start_points)
            self.random_start_points = self.random_start_points.reshape(-1, 3)
            self.n_samples = len(self.random_start_points)
            #print("random start points")
            #print(self.random_start_points)
            print("start point shape: ", str(self.random_start_points.shape))

        self.x = (self.x - self.center) @ np.linalg.inv(self.transformation_matrix)

        if self.box_shift != [0,0,0]:
            print("Shifting box by: ", self.box_shift)
            self.x = self.x - np.array(self.box_shift)

        if self.dtype == "float32":
            self.x = self.x.astype(np.float32)
            self.Q = self.Q.astype(np.float32)
            self.center = self.center.astype(np.float32)
            self.dimensions = self.dimensions.astype(np.float32)

        print("... > Initialized Calculator!")

    def compute_topo_base(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        # print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            result = pool.starmap(task_base, args)
            # print(raw)
            hist = []
            for result in result.get():
                hist.append(result)
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_topo(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        # print("starting pooling")
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            # raw = pool.starmap(task, args)

            result = pool.starmap_async(task, args)
            dist = []
            curve = []
            for result in result.get():
                dist.append(result[0])
                curve.append(result[1])

            hist = np.column_stack((dist, curve))
        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_topo_single(self):
        print("... > Computing Topo(baseline)!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        dist_list, curve_list, init_points_list, final_points_list = [], [], [], []
        endtype_list = []
        for i, n_iter in zip(self.random_start_points, self.random_max_samples):
            
            dist, curve, init_points, final_points, endtype = task_base(i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
            dist_list.append(dist)
            curve_list.append(curve)
            init_points_list.append(init_points)
            final_points_list.append(final_points)
            endtype_list.append(endtype)
            #print(dist, curve)
        hist = np.column_stack((dist_list, curve_list))
        end_time = time.time()
        self.hist = hist
        init_points_list = np.array(init_points_list) #Shape (N, 3, 3)
        final_points_list = np.array(final_points_list) #Shape (N, 3, 3)
        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        #For testing purposes
        np.savetxt(
            "topology_base.txt",
            hist,
            fmt="%.6f",
        )

        np.savetxt(
            "dumped_values_init_base.txt",
            init_points_list.reshape(init_points_list.shape[0], -1),
            fmt="%.6f",
        )
        np.savetxt(
            "dumped_values_final_base.txt",
            final_points_list.reshape(final_points_list.shape[0], -1),
            fmt="%.6f",
        )

        np.savetxt(
            "endtype_base.txt",
            endtype_list,
            fmt="%s",
        )
        return hist

    def compute_topo_complete_c_shared(self):
        print("... > Computing Topo!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        print(f"Start point shape: {self.random_start_points.shape}")
        start_time = time.time()
        # print("starting pooling")
        #print("random start points")
        #print(self.random_max_samples)
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(self.random_start_points, self.random_max_samples)
            ]
            # raw = pool.starmap(task, args)

            result = pool.starmap_async(task_complete_thread, args)
            dist = []
            curve = []
            for result in result.get():
                dist.append(result[0])
                curve.append(result[1])

            hist = np.column_stack((dist, curve))

        end_time = time.time()
        self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_topo_batched(self):
        print("... > Computing Topo in Batches!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")
        start_time = time.time()
        print("num batches: {}".format(len(self.random_start_points_batched)))
        # print(self.random_start_points_batched)
        # print(self.random_max_samples_batched)
        with Pool(self.concur_slip) as pool:
            args = [
                (i, n_iter, self.x, self.Q, self.step_size, self.dimensions)
                for i, n_iter in zip(
                    self.random_start_points_batched, self.random_max_samples_batched
                )
            ]
            raw = pool.starmap(task_batch, args)
            raw_flat = [item for sublist in raw for item in sublist]
            dist = []
            curve = []
            for result in raw_flat:
                dist.append(result[0])
                curve.append(result[1])
            hist = np.column_stack((dist, curve)) 
            

        end_time = time.time()
        # self.hist = hist

        print(
            f"Time taken for {self.n_samples} calculations with N_charges = {len(self.Q)}: {end_time - start_time:.2f} seconds"
        )
        return hist

    def compute_box(self):
        print("... > Computing Box!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        field_box = compute_field_on_grid(self.mesh, self.x, self.Q)
        return field_box, self.mesh.shape

    def compute_box_ESP(self):
        print("... > Computing Box!")
        print(f"Number of charges: {len(self.Q)}")
        print("mesh shape: {}".format(self.mesh.shape))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("Transformation matrix: {}".format(self.transformation_matrix))
        print("Center: {}".format(self.center))
        field_box = compute_ESP_on_grid(self.mesh, self.x, self.Q)
        return field_box

    def compute_point_mag(self):
        print("... > Computing Point Magnitude!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        start_time = time.time()
        #Since x and Q are already rotated and translated, need to supply 0 vector as center
        point_mag = np.norm(
            calculate_electric_field_dev_c_shared(np.array([0,0,0]), self.x, self.Q)
        )
        end_time = time.time()
        print(f"{end_time - start_time:.2f}")
        return point_mag

    def compute_point_field(self):
        print("... > Computing Point Field!")
        print(f"Number of charges: {len(self.Q)}")
        print("point: {}".format(self.center))
        print("x shape: {}".format(self.x.shape))
        print("Q shape: {}".format(self.Q.shape))
        print("First few lines of x: {}".format(self.x[:5]))
        start_time = time.time()
        #Since x and Q are already rotated and translated, need to supply 0 vector as center
        point_field = calculate_electric_field_dev_c_shared(np.array([0,0,0]), self.x, self.Q)
        end_time = time.time()
        print(f"{end_time - start_time}")
        return point_field

    def compute_topo_GPU_batch_filter(self):
        print("... > Computing Topo in Batches on a GPU!")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of charges: {len(self.Q)}")
        print(f"Step size: {self.step_size}")

        Q_gpu = torch.tensor(self.Q, dtype=torch.float32).cuda()
        Q_gpu = Q_gpu.unsqueeze(0)
        x_gpu = torch.tensor(self.x, dtype=torch.float32).cuda()
        dim_gpu = torch.tensor(self.dimensions, dtype=torch.float32).cuda()
        step_size_gpu = torch.tensor([self.step_size], dtype=torch.float32).cuda()
        random_max_samples = torch.tensor(self.random_max_samples).cuda()

        M = self.max_steps
        max_num_batch = int(
            (M + 2 - self.GPU_batch_freq)/(self.GPU_batch_freq - 2)
        ) + 1
        remainder = (M + 2 - self.GPU_batch_freq) % (self.GPU_batch_freq - 2) #Number of remaining propagations
        path_matrix_torch = torch.tensor(np.zeros((self.GPU_batch_freq, self.n_samples, 3)), dtype=torch.float32).cuda()
        path_matrix_torch[0] = torch.tensor(self.random_start_points)
        path_matrix_torch = propagate_topo_matrix_gpu(
            path_matrix_torch,
            torch.tensor([0]).cuda(),
            x_gpu,
            Q_gpu,
            step_size_gpu,
        )
        #Using random_max_samples-1 to convert from max samples to indices
        path_filter = generate_path_filter_gpu(
        random_max_samples, torch.tensor([M + 2], dtype=torch.int64).cuda()
        )

        #Need to augment random_max_samples for smaller streamlines than the batching frequency
        if M+2 < self.GPU_batch_freq:
            path_filter_temp = torch.ones((self.GPU_batch_freq, self.n_samples,1), dtype=torch.bool).cuda()
            path_filter_temp[0:M+2] = path_filter
            path_filter = path_filter_temp

        path_filter = torch.tensor(path_filter, dtype=torch.bool).cuda()
        print(path_matrix_torch.shape)
        print(path_filter.shape)
        print(M+2)

        dumped_values = torch.tensor(
            np.empty((6, 0, 3)), dtype=torch.float32
        ).cuda()

        j = 0
        start_time = time.time()
        for i in range(max_num_batch):
            
            for j in range(self.GPU_batch_freq - 2):
                path_matrix_torch = propagate_topo_matrix_gpu(
                    path_matrix_torch,
                    torch.tensor([j + 1]).cuda(),
                    x_gpu,
                    Q_gpu,
                    step_size_gpu,
                )
                if i == 0 and j == 0:
                    init_points = path_matrix_torch[0:3, ...]
            
            #print("filtering!")
            (
                path_matrix_torch,
                dumped_values,
                path_filter,
                init_points,
            ) = batched_filter_gpu(
                path_matrix=path_matrix_torch,
                dumped_values=dumped_values,
                i=i,
                dimensions=dim_gpu,
                path_filter=path_filter,
                init_points=init_points,
                GPU_batch_freq=self.GPU_batch_freq,
                dtype_str=self.dtype,
            )
            print(dumped_values.shape[1])
            if dumped_values.shape[1] >= self.n_samples:
                print("Finished streamlines early, breaking!")
                break
        print(f"Checking dumped values ({dumped_values.shape[1]}) vs number of samples ({self.n_samples})")
        if (
            dumped_values.shape[1] < self.n_samples
        ):  # Still some samples remaining in the remainder
            print("Streamlines remaining")
            print(remainder)
            print(path_matrix_torch.shape)
            path_matrix_torch_new = torch.zeros(
                (remainder + 2, path_matrix_torch.shape[1], 3),
                dtype=torch.float32,
            ).cuda()
            path_matrix_torch_new[0:2, ...] = path_matrix_torch[-2:, ...]
            del path_matrix_torch
            # For remainder
            for i in range(remainder - 1):
                path_matrix_torch_new = propagate_topo_matrix_gpu(
                    path_matrix_torch_new,
                    torch.tensor([i + 2]).cuda(),
                    x_gpu,
                    Q_gpu,
                    step_size_gpu,
                )
            (
                path_matrix_torch_new,
                dumped_values,
                path_filter,
                init_points,
            ) = batched_filter_gpu(
                path_matrix=path_matrix_torch_new,
                dumped_values=dumped_values,
                i=i,
                dimensions=dim_gpu,
                path_filter=path_filter,
                init_points=init_points,
                GPU_batch_freq=remainder,
                dtype_str=self.dtype,
            )
            # print(path_matrix_torch_new, dumped_values, path_filter)
        else:
            del path_matrix_torch

        print(dumped_values.shape)
        np.savetxt(
            "dumped_values_init.txt",
            dumped_values[0:3]
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .reshape(dumped_values.shape[1], -1),
            fmt="%.6f",
        )
        np.savetxt(
            "dumped_values_final.txt",
            dumped_values[3:6]
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .reshape(dumped_values.shape[1], -1),
            fmt="%.6f",
        )
        distances, curvatures = compute_curv_and_dist_mat_gpu(
            dumped_values[0, :, :],
            dumped_values[1, :, :],
            dumped_values[2, :, :],
            dumped_values[3, :, :],
            dumped_values[4, :, :],
            dumped_values[5, :, :],
        )
        end_time = time.time()
        print(
            f"Time taken for {self.n_samples} calculations with N~{self.Q.shape}: {end_time - start_time:.2f} seconds"
        )
        topology = np.column_stack(
            (distances.cpu().numpy(), curvatures.cpu().numpy())
        )
        print(topology.shape)
        #For dev testing
        np.savetxt(
            "topology_GPU.txt",
            topology,
            fmt="%.6f",
        )
        return topology
