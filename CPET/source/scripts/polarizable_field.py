from CPET.source.CPET import CPET
from CPET.utils import intro_citation
from CPET.source.calculator import calculator
from CPET.utils.calculator import compute_field_on_grid_amoeba
from CPET.utils.io import save_numpy_as_dat

import shutil
import os
import argparse
import time
import numpy as np
import json

"""
This script is for polarizable force field calculations of electric field.

Current status: Testing phase
"""

def parse_coordinates(coordinate_filepath, analyze_tinker_outfile, induced_dipole_path, filter_idxs):
    """
    Parses the coordinate and parameter files to extract the coordinates, charges, dipoles, and quadrupoles
    Takes:
        coordinate_filepath: Path to the coordinate file (includes connectivity information)
        analyze_tinker_outfile: Path to the TINKER analyze output file that contains permanent multipole info for each atom
        induced_dipole_path: Path to the induced dipole file (includes induced dipoles)
        filter_idxs: List of atom indices to filter out, determined from pycpet filtering object
    Returns:
        x (np.ndarray): coordinates of the atoms of shape (N, 3)
        q (np.ndarray): charges of the atoms of shape (N,)
        d (np.ndarray): dipole moments of the polarizable atoms of shape (N, 3)
        t (np.ndarray): quadrupole moments of the polarizable atoms of shape (N, 6) (using the 6 unique components of the quadrupole tensor)
    """

    # First, parse coordinate file
    x = [] #xyz file
    with open(coordinate_filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split()
        print(values)
        try:
            float(values[2])
            x.append([float(values[2]), float(values[3]), float(values[4])])
        except:
            continue
    x = np.array(x)
    # Next, parse TINKER output for charges, dipoles, and multipoles
    q = [] #parameter file
    d = [] #parameter file
    t = [] #parameter file

    """
    Parameter section formatting example:
     1        1       2     10          Z-then-X    0.07219
                                                    0.28913  0.00000  0.29472
                                                    0.20132
                                                    0.00000 -0.54286
                                                    0.05155  0.00000  0.34154

    In first row, 3 or 4 indices may be present after the atom number
    """
    read_index = 0
    start_reading = False
    temp_quad = []
    with open(analyze_tinker_outfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # Start reading 5 lines after the line that contains "Atomic Multipole Parameters". Multipole parameters are in chunks of 5
        if line.startswith(" Atomic Multipole Parameters"):
            start_reading = True
            print("Found line! Continuing to read")
            continue
        if len(line.strip())==0:# Line is empty
            continue
        if "Multipole Moments" in line:
            continue
        if start_reading:
            if "Dipole Polarizability Parameters" in line: #Stop reading when reached the next section
                break
            values = line.split()
            if len(values) == 0: #Skip blank lines
                continue
            if read_index == 0:
                q.append(float(values[-1])) #Charge is always the last value in the first line
                read_index += 1
            elif read_index == 1:
                d.append([float(x) for x in values]) #Dipole values are in the second line
                read_index += 1
            elif read_index == 2:
                temp_quad.extend([float(x) for x in values]) #Quadrupole values are in the next 4 lines, but we will rearrange them later
                read_index += 1
            elif read_index == 3:
                temp_quad.extend([float(x) for x in values])
                read_index += 1
            elif read_index == 4:
                temp_quad.extend([float(x) for x in values])
                # Build quadrupole tensor in the form [[Qxx, Qxy, Qxz], [Qxy, Qyy, Qyz], [Qxz, Qyz, Qzz]]
                quadrupole_array = np.array([[temp_quad[0], temp_quad[1], temp_quad[3]],
                                            [temp_quad[1], temp_quad[2], temp_quad[4]],
                                            [temp_quad[3], temp_quad[4], temp_quad[5]]])
                t.append(quadrupole_array)
                temp_quad = []
                read_index = 0
            
    q = np.array(q) #e
    d = np.array(d) #e * bohr
    t = np.array(t) #e * bohr^2
    
    BOHR_TO_ANG = 0.52917721067121

    d = d * BOHR_TO_ANG
    t = t * BOHR_TO_ANG**2


    if q.shape[0] == 0 or d.shape[0] == 0 or t.shape[0] == 0 or x.shape[0] == 0:
        raise ValueError("Failed to parse coordinates, charges, dipoles, or quadrupoles (lengths: {}, {}, {}, {}). Please check the formatting of the input files.".format(x.shape[0], q.shape[0], d.shape[0], t.shape[0]))

    if not (x.shape[0] == q.shape[0] == d.shape[0] == t.shape[0]):
        raise ValueError(f"Number of atoms in coordinate file ({x.shape[0]}) does not match number of entries in parameter file ({q.shape[0]}, {d.shape[0]}, {t.shape[0]})")

    # Finally, parse induced dipole file and add to dipole moments
    ind_dips = []
    if not induced_dipole_path.endswith(".uind"):
        raise ValueError(f"Expected .uind file for induced dipoles, but got {induced_dipole_path}")
    with open(induced_dipole_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if i < 2: #Skip header lines
            continue
        values = lines[i].split()
        if len(values) < 5: #Expect atom number, type, mux, muy, muz
            raise ValueError(f"Expected at least 5 columns in induced dipole file, but got {len(values)} in line: {lines[i]}")
        try:
            ind_dip = lines[i].split()[2:5]
            ind_dip = [float(x) for x in ind_dip] #Convert to float
            ind_dips.append(ind_dip)
        except:
            raise ValueError(f"Error parsing induced dipole values in line: {lines[i]}")
    ind_dips = np.array(ind_dips, dtype=np.float64)
    if ind_dips.shape[0] != d.shape[0]:
        raise ValueError(f"Number of induced dipoles ({ind_dips.shape[0]}) does not match number of dipole entries from parameter file ({d.shape[0]})")
    if not (x.shape[0] == q.shape[0] == d.shape[0] == t.shape[0]):
        raise ValueError(f"Number of atoms in coordinate file ({x.shape[0]}) does not match number of entries in parameter file ({q.shape[0]}, {d.shape[0]}, {t.shape[0]})")

    DEBYE_TO_ATOMIC = 0.3934303
    ind_dips = ind_dips * DEBYE_TO_ATOMIC * BOHR_TO_ANG

    x = np.delete(x, filter_idxs, axis=0)
    q = np.delete(q, filter_idxs, axis=0)
    d = np.delete(d, filter_idxs, axis=0)
    t = np.delete(t, filter_idxs, axis=0)
    ind_dips = np.delete(ind_dips, filter_idxs, axis=0)
    
    return x, q, d, t, ind_dips


def parse_rotation_matrix(analyze_tinker_outfile, n_atoms, filter_idxs):
    """
    Parses the rotation matrix file (from energy evaluation in TINKER)
    Takes:
        analyze_tinker_outfile: Path to the TINKER analyze output file
        n_atoms (int): Number of atoms
    Returns:
        rot_mats: Rotation matrices of each atom of shape (N, 3, 3)
    """

    rot_mats = []
    start_reading = False
    with open(analyze_tinker_outfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Atom Frame zaxis xaxis yaxis"):
                start_reading = True
                print("Found line for rotation matrices! Continuing to read")
                continue
            if start_reading:
                values = line.split()[6:15]
                rot_mat = np.array(values, dtype=float).reshape(3, 3)
                rot_mats.append(rot_mat)
    if filter_idxs == None:
        rot_mats_filter = rot_mats
    elif len(filter_idxs) == 1:
        rot_mats_filter = [v for i, v in enumerate(rot_mats) if i not in filter_idxs[0]]
    else:
        raise ValueError(f"Filtering indices not supported this way (e.g. radius and atom-based). Please edit options file")
    if len(rot_mats_filter) != n_atoms:
        raise ValueError(f"Expected {n_atoms} rotation matrices, but found {len(rot_mats)} in {analyze_tinker_outfile}")
    return rot_mats_filter


def rotate_dipoles_quadrupoles(r, d, t):
    """
    Rotates dipoles and quadrupoles to align with the field
    Takes:
        r: rotation matrix of each atom of shape (N, 3, 3)
        d: dipole moments of the polarizable atoms of shape (N, 3)
        t: quadrupole moments of the polarizable atoms of shape (N, 3, 3)
    Returns:
        d: rotated dipole moments
        t: rotated quadrupole moments
    """
    print("ROTATING DIPOLE AND QUADRUPOLE TO GLOBAL FRAME")
    print(d.shape, t.shape, r)
    d = np.einsum('nij,nj->ni', r, d)
    t = np.einsum('nik,nkl,njl->nij', r, t, r)

    return d, t


def rotate_and_translate_to_box_reference(x, d, t, center, rot):
    """
    Rotates and translatescoordinates, dipoles, and quadrupoles to align with the box reference frame
    Takes:
    x: coordinates of the atoms of shape (N, 3)
    d: dipole moments of the polarizable atoms of shape (N, 3)
    t: quadrupole moments of the polarizable atoms of shape (N, 3
    center: center of the box for translation
    rot: rotation matrix for rotation of shape (3,3)
    Returns:
    x: rotated and translated coordinates of the atoms of shape (N, 3)
    d: rotated and translated dipole moments of the polarizable atoms of shape (N, 3)
    t: rotated and translated quadrupole moments of the polarizable atoms of shape (N, 3, 3)
    """
    print("ROTATING AND TRANSLATING TO BOX")
    print(x.shape, d.shape, t.shape, rot.shape)
    x = (x - center) @ np.linalg.inv(rot)
    Rinv = np.linalg.inv(rot)
    Rinv = np.broadcast_to(Rinv, (x.shape[0], 3, 3))
    d = np.einsum('nij,nj->ni', Rinv, d)
    t = np.einsum('nik,nkl,njl->nij', Rinv, t, Rinv)
    return x, d, t


def tinker_energy_eval(parameter_path, coordinate_path, analyze_tinker_outfile):
    """
    Runs TINKER analyze.x to obtain an output file that contains permanent multipole and rotation matrices for all atoms
    Takes:
        parameter_path: File that ends in ".prm" that contains AMOEBA FF parameters
        coordinate_path: File that ends in ".xyz" that contains atomic coordinates
        analyze_tinker_outfile: Output file from tinker
    """

    # Verify that analyze.x is in the user's path
    if not shutil.which("analyze.x"):
        raise EnvironmentError("TINKER analyze.x executable not found in PATH. Please ensure TINKER is installed and analyze.x is accessible.")
    
    # Verify that input files are of the right type
    if not parameter_path.endswith(".prm"):
        raise ValueError(f"Expected .prm file for parameters, but got {parameter_path}")
    if not coordinate_path.endswith(".xyz"):
        raise ValueError(f"Expected .xyz file for coordinates, but got {coordinate_path}")
    
    # Construct the TINKER command
    command = f"analyze.x {coordinate_path} {parameter_path} E > {analyze_tinker_outfile}"
    os.system(command)
    print(f"Finished TINKER energy evaluation. Output saved to {analyze_tinker_outfile}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Calculate electric field on an oriented grid from a AMOEBA force field")
    parser.add_argument("-n","--name", type=str, help="Name for the system, used for saving output files", required=True)
    parser.add_argument("-o", "--options", type=str, help="Options for CPET", default="./options/options.json")
    parser.add_argument("--xyz", type=str, help="Path to the .xyz coordinate file", required=True)
    parser.add_argument("--prm", type=str, help="Path to the .prm parameter file", required=True)
    parser.add_argument("--uind", type=str, help="Path to the .uind induced dipole file", required=True)
    parser.add_argument("--pdb", type=str, help="Path to the .pdb file used for parsing and zeroing point charges", required=True)
    parser.add_argument("--no-permanent", help="Specify whether permanent dipoles + quadrupoles should be considered", action="store_true")
    parser.add_argument("--no-induced", help="Specify whether induced dipoles should be considered", action="store_true")
    args = parser.parse_args()

    """
    File input section
    """
    name = args.name
    options = json.load(open(args.options))
    outputpath = options["outputpath"]
    parameter_path = args.prm #Includes permanent charges, dipoles, and quadrupoles
    analyze_tinker_outfile = f"{outputpath}/{name}.out" #From energy evaluation in TINKER
    coordinate_path = args.xyz #Includes connectivity information
    induced_dipole_path = args.uind #Includes induced dipoles
    path_to_pdb = args.pdb #Used for parsing and zeroing point charges as well as defining local molecule reference frame, but not for coordinates/charge/dipole/quadrupole values
    field_output_path = os.path.join(options["outputpath"], coordinate_path.split("/")[-1].split(".")[0] + "_efield.dat")
    no_permanent = args.no_permanent
    no_induced = args.no_induced
    if no_permanent and not no_induced:
        ValueError("Can not turn off permanent dipoles/quadrupoles and keep induced dipoles on!")
    """
    Pycpet initialization section, need grid, box dimension info, x indices, and rotation vectors
    """
    calculator_object = calculator(options, path_to_pdb)
    global2local_rotmat = calculator_object.transformation_matrix
    local_mesh = calculator_object.mesh
    dimensions = calculator_object.dimensions
    step_size = calculator_object.step_size
    center = calculator_object.center
    filter_idxs = None
    filter_idxs = calculator_object.filter_idxs
    print(filter_idxs)
    """
    Calculation section
    """
    if not os.path.exists(analyze_tinker_outfile):
        tinker_energy_eval(parameter_path, coordinate_path, analyze_tinker_outfile)
    else:
        print(f"TINKER output file {analyze_tinker_outfile} already exists. Skipping energy evaluation.")
    x, q, d, t, ind_dips = parse_coordinates(coordinate_path, analyze_tinker_outfile, induced_dipole_path, filter_idxs)
    print(x.shape, q.shape, d.shape, t.shape)
    print(x[:5], q[:5], d[:5], t[:5])
    r = parse_rotation_matrix(analyze_tinker_outfile, n_atoms=len(x), filter_idxs=filter_idxs)
    print(len(r))
    print(r[:5])
    d, t = rotate_dipoles_quadrupoles(r, d, t)
    if not no_induced:
        d += ind_dips
    print(d.shape, t.shape)
    print(d[:5], t[:5])
    x, d, t = rotate_and_translate_to_box_reference(x, d, t, center, global2local_rotmat)
    
    # Flatten t so that each t only has Qxx Qxy Qxz Qyy Qyz Qzz (right now it has shape Nx3x3)
    t_flat = np.zeros((t.shape[0], 6), dtype=np.float32)
    t_flat[:, 0] = t[:, 0, 0] #Qxx
    t_flat[:, 1] = t[:, 0, 1] #Qxy
    t_flat[:, 2] = t[:, 0, 2] #Qxz
    t_flat[:, 3] = t[:, 1, 1] #Qyy
    t_flat[:, 4] = t[:, 1, 2] #Qyz
    t_flat[:, 5] = t[:, 2, 2] #Qzz
    print("T flattened")
    print(local_mesh.shape, x.shape, q.shape, d.shape, t_flat.shape)
    override_points = False # ALWAYS LEAVE AS FALSE UNLESS YOU ARE RUNNING SPECIAL TESTS
    if override_points == True:
        #x_0 is a grid from -1 to 1 in each dimension with N points in each dimension
        N = 10 #Total test points in each dimension
        x_0 = np.zeros((N**3, 3), dtype=np.float32)
        grid_points = np.linspace(-1, 1, N)
        index = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    x_0[index] = [grid_points[i], grid_points[j], grid_points[k]]
                    index += 1
        print("x_0 generated")
        #q and d will be 0 for all points
        x = np.array([[0,0,0]], dtype=np.float32) #Single point at origin
        q = np.array([0.0]) #Single charge (0)
        d = np.zeros_like(x) #Single dipole (0,0,0)
        #t will be a 1, 0, 0, -1, 0, 0
        t_flat = np.array([[0.5, 0.0, 0.0, -0.25, 0.0, -0.25]], dtype=np.float32) #Single quadrupole with Qxx=1, Qyy=-1, Qzz=0
        print("Test points overridden")
    if no_permanent:
        d = np.zeros_like(d)
        t_flat = np.zeros_like(t_flat)
    time_start = time.time()
    field = compute_field_on_grid_amoeba(local_mesh, x, q, d, t_flat)
    time_end = time.time()
    print(f"Time for grid calculation: {time_end - time_start} seconds. Time per grid point: {(time_end - time_start) / (local_mesh.shape[0]*local_mesh.shape[1]*local_mesh.shape[2])} seconds")
    print(field.shape)
    meta_data = {
        "dimensions": dimensions,
        "step_size": [step_size, step_size, step_size],
        "num_steps": [local_mesh.shape[0], local_mesh.shape[1], local_mesh.shape[2]],
        "transformation_matrix": global2local_rotmat,
        "center": center,
    }
    save_numpy_as_dat(
        name=field_output_path,
        volume=field,
        meta_data=meta_data,
    )

if __name__ == "__main__":
    main()
