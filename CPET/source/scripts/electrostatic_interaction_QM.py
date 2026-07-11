from CPET.source.calculator import calculator

import argparse
import json
import numpy as np
from tqdm import tqdm
from pyscf import gto, tools
from pathlib import Path


def E_elec_qmmm(mol, x, q, dm, verbose, track=False):
    E_elec = 0.0
    if not track:
        for i in range(len(x)):
            mol.set_rinv_origin(x[i])
            V = (-1) * mol.intor("int1e_rinv")
            E_elec += q[i] * np.einsum("ij,ij", dm, V)
    else:
        for i in tqdm(range(len(x))):
            mol.set_rinv_origin(x[i])
            V = (-1) * mol.intor("int1e_rinv")
            E_elec += q[i] * np.einsum("ij,ij", dm, V)
    if verbose:
        print(E_elec)
    return E_elec


def E_nuc_qmmm(x, x_qm, q, q_qm, verbose, track=False):
    E_nuc = 0.0
    if not track:
        for i in range(len(x)):
            for j in range(len(x_qm)):
                r = np.linalg.norm(
                    x_qm[j] - x[i]
                )  # Take care of cases where nuclei overlap in case filtering failed
                if r < 1e-4:
                    print(
                        f"Warning: Overlapping nuclei detected at QM index {i} and PDB index {j}. Skipping this interaction, make sure your filtering is set up correctly"
                    )
                    continue
                E_nuc += q[i] * q_qm[j] / r
    else:
        for i in range(len(x)):
            for j in range(len(x_qm)):
                r = np.linalg.norm(
                    x_qm[j] - x[i]
                )  # Take care of cases where nuclei overlap in case filtering failed
                if r < 1e-4:
                    print(
                        f"Warning: Overlapping nuclei detected at QM index {i} and PDB index {j}. Skipping this interaction, make sure your filtering is set up correctly"
                    )
                    continue
                E_nuc += q[i] * q_qm[j] / r
    if verbose:
        print(E_nuc)
    return E_nuc


def density_matrix(mo_coeff, mo_occ):
    if isinstance(mo_coeff, np.ndarray):  # closed shell
        dm = (mo_coeff * np.sqrt(mo_occ)) @ (mo_coeff * np.sqrt(mo_occ)).T
    elif isinstance(mo_coeff, (tuple, list)) and len(mo_coeff) == 2:  # open shell
        Ca, Cb = mo_coeff  # alpha and beta coeffs
        occ_a, occ_b = mo_occ
        dm_a = (Ca * np.sqrt(occ_a)) @ (Ca * np.sqrt(occ_a)).T
        dm_b = (Cb * np.sqrt(occ_b)) @ (Cb * np.sqrt(occ_b)).T
        dm = dm_a + dm_b  # total density
    else:
        raise ValueError("unexpected mo_coeff format")

    return dm


def main():
    parser = argparse.ArgumentParser(
        description="Electrostatic Interaction Analysis of Charge Density with Surrouding Point Charges"
    )
    parser.add_argument("-m", "--molden", help="Molden Input File", required=True)
    parser.add_argument("-p", "--pdb", help="PDB file with charges", required=True)
    parser.add_argument(
        "-r",
        "--res",
        help="Flag for residue breakdown. If provided, analysis will be done by residue as well",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--options",
        help="Path to options file (mainly for filtering purposes).",
        required=True,
    )
    parser.add_argument(
        "-v", "--verbose", help="Verbose output", action="store_true", default=False
    )
    parser.add_argument("-s", "--spin", help="Provide the total S", default=0.5)
    parser.add_argument(
        "-c",
        "--charge",
        help="Provide the total charge of the QM region as an integer",
        default=0,
    )
    ANGSTROM_TO_BOHR = 1.8897259886

    args = parser.parse_args()
    molden = args.molden
    res = args.res
    verbose = args.verbose
    options = json.load(open(args.options, "r"))
    options["dtype"] = (
        "float64"  # Ensure that the dtype is set to float64 for consistency
    )
    options["CPET_method"] = (
        "point_field"  # Preventative to ensure that time isn't wasted on creating other attributes
    )
    options["center"] = [0, 0, 0]
    calculator_object = calculator(
        options, args.pdb
    )  # Creates object, filters, etc. based on options file
    x = calculator_object.x * ANGSTROM_TO_BOHR  # Convert Angstrom to Bohr
    q = calculator_object.Q  # Charges in e
    print(x.dtype, q.dtype)  # Print data types for debugging
    atom_number = calculator_object.atom_number  # Atomic numbers
    atom_type = calculator_object.atom_type  # Atomic types
    resids = calculator_object.resids  # Residue names
    residue_number = calculator_object.residue_number  # Residue numbers

    print("Parsing Molden file...")
    parsed_molden = tools.molden.parse(molden, verbose=1)
    mol, mo_energy, mo_coeff, mo_occ = parsed_molden[0:4]
    x_qm = mol.atom_coords()  # Get coordinates of atoms in Bohr
    q_qm = mol.atom_charges()  # Get atomic numbers to get nuclear charges
    dm = density_matrix(mo_coeff, mo_occ)  # Density matrix in AO basis
    print(f"Molden file parsed successfully. MO coefficients shape (AOxMO):")

    print(x[0:5], q[0:5])  # Print first 5 coordinates and charges for debugging
    print(
        x_qm[0:5], q_qm[0:5]
    )  # Print first 5 QM coordinates and charges for debugging
    print(len(x), len(q))
    print(f"Number of unpaired electrons: {int(float(args.spin)*2)}")
    mol.spin = int(float(args.spin) * 2)
    mol.charge = int(args.charge)
    # Create a PySCF molecule object
    mol.build()
    print("Molecule built successfully.")

    if res:
        print(
            "Residue breakdown enabled. Analysis will be done by residue and saved to file residue_breakdown.txt"
        )
        res_breakdown_dict = {}

        # Loop to build dictionary of residues
        print("Building residue breakdown dictionary...")
        previous_resn = None
        current_key = None
        duplicate_count = {}
        for i in tqdm(range(len(x))):
            resn = residue_number[i]
            if resn not in duplicate_count:
                duplicate_count[resn] = 0
                current_key = resn
            else:
                duplicate_count[resn] += 1
                current_key = f"{resn}_{duplicate_count[resn]}"
                
                """
                x captures the coordinates of atoms in the residue.
                count keeps track of the number of atoms in the residue.
                E_elec and E_nuc are initialized to 0.0, they will be calculated later.
                """
            res_breakdown_dict[current_key] = {
                    "E_elec": 0.0,
                    "E_nuc": 0.0,
                    "V_qmmm": 0.0,
                    "count": 0,
                    "x": [],
                    "q": [],
            }
            res_breakdown_dict[current_key]["x"].append(x[i])
            res_breakdown_dict[current_key]["q"].append(
                q[i]
            )  # Store charge for the residue
            res_breakdown_dict[current_key]["count"] += 1

        # Loop to calculate interaction energies for each residue
        for resn, data in tqdm(res_breakdown_dict.items()):
            x_temp = np.array(data["x"])  # Coordinates of atoms in the residue
            q_temp = np.array(data["q"])
            count = data["count"]
            if count == 0:
                raise ValueError(
                    f"No atoms found for residue {resn}. Please check your PDB file and options."
                )
            # Interaction energy of PC with charge density
            if verbose:
                print(
                    f"Calculating interaction energy for residue {resn} with {count} atoms..."
                )
            res_breakdown_dict[resn]["E_elec"] = E_elec_qmmm(
                mol, x_temp, q_temp, dm, verbose
            )
            res_breakdown_dict[resn]["E_nuc"] = E_nuc_qmmm(
                x_temp, x_qm, q_temp, q_qm, verbose
            )
            res_breakdown_dict[resn]["V_qmmm"] = (
                res_breakdown_dict[resn]["E_elec"] + res_breakdown_dict[resn]["E_nuc"]
            )

        # Write the breakdown to a file from the dictionary, sorted by residue number
        print("Writing residue breakdown to file residue_breakdown.txt")
        with open("residue_breakdown.txt", "w") as f:
            f.write("Residue\tE_elec\tE_nuc\tV_qmmm\tCount\n")
            for resn in sorted(res_breakdown_dict.keys(), key=lambda x: int(x)):
                data = res_breakdown_dict[resn]
                f.write(
                    f"{resn}\t{float(data['E_elec']):.8f}\t{float(data['E_nuc']):.8f}\t{float(data['V_qmmm']):.8f}\t{data['count']}\n"
                )
        print(
            f"Writing complete. Overall interaction energy V_qmmm: {sum(data['V_qmmm'] for data in res_breakdown_dict.values())} Hartree"
        )

    else:
        print(
            "Residue breakdown disabled. Analysis will be done for the entire molecule."
        )
        # Interaction energy of PC with charge density
        print("E_elec")
        # Print floating point precision of the variables
        print(f"x: {x.dtype}, q: {q.dtype}")

        E_elec = E_elec_qmmm(mol, x, q, dm, verbose, track=True)
        # Interaction energy of PC with nuclear charges
        print("E_nuc")
        E_nuc = E_nuc_qmmm(x, x_qm, q, q_qm, verbose, track=True)
        V_qmmm = E_elec + E_nuc
        print(f"Total interaction energy V_qmmm: {V_qmmm} Hartree")


if __name__ == "__main__":
    main()
