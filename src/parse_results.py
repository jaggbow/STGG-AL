import os, sys
import glob
from pathlib import Path
import shutil
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse
au2ev = 27.2114

class RESULT:
    def __init__(self, file_name, norm_line=99999999):
        self.file_name = file_name
        self.file2list(norm_line=norm_line)

        self.dipole_s = None
        self.dipole = None
        self.tot_en_list = []
        self.tot_en = None
        self.tot_enev = None
        self.excited_en_list = []
        self.excited_en = None
        self.homo = None
        self.lumo = None
        self.S1 = None
        self.S1_f = None
        self.S1_en = None
        self.T1 = None
        self.T1_f = None
        self.T1_en = None
        self.spin = None
        self.aniso = None
        self.molvolume = None
        self.tran_dipole = None
        self.sn_list = []
        self.tn_list = []
        self.mol_weight = None

    def file2list(self, norm_line=99999999):
        i_norm = 0
        self.log_list = []
        self.key = True
        for line in open(self.file_name):
            line = line.strip()
            if line == '':
                continue
            self.log_list.append(line)
            if 'Error termination' in line:
                self.key = False
            if 'Normal termination' in line:
                i_norm += 1
                if i_norm >= norm_line:
                    break

    def get_dipole(self):
        dipole_s = []
        i_line = 999
        for line in self.log_list:
            if 'Dipole moment' in line:
                i_line = 0
                continue
            i_line += 1
            if i_line == 1:
                linesp = line.strip().split()
                dipole = float(linesp[-1])
                dipole_s.append(dipole)
        self.dipole_s = dipole_s
        self.dipole = dipole_s[-1]

    def get_tran_dipole(self):
        vec_s = []
        dipole_s = []
        flag = False
        for line in self.log_list:
            if 'transition electric dipole moments' in line:
                flag = True
                sub_s = []
                sub2_s = []
                continue
            if 'transition velocity dipole moments' in line:
                flag = False
                dipole_s.append(sub_s)
                vec_s.append(sub2_s)
                continue
            if 'state' in line:
                continue
            if not flag:
                continue
            linesp = line.strip().split()
            sub_s.append(float(linesp[4]))
            sub2_s.append([float(linesp[1]), float(linesp[2]), float(linesp[3])])
        if len(dipole_s) > 0:
            # sub_s = dipole_s[-1]
            sub_s = dipole_s[0]
            sub2_s = vec_s[0]
            if len(sub_s) > 0:
                self.tran_dipole = sub_s[0]
                self.vec = sub2_s[0]

    def get_toten(self):
        for line in self.log_list:
            if not 'SCF Done' in line:
                continue
            linesp = line.strip().split()
            tot_en = float(linesp[4])
            self.tot_en_list.append(tot_en)
        self.tot_en = self.tot_en_list[-1]
        self.tot_enev = self.tot_en*au2ev

    def get_exciteden(self):
        for line in self.log_list:
            if not 'Total Energy,' in line:
                continue
            linesp = line.strip().split()
            excited_en = float(linesp[4])
            self.excited_en_list.append(excited_en)
        if len(self.excited_en_list) > 0:
            self.excited_en = self.excited_en_list[-1]

    def get_homolumo(self):
        idx_homo = None
        for i_line, line in enumerate(self.log_list):
            if 'occ. eigenvalues' in line:
                idx_homo = i_line
        homo_line = self.log_list[idx_homo]
        lumo_line = self.log_list[idx_homo+1]
        linesp = homo_line.strip().split()
        self.homo = float(linesp[-1])
        linesp = lumo_line.strip().split()
        self.lumo = float(linesp[4])

    def get_excited(self):
        sn_sub = []
        tn_sub = []
        sn_list = []
        tn_list = []
        en_list = []
        flag = False
        is_first = True
        for line in self.log_list:
            if 'Excitation energies and oscillator strengths' in line:
                if is_first:
                    is_first = False
                else:
                    if len(sn_sub) > 0:
                        sn_list.append(sn_sub)
                    if len(tn_sub) > 0:
                        tn_list.append(tn_sub)
                sn_sub = []
                tn_sub = []
                continue
            if 'Excited State' in line:
                linesp = line.strip().split()
                sn = float(linesp[4])
                f = float(linesp[8].strip('f='))
                if 'Singlet' in line:
                    sn_sub.append((sn,f))
                elif 'Triplet' in line:
                    tn_sub.append((sn,f))
                flag = True
            if flag and 'Total Energy,' in line:
                linesp = line.strip().split()
                try:
                    en = float(linesp[4])
                except:
                    en = None
                en_list.append(en)
                flag = False

        if len(sn_sub) > 0:
            sn_list.append(sn_sub)
        if len(tn_sub) > 0:
            tn_list.append(tn_sub)

        return sn_list, tn_list, en_list

    def get_S1(self):
        sn_list, tn_list, en_list = self.get_excited()
        if len(sn_list) > 0:
            sn_sub = sn_list[-1]
            self.S1 = sn_sub[0][0]
            self.S1_f = sn_sub[0][1]
            self.sn_list = []
            for sn in sn_sub:
                self.sn_list.append(sn[0])
        if len(en_list) > 0:
            self.S1_en = en_list[0]

    def get_T1(self):
        sn_list, tn_list, en_list = self.get_excited()
        if len(tn_list) > 0:
            tn_sub = tn_list[-1]
            self.T1 = tn_sub[0][0]
            self.T1_f = tn_sub[0][1]
            self.tn_list = []
            for tn in tn_sub:
                self.tn_list.append(tn[0])
        if len(en_list) > 0:
            self.T1_en = en_list[0]

    def get_spin_density(self):
        flag = False
        for line in self.log_list:
            if 'Mulliken charges and spin densities:' in line:
                flag = True
                continue
            if not flag:
                continue
            linesp = line.strip().split()
            if linesp[1] in ['Ir', 'Pt', 'Pd', 'Os', 'Au']:
                self.spin = float(linesp[3])
                flag = False

    def get_anisotropy(self):
        for line in self.log_list:
            if not 'Rotational constants (GHZ):' in line:
                continue
            linesp = line.strip().split()
            x = float(linesp[3])
            y = float(linesp[4])
            z = float(linesp[5])
            max_len = max(x,y,z)
            min_len = min(x,y,z)
            self.x_axis = x
            self.y_axis = y
            self.z_axis = z
            self.aniso = min_len / max_len

    def get_molvolume(self):
        for line in self.log_list:
            if not 'Molar volume =' in line:
                continue
            linesp = line.strip().split()
            self.molvolume = float(linesp[3])
        return self.molvolume

    def get_radius(self):
        self.radius = None
        for line in self.log_list:
            if not 'Recommended a0 for SCRF calculation' in line:
                continue
            linesp = line.strip().split()
            self.radius = float(linesp[6])
        return self.radius

    def get_polar(self):
        self.polar = None
        for line in self.log_list:
            if not 'Isotropic polarizability for' in line:
                continue
            linesp = line.strip().split()
            self.polar = float(linesp[5])
        return self.polar

    def get_dielectric_const(self):
        self.dielec_const = None
        self.molvolume = self.get_molvolume()
        self.polar = self.get_radius()
        if self.molvolume == '' or self.polar == '':
            return self.dielec_const
        mid_val = 4.188790 * self.polar / self.molvolume #4.188790 is 4*pi/3
        self.dielec_const = (2.0*mid_val + 1.0) / (1.0 - mid_val)
        return self.dielec_const
    
    def get_elapsed_time(self):
        idx = None
        for i in range(len(self.log_list)):
            if "Elapsed" in self.log_list[i]:
                idx = i
                break
        if idx is not None:
            days, hours, minutes, seconds = map(lambda x: float(x), self.log_list[idx].split()[2::2])
            total_seconds = seconds+60*minutes+3600*hours+3600*24*days
            return total_seconds
        return None

def main(log_file):
    name = os.path.basename(log_file).replace('.log','')
    result = RESULT(log_file)
    result.get_toten()
    result.get_homolumo()
    result.get_S1()
    result.get_T1()
    result.homo = result.homo*au2ev
    result.lumo = result.lumo*au2ev
    elapsed_time = result.get_elapsed_time()
    if result.S1 is None or result.T1 is None:
        delta = None
    else:
        delta = result.S1-result.T1
    return name, result.S1, delta, result.homo, result.lumo, result.S1_f, elapsed_time

s1_slope = 0.9527960191042272
s1_intercept = -0.40221058
t1_slope = 0.889516986
t1_intercept = -0.03654902

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--gaussian_dir", type=str)
    parser.add_argument("--coordinates_dir", type=str)
    parser.add_argument("--generator_data_path", type=str)
    parser.add_argument("--property_predictor_data_path", type=str)
    args = parser.parse_args()
   
    csv_path = Path(args.csv_path)
    predictions_df = pd.read_csv(csv_path)
    gaussian_dir = Path(args.gaussian_dir)
    coordinates_dir = Path(args.coordinates_dir)

    predictions_df["log_path"] = predictions_df["id"].apply(lambda x: gaussian_dir / f"{x}.log")

    smiles = predictions_df["SMILES"].values
    log_paths = predictions_df["log_path"].values

    n = 0
    c = 0
    result = {
    "name": [],
    "smiles": [],
    "aS1": [],
    "vS1": [],
    "vdelta": [],
    "adelta": [],
    "homo": [],
    "lumo": [],
    "oscillator_strength": [],
    "elapsed_time": []
}
    youhou = 0
    for lg_path in log_paths:
        if lg_path.exists():
            try:
                name, vs1, vdelta, homo, lumo, oscillator_strength, elapsed_time = main(lg_path)
                if vs1 is None or vdelta is None:
                    continue
                vt1 = vs1 - vdelta
                idx = int(name[4:])
                print(lg_path)
                result["name"].append(name)
                result["smiles"].append(predictions_df[predictions_df["id"] == name]["SMILES"].item())
        
                as1_hat = s1_slope * vs1 + s1_intercept
                at1_hat = t1_slope * vt1 + t1_intercept
                adelta_hat = as1_hat - at1_hat

                result["aS1"].append(as1_hat)
                result["adelta"].append(adelta_hat)
                result["vS1"].append(vs1)
                result["vdelta"].append(vdelta)
                result["homo"].append(homo)
                result["lumo"].append(lumo)
                result["oscillator_strength"].append(oscillator_strength)
                if elapsed_time is None:
                    result["elapsed_time"].append(None)
                else:
                    result["elapsed_time"].append(elapsed_time * 60 / 3600)
                c += 1
                if (vs1 is not None) and (vdelta is not None) and (as1_hat > 2.6) and (as1_hat < 2.8) and (adelta_hat < 0.2):
                    youhou += 1
            except:
                pass    
        n += 1

    print(f"{c} molecules made it ouf of {n} for a ratio of {c/n}")
    print(f"{youhou} molecules satisfy the specifications out of {c} valid ones for a ratio of {youhou/c}")
    gaussian_df = pd.DataFrame.from_dict(result)

    error = {"molecule": [], "aS1": [], "adelta": []}
    for i in range(len(gaussian_df)):
        row = gaussian_df.iloc[i]
        name = row["name"]
        predicted_row = predictions_df[predictions_df["id"] == name]
        if len(predicted_row) > 0:
            predicted_s1 = predicted_row["aS1"].item()
            predicted_delta = predicted_row["adelta"].item()
            error["molecule"].append(name)
            error["aS1"].append(predicted_s1 - row["aS1"].item())
            error["adelta"].append(predicted_delta - row["adelta"].item())
    error_df = pd.DataFrame.from_dict(error)
    print("aS1 MAE:", error_df["aS1"].abs().mean())
    print("adelta MAE:", error_df["adelta"].abs().mean())

    smiles_path = csv_path.parent / f"{csv_path.stem}.pkl"
    smiles_data = pickle.load(open(smiles_path, "rb"))
    smiles_data["statistics"]["num_pass_property"] = youhou
    error_df.to_csv(csv_path.parent / f"property_predictor_error_{csv_path.stem}.csv", index=False)
    print(smiles_data["statistics"])
   
    # Move old data and update it with new entries
    generator_data_path = Path(args.generator_data_path)
    data = pd.read_csv(generator_data_path)
    
    subset = gaussian_df[["smiles", "aS1", "adelta", "vS1", "vdelta"]]
    subset["target_core"] = 1
    subset["molecule_id"] = gaussian_df["name"].apply(lambda x: f"{csv_path.stem}_{x}")
    result_df = pd.concat([data, subset], ignore_index=True)
    result_df = result_df.drop("Unnamed: 0", axis=1)
    print(result_df.describe())
    shutil.move(args.generator_data_path, csv_path.parent / f"generator_data_{csv_path.stem}.csv")
    result_df.to_csv(generator_data_path)

    property_predictor_data_path = Path(args.property_predictor_data_path)
    data = pd.read_csv(property_predictor_data_path)
    
    subset = gaussian_df[["smiles", "aS1", "adelta", "vS1", "vdelta"]]
    subset["xtb_coordinates_path"] = gaussian_df["name"].apply(lambda x: coordinates_dir / f"{x[4:]}")
    subset["molecule_id"] = gaussian_df["name"].apply(lambda x: f"{csv_path.stem}_{x}")
    result_df = pd.concat([data, subset], ignore_index=True)
    result_df = result_df.drop("Unnamed: 0", axis=1)
    print(result_df.describe())
    shutil.move(property_predictor_data_path, csv_path.parent / f"property_predictor_data_{csv_path.stem}.csv")
    result_df.to_csv(property_predictor_data_path)
