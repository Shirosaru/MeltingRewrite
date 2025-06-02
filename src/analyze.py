#!/usr/bin/env python3

#    MIT License

#    COPYRIGHT (C) 2024 MERCK SHARP & DOHME CORP. A SUBSIDIARY OF MERCK & CO., 
#    INC., RAHWAY, NJ, USA. ALL RIGHTS RESERVED

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from pymbar import timeseries
from os.path import basename
os.chdir('/home2/AbMelt/src')
from res_sasa import get_core_surface
from res_sasa import get_slope
import subprocess

# Set this to whatever top-level path you're analyzing
base_dir = '/home2/AbMelt/project/AbMelt'
os.chdir(base_dir)
print("Working directory:", os.getcwd())
cwd=os.getcwd()

# Find subdirectories
dirs = [d for d in glob.glob('./*/', recursive=True) if os.path.isdir(d)]
print("Subdirectories found:", dirs)

# Data collection dictionary
eq_parameters = {'mAb': [], 'TEMP': [], 'metric': [], 'eq_time': [], 'eq_mu': [], 'eq_std': []}

def load_xvg_data(filepath):
    """Load .xvg file and return a NumPy array of the second column (y-values)."""
    y_vals = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@') or line.startswith('#') or line == '':
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    y_vals.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(y_vals)

# Loop through directories
for subdir in dirs:
    full_path = os.path.join(base_dir, subdir)
    os.chdir(full_path)
    
    xvgs = glob.glob('*.xvg')
    if not xvgs:
        print(f"Skipping {subdir} — no .xvg files found.")
        continue

    print(f"\nProcessing {subdir}")
    print("Found .xvg files:", xvgs)

    for xvg in range(len(xvgs)):
        temps = ['310','350','373']
        metric = xvgs[xvg].split('.')[0]
        t, x, y, z, r= [], [], [], [] , []
        try:
            temp = [temp for temp in temps if temp in xvgs[xvg]][0]
        except:
            print('skip %s' % xvgs[xvg])
            continue
        with open("%s" % xvgs[xvg]) as f:
            #lines = (line for line in f if not line.startsiwith('#') if not line.startswith('$'))
            for line in f:
                if line.startswith('#'):
                    continue
                if line.startswith('@'):
                    continue
                cols = line.split()

                if len(cols) == 2:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                elif len(cols) == 3:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                elif len(cols) == 4:
                    t.append(float(cols[0]))
                    x.append(float(cols[1]))
                    y.append(float(cols[2]))
                    z.append(float(cols[3]))
                elif len(cols) == 5:
                    t.append(float(cols[0]))
                    r.append(float(cols[1]))
                    x.append(float(cols[2]))
                    y.append(float(cols[3]))
                    z.append(float(cols[4]))
                else:
                    raise ValueError("Invalid number of columns: %d" % len(cols))

        # === Use fixed t0 and eq_time for all cases ===
        t0 = 2000

        if "bonds" in metric:
            x = np.array(x)
            x_equlibrated = x[t0:]
            x_mu, x_std = np.mean(x_equlibrated), np.std(x_equlibrated)
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric + '_hbonds', 'eq_time': 2000, 'eq_mu': x_mu, 'eq_std': x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            # Load hydrogen bond data from .xvg file
            y = load_xvg_data('bonds_lh_400_contacts.xvg')  # <- update with your dynamic path if needed

            # Handle equilibration time
            t0 = 2000
            if len(y) < t0:
                print(f"⚠️ Warning: contacts data for {metric} has fewer than {t0} frames, using t0 = 0 instead")
                t0 = 0

            # Slice and compute statistics
            y_equlibrated = y[t0:]
            y_mu = np.mean(y_equlibrated)
            y_std = np.std(y_equlibrated)

            # Package results
            params = {
                'mAb': subdir.strip('./').split('/')[0],
                'TEMP': temp,
                'metric': metric + '_contacts',
                'eq_time': t0,
                'eq_mu': y_mu,
                'eq_std': y_std
            }

            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "gyr" in metric:
            r = np.array(r)
            r_mu, r_std = np.mean(r[t0:]), np.std(r[t0:])
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric + '_Rg', 'eq_time': 2000, 'eq_mu': r_mu, 'eq_std': r_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

            x_mu, x_std = np.mean(np.array(x)[t0:]), np.std(np.array(x)[t0:])
            y_mu, y_std = np.mean(np.array(y)[t0:]), np.std(np.array(y)[t0:])
            z_mu, z_std = np.mean(np.array(z)[t0:]), np.std(np.array(z)[t0:])

            for suffix, mu, std in zip(['Rx', 'Ry', 'Rz'], [x_mu, y_mu, z_mu], [x_std, y_std, z_std]):
                params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric + '_' + suffix, 'eq_time': 2000, 'eq_mu': mu, 'eq_std': std}
                for key, value in params.items():
                    eq_parameters[key].append(value)

        elif "rmsd" in metric:
            x = np.array(x)
            x_mu, x_std = np.mean(x[t0:]), np.std(x[t0:])
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric, 'eq_time': 2000, 'eq_mu': x_mu, 'eq_std': x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "rmsf" in metric:
            x_mu, x_std = np.mean(x), np.std(x)
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric, 'eq_time': 2000, 'eq_mu': x_mu, 'eq_std': x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "sasa" in metric:
            x = np.array(x)
            x_mu, x_std = np.mean(x[t0:]), np.std(x[t0:])
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric, 'eq_time': 2000, 'eq_mu': x_mu, 'eq_std': x_std}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "potential" in metric:
            x = np.array(x)
            if len(x) > 5:
                x_mu = x[5]  # assuming radius = 5 is desired
            else:
                x_mu = x[-1]
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric, 'eq_time': 2000, 'eq_mu': x_mu, 'eq_std': 0}
            for key, value in params.items():
                eq_parameters[key].append(value)

        elif "dipole" in metric:
            z = np.array(z)
            z_mu, z_std = np.mean(z[t0:]), np.std(z[t0:])
            params = {'mAb': subdir.strip('./').split('/')[0], 'TEMP': temp, 'metric': metric, 'eq_time': 2000, 'eq_mu': z_mu, 'eq_std': z_std}
            for key, value in params.items():
                eq_parameters[key].append(value)
        else:
            continue
    # add core/surface sasa metrics 
    nps = glob.glob('*.np')
    res_sasa = [np for np in nps if "res_sasa" in np]
    temps = [np.split('res_sasa_')[1] for np in res_sasa]
    temps = [temp.split('.np')[0] for temp in temps]
    temps = [int(x) for x in temps]
    sasa_dict = {temp: {} for temp in temps}
    sasa_dict['dSASA/dT'] = {}
    sections = ['total_mean','core_mean','surface_mean','total_std','core_std','surface_std']
    for temp in temps:
        sasa_dict = get_core_surface(sasa_dict, temp, k = 20, start = 20)
    for sec in sections:
        sasa_dict['dSASA/dT'][sec] = get_slope([(temp, sasa_dict[temp][sec]) for temp in temps])
    # iterate through sasa_dict and append values to eq_parameters
    for key, value in sasa_dict.items():
        if key == 'dSASA/dT':
            for sec, val in value.items():
                params = {'mAb':subdir.strip('./').split('/')[0],'TEMP':'all','metric':sec + '_dSASA_dT','eq_time':2000,'eq_mu':val,'eq_std':0}
                for k, value in params.items():
                    eq_parameters[k].append(value)
        elif key != 'dSASA/dT':
            for sec, val in value.items():
                params = {'mAb':subdir.strip('./').split('/')[0],'TEMP':str(key),'metric':sec + '_' + str(key),'eq_time':2000,'eq_mu':val,'eq_std':0}
                for k, value in params.items():
                    eq_parameters[k].append(value)

    # add order parameter metrics
    csvs = glob.glob('*.csv')
    ops = [csv for csv in csvs if "order_" in csv]
    # iterate through csvs and collect order parameter data
    for op in ops:
        if 'order_lambda' in op:
            df = pd.read_csv(op).drop(columns = ['Unnamed: 0'])
            # split left of csv
            name = op.split('.csv')[0]
            # mean of lambda and mean of r 
            lambda_mean = np.mean(df['lamda'])
            params = {'mAb':subdir.strip('./').split('/')[0],'TEMP':'all','metric':name,'eq_time':2000,'eq_mu':lambda_mean,'eq_std':0}
            for key, value in params.items():
                eq_parameters[key].append(value)
            r_mean = np.mean(df['r'])
            params = {'mAb':subdir.strip('./').split('/')[0],'TEMP':'all','metric':name + '_r','eq_time':2000,'eq_mu':r_mean,'eq_std':0}
            for key, value in params.items():
                eq_parameters[key].append(value)
        elif 'order_s2' in op:
            df = pd.read_csv(op).drop(columns = ['Unnamed: 0'])
            # split left of csv
            name = op.split('.csv')[0]
            # mean and std of first column by index
            s2_mean = np.mean(df.iloc[:,0])
            s2_std = np.std(df.iloc[:,0])
            # split name to get value before K
            temp = name.split('order_s2_')[1]
            temp = temp.split('K_')[0]
            params = {'mAb':subdir.strip('./').split('/')[0],'TEMP':str(temp),'metric':name,'eq_time':2000,'eq_mu':s2_mean,'eq_std':s2_std}
    # read output_analyze.log and find lines containing "Entropy"
    # if file contains output_analyze.507708.* then read file
    log_list = glob.glob('output_analyze.*.log')

    if not log_list:
        print(f"No log file found in {os.getcwd()}")
    else:
        for log_file in log_list:
            with open(log_file) as f:
                temp = None
                for line in f:
                    if 'gmx_mpi anaeig -f md_final_covar_' in line:
                        line = line.split()
                        temp = line[8] if len(line) > 8 else "NA"

                    if all(keyword in line for keyword in ['Entropy', 'J/mol K', 'Schlitter']):
                        line = line.split()
                        entropy = line[8] if len(line) > 8 else "NA"
                        params = {
                            'mAb': os.path.basename(os.path.normpath(dir)),
                            'TEMP': str(temp),
                            'metric': 'sconf_schlitter',
                            'eq_time': 2000,
                            'eq_mu': entropy,
                            'eq_std': 0
                        }
                        for key, value in params.items():
                            eq_parameters[key].append(value)

                    elif all(keyword in line for keyword in ['Entropy', 'J/mol K', 'Quasiharmonic']):
                        line = line.split()
                        entropy = line[8] if len(line) > 8 else "NA"
                        params = {
                            'mAb': os.path.basename(os.path.normpath(dir)),
                            'TEMP': str(temp),
                            'metric': 'sconf_quasiharmonic',
                            'eq_time': 2000,
                            'eq_mu': entropy,
                            'eq_std': 0
                        }
                        for key, value in params.items():
                            eq_parameters[key].append(value)
    os.chdir(cwd)
    df = pd.DataFrame(eq_parameters)
    print(df)
    df.to_csv(os.path.join(full_path, '_abmelt_eq_20ns_parameters.csv'))
    # Call get_features.py for the current subdir
    subprocess.run(["python3", "/home2/AbMelt/src/get_features.py"], cwd=full_path)
#os.chdir(cwd)
#df = pd.DataFrame(eq_parameters)
#print(df)
#df.to_csv('_abmelt_eq_20ns_parameters.csv')
