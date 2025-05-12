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

############################################################################################################################################
############################################ run multi-temperature equilibrium AbMelt simulations ##########################################
############################################################################################################################################
############################################# compute dynamic descriptors from AbMelt methodology ##########################################
############################################################################################################################################

# import libraries, modules, and functions
import os
print(os.environ['CONDA_DEFAULT_ENV'])
import sys
print(sys.executable)
print(sys.path)
import gromacs
gromacs.config.setup()
print(os.environ["GMXBIN"])
import propka.run as pk
import argparse
from preprocess import *
from order_param import *
from res_sasa import *
# set gromacs wrapper config
gromacs.config.set_gmxrc_environment('/opt/gromacs/bin/GMXRC')
gromacs.config.get_configuration()
gromacs.tools.registry
gromacs.config.check_setup()
print(gromacs.release())
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gromacs wrapper for pbs array jobs submission')
    parser.add_argument('--project', type=str, required=True, help='specify project directory with subdirectories containing pdb files (AbMelt)')
    parser.add_argument('--pdb_dir', type=str, required=True, help='specify subdirectory containing the PDB file(s) to execute MD simulations')    
    #parser.add_argument('--dir', type=str, required=True, help='specify subdirectory to execute MD simulations (should be specified by the pbs array)' )
    parser.add_argument('--pH', type=float, required=False, default=7.4, help='specify pH to determine protonation states of histidines with propka3')
    parser.add_argument('--p_salt', type=str, required=False, default='NA', help='specify ion to add to system (e.g., "NA+", "K+", "MG2+", "CA2+")')
    parser.add_argument('--n_salt', type=str, required=False, default='CL', help='specify ion to add to system (e.g., "CL-")')
    parser.add_argument('--conc_salt', type=int, required=False, default=150, help='specify salt concentration in mM (e.g., PBS ~150 mM)')
    parser.add_argument('--temp', type=str, default='310', help='specify temperatures to run 100 ns MD simulations as string separated by commas (e.g., "300, 310, 350, 373, 400")')
    parser.add_argument('--md', action='store_true', help='run md simulations for default 100 ns')
    parser.add_argument('--time', type=int, required=False, default=100, help='specify time to run md simulations in ns if different than 100 ns (e.g., 500) or evaluate after 100 ns and extend simulations with --ext ')
    parser.add_argument('--ff', type=str, required=False, default='charmm36-jul2022', help='specify force field to use for md simulations (e.g., "charmm27, charmm36-jul2022, amber03, amber94, amber96, amber99, amber99sb, amber99sb-ildn, amberGS, oplsaa, gromos43a1, gromos43a2, gromos45a3, gromos53a5, gromos53a6, gromos54a7")')
    parser.add_argument('--water', type=str, required=False, default='tip3p', help='specify water model to use for simulations defaulted to tip3p (e.g., "CHARMM > tip3p, AMBER > tip3p or tip4p, GROMOS > spc or spc/e, OPLS--> tip3p or tip4p")')
    parser.add_argument('--ext', type=int, required=False, help='extend md simulations in ns' )
    parser.add_argument('--fix_trj', action='store_true', help='remove periodic boundary conditions from md trajectories')
    parser.add_argument('--analyze', action='store_true', help='compute descriptors from AbMelt methodology on md trajecotries')
    parser.add_argument('--eq_time', type=int, required=False, default=20, help='specify equilibration time in ns to remove from analysis')

    parser.add_argument('--cluster', action='store_true', help='cluster md trajectories using gromos algorithm')
    parser.add_argument('--cluster_n', type=int, required=False, default=10, help='specify max number of most populated clusters to output from trajectories')
    parser.add_argument('--cluster_group', type=str, required=False, default='3', help='specify index group to cluster on (e.g., "1" for all atoms, "3" for C-alpha atoms, "4" for backbone atoms, "17" for cdr3h atoms, "18" for all cdr atoms)')
    parser.add_argument('--cluster_cutoff', type=float, required=False, default=0.10, help='specify cutoff for gromos clustering algorithm in nm')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit()

    d = '/home2/AbMelt'
    pd = args.project
    wd = args.pdb_dir
    mabs = args.pdb_dir

    project_dir = os.path.join(d, 'project', pd, wd)
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Directory does not exist: {project_dir}")
    
    os.chdir(project_dir)

    if args.md:

        # latest CHARMM ff ports for GROMACS --> http://mackerell.umaryland.edu/charmm_ff.shtml
        # latest AMBER ff ports for GROMACS --> https://github.com/alanwilter/acpype
        
        # protonate pdb and convert to gromacs topology
        #gromacs_input = protonation_state(pdb= wd + '.pdb', path=d + 'project/' + pd + '/' + wd + '/' + wd + '.pdb', pH= args.pH)

        pdb_dir = os.path.join(d, 'project', pd, wd)

        # Confirm directory exists
        if not os.path.isdir(pdb_dir):
            raise FileNotFoundError(f"PDB directory does not exist: {pdb_dir}")

        # List all .pdb files
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

        if not pdb_files:
            raise FileNotFoundError(f"No .pdb files found in {pdb_dir}")

        # Process each pdb file
        for pdb_file in pdb_files:
            pdb_path = os.path.join(pdb_dir, pdb_file)
            pdb_stem = os.path.splitext(pdb_file)[0]

            # Create and switch to dedicated subdirectory
            run_dir = os.path.join(pdb_dir, pdb_stem)
            os.makedirs(run_dir, exist_ok=True)
            os.chdir(run_dir)

            # Localize pdb file
            pdb_local = os.path.basename(pdb_path)
            if not os.path.exists(pdb_local):
                shutil.copy(pdb_path, pdb_local)  # ensure the file is present in the run_dir

            print(f"Processing: {pdb_path}")
            
            # Generate .pka file using propka in run_dir
            pka_cmd = f"propka {pdb_local} --output-dir ."
            print(f"[DEBUG] Running: {pka_cmd}")
            os.system(pka_cmd)

            # Construct local paths in run_dir
            local_pdb_path = os.path.abspath(pdb_local)

            # Run protonation using the localized .pdb
            gromacs_input = protonation_state(pdb=local_pdb_path, path=local_pdb_path, pH=args.pH)

            # Run protonation
            #gromacs_input = protonation_state(pdb=pdb_path, path=pdb_path, pH=args.pH)

            # Unique filenames
            processed_pdb = f"{pdb_stem}_processed.pdb"
            processed_gro = f"{pdb_stem}_processed.gro"
            top_file = f"{pdb_stem}_topol.top"
            index_file = f"{pdb_stem}_index.ndx"
            box_gro = f"{pdb_stem}_box.gro"
            solv_gro = f"{pdb_stem}_solv.gro"
            ions_tpr = f"{pdb_stem}_ions.tpr"
            solv_ions_gro = f"{pdb_stem}_solv_ions.gro"
            em_tpr = f"{pdb_stem}_em.tpr"
            em_out = f"{pdb_stem}_em"

            # Step 1: Run pdb2gmx with PDB input to preserve chains and output PDB
            gromacs.pdb2gmx(
                f=pdb_path,
                o=processed_pdb,
                p=top_file,
                ff=args.ff,
                water=args.water,
                ignh=True,
                his=True,
                renum=True,
                input=gromacs_input
            )

            # Convert the processed PDB to GRO (no need to regenerate topology)
            gromacs.pdb2gmx(
                f=processed_pdb,
                o=processed_gro,
                ff=args.ff,
                water=args.water,
                ignh=True,
                his=True,
                renum=True,
                input=gromacs_input
            )

            # Generate a .pdb file from the processed .gro
            #gromacs.editconf(f=processed_gro, o=processed_pdb)

            # Make index from processed structure
            annotation = canonical_index(pdb=processed_pdb)
            gromacs.make_ndx(f=processed_gro, o=index_file, input=annotation)

            # Prepare box and solvate
            gromacs.editconf(f=processed_gro, o=box_gro, c=True, d='1.0', bt='triclinic')
            gromacs.solvate(cp=box_gro, cs="spc216.gro", p=top_file, o=solv_gro)

            # Prepare ion system
            ions = gromacs.config.get_templates(os.path.join(d, 'mdp', 'ions.mdp'))
            em = gromacs.config.get_templates(os.path.join(d, 'mdp', 'em.mdp'))
            em_gro = f"{pdb_stem}_em.gro"
            topol_top = f"{pdb_stem}_topol.top"

            gromacs.grompp(f=ions[0], c=solv_gro, p=top_file, o=ions_tpr, maxwarn="2")
            gromacs.genion(s=ions_tpr, o=solv_ions_gro, p=top_file,
                        pname=args.p_salt, nname=args.n_salt,
                        conc=args.conc_salt / 1000, neutral=True, input=['13'])

            # Energy minimization
            gromacs.grompp(f=em, c=solv_ions_gro, p=top_file, o=em_tpr)
            gromacs.mdrun(v=True, deffnm=em_out)


            def run_md_stage(mdp_file, deffnm, structure_gro, topol_top="topol.top", tpr_input=None):
                gpp_args = {
                    "f": mdp_file,
                    "o": f"{deffnm}.tpr",
                    "c": structure_gro,
                    "r": structure_gro,
                    "p": topol_top,
                    "maxwarn": "1",  # <-- Add this line
                }

                if tpr_input:
                    gpp_args["t"] = tpr_input

                gromacs.grompp(**gpp_args)

                gromacs.mdrun(
                    deffnm=deffnm,
                    ntomp="16",
                    nb="gpu",
                    pme="gpu",
                    bonded="cpu",
                    pin="on"
                )

            def generate_custom_mdp(template, output, temp, nsteps=None):
                kwargs = {'ref_t': [temp, temp]}
                if nsteps:
                    kwargs['nsteps'] = [nsteps]
                edit_mdp(template, new_mdp=output, **kwargs)
                return gromacs.config.get_templates(output)[0]

            # Predefined MDPs available
            #avail_temps = ['300', '310', '350', '373', '400']
            #temps = args.temp.split(',')

            # get temps to simulate
            avail_temps = ['300', '310', '350', '373', '400']
            temps = args.temp
            temps = temps.split(',')
            temps = [str(temp.strip()) for temp in temps]

            for temp in temps:
                temp = temp.strip()
                temp_int = int(temp)
                temp_prefix = f"{pdb_stem}_{temp}"

                if temp in avail_temps:
                    nvt_mdp = gromacs.config.get_templates(os.path.join(d, f"mdp/nvt_{temp}.mdp"))[0]
                    npt_mdp = gromacs.config.get_templates(os.path.join(d, f"mdp/npt_{temp}.mdp"))[0]
                    md_mdp = gromacs.config.get_templates(os.path.join(d, f"mdp/md_{temp}.mdp"))[0]
                else:
                    # Generate custom MDPs for non-predefined temperatures
                    base = os.path.join(d, "mdp")
                    nvt_mdp = generate_custom_mdp(os.path.join(base, "nvt_300.mdp"), os.path.join(base, f"nvt_{temp}.mdp"), temp_int)
                    npt_mdp = generate_custom_mdp(os.path.join(base, "npt_300.mdp"), os.path.join(base, f"npt_{temp}.mdp"), temp_int)
                    md_mdp  = generate_custom_mdp(os.path.join(base, "md_300.mdp"),  os.path.join(base, f"md_{temp}.mdp"),  temp_int)

                # Run NVT
                #run_md_stage(nvt_mdp, f"{temp_prefix}_nvt", em_gro)
                run_md_stage(nvt_mdp, f"{temp_prefix}_nvt", em_gro, topol_top)
                
                # Run NPT
                #run_md_stage(npt_mdp, f"{temp_prefix}_npt", f"{temp_prefix}_nvt.gro", tpr_input=f"{temp_prefix}_nvt.cpt")
                run_md_stage(npt_mdp, f"{temp_prefix}_npt", f"{temp_prefix}_nvt.gro", topol_top, tpr_input=f"{temp_prefix}_nvt.cpt")

                # Run MD (default or custom duration)
                if args.time == 100:
                    run_md_stage(md_mdp, f"{temp_prefix}_md", f"{temp_prefix}_npt.gro", topol_top, tpr_input=f"{temp_prefix}_npt.cpt")
                else:
                    md_mdp_custom = os.path.join(d, "mdp", f"md_{temp}_{args.time}.mdp")
                    nsteps = int(args.time * 1000 * 1000 / 2)
                    md_mdp_path = generate_custom_mdp(md_mdp, md_mdp_custom, temp_int, nsteps=nsteps)
                    run_md_stage(md_mdp_path, f"{temp_prefix}_md_{args.time}", f"{temp_prefix}_npt.gro", topol_top, tpr_input=f"{temp_prefix}_npt.cpt")
    elif args.ext:
                avail_temps = ['300', '310', '350', '373', '400']
                temps = args.temp
                temps = temps.split(',')
                temps = [str(temp.strip()) for temp in temps]

                print('extending trajectories %sns for %s temperatures...' % (str(args.ext), str(len(temps))))

                for temp in temps:
                    if args.time == 100:
                        gromacs.convert_tpr(s='md_' + temp + '.tpr',extend=args.ext*1000, o='md_' + temp + '.tpr')
                        gromacs.mdrun(deffnm='md_' + temp , cpi='md_' + temp + '.cpt', noappend=True)
                        # combine trajectories & overwrite original
                        gromacs.trjcat(f='md_' + temp + '.xtc' + ' ' + 'md_' + temp + '.part0002' + '.xtc', o='md_' + temp + '.xtc', input=['0'])
                    elif args.time != 100:
                        gromacs.convert_tpr(s='md_' + temp + '_' + str(args.time) + '.tpr',extend=args.ext*1000, o='md_' + temp + '.tpr')
                        gromacs.mdrun(deffnm='md_' + temp + '_' + str(args.time) , cpi='md_' + temp + '_' + str(args.time) + '.cpt', noappend=True)
                        # combine trajectories & overwrite original
                        gromacs.trjcat(f='md_' + temp + '_' + str(args.time) + '.xtc' + ' ' + 'md_' + temp + '_' + str(args.time) + '.part0002' + '.xtc', o='md_' + temp + '_' + str(args.time) + '.xtc', input=['0'])
                    else:
                        raise ValueError('%sns trajectory specified at %sK was not found (check .xtc filename)' % (args.time,temp))
    elif args.fix_trj:
            avail_temps = ['300', '310', '350', '373', '400']
            temps = args.temp
            temps = temps.split(',')
            temps = [str(temp.strip()) for temp in temps]

            print('removing periodic boundary conditions on trajectories for %s temperatures...' % str(len(temps)))

            for temp in temps:
                if args.time == 100:
                    gromacs.trjconv(f='md_' + temp + '.xtc', s='md_' + temp + '.tpr', pbc='whole', o='md_whole_' + temp + '.xtc', input=['0'])
                    gromacs.trjconv(f='md_whole_' + temp + '.xtc', s='md_' + temp + '.tpr', pbc='nojump', o='md_nopbcjump_' + temp + '.xtc', input=['1'])
                    gromacs.trjconv(f='md_nopbcjump_' + temp + '.xtc', s='md_' + temp + '.tpr', b='0', e='0', o='md_final_' + temp + '.gro', input=['1'])
                    gromacs.trjconv(f='md_nopbcjump_' + temp + '.xtc', s='md_' + temp + '.tpr', dt='0', o='md_final_' + temp + '.xtc', input=['1'])
                elif args.time != 100:
                    gromacs.trjconv(f='md_' + temp + '_' + str(args.time) + '.xtc', s='md_' + temp + '_' + str(args.time) + '.tpr', pbc='whole', o='md_whole_' + temp + '_' + str(args.time) + '.xtc', input=['0'])
                    gromacs.trjconv(f='md_whole_' + temp + '_' + str(args.time) + '.xtc', s='md_' + temp + '_' + str(args.time) + '.tpr', pbc='nojump', o='md_nopbcjump_' + temp + '_' + str(args.time) + '.xtc', input=['1'])
                    gromacs.trjconv(f='md_nopbcjump_' + temp + '_' + str(args.time) + '.xtc', s='md_' + temp + '_' + str(args.time) + '.tpr', b='0', e='0', o='md_final_' + temp + '.gro', input=['1'])
                    gromacs.trjconv(f='md_nopbcjump_' + temp + '_' + str(args.time) + '.xtc', s='md_' + temp + '_' + str(args.time) + '.tpr', dt='0', o='md_final_' + temp, input=['1'])
                    # change name of md_temp_time.tpr to md_temp.tpr for analysis
                    os.rename('md_' + temp + '_' + str(args.time) + '.tpr', 'md_' + temp + '.tpr')
                else:
                        raise ValueError('%sns trajectory specified at %sK was not found' % (args.time, temp))
    elif args.analyze:
            avail_temps = ['300', '310', '350', '373', '400']
            temps = args.temp
            temps = temps.split(',')
            temps = [str(temp.strip()) for temp in temps]

            print('analyzing trajectories for %s temperatures...' % str(len(temps)))

            master_s2_dict = {int(temp):{} for temp in temps}
            for temp in temps:
                # if md_final exists 
                if os.path.exists('md_final_' + temp + '.xtc'):
                    # global features
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_' + temp + '.xvg', input=['1'])
                    gromacs.hbond(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', num='bonds_' + temp + '.xvg', input=['1', '1'])
                    gromacs.rms(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsd_' + temp + '.xvg', input=['3', '3'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_' + temp + '.xvg', n='index.ndx', input=['1'])
                    # loop features
                    ## sasa
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrl1_' + temp + '.xvg', n='index.ndx', input=['12'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrl2_' + temp + '.xvg', n='index.ndx', input=['13'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrl3_' + temp + '.xvg', n='index.ndx', input=['14'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrh1_' + temp + '.xvg', n='index.ndx', input=['15'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrh2_' + temp + '.xvg', n='index.ndx', input=['16'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrh3_' + temp + '.xvg', n='index.ndx', input=['17'])
                    gromacs.sasa(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='sasa_cdrs_' + temp + '.xvg', n='index.ndx', input=['18'])
                    ## bonds
                    gromacs.hbond(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', num='bonds_lh_' + temp + '.xvg', n='index.ndx', input=['10', '11'])
                    ## rmsf
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrl1_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['12', '12'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrl2_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['13', '13'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrl3_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['14', '14'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrh1_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['15', '15'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrh2_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['16', '16'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrh3_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['17', '17'])
                    gromacs.rmsf(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='rmsf_cdrs_' + temp + '.xvg', n='index.ndx', b=str(args.eq_time*1000), input=['18', '18'])
                    ## gyration
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrl1_' + temp + '.xvg', n='index.ndx', input=['12'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrl2_' + temp + '.xvg', n='index.ndx', input=['13'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrl3_' + temp + '.xvg', n='index.ndx', input=['14'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrh1_' + temp + '.xvg', n='index.ndx', input=['15'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrh2_' + temp + '.xvg', n='index.ndx', input=['16'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrh3_' + temp + '.xvg', n='index.ndx', input=['17'])
                    gromacs.gyrate(f='md_final_' + temp + '.xtc', s='md_final_' + temp + '.gro', o='gyr_cdrs_' + temp + '.xvg', n='index.ndx', input=['18'])
                    ## S_conf
                    gromacs.trjconv(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', dt='0', fit='rot+trans', n='index.ndx', o='md_final_covar_' + temp + '.xtc', input=['1','1'])
                    gromacs.covar(f='md_final_covar_' + temp + '.xtc', s='md_' + temp + '.tpr', n='index.ndx', o='covar_' + temp + '.xvg', av='avg_covar' + temp + '.pdb', ascii='covar_matrix_' + temp + '.dat', v='covar_' + temp + '.trr',  input=['4', '4'])
                    gromacs.anaeig(f='md_final_covar_' + temp + '.xtc', v='covar_' + temp + '.trr', entropy=True, temp=temp, s='md_' + temp + '.tpr', nevskip='6', n='index.ndx', b=str(args.eq_time*1000), input=['> sconf_' + temp + '.log'])
                    ## cdr potential energy in spherical coordinates integrated over 10 slices of simulation box from index group
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrl1_' + temp + '.xvg', oc='charge_cdrl1_' + temp + '.xvg', of='field_cdrl1_' + temp + '.xvg', n='index.ndx', input=['12'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrl2_' + temp + '.xvg', oc='charge_cdrl2_' + temp + '.xvg', of='field_cdrl2_' + temp + '.xvg', n='index.ndx', input=['13'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrl3_' + temp + '.xvg', oc='charge_cdrl3_' + temp + '.xvg', of='field_cdrl3_' + temp + '.xvg', n='index.ndx', input=['14'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrh1_' + temp + '.xvg', oc='charge_cdrh1_' + temp + '.xvg', of='field_cdrh1_' + temp + '.xvg', n='index.ndx', input=['15'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrh2_' + temp + '.xvg', oc='charge_cdrh2_' + temp + '.xvg', of='field_cdrh2_' + temp + '.xvg', n='index.ndx', input=['16'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrh3_' + temp + '.xvg', oc='charge_cdrh3_' + temp + '.xvg', of='field_cdrh3_' + temp + '.xvg', n='index.ndx', input=['17'])
                    gromacs.potential(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', spherical=True, sl='10', o='potential_cdrs_' + temp + '.xvg', oc='charge_cdrs_' + temp + '.xvg', of='field_cdrs_' + temp + '.xvg', n='index.ndx', input=['18'])
                    ## dipole moment
                    gromacs.dipoles(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', o='dipole_' + temp + '.xvg', n='index.ndx', input=['1'])
                    ## N-H bond vector order parameter
                    block_length = 10 # ns; corresponds to Stokes-Einstein tumbling time of Fv
                    start = args.eq_time # ns; corresponds to defaulted equilibration time of Fv
                    s2_blocks_dict = order_s2(mab=mabs, temp=temp, block_length=block_length, start=start)
                    master_s2_dict[int(temp)] = avg_s2_blocks(s2_blocks_dict)
                    ## core/surface sasa
                    core_surface(temp)
                else:
                    raise ValueError('md_final_%s.xtc trj at %sK was not found (ensure the trjs were fixed for PBCs --fix_trj)' % (temp, temp))
            # multi-temperature features
            ## Lambda
            order_lambda(master_dict= master_s2_dict, mab=mabs, temps=temps, block_length=block_length, start=start)
    elif args.cluster:
            avail_temps = ['300', '310', '350', '373', '400']
            temps = args.temp
            temps = temps.split(',')
            temps = [str(temp.strip()) for temp in temps]

            print('clustering trajectories for %s temperatures...' % str(len(temps)))
            print('    temps (K): %s' % ' '.join(temps))
            print('    clustering on index group %s' % str(args.cluster_group))
            print('    clustering cutoff %s nm' % str(args.cluster_cutoff))

            for temp in temps:
                if os.path.exists('md_final_' + temp + '.xtc'):
                    # cluster on all atoms(1), C-alpha atoms (3), backbone atoms (4), cdr3h atoms (17), all cdr atoms (18) 
                    gromacs.cluster(f='md_final_' + temp + '.xtc', s='md_' + temp + '.tpr', method='gromos', cutoff=args.cluster_cutoff, n='index.ndx', o=True, g='clusters_' + temp, sz='cluster-size_' + temp, ntr='cluster-trans_' + temp, clid='cluster-time_' + temp, cl='clusters_' + temp, wcl=args.cluster_n , input=[args.cluster_group,'1'])
                else:
                    raise ValueError('md_final_%s.xtc trj at %sK was not found (ensure the trjs were fixed for PBCs --fix_trj)' % (temp, temp))

