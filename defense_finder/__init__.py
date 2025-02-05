import os
import colorlog

from macsypy.scripts import macsyfinder
from warnings import simplefilter
import pandas as pd
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def run(protein_file_name, dbtype, workers, coverage, adf, adf_only, tmp_dir, models_dir, nocut_ga, loglevel, index_dir):
    scripts=[]

    if adf_only == False:
        scripts.append(['--db-type', dbtype, '--sequence-db',protein_file_name, '--models', 'defense-finder-models/DefenseFinder', 'all',
                        '--out-dir', os.path.join(tmp_dir, 'DefenseFinder'), '--w', str(workers),
                        '--coverage-profile', str(coverage), '--exchangeable-weight', '1'])
        
        scripts.append(['--db-type', dbtype, '--sequence-db',protein_file_name, '--models', 'defense-finder-models/RM', 'all',
                        '--out-dir', os.path.join(tmp_dir, 'RM'), '--w', str(workers),
                        '--coverage-profile', str(coverage), '--exchangeable-weight', '1'])

        scripts.append(['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'CasFinder', 'all',
                        '--out-dir', os.path.join(tmp_dir, 'Cas'), '-w', str(workers)])

    
    if (adf == True) or (adf_only == True):
        scripts.append(['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'defense-finder-models/ADF', 'all',
                     '--out-dir', os.path.join(tmp_dir, 'AntiDefenseFinder'), '-w', str(workers)])

    for msf_cmd in scripts:
        if nocut_ga:
            msf_cmd.append("--no-cut-ga")
        if models_dir:
            msf_cmd.extend(("--models-dir", models_dir))
        if index_dir:
            if not os.path.exists(index_dir):
                os.makedirs(index_dir)
            msf_cmd.extend(("--index-dir", index_dir))
        if loglevel != "DEBUG":
            msf_cmd.append("--mute")

        macsyfinder.main(args=msf_cmd)

        # to avoid that the macsyfinder log messages
        # appear 7 times (one by msf call
        logger2 = colorlog.getLogger('macsypy')

        for h in logger2.handlers[:]:
            logger2.removeHandler(h)

