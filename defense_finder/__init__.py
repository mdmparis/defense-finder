import os
import colorlog

from macsypy.scripts import macsyfinder
from macsypy.scripts import macsydata


def run(protein_file_name, dbtype, workers, coverage, tmp_dir, models_dir, nocut_ga, loglevel):

    gen_args = ['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'defense-finder-models/DefenseFinder_{i}', 'all',
                '--out-dir', os.path.join(tmp_dir, 'DF_{i}'), '--w', str(workers),
                '--coverage-profile', str(coverage), '--exchangeable-weight', '1']
    scripts = [[f.format(i=i) for f in gen_args] for i in range(1, 6)]

    scripts.append(['--db-type', dbtype, '--sequence-db',protein_file_name, '--models', 'defense-finder-models/RM', 'all',
                     '--out-dir', os.path.join(tmp_dir, 'RM'), '--w', str(workers),
                     '--coverage-profile', str(coverage), '--exchangeable-weight', '1'])

    scripts.append(['--db-type', dbtype, '--sequence-db', protein_file_name, '--models', 'CasFinder', 'all',
                     '--out-dir', os.path.join(tmp_dir, 'Cas'), '-w', str(workers)])

    for msf_cmd in scripts:
        if nocut_ga:
            msf_cmd.append("--no-cut-ga")
        if models_dir:
            msf_cmd.extend(("--models-dir", models_dir))
        if loglevel != "DEBUG":
            msf_cmd.append("--mute")

        macsydata.main(args=["list"])
        macsyfinder.main(args=msf_cmd)

        # to avoid that the macsyfinder log messages
        # appear 7 times (one by msf call
        logger2 = colorlog.getLogger('macsypy')

        for h in logger2.handlers[:]:
            logger2.removeHandler(h)

