import subprocess

def run(f, dbtype, workers, tmp_dir):
    script = """
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_1 all --out-dir {dir}/DF_1 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_2 all --out-dir {dir}/DF_2 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_3 all --out-dir {dir}/DF_3 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_4 all --out-dir {dir}/DF_4 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_5 all --out-dir {dir}/DF_5 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/RM all --out-dir {dir}/RM --coverage-profile 0.1  --w {workers} --exchangeable-weight 1 --models-dir models
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/Cas all --out-dir {dir}/Cas --accessory-weight 1 --exchangeable-weight 1 --coverage-profile 0.4 -w {workers} --models-dir models
""".format(workers=workers, file=f.name, dbtype=dbtype, dir=tmp_dir)

    subprocess.run(script, shell=True)
