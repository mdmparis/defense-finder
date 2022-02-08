import subprocess

def run(f, dbtype, workers, tmp_dir):
    script = """
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_1 all --out-dir {dir}/DF_1 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_2 all --out-dir {dir}/DF_2 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_3 all --out-dir {dir}/DF_3 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_4 all --out-dir {dir}/DF_4 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/DefenseFinder_5 all --out-dir {dir}/DF_5 --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/RM all --out-dir {dir}/RM --coverage-profile 0.1  --w {workers} --exchangeable-weight 1
macsyfinder --db-type {dbtype} --sequence-db {file} --models defense-finder-models/Cas all --out-dir {dir}/Cas --accessory-weight 1 --exchangeable-weight 1 --coverage-profile 0.4 -w {workers}
""".format(workers=workers, file=escapeFileName(f.name), dbtype=dbtype, dir=tmp_dir)

    subprocess.run(script, shell=True)

def escapeFileName(f):
    r = ""
    for char in f:
        if char >= "a" and char <= "z":
            r += char
        elif char >= "A" and char <= "Z":
            r += char
        elif char >= "0" and char <= "9":
            r += char
        elif char in ["-", "_", ".", "/"]:
            r += char
        else:
            r += "\\" + char
    return r
