import subprocess

def run(f, dbtype, workers, tmp_dir):
    script = """
macsyfinder \
  --db-type {dbtype} \
  --sequence-db {file} \
  --models defense-finder-models all \
  --out-dir {dir} \
  --coverage-profile 0.1 \
  --w {workers}
    """.format(file=f.name, dbtype=dbtype, workers=workers, dir=tmp_dir)

    subprocess.run(script, shell=True)
