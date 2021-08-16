import subprocess

def run(f, dbtype, workers):
    script = """
DIR="/tmp/defense-finder"
rm -rf $DIR
mkdir $DIR
macsyfinder \
  --db-type {dbtype} \
  --sequence-db {file} \
  --models defense-finder-models all \
  --out-dir $DIR \
  --coverage-profile 0.1 \
  --w {workers}
    """.format(file=f.name, dbtype=dbtype, workers=workers)

    subprocess.run(script, shell=True)
