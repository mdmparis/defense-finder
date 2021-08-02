import subprocess

def run(f, dbtype):
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
  --w 4
    """.format(file=f.name, dbtype=dbtype)

    subprocess.run(script, shell=True)
