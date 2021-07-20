import subprocess

def run(f):
    script = """
DIR="/tmp/defense-finder"
rm -rf $DIR
mkdir $DIR
macsyfinder \
  --db-type ordered_replicon \
  --sequence-db {file} \
  --models defense-finder-models all \
  --out-dir $DIR \
  --coverage-profile 0.1 \
  --w 4
    """.format(file=f.name)

    subprocess.run(script, shell=True)

    print("Analysis results are saved in /tmp/defense-finder/")
