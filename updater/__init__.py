from subprocess import run
import os

script = """
DIR="/tmp/defense-finder"
rm -rf $DIR
mkdir $DIR
macsydata download --org mdmparis -d $DIR defense-finder-models 
ARCHIVE=$(ls $DIR | head -n 1)
macsydata install --org mdmparis "$DIR/$ARCHIVE"
rm -rf $DIR
"""

def update_models():
    run(script, shell=True)
