from subprocess import run

script = """
macsydata install -U -u --org mdmparis defense-finder-models
"""

def update_models():
    run(script, shell=True)
