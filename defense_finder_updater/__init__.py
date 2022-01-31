from subprocess import run

script = """
macsydata install -U --org mdmparis defense-finder-models
"""

def update_models():
    run(script, shell=True)
