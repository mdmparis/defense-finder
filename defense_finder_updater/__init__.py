from subprocess import run

def update_models(models_dir):
    args_models_dir = f"-t {models_dir}" if models_dir is not None else ""
    script = f"""
    macsydata install -U {args_models_dir} --org mdmparis defense-finder-models
    """
    run(script, shell=True)
