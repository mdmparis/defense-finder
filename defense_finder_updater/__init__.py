import shlex
from macsypy.scripts import macsydata

def update_models(models_dir):
    # Updating DefenseFinder models
    args_models_dir = f"-t {models_dir}" if models_dir is not None else "-u"
    cmd_args = f"install -U {args_models_dir} --org mdmparis defense-finder-models"
    macsydata.main(shlex.split(cmd_args))

    # Updating CASFinder models
    args_models_dir = f"-t {models_dir}" if models_dir is not None else "-u"
    cmd_args = f"install -U {args_models_dir} CasFinder"
    macsydata.main(shlex.split(cmd_args))

