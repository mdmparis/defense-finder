import shlex

from macsypy.scripts.macsydata import main as mdmain


def update_models(models_dir, force_reinstall: bool):
    # Updating DefenseFinder models
    args_models_dir = f"-t {models_dir}" if models_dir is not None else "-u"
    args_force = "-f" if force_reinstall else ""
    cmd_args = f"install -U {args_models_dir} {args_force} --org mdmparis defense-finder-models"
    mdmain(shlex.split(cmd_args))

    # Updating CASFinder models
    args_models_dir = f"-t {models_dir}" if models_dir is not None else "-u"
    cmd_args = f"install -U {args_models_dir} {args_force} CasFinder"
    mdmain(shlex.split(cmd_args))

