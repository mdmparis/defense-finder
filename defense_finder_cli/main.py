import click
import defense_finder
import defense_finder_updater

@click.group()
def cli():
    """Systematic search of all known anti-phage systems by MDM Labs, Paris.

    Run this command on first install to get the models:

    $ defense-finder update

    Then run it every so often to get the latest models from us.

    Tool repository: https://github.com/mdmparis/defense-finder.
    """
    pass

@cli.command()
def update():
    """Fetches the latest defense finder models.

    The models will be downloaded from mdmparis repositories and installed on macsydata.

    This will make them available to macsyfinder and ultimately to defense-finder.

    Models repository: https://github.com/mdmparis/defense-finder-models.
    """
    defense_finder_updater.update_models()

@cli.command()
@click.argument('file')
def run(file: str):
    """Search for known anti-phage defense systems in a protein.

    Point the 'file' argument to the file where the .faa protein sequence is defined.

    Output is written to /tmp/defense-finder/
    """
    with open(file) as f:
        defense_finder.run(f)

