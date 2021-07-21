import click
import defense_finder
import defense_finder_updater

@click.group()
def cli():
    """Systematic search of all known anti-phage systems by MDM Labs, Paris.

    Run this command at first install to get the models:

    $ defense-finder update

    Then run it every so often to the the latest updates from us.
    """
    pass

@cli.command()
def update():
    """Fetches the latest defense finder models from mdmparis repositories.

    The models will be downloaded and installed on macsydata.
    This will make them available to macsyfinder and ultimately to defense-finder.
    """
    defense_finder_updater.update_models()

@cli.command()
@click.argument('file')
def run(file: str):
    """Search for all known anti-phage defense systems in a protein sequence.

    Point the 'file' argument to the file where the .faa protein sequence is defined.
    """
    with open(file) as f:
        defense_finder.run(f)

