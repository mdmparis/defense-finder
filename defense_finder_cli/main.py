import click
import defense_finder
import defense_finder_updater
import defense_finder_posttreat

@click.group()
def cli():
    """Systematic search of all known anti-phage systems by MDM Labs, Paris.

    Prior to using defense-finder:

    - install hmmsearch: http://hmmer.org/documentation.html

    - get the models (run this every so often to stay up to date):

        $ defense-finder update

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
@click.option('-o', '--out-dir', 'outdir', help='The target directory where to store the results. Defaults to the current directory.')
def run(file: str, outdir: str):
    """Search for known anti-phage defense systems in a protein.

    Point the 'file' argument to the file where the .faa protein sequence is defined.
    """
    with open(file) as f:
        defense_finder.run(f)

    defense_finder_posttreat.run(outdir)

