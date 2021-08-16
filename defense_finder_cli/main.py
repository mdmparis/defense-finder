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
@click.argument('file', type=click.Path(exists=True))
@click.option('-o', '--out-dir', 'outdir', help='The target directory where to store the results. Defaults to the current directory.')
@click.option('-w', '--workers', 'workers', default=0, help='The workers count. By default all cores will be used (w=0).')
@click.option('--db-type', 'dbtype', default='ordered_replicon', help='The macsyfinder --db-type option. Run macsyfinder --help for more details. Possible values are ordered_replicon, gembase, unordered, defaults to ordered_replicon.')
def run(file: str, outdir: str, dbtype: str, workers: int):
    """Search for all known anti-phage defense systems in the target .faa protein file.
    """
    filename = click.format_filename(file)
    with open(filename) as f:
        defense_finder.run(f, dbtype, workers)

    defense_finder_posttreat.run(outdir)

