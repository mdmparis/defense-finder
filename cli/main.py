import click
import defense_finder
import updater

@click.group()
def cli():
    pass

@cli.command()
def update():
    """Fetches the latest defense finder models from mdmlabs repositories."""
    updater.update_models()

@cli.command()
@click.argument('file')
def run(file: str):
    """Search for all known anti-phage defense systems in a protein sequence.

    The 'file' argument should point to the file where the protein sequence is defined.
    """
    with open(file) as f:
        defense_finder.run(f)

