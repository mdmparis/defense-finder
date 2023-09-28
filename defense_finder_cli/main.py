import os
import shutil
import click
import defense_finder
import defense_finder_updater
import defense_finder_posttreat
from pyhmmer.easel import SequenceFile, TextSequence, Alphabet
import pyrodigal

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
@click.option('--models-dir', 'models_dir', required=False, help='Specify a directory containing your models.')
def update(models_dir=None):
    """Fetches the latest defense finder models.

    The models will be downloaded from mdmparis repositories and installed on macsydata.

    This will make them available to macsyfinder and ultimately to defense-finder.

    Models repository: https://github.com/mdmparis/defense-finder-models.
    """
    defense_finder_updater.update_models(models_dir)


@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('-o', '--out-dir', 'outdir',
              help='The target directory where to store the results. Defaults to the current directory.')
@click.option('-w', '--workers', 'workers', default=0,
              help='The workers count. By default all cores will be used (w=0).')
@click.option('-c', '--coverage', 'coverage', default=0.4,
              help='Minimal percentage of coverage for each profiles. By default set to 0.4')
@click.option('--db-type', 'dbtype', default='ordered_replicon',
              help='The macsyfinder --db-type option. Run macsyfinder --help for more details. Possible values are\
               ordered_replicon, gembase, unordered, defaults to ordered_replicon.')
@click.option('--preserve-raw', 'preserve_raw', is_flag=True, default=False,
              help='Preserve raw MacsyFinder outputs alongside Defense Finder results inside the output directory.')
@click.option('--models-dir', 'models_dir', required=False, help='Specify a directory containing your models.')
@click.option('--no-cut-ga', 'no_cut_ga', is_flag=True, default=False,
              help='Advanced! Run macsyfinder in no-cut-ga mode. The validity of the genes and systems found is not guaranteed!')

def run(file: str, outdir: str, dbtype: str, workers: int, coverage: float, preserve_raw: bool, no_cut_ga: bool, models_dir: str = None ):
    """Search for all known anti-phage defense systems in the target .faa protein file.
    """
    filename = click.format_filename(file)

    # Prepare output folder
    default_outdir = os.getcwd()
    outdir = outdir if outdir != None else default_outdir
    if not os.path.isabs(outdir):
        outdir = os.path.join(os.getcwd(), outdir)
    os.makedirs(outdir, exist_ok=True)

    tmp_dir = os.path.join(outdir, 'defense-finder-tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)

    with SequenceFile(filename) as sf:
        seq = TextSequence()
        dic_genes = {}
        if sf.guess_alphabet() == Alphabet.dna():
            while sf.readinto(seq) is not None: # iterate over sequences in case multifasta
                sseq = bytes(seq.sequence, encoding="utf-8")
                sname = seq.name.decode()
                if len(sseq) < 100000: # it is recommended to use the mode meta when seq is less than 100kb
                    orf_finder = pyrodigal.GeneFinder(meta=True)
                    dic_genes[sname] = orf_finder.find_genes(sseq)
                else:
                    orf_finder = pyrodigal.GeneFinder()
                    orf_finder.train(sseq)
                    dic_genes[sname] = orf_finder.find_genes(sseq)
            protein_file_name = f"{os.path.splitext(filename)[0]}.prt"
            if os.path.exists(protein_file_name):
                protein_file_name += "_defensefinder.prt"
            with open(protein_file_name, "w") as protein_file:
                for key, genes in dic_genes.items():
                    # proteins will be like `> contig_name_i` with i being increasing integer 
                    genes.write_translations(protein_file, key)
            # prodigal correctly numbered the protein in a gembase like format (cf above)
            dbtype = "gembase"
        else:
            protein_file_name = filename


    defense_finder.run(protein_file_name, dbtype, workers, coverage, tmp_dir, models_dir, no_cut_ga)

    defense_finder_posttreat.run(tmp_dir, outdir)

    if not preserve_raw:
        shutil.rmtree(tmp_dir)
