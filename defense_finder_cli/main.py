import os
import shutil
import click
import defense_finder
import defense_finder_updater
import defense_finder_posttreat
from pyhmmer.easel import SequenceFile, TextSequence, Alphabet
import pyrodigal
from macsypy.scripts.macsydata import get_version_message
from macsypy.scripts.macsydata import _find_all_installed_packages

import colorlog
try:
    logging = colorlog.logging.logging
except AttributeError:
    logging = colorlog.wrappers.logging


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
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
def version():
    """Get the version of DefenseFinder (software)
    """
    print(f"Using DefenseFinder version {__version__}")


@cli.command()
@click.option('--models-dir', 'models_dir', required=False, help='Specify a directory containing your models.')
@click.option('--force-reinstall', '-f', 'force_reinstall', is_flag=True,
              help='Force update even if models in already there.', default=False)
def update(models_dir=None, force_reinstall: bool = False):
    """Fetches the latest defense finder models.

    The models will be downloaded from mdmparis repositories and installed on macsydata.

    This will make them available to macsyfinder and ultimately to defense-finder.

    Models repository: https://github.com/mdmparis/defense-finder-models.
    """
    defense_finder_updater.update_models(models_dir, force_reinstall)


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
@click.option('--log-level', 'loglevel', default="INFO",
              help='set the logging level among DEBUG, [INFO], WARNING, ERROR, CRITICAL')
@click.option('--index-dir', 'index_dir', required=False, help='Specify a directory to write the index files required by macsyfinder when the input file is in a read-only folder')
def run(file: str, outdir: str, dbtype: str, workers: int, coverage: float, preserve_raw: bool, 
        no_cut_ga: bool, models_dir: str = None, loglevel : str = "INFO",
        index_dir: str = None):
    """
    Search for all known anti-phage defense systems in the target fasta file.
    """
    global logger
    LOGFORMAT = " %(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    #logging.root.setLevel(LOG_LEVEL)
    formatter = colorlog.ColoredFormatter(LOGFORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    stream = logging.StreamHandler()
    stream.setLevel(loglevel)
    stream.setFormatter(formatter)
    logger = colorlog.getLogger("Defense_Finder")
    logger.setLevel(loglevel)
    logger.addHandler(stream)
       
    filename = click.format_filename(file)
    # Prepare output folder

    logger.info(f"Received file {filename}")

    default_outdir = os.getcwd()
    logger.debug(f"Defaut outdir : {default_outdir}")
    outdir = outdir if outdir != None else default_outdir
    if not os.path.isabs(outdir):
        outdir = os.path.join(os.getcwd(), outdir)
    logger.debug(f"outdir : {outdir}")

    if os.path.exists(outdir):
        logger.warning(f"Out directory {outdir} already exists. Existing DefenseFinder output will be overwritten")
    os.makedirs(outdir, exist_ok=True)

    tmp_dir = os.path.join(outdir, 'defense-finder-tmp')
    if os.path.exists(tmp_dir):
        logger.warning(f"Temporary directory {tmp_dir} already exists. Overwriting it.")
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)

    with SequenceFile(filename) as sf:
        seq = TextSequence()
        dic_genes = {}
        if sf.guess_alphabet() == Alphabet.dna():
            logger.info(f"{filename} is a nucleotide fasta file. Prodigal will annotate the CDS")
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
                seq.clear()

            protein_file_name = os.path.join(outdir, f"{os.path.splitext(os.path.basename(filename))[0]}.prt")

            if os.path.exists(protein_file_name):
                logger.warning(f"{protein_file_name} already exists, writing in {protein_file_name + '_defensefinder.prt'} -- Overwriting it if already existing")
                protein_file_name += "_defensefinder.prt"
            logger.info(f"Prodigal annotated {len(dic_genes.keys())} replicons")
            with open(protein_file_name, "w") as protein_file:
                for key, genes in dic_genes.items():
                    # proteins will be like `> contig_name_i` with i being increasing integer 
                    genes.write_translations(protein_file, key)
            # prodigal correctly numbered the protein in a gembase like format (cf above)
            dbtype = "gembase"
            logger.info(f"Protein files written in {protein_file_name}")
            logger.info(f"{sum([len(v) for v in dic_genes.values()])} CDS were annotated")

        else:
            protein_file_name = filename

    logger.info(f"Running DefenseFinder version {__version__}")
    defense_finder.run(protein_file_name, dbtype, workers, coverage, tmp_dir, models_dir, no_cut_ga, loglevel, index_dir)
    logger.info("Post-treatment of the data")
    defense_finder_posttreat.run(tmp_dir, outdir, os.path.splitext(os.path.basename(filename))[0])

    if not preserve_raw:
        shutil.rmtree(tmp_dir)

    models = _find_all_installed_packages().models()
    versions_models = []
    for m in models:
        if "cas" in m.path.lower() or "defense-finder" in m.path.lower():
            versions_models.append([m.path, m.version])
    nl = "\n"
    tab = "\t"

    logger.info(f"""\
Analysis done. Please cite :

Tesson F., Hervé A. , Mordret E., Touchon M., d’Humières C., Cury J., Bernheim A., 2022, Nature Communication
Systematic and quantitative view of the antiviral arsenal of prokaryotes

Using DefenseFinder version {__version__}.

DefenseFinder relies on MacSyFinder : 

{get_version_message().split("and don't")[0]}

Using the following models:

{nl.join([f"{path+tab+version}" for path, version in versions_models])}

""")

if __name__ == "__main__":
    __version__ = "Version_from_the_command_line"
    cli()
else:
    from ._version import __version__
