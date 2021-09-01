# Documentation DefenseFinder

DefenseFinder is a program to systematically detect known anti-phage systems.

DefenseFinder uses Macsyfinder. For more details regarding several MacsyFinder options and details, go here.

## Installing DefenseFinder command line interface

### Install dependency

DefenseFinder has one program dependency:
the Hmmer program, version 3.1 or greater (http://hmmer.org/).
The hmmsearch program should be installed (e.g., in the PATH) in order to use MacSyFinder.
DefenseFinder also relies on Python library dependencies:

- macsyfinder
- colorlog
- pyyaml
- packaging
- networkx
- These dependencies will be automatically retrieved and installed when using pip for installation (see below).

### Install DefenseFinder

DefenseFinder is installable through pip
Before starting, if you can, it is recommended to install DefenseFinder in a virtualenv (such as condas)

```sh
conda create –name defensefinder
conda activate defensefinder
pip install mdmparis-defense-finder
```

But you can just chose to install it wherever using pip

```sh
pip install mdmparis-defense-finder
```

if at this stage you are running into issues, it is very often due to a problem with your pip installer. Check the following [webpage](https://stackoverflow.com/questions/49748063/pip-install-fails-for-every-package-could-not-find-a-version-that-satisfies/49748494#49748494) for details on how to solve it

After installing DefenseFinder, you need to get the rules. Run the command

```sh
defense-finder update
```

### Updating DefenseFinder

In general, before running DefenseFinder, make sure to get the most uptodate rules by running

```sh
defense-finder update
```

If you have an outdated version of DefenseFinder, you can use the following line to get the most uptodate version

```sh
pip install -U mdmparis-defense-finder
defense-finder update
```

## Running Defense Finder

### Quick run (typically on one genome)

```sh
defense-finder run genome.faa
```

### Input.

The input file, here “genome.faa” has to be under the format of protein fasta, which should be “ordered”. Indeed DefenseFinder takes into account the order of the proteins.

A run on a genome (few thousands proteins) should take less than two minutes on a standard laptop. If more,make sure everything is installed properly.

ATTENTION, for more than one genome/replicon, either run one genome at a time, or format the database as described in a following section. DefenseFinder will not work on a “big” multifasta not formatted as described.

### Outputs

DefenseFinder will generate two types of files (and one optional), detailed below as well as provides the results from macsyfinder. Everything will be stored in a defined folder.

_defense_finder_systems.tsv_ : In this file, each line corresponds to a system found in the given genomes. This is a summary of what was found and gives the following information

- type: Type of the anti-phage system found
- subtype : Subtype of the anti-phage system found
- sys_beg : Protein where the systems begins (name found in the fasta file)
- sys_end : Protein where the systems ends (name found in the fasta file)
- protein_in_syst Proteins founds in the systems (name found in the fasta file)
- genes_count Number of genes found in the system
- name_of_profiles_in_sys List of the protein profiles found in the system (name from the HMM)

_defense_finder_genes.tsv_ : In this file, each line corresponds to a gene found in a system. This is a summary of what was found and gives the following information.
This follows MacsyFinder nomenclature (all_best_solution.tsv) and more can be found in the macsyfinder [documentation](https://macsyfinder.readthedocs.io/en/latest/user_guide/index.html#running-macsyfinder).

_defense_finder_hmmer.tsv_ : In this file, each line corresponds to a hit to any of the protein profiles involved in defense. Beware, a single protein can have several hits. This file is for a deep infection, of any proteins potentially linked to defense. However, biologically, it was shown that only a full system will be anti phage. So this should be interpreted with cautions.

### Running DefenseFinder on several genomes

When running DefenseFInder on several genomes, like Macsyfinder, we propose to adopt the following convention to fulfill the requirements for the “gembase db_type”.

It consists in providing for each protein, both the replicon name and a protein identifier separated by a “_” in the first field of fasta headers. “_” are accepted in the replicon name, but not in the protein identifier. Hence, the last “\_” isthe separator between the replicon name and the protein identifier. As such, MacSyFinder will be able to treat eachreplicon separately to assess macromolecular systems’ presence.

```sh
Example: esco_genomes.faa
> ESCO388_0001
XXXXXXX
> ESCO388_0002
XXXXXXX
…..
> ESCO388_3603
XXXXXXX
> ESCO389_0001
XXXXXXX
> ESCO388_0002
XXXXXXX
> ESCO388_3555
XXXXXXX
```

Then run

```sh
defense-finder run –dbtype gembase esco_genomes.faa
```

## DefenseFinder options

Help

```sh
defense-finder run --help
```

core-macsyfinder options
_-o, --out-dir_ TEXT The target directory where to store the results.Defaults to the current directory.
_-w, --workers_ INTEGER The workers count. By default all cores will be used (w=0).
_--db-type_ TEXT The macsyfinder --db-type option. Possible values are ordered_replicon, gembase, unordered, defaults to ordered_replicon. Run macsyfinder --help for more details

For questions: you can contact aude.bernheim@inserm.fr
