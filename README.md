
# Documentation DefenseFinder

DefenseFinder is a program to systematically detect known anti-phage systems. DefenseFinder uses MacSyFinder.

If you are using DefenseFinder please cite

- "Systematic and quantitative view of the antiviral arsenal of prokaryotes" [Nature Communication](https://www.nature.com/articles/s41467-022-30269-9.pdf), 2022, _Tesson F., Hervé A. , Mordret E., Touchon M., d’Humières C., Cury J., Bernheim A._
- "MacSyFinder: A Program to Mine Genomes for Molecular Systems with an Application to CRISPR-Cas Systems." PloS one 2014
  _Abby S., Néron B.,Ménager H., Touchon M. Rocha EPC._

## DefenseFinder Models

This repository contains DefenseFinder a tool allowing for a systematic search of anti-phage systems.
The DefenseFinder models based on MacSyFinder architecture can be [here](https://github.com/mdmparis/defense-finder-models)

# Installing DefenseFinder command-line interface
	
### Install dependency

DefenseFinder has one program dependency:
the Hmmer program, version 3.1 or greater (http://hmmer.org/).
The hmmsearch program should be installed (e.g., in the PATH) to use MacSyFinder.
DefenseFinder also relies on Python library dependencies:

- macsyfinder
- colorlog
- pyyaml
- packaging
- networkx
- These dependencies will be automatically retrieved and installed when using pip for installation (see below).

### Install DefenseFinder

DefenseFinder is installable through pip.
Before starting, if you can, it is recommended to install DefenseFinder in a virtualenv (such as condas).

```sh
conda create –name defensefinder
conda activate defensefinder
pip install mdmparis-defense-finder
```

However, you can also install DefenseFinder using only pip.

```sh
pip install mdmparis-defense-finder
```

At this stage, if you have an issue, this could be due to a problem with your pip installer.
Check the following [webpage](https://stackoverflow.com/questions/49748063/pip-install-fails-for-every-package-could-not-find-a-version-that-satisfies/49748494#49748494) for details on how to solve it

After installing DefenseFinder, you need to retrieve the DefenseFinder models.
To retrieve it run:

```sh
defense-finder update
```

### Updating DefenseFinder
When you have not used DefenseFinder in the last days, make sure you have the latest versions of the models.
To verify and downloaded if necessary the latest models run:

```sh
defense-finder update
```

When the DefenseFinder models are updated you only have to update the models and not the tool.
However, if you have an outdated version of the DefenseFinder tool, you can use the following line to get the most recent version

```sh	
pip install -U mdmparis-defense-finder
defense-finder update
```
### defense-finder update options
To check the different DefenseFinder update options run

```bash
$ defense-finder update --help

Usage: defense-finder update [OPTIONS]

  Fetches the latest defense finder models.

  The models will be downloaded from mdmparis repositories and installed on
  macsydata.

  This will make them available to macsyfinder and ultimately to defense-
  finder.

  Models repository: https://github.com/mdmparis/defense-finder-models.

Options:
  --models-dir TEXT  Specify a directory containing your models.
  --help             Show this message and exit.
```


## Running Defense Finder

### Quick run
If you want to run DefenseFinder on a small set of genomes (< 30 000 proteins).
You can run the following command.

```sh
defense-finder run genome.faa
```

### Input.

The input file, here “genome.faa” has to be under the format of protein fasta, where all proteins are in the order of their position in the genome. Indeed DefenseFinder takes into account the order of the proteins.

A run on a genome (few thousand proteins) should take less than two minutes on a standard laptop. If more, make sure everything is installed properly.
In this configuration, all the replicon will be named UserReplicon.
ATTENTION, If you want to run DefenseFinder on a larger set of genomes you need to format your dataset as described in "Larger dataset and Gembase Format".

### Outputs

DefenseFinder will generate three types of files (and an option to conserve MacSyFinder options).
All the files are described below.

`defense_finder_systems.tsv` : In this file, each line corresponds to a system found in the given genomes. This is a summary of what was found and gives the following information

- `sys_id` : Each system detected by DefenseFinder have a unique ID based on the replicon where it was found and the type of systems
- `type`: Type of the anti-phage system found (such as RM, Cas...)
- `subtype` : Subtype of the anti-phage system found (such as RM_type_I, CAS_Class1-Subtype-I-E)
- `sys_beg` : Protein where the system begins (name found in the input file)
- `sys_end` : Protein where the system ends (name found in the input file)
- `protein_in_syst` : List of all protein(s) present in this system (name found in the input file)
- `genes_count` : Number of genes found in the system
- `name_of_profiles_in_sys`:  List of the protein profiles that hit the protein of the system (name from the HMM).

`defense_finder_genes.tsv` : In this file, each line corresponds to a gene found in a system.
For each gene, there is several information such as the replicon, the position, the system..
All the information comes from MacSyFinder and follows MacSyFinder nomenclature (best_solution.tsv) and more can be found in the MacSyFinder Ma [documentation](https://macsyfinder.readthedocs.io/en/latest/user_guide/outputs.html).

`defense_finder_hmmer.tsv` : In this file, each line corresponds to an HMM hit. This file show all hit of HMM regardless if they are in a complete system or not. Those results have to be used cautiously for deep inspection. Indeed, biologically, it was shown that only a full system will be anti phage. This function can be used to found defense gene in small portion of genomes.
Beware, a single protein can have several hits. The output is a part of the result of HMMer results table.

- `hit_id` : the protein name (name found in the input file)
- `replicon` : The name of the replicon
- `position_hit`: The position in the input file
- `Gene_name` : the name of the HMM

By using the argument --preserve-raw , you will have all the results from MacSyFinder. Those results are explained [here](https://macsyfinder.readthedocs.io/en/latest/user_guide/outputs.html)

### Running DefenseFinder on several genomes

When running DefenseFInder on several genomes, like MacSyFinder, we propose to adopt the following convention to fulfill the requirements for the “gembase db_type”.
The difference between any fasta file and the gembase format is the name of the protein (Protein name = the text after \> in a fasta file). For both type, protein have to be ordered but in the first case the name of the protein do not matter (except no repetition).
In the gembase format, protein name are composeded of two part: the replicon and the position. The replicon name is the same for all the protein that the user want to be analyse simultaneously (for example a complete genome, a plasmid...)
The second component is the position. Those two component must be separated by "_". It is possible to use "_" in the replicon name, only the last instance will be used as the separator between replicon name and position.
With this format of file, MacSyFinder will be able to treat each replicon separately to assess macromolecular systems presence and reduce ressource use. 


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
> ESCO389_0002
XXXXXXX
…..
> ESCO389_3555
XXXXXXX
```

To use DefenseFinder with gembase format file on larger dataset of genomes run

```sh
defense-finder run –dbtype gembase esco_genomes.faa
```

## defense-finder run options
To check the different DefenseFinder options run

```bash
$ defense-finder run --help

Usage: defense-finder run [OPTIONS] FILE

  Search for all known anti-phage defense systems in the target .faa protein
  file.

Options:
  -o, --out-dir TEXT     The target directory where to store the results.
                         Defaults to the current directory.
  -w, --workers INTEGER  The workers count. By default all cores will be used
                         (w=0).
  --db-type TEXT         The macsyfinder --db-type option. Run macsyfinder
                         --help for more details. Possible values are
                         ordered_replicon, gembase, unordered, defaults to
                         ordered_replicon.
  --preserve-raw         Preserve raw MacsyFinder outputs alongside Defense
                         Finder results inside the output directory.
  --models-dir TEXT      Specify a directory containing your models.
  --help                 Show this message and exit.
```

## Development

To install defense-finder in development mode (so when you edit a file the changes are directly visible without reinstalling), one can do :

```bash
conda create -n defensefinder_dev
conda activate defensefinder_dev
pip install -e .
defense-finder update
```

To test that changes in the code are not breaking the output, you can compare your results with the test dataset : 

```bash
defense-finder run test/df_test_prot.faa
defense-finder run test/df_test_nt.fna

#
for i in systems genes hmmer; do
echo "Verifying $i results in : " 
echo -n "Protein file :"
diff -q df_test_prot_defense_finder_$i.tsv test/expected_results/df_test_prot_defense_finder_$i.tsv && echo " > Tests OK" || echo " >> Test Failed <<"
echo -n "Nucleotide file :"
diff -q df_test_nt_defense_finder_$i.tsv test/expected_results/df_test_nt_defense_finder_$i.tsv && echo " > Tests OK" || echo " >> Test Failed <<"
echo
done
```

---
For questions: you can contact aude.bernheim@inserm.fr
