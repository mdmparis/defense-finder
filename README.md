# Defense Finder

A cli utility enabling systematic search of all anti-phage systems in proteins using MacSyData and Mdm labs models.

## Modules

### cli
The cli logic, using click.

### defense-finder
The defense finder logic, calling macsydata with the models.

### updater
Updates the defense-finder models. Called during post-install and anytime later on.
