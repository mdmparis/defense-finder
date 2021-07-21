# Defense Finder

A cli utility enabling systematic search of all anti-phage systems in proteins using MacSyData and Mdm labs models.

## Structure

### cli
The cli logic, using click.

### defense-finder
The defense finder logic, calling macsydata with the models.

### updater
Updates the defense-finder models. Called during post-install and anytime later on.

## Publishing

A [Github action](https://github.com/mdmparis/defense-finder/actions) is setup to trigger a package release everytime a tagged commit is pushed.

Note that you don't need to publish defense-finder everytime the models change: only changes in this repository are pertinent.

Here are the steps to follow in order to publish defense-finder:
- find the current version in the `setup` function of `setup.py`.
- get a new version number according to [semantic versionning](https://semver.org/)
- update the version un `setup.py`
- commit this change, and tag the commit with `git tag -a v<your version number> -m '<your version number> <an optional message>'
- push your commits to master
- run `git push origin v<your version>` to sync the tags
- wait for the Github actions task to complete.
- all set! install the new version with `pip install mdmparis-defense-finder==v<your-version>`
