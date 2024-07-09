from setuptools import setup
from setuptools.command.install import install

def read_md(f):
    import codecs
    with codecs.open(f, 'r', encoding='utf8') as f:
        text = f.read()
    return text


def read_req(req: str):
    return [i for i in [l.strip() for l in open(req).read().split('\n')] if i]

exec(open('defense_finder_cli/_version.py').read())

setup(name='mdmparis-defense-finder',
        version=__version__, # from 'defense_finder_cli/_version.py'
        description="Defense Finder: allow for a systematic search of all known anti-phage systems.",
        long_description=read_md('README.md'),
        long_description_content_type="text/markdown",
        author="Jean Cury",
        author_email="jean.cury@normalesup.org",
        url="https://github.com/mdmparis/defense-finder",
        license="GPLv3",
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Operating System :: POSIX',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics'
            ],
        python_requires='>=3.7',
        install_requires=read_req('requirements.txt'),
        extras_require=dict(dev=read_req('requirements-dev.txt')),
        packages=[
            'defense_finder',
            'defense_finder_cli',
            'defense_finder_updater',
            'defense_finder_posttreat'
        ],
        entry_points='''
          [console_scripts]
          defense-finder=defense_finder_cli.main:cli
        '''
      )
