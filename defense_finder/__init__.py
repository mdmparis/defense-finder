import macsypy

def run(f):
    content = f.readlines()
    for l in content:
        print(f'test: {l}')
