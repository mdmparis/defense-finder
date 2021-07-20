from subprocess import check_call

def update_models():
    bashCommand = 'touch test.txt'
    check_call(bashCommand.split())
