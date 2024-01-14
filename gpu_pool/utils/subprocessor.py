import subprocess as sp

def train(command) -> sp.CompletedProcess[str]:
    return sp.run(command, capture_output=True, text=True, shell=True)

def handle_setup(command) -> sp.CompletedProcess[str]:
    return sp.run(command, capture_output=True, text=True, shell=True)