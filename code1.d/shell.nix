{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Define the Python version and packages
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.numpy
    pkgs.python3Packages.matplotlib
    pkgs.texliveFull
  ];

  # Environment variables
  VENV_DIR = ".venv";

  # Shell hook to set up virtual environment
  shellHook = ''
    # Check if virtual environment directory exists
    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtual environment..."
      python -m venv $VENV_DIR
    fi

    # Activate the virtual environment
    source $VENV_DIR/bin/activate

    echo "Virtual environment activated with numpy and matplotlib."
  '';
}

