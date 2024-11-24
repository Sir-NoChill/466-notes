{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python310               # Specify your desired Python version
    python310Packages.virtualenv  # Virtualenv package
  ];

  shellHook = ''
    # Create the virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      virtualenv .venv
      echo "Virtual environment created."
    fi

    # Activate the virtual environment
    source .venv/bin/activate

    # Ensure pip is up-to-date in the virtual environment
    pip install --upgrade pip
    echo "Python virtual environment activated!"
  '';
}
