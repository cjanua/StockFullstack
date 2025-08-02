# flake.nix

{
  description = "StockFullstack Environment";

  inputs = {
    # nixpkgs.url = "tarball+https://nixos.org/channels/nixos-25.05/nixexprs.tar.xz";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        buildInputs = with pkgs; [
          nodejs_24 # For Next.js application
          nodePackages.npm
          sqlite      # SQLite database
          toml-cli
        ];
        pythonPkgs = ps: with ps; [
          # fastapi
          uvicorn
          # pandas
          # numpy
          # scipy
          # yfinance
          # torch
          # redis
          # tabulate
          # seaborn
          # python-dotenv
          # scikit-learn
          # # Add packages that can be managed by nix; others via pip
        ];
        tools = with pkgs; [
          visidata    # For viewing SQLite database
          git         # Version control
          curl        # Basic utility for fetching
        ];
        sys = with pkgs; [
          stdenv.cc.cc.lib
          pkg-config
        ];
      in {
        devShells.default = pkgs.mkShell {

          buildInputs = with pkgs; [
            (python310.withPackages pythonPkgs)
            python310Packages.venvShellHook          
          ] ++ buildInputs ++ tools ++ sys;
          venvDir = "./.venv";
          postVenvCreation = ''
            pip install -r requirements.txt
          '';

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib]}:$LD_LIBRARY_PATH
            export PKG_CONFIG_PATH=${pkgs.sqlite.dev}/lib/pkgconfig:$PKG_CONFIG_PATH

            echo "Env's been setup!"
          '';
        };
      });
}