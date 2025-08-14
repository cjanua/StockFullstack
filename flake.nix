# flake.nix

{
  description = "StockFullstack Environment";

  inputs = {
    # nixpkgs.url = "tarball+https://nixos.org/channels/nixos-25.05/nixexprs.tar.xz";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
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
          uvicorn
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
          openssl
        ];
      in {
        devShells.default = pkgs.mkShell {

          buildInputs = with pkgs; [
            # Python - Packages, Interpreter, and 
            uv
            (python313.withPackages pythonPkgs)
          ] ++ buildInputs ++ tools ++ sys;



          env = {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.openssl
            ];
            PKG_CONFIG_PATH = "${pkgs.sqlite.dev}/lib/pkgconfig:${pkgs.openssl.dev}/lib/pkgconfig";
          };

          # shellHook = ''
          #   venvDir="./.venv"
          #   if [ ! -d "$venvDir" ]; then
          #     echo "Creating venv with system-site-packages..."
          #     uv venv --system-site-packages "$venvDir"
          #     source "$venvDir/bin/activate"
          #     echo "Syncing dependencies from lock file..."
          #     uv pip sync pylock.toml
          #     uv pip install -e .
          #     deactivate
          #   fi
          #   source "$venvDir/bin/activate"
          # '';
        };
      });
}