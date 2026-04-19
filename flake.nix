{
  description = "TSDF Fusion & Mesh Extraction Web Demo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            just
            entr
            python3
          ];

          shellHook = ''
            echo "tsdf-demo dev shell ready"
            echo "  just dev  — serve on :8080 with entr watch"
          '';
        };
      });
}
