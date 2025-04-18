{ pkgs }: {
  deps = with pkgs; [
    python312
    
    # Core system libraries
    stdenv.cc.cc.lib
    libstdcxx5

    # Make sure gcc and its runtime are available, needed for fasttext
    gcc
    gcc.cc.lib

    # Other dependencies
    bash
    hydrus
    gitFull
    glibcLocales
  ];

  env = {
    APPEND_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    NIXPKGS_ALLOW_UNFREE = "1";
    LD_LIBRARY_PATH = let
      libraries = [
        pkgs.stdenv.cc.cc.lib
        pkgs.libstdcxx5
        pkgs.gcc.cc.lib
      ];
    in "${pkgs.lib.makeLibraryPath libraries}";
  };
}