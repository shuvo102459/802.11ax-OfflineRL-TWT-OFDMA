# Registers the joint TWT + uplink OFDMA scenario with the ns-3 build.
#
# Add to contrib/ai/examples/MobiCom/twt/CMakeLists.txt:
#     add_subdirectory(802.11ax-OfflineRL-TWT-OFDMA)
#
# The scenario also runs from scratch/ -- see README.md Quick Start, which is
# the path the reported results used.
build_lib_example(
    NAME twt_ofdma_main_simulation
    SOURCE_FILES twt-ofdma-main-simulation.cc
    LIBRARIES_TO_LINK
        ${libai} ${libwifi} ${libnetwork} ${libinternet} ${libapplications}
        ${libenergy} ${libmobility} ${libflow-monitor} ${libcore} ${libstats}
        ${libpropagation} ${libspectrum}
)
