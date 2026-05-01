import sys
import geouned

# path to FreeCAD lib
PATH_TO_FREECAD_LIBDIR = r"...\FreeCAD 1.0\lib"
print("PATH_TO_FREECAD_LIBDIR changed to", PATH_TO_FREECAD_LIBDIR)

# --------- arguments ----------
if len(sys.argv) < 2:
    print("Usage: python run_geouned.py CAD_file [matFile] [outFormat]")
    print("Available outFormat: openmc_xml, openmc_py, serpent, phits, mcnp")
    sys.exit(1)

CAD_file = sys.argv[1]

if len(sys.argv) > 2:
    matFile = sys.argv[2]
else:
    matFile = ""

# Available output formats
available_formats = ("openmc_xml", "openmc_py", "serpent", "phits", "mcnp")

# Output format argument (optional)
if len(sys.argv) > 3:
    outFormat_arg = sys.argv[3]
    if outFormat_arg not in available_formats:
        print(f"Unknown outFormat: '{outFormat_arg}'")
        print("Available outFormat options are:")
        for f in available_formats:
            print("  -", f)
        sys.exit(1)
    outFormats = (outFormat_arg,)
else:
    # Default output format
    outFormats = ("openmc_xml",)

# --------- options ----------
my_options = geouned.Options(
    forceCylinder=False,
    newSplitPlane=True,
    delLastNumber=False,
    enlargeBox=2,
    nPlaneReverse=0,
    splitTolerance=0,
    scaleUp=True,
    quadricPY=False,
    Facets=False,
    prnt3PPlane=False,
    forceNoOverlap=False,
)

my_settings = geouned.Settings(
    outPath=".",
    matFile=matFile,
    voidGen=True,
    debug=False,
    compSolids=False,
    simplify="no",
    exportSolids="",
    minVoidSize=200.0,
    maxSurf=50,
    maxBracket=30,
    voidMat=[],
    voidExclude=[],
    startCell=1,
    startSurf=1,
    sort_enclosure=False,
)

my_tolerances = geouned.Tolerances(
    relativeTol=False,
    relativePrecision=0.000001,
    value=0.000001,
    distance=0.0001,
    angle=0.0001,
    pln_distance=0.0001,
    pln_angle=0.0001,
    cyl_distance=0.0001,
    cyl_angle=0.0001,
    sph_distance=0.0001,
    kne_distance=0.0001,
    kne_angle=0.0001,
    tor_distance=0.0001,
    tor_angle=0.0001,
    min_area=0.01,
)

my_numeric_format = geouned.NumericFormat(
    P_abc="14.7e",
    P_d="14.7e",
    P_xyz="14.7e",
    S_r="14.7e",
    S_xyz="14.7e",
    C_r="12f",
    C_xyz="12f",
    K_xyz="13.6e",
    K_tan2="12f",
    T_r="14.7e",
    T_xyz="14.7e",
    GQ_1to6="18.15f",
    GQ_7to9="18.15f",
    GQ_10="18.15f",
)

# --------- run ----------
geo = geouned.CadToCsg(
    options=my_options,
    settings=my_settings,
    tolerances=my_tolerances,
    numeric_format=my_numeric_format,
)

geo.load_step_file(
    filename=CAD_file,
    skip_solids=[],
    spline_surfaces="stop",
)

geo.start()

geo.export_csg(
    title="Converted with GEOUNED",
    geometryName="csg",
    outFormat=outFormats,
    volSDEF=True,
    volCARD=False,
    UCARD=None,
    dummyMat=True,
    cellCommentFile=False,
    cellSummaryFile=False,
)

print("Done.")
