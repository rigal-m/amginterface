#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "amgio.h"
#include "amgio_py.h"
#include "../tools.h"


int py_ConvertSU2toLibmeshb( char *MshNam, char *SolNam, char *OutNam ) 
{
    char error_msg[BUFSIZE];
    Options *mshopt = AllocOptions();

    strcpy(mshopt->OutNam,OutNam);
    strcpy(mshopt->InpNam,MshNam);
    if (NULL != SolNam) strcpy(mshopt->SolNam,SolNam);

    mshopt->clean = 0; /* remove unconnected vertices */

    if ( !CheckOptions(mshopt) ) {
	sprintf(error_msg, "py_ConvertSU2toLibmeshb: mesh options incorrect\n");
	handle_err_(PyExc_RuntimeError, error_msg, err);
    }

    return ConvertSU2SolToGMF (mshopt);


 err:
    return 0;
}


int py_ConvertLibmeshbtoSU2( char *MshNam, char *SolNam, char *OutNam ) 
{
    char error_msg[BUFSIZE];
    Options *mshopt = AllocOptions();
	
    strcpy(mshopt->OutNam,OutNam);
    strcpy(mshopt->InpNam,MshNam);
    strcpy(mshopt->SolNam,SolNam);
	
    mshopt->clean = 0; /* remove unconnected vertices */
	
    if ( !CheckOptions(mshopt) ) {
	sprintf(error_msg, "py_ConvertLibmeshbtoSU2: mesh options incorrect\n");
	handle_err_(PyExc_RuntimeError, error_msg, err);
    }
	
    return ConvertGMFtoSU2Sol (mshopt);
	
	
    return 1;


 err:
    return 0;
}


int py_SplitSolution(char *SolNam, int dim, char *prefix, char *adap_sensor)
{
    int SizMsh[GmfMaxSizMsh+1];
    memset(SizMsh,0,sizeof(int)*(GmfMaxSizMsh+1));

    Mesh *Msh = AllocMesh(SizMsh);

    Msh->NbrVer = GetSU2SolSize(SolNam);

    LoadSU2Solution(SolNam, Msh);

    Msh->Dim = dim;
    SplitSolution(Msh, prefix, adap_sensor);
}


int py_ReadMesh__ (char *MshNam, char *SolNam, PyObject *pyVer,
		    PyObject *pyTri, PyObject *pyTet, PyObject *pyEdg,
		    PyObject *pyHex, PyObject *pyQua, PyObject *pyPyr,
		    PyObject *pyPri, PyObject *pySol, PyObject *pySolHeader,
		    PyObject *pyMarkers)
{
    int i, j;

    Options *mshopt = AllocOptions();

    strcpy(mshopt->InpNam,MshNam);
    strcpy(mshopt->SolNam,SolNam);

    /* Open mesh/solution file */

    Mesh *Msh = NULL;
    Msh = SetupMeshAndSolution (mshopt->InpNam, mshopt->SolNam);

    for (i=1; i<=Msh->NbrVer; ++i){
	for (j=0; j<3; ++j)
	    if (-1 == PyList_Append(pyVer, PyFloat_FromDouble(Msh->Ver[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrTri; ++i){
	for (j=0; j<4; ++j)
	    if (-1 == PyList_Append(pyTri, PyInt_FromLong(Msh->Tri[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrTet; ++i){
	for (j=0; j<5; ++j)
	    if (-1 == PyList_Append(pyTet, PyInt_FromLong(Msh->Tet[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrEfr; ++i){
	for (j=0; j<3; ++j)
	    if (-1 == PyList_Append(pyEdg, PyInt_FromLong(Msh->Efr[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrHex; ++i){
	for (j=0; j<9; ++j)
	    if (-1 == PyList_Append(pyHex, PyInt_FromLong(Msh->Hex[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrQua; ++i){
	for (j=0; j<5; ++j)
	    if (-1 == PyList_Append(pyQua, PyInt_FromLong(Msh->Qua[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrPyr; ++i){
	for (j=0; j<6; ++j)
	    if (-1 == PyList_Append(pyPyr, PyInt_FromLong(Msh->Pyr[i][j])))
		goto err;
    }

    for (i=1; i<=Msh->NbrPri; ++i) {
	for (j=0; j<7; ++j)
	    if (-1 == PyList_Append(pyPri, PyInt_FromLong(Msh->Pri[i][j])))
		goto err;
    }

    /* First row of Markers contains dimension */
    if (-1 == PyList_Append(pyMarkers, PyInt_FromLong(Msh->Dim))) goto err;
    for (i=1; i<=Msh->NbrMarkers; ++i){
	if (-1 ==
	    PyList_Append(pyMarkers, PyString_FromString(Msh->Markers[i])))
	    goto err;
    }

    for (i=0; i<=Msh->SolSiz; i++){
	if (-1 ==
	    PyList_Append(pySolHeader, PyString_FromString(Msh->SolTag[i])))
	    goto err;
    }
	
    if ( Msh->Sol ) {
		
	/* Output solution */
	int iVer;
	for (iVer=1; iVer<=Msh->NbrVer; iVer++) {
	    for (i=0; i<Msh->SolSiz; i++) {
		if (-1 ==
		    PyList_Append(
		        pySol,
			PyFloat_FromDouble(Msh->Sol[iVer*Msh->SolSiz+i])
		    ))
		    goto err;
	    }
	}
		
    }

    if ( Msh )
	FreeMesh(Msh);


    return 1;
    

 err:
    return 0;
}


PyObject *py_GetMeshToDict__ (PyObject *mesh_name, PyObject *solution_name)
{
    _import_array();
    
    PyObject *mesh = NULL;
    PyObject *item = NULL;
    PyObject *item_ = NULL;
    PyObject *value = NULL;
    PyObject *dimension  = NULL;

    npy_intp dim[2];
    int i;

    /* Converting the Python names into strings */
    char *mesh_name_ = NULL;
    char *solution_name_ = NULL;

    mesh_name_ = PyString_AsString(mesh_name);
    solution_name_ = PyString_AsString(solution_name);

    if (NULL == mesh_name_ || NULL == solution_name_) return NULL;

    /* Initializing the lists of elements */
    PyObject *VerList     = NULL;
    PyObject *TriList     = NULL;
    PyObject *TetList     = NULL;
    PyObject *EdgList     = NULL;
    PyObject *HexList     = NULL;
    PyObject *QuaList     = NULL;
    PyObject *PyrList     = NULL;
    PyObject *PriList     = NULL;
    PyObject *SolList     = NULL;
    PyObject *SolTagList  = NULL;
    PyObject *MarkersList = NULL;

    PyObject *TetArray = NULL;
    PyObject *TriArray = NULL;
    PyObject *EdgArray = NULL;
    PyObject *HexArray = NULL;
    PyObject *QuaArray = NULL;
    PyObject *PyrArray = NULL;
    PyObject *PriArray = NULL;
    PyObject *VerArray = NULL;
    PyObject *SolArray = NULL;

    int (*TetPtr)[5] = NULL;
    int (*TriPtr)[4] = NULL;
    int (*EdgPtr)[3] = NULL;
    int (*HexPtr)[9] = NULL;
    int (*QuaPtr)[5] = NULL;
    int (*PyrPtr)[6] = NULL;
    int (*PriPtr)[7] = NULL;
    int (*VerPtr)[3] = NULL;

    VerList     = PyList_New(0);
    TriList     = PyList_New(0);
    TetList     = PyList_New(0);
    EdgList     = PyList_New(0);
    HexList     = PyList_New(0);
    QuaList     = PyList_New(0);
    PyrList     = PyList_New(0);
    PriList     = PyList_New(0);
    SolList     = PyList_New(0);
    SolTagList  = PyList_New(0);
    MarkersList = PyList_New(0);

    if (NULL == VerList || NULL == TriList || NULL == TetList ||
	NULL == EdgList || NULL == HexList || NULL == QuaList ||
	NULL == PyrList || NULL == PriList || NULL == SolList ||
	NULL == SolTagList || NULL == MarkersList)
	goto err;

    /* Reading the mesh and solution */
    if (!py_ReadMesh__ (mesh_name_, solution_name_, VerList, TriList, TetList,
			EdgList, HexList, QuaList, PyrList, PriList, SolList,
			SolTagList, MarkersList))
	goto err;

    dimension = PyList_GetItem(MarkersList, 0);
    if (NULL == dimension) goto err;


    {

	/* Determining the number of elements */
	int NbrTet = PyList_Size(TetList)/5;
	int NbrTri = PyList_Size(TriList)/4;
	int NbrEdg = PyList_Size(EdgList)/3;
	int NbrHex = PyList_Size(HexList)/9;
	int NbrQua = PyList_Size(QuaList)/5;
	int NbrPyr = PyList_Size(PyrList)/6;
	int NbrPri = PyList_Size(PriList)/7;
	int NbrVer = PyList_Size(VerList)/3;

	int SolSiz = PyList_Size(SolList)/NbrVer;
	double (*SolPtr)[SolSiz] = NULL;

	/* Extracting the tetrahdra from Python list as an array */
	if (0 < NbrTet) {
	    dim[0] = NbrTet;
	    dim[1] = 5;
	    TetPtr = unflat_int_PyList(TetList, dim);
	    Py_DECREF(TetList);
	    if (NULL == TetPtr) goto err;
	    TetArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, TetPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)TetArray, NPY_ARRAY_OWNDATA);
	    if (NULL == TetArray) goto err;
	}
	else {
	    Py_DECREF(TetList);
	    dim[0] = 0;
	    dim[1] = 0;
	    TetArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)TetArray, NPY_ARRAY_OWNDATA);
	    if (NULL == TetArray) goto err;
	}

	/* Extracting the triangles from Python list as an array */
	if (0 < NbrTri) {
	    dim[0] = NbrTri;
	    dim[1] = 4;
	    TriPtr = unflat_int_PyList(TriList, dim);
	    Py_DECREF(TriList);
	    if (NULL == TriPtr) goto err;
	    TriArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, TriPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)TriArray, NPY_ARRAY_OWNDATA);
	    if (NULL == TriArray) goto err;
	}
	else {
	    Py_DECREF(TriList);
	    dim[0] = 0;
	    dim[1] = 0;
	    TriArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)TriArray, NPY_ARRAY_OWNDATA);
	    if (NULL == TriArray) goto err;
	}
    
	/* Extracting the edges from Python list as an array */
	if (0 < NbrEdg) {
	    dim[0] = NbrEdg;
	    dim[1] = 3;
	    EdgPtr = unflat_int_PyList(EdgList, dim);
	    Py_DECREF(EdgList);
	    if (NULL == EdgPtr) goto err;
	    EdgArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, EdgPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)EdgArray, NPY_ARRAY_OWNDATA);
	    if (NULL == EdgArray) goto err;
	}
	else {
	    Py_DECREF(EdgList);
	    dim[0] = 0;
	    dim[1] = 0;
	    EdgArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)EdgArray, NPY_ARRAY_OWNDATA);
	    if (NULL == EdgArray) goto err;
	}

	/* Extracting the hexahedra from Python list as an array */
	if (0 < NbrHex) {
	    dim[0] = NbrHex;
	    dim[1] = 9;
	    HexPtr = unflat_int_PyList(HexList, dim);
	    Py_DECREF(HexList);
	    if (NULL == HexPtr) goto err;
	    HexArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, HexPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)HexArray, NPY_ARRAY_OWNDATA);
	    if (NULL == HexArray) goto err;
	}
	else {
	    Py_DECREF(HexList);
	    dim[0] = 0;
	    dim[1] = 0;
	    HexArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)HexArray, NPY_ARRAY_OWNDATA);
	    if (NULL == HexArray) goto err;
	}

	/* Extracting the quadrilaterals from Python list as an array */
	if (0 < NbrQua) {
	    dim[0] = NbrQua;
	    dim[1] = 5;
	    QuaPtr = unflat_int_PyList(QuaList, dim);
	    Py_DECREF(QuaList);
	    if (NULL == QuaPtr) goto err;
	    QuaArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, QuaPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)QuaArray, NPY_ARRAY_OWNDATA);
	    if (NULL == QuaArray) goto err;
	}
	else {
	    Py_DECREF(QuaList);
	    dim[0] = 0;
	    dim[1] = 0;
	    QuaArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)QuaArray, NPY_ARRAY_OWNDATA);
	    if (NULL == QuaArray) goto err;
	}

	/* Extracting the pyramids from Python list as an array */
	if (0 < NbrPyr) {
	    dim[0] = NbrPyr;
	    dim[1] = 6;
	    PyrPtr = unflat_int_PyList(PyrList, dim);
	    Py_DECREF(PyrList);
	    if (NULL == PyrPtr) goto err;
	    PyrArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, PyrPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)PyrArray, NPY_ARRAY_OWNDATA);
	    if (NULL == PyrArray) goto err;
	}
	else {
	    Py_DECREF(PyrList);
	    dim[0] = 0;
	    dim[1] = 0;
	    PyrArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)PyrArray, NPY_ARRAY_OWNDATA);
	    if (NULL == PyrArray) goto err;
	}

	/* Extracting the prisms from Python list as an array */
	if (0 < NbrPri) {
	    dim[0] = NbrPri;
	    dim[1] = 7;
	    PriPtr = unflat_int_PyList(PriList, dim);
	    Py_DECREF(PriList);
	    if (NULL == PriPtr) goto err;
	    PriArray = PyArray_SimpleNewFromData(2, dim, NPY_INT, PriPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)PriArray, NPY_ARRAY_OWNDATA);
	    if (NULL == PriArray) goto err;
	}
	else {
	    Py_DECREF(PriList);
	    dim[0] = 0;
	    dim[1] = 0;
	    PriArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)PriArray, NPY_ARRAY_OWNDATA);
	    if (NULL == PriArray) goto err;
	}
    
	/* Extracting the vertices from Python list as an array */
	if (0 < NbrVer) {
	    dim[0] = NbrVer;
	    dim[1] = 3;
	    VerPtr = unflat_double_PyList(VerList, dim);
	    Py_DECREF(VerList);
	    if (NULL == VerPtr) goto err;
	    VerArray = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, VerPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)VerArray, NPY_ARRAY_OWNDATA);
	    if (NULL == VerArray) goto err;
	}
	else {
	    Py_DECREF(VerList);
	    dim[0] = 0;
	    dim[1] = 0;
	    VerArray = PyArray_SimpleNew(2, dim, NPY_DOUBLE);
	    PyArray_ENABLEFLAGS((PyArrayObject *)VerArray, NPY_ARRAY_OWNDATA);
	    if (NULL == VerArray) goto err;
	}
    
	/* Extracting the solution from Python list as an array */
	if (0 < NbrVer) {
	    dim[0] = NbrVer;
	    dim[1] = SolSiz;
	    SolPtr = unflat_double_PyList(SolList, dim);
	    Py_DECREF(SolList);
	    if (NULL == SolPtr) goto err;
	    SolArray = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, SolPtr);
	    PyArray_ENABLEFLAGS((PyArrayObject *)SolArray, NPY_ARRAY_OWNDATA);
	    if (NULL == SolArray) goto err;
	}
	else {
	    Py_DECREF(SolList);
	    dim[0] = 0;
	    dim[1] = 0;
	    SolArray = PyArray_SimpleNew(2, dim, NPY_INT);
	    PyArray_ENABLEFLAGS((PyArrayObject *)SolArray, NPY_ARRAY_OWNDATA);
	    if (NULL == SolArray) goto err;
	}

	/* Filling the mesh dictionary */
	mesh = PyDict_New();
	if (NULL == mesh) goto err;

	if (NULL != dimension) {
	    if (-1 == PyDict_SetItemString(mesh, "dimension", dimension))
		goto err;
	    Py_DECREF(dimension);
	}
	else {
	    fprintf(stderr, "Error: no dimension was found\n");
	    goto err;
	}

	if (NULL != VerArray) {
	    if (-1 == PyDict_SetItemString(mesh, "xyz", VerArray))
		goto err;
	    /* Py_DECREF(VerArray); */ /* Bug occuring when VerArray is decref. *
	                                * If uncommented, a description of      *
					* this error will be printed in the log *
	                                * file.                                 */
	}

	if (NULL != TriArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Triangles", TriArray))
		goto err;
	    Py_DECREF(TriArray);
	}
    
	if (NULL != TetArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Tetrahedra", TetArray))
		goto err;
	    Py_DECREF(TetArray);
	}
    
	if (NULL != EdgArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Edges", EdgArray))
		goto err;
	    Py_DECREF(EdgArray);
	}

	if (NULL != HexArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Hexahedra", HexArray))
		goto err;
	    Py_DECREF(HexArray);
	}

	if (NULL != QuaArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Quadrilaterals", QuaArray))
		goto err;
	    Py_DECREF(QuaArray);
	}

	if (NULL != PyrArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Pyramids", PyrArray))
		goto err;
	    Py_DECREF(PyrArray);
	}

	if (NULL != PriArray) {
	    if (-1 == PyDict_SetItemString(mesh, "Prisms", PriArray))
		goto err;
	    Py_DECREF(PriArray);
	}

	if (NULL != SolArray){ 
	    if (-1 == PyDict_SetItemString(mesh, "solution", SolArray))
		goto err;
	    Py_DECREF(SolArray);
	}

	if (NULL != SolTagList) {
	    if (-1 == PyDict_SetItemString(mesh, "solution_tag", SolTagList))
		goto err;
	    Py_DECREF(SolTagList);
	}

	if (NULL != MarkersList) {
	    if (-1 == PyDict_SetItemString(mesh, "markers", MarkersList))
		goto err;
	    Py_DECREF(MarkersList);
	}


	item = PyList_New(0);
	if (NULL == item) goto err;
	if (-1 == PyDict_SetItemString(mesh, "Corners", item)) goto err;
	Py_DECREF(item);

	item = PyDict_New();
	if (NULL == item) goto err;
	int SolTag_len = PyList_Size(SolTagList);
	for (i = 0 ; i < SolTag_len ; ++i) {
	    item_ = PyList_GetItem(SolTagList, i);
	    if (NULL == item_) goto err;
	    value = Py_BuildValue("i", i);
	    if (NULL == value) goto err;
	    if (-1 == PyDict_SetItem(item, item_, value)) goto err;
	    Py_DECREF(item_);
	}
	if (-1 == PyDict_SetItemString(mesh, "id_solution_tag", item))
	    goto err;
	Py_DECREF(item);
	
    }


    return mesh;


 err:
    Py_XDECREF(VerList);
    Py_XDECREF(TriList);
    Py_XDECREF(TetList);
    Py_XDECREF(EdgList);
    Py_XDECREF(HexList);
    Py_XDECREF(QuaList);
    Py_XDECREF(PyrList);
    Py_XDECREF(PriList);
    Py_XDECREF(SolList);
    Py_XDECREF(SolTagList);
    Py_XDECREF(MarkersList);

    Py_XDECREF(VerArray);
    Py_XDECREF(TetArray);
    Py_XDECREF(TriArray);
    Py_XDECREF(EdgArray);
    Py_XDECREF(HexArray);
    Py_XDECREF(QuaArray);
    Py_XDECREF(PyrArray);
    Py_XDECREF(PriArray);
    
    Py_XDECREF(SolArray);

    Py_XDECREF(item);
    Py_XDECREF(mesh);
    Py_XDECREF(dimension);


    return NULL;
}


int py_WriteMeshFromDict__ (PyObject *mesh, PyObject *mesh_name,
			    PyObject *solution_name)
{
    _import_array();
    
    PyObject *key = NULL;
    PyObject *value = NULL;
    PyObject *item = NULL;

    npy_intp *siz = NULL;

    /* Converting the Python names into strings */
    char *mesh_name_ = NULL;
    char *solution_name_ = NULL;

    mesh_name_ = PyString_AsString(mesh_name);
    solution_name_ = PyString_AsString(solution_name);

    if (NULL == mesh_name_ || NULL == solution_name_) goto err;

    /* Initializing the lists of elements */
    PyObject *VerArray = NULL;
    PyObject *TriArray = NULL;
    PyObject *TetArray = NULL;
    PyObject *EdgArray = NULL;
    PyObject *HexArray = NULL;
    PyObject *QuaArray = NULL;
    PyObject *PyrArray = NULL;
    PyObject *PriArray = NULL;
    PyObject *SolArray = NULL;

    PyObject *TetList = NULL;
    PyObject *TriList = NULL;
    PyObject *EdgList = NULL;
    PyObject *HexList = NULL;
    PyObject *QuaList = NULL;
    PyObject *PyrList = NULL;
    PyObject *PriList = NULL;
    PyObject *VerList = NULL;
    PyObject *SolList = NULL;
    PyObject *MarkersList = NULL;

    /* We extract all the items from the mesh representation */
    int Dim = 3;
    int i;
    int j;
    
    key = PyUnicode_FromString("Triangles");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	TriArray = PyDict_GetItem(mesh, key);
	if (NULL == TriArray) goto err;
	TriList = flatten_int_PyArray((PyArrayObject *)TriArray);
	Py_DECREF(TriArray);
	if (NULL == TriList) goto err;
    }
    else {
	TriList = PyList_New(0);
	if (NULL == TriList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Tetrahedra");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	TetArray = PyDict_GetItem(mesh, key);
	if (NULL == TetArray) goto err;
	TetList = flatten_int_PyArray((PyArrayObject *)TetArray);
	Py_DECREF(TetArray);
	if (NULL == TetList) goto err;
    }
    else {
	TetList = PyList_New(0);
	if (NULL == TetList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Edges");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	EdgArray = PyDict_GetItem(mesh, key);
	if (NULL == EdgArray) goto err;
	EdgList = flatten_int_PyArray((PyArrayObject *)EdgArray);
	Py_DECREF(EdgArray);
	if (NULL == EdgList) goto err;
    }
    else {
	EdgList = PyList_New(0);
	if (NULL == EdgList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Hexahedra");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	EdgArray = PyDict_GetItem(mesh, key);
	if (NULL == EdgArray) goto err;
	HexList = flatten_int_PyArray((PyArrayObject *)HexArray);
	Py_DECREF(HexArray);
	if (NULL == HexList) goto err;
    }
    else {
	HexList = PyList_New(0);
	if (NULL == HexList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Quadrilaterals");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	QuaArray = PyDict_GetItem(mesh, key);
	if (NULL == QuaArray) goto err;
	QuaList = flatten_int_PyArray((PyArrayObject *)QuaArray);
	Py_DECREF(QuaArray);
	if (NULL == QuaList) goto err;
    }
    else {
	QuaList = PyList_New(0);
	if (NULL == QuaList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Pyramids");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	PyrArray = PyDict_GetItem(mesh, key);
	if (NULL == PyrArray) goto err;
	PyrList = flatten_int_PyArray((PyArrayObject *)PyrArray);
	Py_DECREF(PyrArray);
	if (NULL == PyrList) goto err;
    }
    else {
	PyrList = PyList_New(0);
	if (NULL == PyrList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("Prisms");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	PriArray = PyDict_GetItem(mesh, key);
	if (NULL == PriArray) goto err;
	PriList = flatten_int_PyArray((PyArrayObject *)PriArray);
	Py_DECREF(PriArray);
	if (NULL == PriList) goto err;
    }
    else {
	PriList = PyList_New(0);
	if (NULL == PriList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("solution");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	SolArray = PyDict_GetItem(mesh, key);
	if (NULL == SolArray) goto err;
	siz = PyArray_DIMS((PyArrayObject *)SolArray);
	if (1 < siz[0]) {
	    SolList = flatten_double_PyArray((PyArrayObject *)SolArray);
	}
	else {
	    SolList = PyList_New(0);
	}
	Py_DECREF(SolArray);
	if (NULL == SolList) goto err;
    }
    else {
    	SolList = PyList_New(0);
    	if (NULL == SolList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("markers");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	MarkersList = PyDict_GetItem(mesh, key);
	if (NULL == MarkersList) goto err;
    }
    else {
	MarkersList = PyList_New(0);
	if (NULL == MarkersList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("dimension");
    if (NULL == key) goto err;
    if (PyDict_Contains(mesh, key)) {
	value = PyDict_GetItem(mesh, key);
	if (NULL == value) goto err;
	Dim = PyLong_AsLong(value);
	Py_DECREF(value);
	if (-1 == Dim) goto err;
    }
    else {
	fprintf(stderr, "Error: key 'dimension' is missing\n");
	goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("xyz");
    if (NULL == key) return 0;
    if (PyDict_Contains(mesh, key)) {
	VerArray = PyDict_GetItem(mesh, key);
	if (NULL == VerArray) goto err;
	PyArray_ENABLEFLAGS((PyArrayObject *)VerArray, NPY_ARRAY_OWNDATA);
	VerList = flatten_double_PyArray((PyArrayObject *)VerArray);
	Py_DECREF(VerArray);
	if (NULL == VerList) goto err;
    }
    Py_DECREF(key);

    key = PyUnicode_FromString("xy");
    if (NULL == key) return 0;
    if (PyDict_Contains(mesh, key)) {
	VerArray = PyDict_GetItem(mesh, key);
	if (NULL == VerArray) goto err;
	PyArray_ENABLEFLAGS((PyArrayObject *)VerArray, NPY_ARRAY_OWNDATA);
	siz = PyArray_DIMS((PyArrayObject *)VerArray);
	VerList = PyList_New(siz[0]*3);
	if (NULL == VerList) goto err;
	for (i = 0 ; i < siz[0] ; ++i) {
	    for (j = 0 ; j < 2 ; ++j) {
		double val_ = *((double *)PyArray_GETPTR2((PyArrayObject *)VerArray, i, j));
		item = PyFloat_FromDouble(val_);
		if (NULL == item) goto err;
		if (-1 == PyList_SetItem(VerList, 3*i+j, item)) goto err;
	    }
	    item = PyFloat_FromDouble(0.);
	    if (NULL == item) goto err;
	    if (-1 == PyList_SetItem(VerList, 3*i+2, item)) goto err;
	}
	Py_DECREF(VerArray);
    }
    Py_DECREF(key);

    if (!py_WriteMesh__ (mesh_name_, solution_name_, VerList, TriList, TetList,
			EdgList, HexList, QuaList, PyrList, PriList, 
			SolList, MarkersList, Dim)) goto err;

    Py_DECREF(TetList);
    Py_DECREF(TriList);
    Py_DECREF(EdgList);
    Py_DECREF(HexList);
    Py_DECREF(QuaList);
    Py_DECREF(PyrList);
    Py_DECREF(PriList);
    Py_DECREF(VerList);
    Py_DECREF(SolList);
    Py_DECREF(VerList);

    return 1;


 err:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(item);

    Py_XDECREF(VerArray);
    Py_XDECREF(TriArray);
    Py_XDECREF(TetArray);
    Py_XDECREF(EdgArray);
    Py_XDECREF(HexArray);
    Py_XDECREF(QuaArray);
    Py_XDECREF(PyrArray);
    Py_XDECREF(PriArray);
    Py_XDECREF(SolArray);

    Py_XDECREF(TetList);
    Py_XDECREF(TriList);
    Py_XDECREF(EdgList);
    Py_XDECREF(HexList);
    Py_XDECREF(QuaList);
    Py_XDECREF(PyrList);
    Py_XDECREF(PriList);
    Py_XDECREF(VerList);
    Py_XDECREF(SolList);
    Py_XDECREF(VerList);
    
    return 0;
}


int py_WriteSolFromDict__ (PyObject *SolDict, PyObject *solution_name)
{
    _import_array();

    PyObject *dim = NULL;    /* Dimension */
    PyObject *soltag = NULL; /* Solution tag */

    PyObject *VerList = NULL; /* Flattened list of vertices */
    PyObject *SolList = NULL; /* Flattened list of vertices */

    PyArrayObject *VerArray = NULL;
    PyArrayObject *SolArray = NULL;

    /* Converting the Python name into string */
    char *solution_name_ = NULL;
    solution_name_ = PyString_AsString(solution_name);

    if (NULL == solution_name_) goto err;

    dim = PyDict_GetItemString(SolDict, "dimension");
    if (NULL == dim) goto err;

    SolArray = (PyArrayObject *)PyDict_GetItemString(SolDict, "solution");
    if (NULL == SolArray) goto err;

    VerArray = (PyArrayObject *)PyDict_GetItemString(SolDict, "xyz");
    if (NULL == VerArray) goto err;

    soltag = PyDict_GetItemString(SolDict, "solution_tag");
    if (NULL == soltag) goto err;

    int nbVertices = *PyArray_DIMS(VerArray);
    VerList = flatten_double_PyArray(VerArray);
    Py_DECREF(VerArray);
    if (NULL == VerList) goto err;

    int nbSol = *PyArray_DIMS(SolArray);
    
    if (1 < nbSol) {
	SolList = flatten_double_PyArray(SolArray);
	Py_DECREF(SolArray);
	if (NULL == SolList) goto err;
    }
    else {
	fprintf(stderr, "## ERROR write_solution: No solution.\n");
	goto err;
    }

    int dim_ = PyLong_AsLong(dim);

    py_WriteSolution__ (solution_name_, VerList, SolList, soltag, nbVertices, dim_);

    return 1;


 err:
    Py_XDECREF(dim);
    Py_XDECREF(soltag);
    Py_XDECREF(VerList);
    Py_XDECREF(VerArray);
    Py_XDECREF(SolArray);
    return 0;
}


PyObject *py_CreateSensor__ (PyObject *SolDict, PyObject *sensor)
{
    _import_array();

    PyObject *str = NULL;
    
    PyObject *dim = NULL;

    PyObject *VerList = NULL;
    PyObject *SensorHeader = NULL;
    PyObject *SensorWrap = NULL;
    PyObject *SolTag = NULL;
    PyObject *iMach = NULL;
    PyObject *iPres = NULL;

    PyArrayObject *VerArray = NULL;
    PyArrayObject *SolArray = NULL;
    PyArrayObject *SensorArray = NULL;

    npy_intp arrayDims[2];
    int i;

    VerArray = (PyArrayObject *)PyDict_GetItemString(SolDict, "xyz");
    if (NULL == VerArray) goto err;

    VerList = flatten_double_PyArray(VerArray);
    Py_DECREF(VerArray);
    if (NULL == VerList) goto err;

    dim = PyDict_GetItemString(SolDict, "dimension");
    if (NULL == dim) goto err;

    SolArray = (PyArrayObject *)PyDict_GetItemString(SolDict, "solution");
    if (NULL == SolArray) goto err;

    str = Py_BuildValue("s", "MACH");
    if (NULL == str) goto err;
    if (PyObject_RichCompareBool(str, sensor, Py_EQ)) {
    	SolTag = PyDict_GetItemString(SolDict, "id_solution_tag");
    	if (NULL == SolTag) goto err;
    	iMach = PyDict_GetItemString(SolTag, "Mach");
    	if (NULL == iMach) goto err;
    	arrayDims[0] = PyArray_DIMS(SolArray)[0];
    	arrayDims[1] = 1;
    	SensorArray = (PyArrayObject *)PyArray_SimpleNew(2, arrayDims, NPY_DOUBLE);
    	if (NULL == SensorArray) goto err;
    	int j = PyLong_AsLong(iMach);
    	Py_DECREF(iMach);
    	for (i = 0 ; i < arrayDims[0] ; ++i) {
    	    double val = *((double *)PyArray_GETPTR2(SolArray, i, j));
    	    *((double *)PyArray_GETPTR2(SensorArray, i, 0)) = val;
    	}
    	SensorHeader = PyList_New(1);
    	if (NULL == SensorHeader) goto err;
    	if (-1 == PyList_SetItem(SensorHeader, 0, Py_BuildValue("s", "Mach"))) goto err;
    }

    str = Py_BuildValue("s", "PRES");
    if (NULL == str) goto err;
    if (PyObject_RichCompareBool(str, sensor, Py_EQ)) {
    	SolTag = PyDict_GetItemString(SolDict, "id_solution_tag");
    	if (NULL == SolTag) goto err;
    	iPres = PyDict_GetItemString(SolTag, "Pressure");
    	if (NULL == iPres) goto err;
    	arrayDims[0] = PyArray_DIMS(SolArray)[0];
    	arrayDims[1] = 1;
    	SensorArray = (PyArrayObject *)PyArray_SimpleNew(2, arrayDims, NPY_DOUBLE);
    	if (NULL == SensorArray) goto err;
    	int j = PyLong_AsLong(iPres);
    	Py_DECREF(iPres);
    	for (i = 0 ; i < arrayDims[0] ; ++i) {
    	    double val = *((double *)PyArray_GETPTR2(SolArray, i, j));
    	    *((double *)PyArray_GETPTR2(SensorArray, i, 0)) = val;
    	}
    	SensorHeader = PyList_New(1);
    	if (NULL == SensorHeader) goto err;
    	if (-1 == PyList_SetItem(SensorHeader, 0, Py_BuildValue("s", "Pres"))) goto err;
    }

    str = Py_BuildValue("s", "MACH_PRES");
    if (NULL == str) goto err;
    if (PyObject_RichCompareBool(str, sensor, Py_EQ)) {
    	SolTag = PyDict_GetItemString(SolDict, "id_solution_tag");
    	if (NULL == SolTag) goto err;
    	iMach = PyDict_GetItemString(SolTag, "Mach");
    	if (NULL == iMach) goto err;
    	iPres = PyDict_GetItemString(SolTag, "Pressure");
    	if (NULL == iPres) goto err;
    	arrayDims[0] = PyArray_DIMS(SolArray)[0];
    	arrayDims[1] = 2;
    	SensorArray = (PyArrayObject *)PyArray_SimpleNew(2, arrayDims, NPY_DOUBLE);
    	if (NULL == SensorArray) goto err;
    	int j = PyLong_AsLong(iMach);
    	int k = PyLong_AsLong(iPres);
    	Py_DECREF(iMach);
    	Py_DECREF(iPres);
    	for (i = 0 ; i < arrayDims[0] ; ++i) {
    	    double val = *((double *)PyArray_GETPTR2(SolArray, i, j));
    	    *((double *)PyArray_GETPTR2(SensorArray, i, 0)) = val;
    	    val = *((double *)PyArray_GETPTR2(SolArray, i, k));
    	    *((double *)PyArray_GETPTR2(SensorArray, i, 1)) = val;
    	}
    	SensorHeader = PyList_New(2);
    	if (NULL == SensorHeader) goto err;
    	if (-1 == PyList_SetItem(SensorHeader, 0, Py_BuildValue("s", "Mach"))) goto err;
    	if (-1 == PyList_SetItem(SensorHeader, 1, Py_BuildValue("s", "Pres"))) goto err;
    }

    if (NULL == SensorHeader) {
	fprintf(stderr, "## ERROR : Unknown sensor.\n");
	goto err;
    }

    SensorWrap = PyDict_New();
    if (NULL == SensorWrap) goto err;

    if (-1 == PyDict_SetItemString(SensorWrap, "solution_tag", SensorHeader)) goto err;
    if (-1 == PyDict_SetItemString(SensorWrap, "xyz",
    				   PyDict_GetItemString(SolDict, "xyz"))) goto err;
    if (-1 == PyDict_SetItemString(SensorWrap, "dimension",
    				   PyDict_GetItemString(SolDict, "dimension"))) goto err;
    if (-1 == PyDict_SetItemString(SensorWrap, "solution", (PyObject *)SensorArray)) goto err;

    return SensorWrap;


 err:
    Py_XDECREF(str);
    Py_XDECREF(dim);
    Py_XDECREF(VerList);
    Py_XDECREF(SensorHeader);
    Py_XDECREF(SolTag);
    Py_XDECREF(iMach);
    Py_XDECREF(iPres);
    Py_XDECREF(VerArray);
    Py_XDECREF(SolArray);
    Py_XDECREF(SensorArray);
    return NULL;
}


int py_WriteMesh__ (char *MshNam, char *SolNam, PyObject *pyVer,
		    PyObject *pyTri, PyObject *pyTet, PyObject *pyEdg,
		    PyObject *pyHex, PyObject *pyQua, PyObject *pyPyr,
		    PyObject *pyPri, PyObject *pySol, PyObject *pyMarkers,
		    int Dim)
{
    int i, j;
    Mesh *Msh= NULL;
    int SizMsh[GmfMaxKwd+1];
	
    int is[5], siz, ref, idx;
    double crd[3];
	
    int NbrMarkers = 0;
	
    for (i=0; i<GmfMaxKwd; i++)
	SizMsh[i] = 0;
	
    /* Get mesh size */

    if ( PyList_Check(pyVer) )
	SizMsh[GmfVertices] = PyList_Size(pyVer);
	
    if ( PyList_Check(pyTri) )
	SizMsh[GmfTriangles] = PyList_Size(pyTri);
	
    if ( PyList_Check(pyTet) )
	SizMsh[GmfTetrahedra] = PyList_Size(pyTet);
	
    if ( PyList_Check(pyEdg) )
	SizMsh[GmfEdges] = PyList_Size(pyEdg);

    if ( PyList_Check(pyHex) )
	SizMsh[GmfHexahedra] = PyList_Size(pyHex);
	
    if ( PyList_Check(pyQua) )
	SizMsh[GmfQuadrilaterals] = PyList_Size(pyQua);
	
    if ( PyList_Check(pyPyr) )
	SizMsh[GmfPyramids] = PyList_Size(pyPyr);
	
    if ( PyList_Check(pyPri) )
	SizMsh[GmfPrisms] = PyList_Size(pyPri);

    if ( PyList_Check(pyMarkers) )
	NbrMarkers = PyList_Size(pyMarkers);
	
    /* Allocate mesh */
	
    Msh = AllocMesh(SizMsh);
	
    Msh->Dim = Dim;
	
    /* Fill mesh */
	
    if ( PyList_Check(pyTri) )
	{
	    siz = PyList_Size(pyTri);
			
	    for (i=0; i<siz/4; i++)
		{
		    idx = 4*i;
				
		    for (j=0; j<3; j++) {
			PyObject *oo = PyList_GetItem(pyTri,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyTri,idx+3);
		    ref = (int) PyInt_AS_LONG(oo);
				
		    Msh->NbrTri++;
		    AddTriangle(Msh,Msh->NbrTri,is,ref);
				
		    /* printf("-- Add tri %d : %d %d %d (ref %d)\n", Msh->NbrTri, is[0], is[1], is[2], ref); */
		    /* exit(1); */
		}
	}
	
    if ( PyList_Check(pyTet) )
	{
	    siz = PyList_Size(pyTet);
			
	    for (i=0; i<siz/5; i++)
		{
		    idx = 5*i;
				
		    for (j=0; j<5; j++) {
			PyObject *oo = PyList_GetItem(pyTet,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    Msh->NbrTet++;
		    AddTetrahedron(Msh,Msh->NbrTet,is,is[4]);
				
		}
	}
		
    if ( PyList_Check(pyEdg) )
	{
	    siz = PyList_Size(pyEdg);
			
	    for (i=0; i<siz/3; i++)
		{
		    idx = 3*i;
				
		    for (j=0; j<2; j++) {
			PyObject *oo = PyList_GetItem(pyEdg,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyEdg,idx+2);
		    ref = (int) PyInt_AS_LONG(oo);
				
		    Msh->NbrEfr++;
		    AddEdge(Msh,Msh->NbrEfr,is,ref);
		}
	}
	
	
    if ( PyList_Check(pyHex) )
	{
	    siz = PyList_Size(pyHex);
			
	    for (i=0; i<siz/9; i++)
		{
		    idx = 9*i;
				
		    for (j=0; j<8; j++) {
			PyObject *oo = PyList_GetItem(pyHex,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyHex,idx+8);
		    ref = (int) PyInt_AS_LONG(oo);
				
		    Msh->NbrHex++;
		    AddHexahedron(Msh,Msh->NbrHex,is,ref);
		}
	}
	
    if ( PyList_Check(pyQua) )
	{
	    siz = PyList_Size(pyQua);
			
	    for (i=0; i<siz/5; i++)
		{
		    idx = 5*i;
				
		    for (j=0; j<4; j++) {
			PyObject *oo = PyList_GetItem(pyQua,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyQua,idx+4);
		    ref = (int) PyInt_AS_LONG(oo);
								
		    Msh->NbrQua++;
		    AddQuadrilateral(Msh,Msh->NbrQua,is,ref);
		}
	}
		
		
    if ( PyList_Check(pyPyr) )
	{
	    siz = PyList_Size(pyPyr);
			
	    for (i=0; i<siz/6; i++)
		{
		    idx = 6*i;
				
		    for (j=0; j<5; j++) {
			PyObject *oo = PyList_GetItem(pyPyr,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyPyr,idx+5);
		    ref = (int) PyInt_AS_LONG(oo);
				
		    Msh->NbrPyr++;
		    AddPyramid(Msh,Msh->NbrPyr,is,ref);
		}
	}
	
    if ( PyList_Check(pyPri) )
	{
	    siz = PyList_Size(pyPri);
			
	    for (i=0; i<siz/7; i++)
		{
		    idx = 7*i;
				
		    for (j=0; j<6; j++) {
			PyObject *oo = PyList_GetItem(pyPri,idx+j);
			if ( PyInt_Check(oo) )
			    {
				is[j] = (int) PyInt_AS_LONG(oo);
			    }
		    }
				
		    PyObject *oo = PyList_GetItem(pyPri,idx+6);
		    ref = (int) PyInt_AS_LONG(oo);
				
		    Msh->NbrPri++;
		    AddPrism(Msh,Msh->NbrPri,is,ref);
		}
	}	
	
    if ( PyList_Check(pyVer) )
	{
	    siz = PyList_Size(pyVer);
			
	    for (i=0; i<siz/3; i++)
		{
		    idx = 3*i;
				
		    for (j=0; j<3; j++) {
			PyObject *oo = PyList_GetItem(pyVer,idx+j);
			if ( PyFloat_Check(oo) )
			    {
				crd[j] = (double) PyFloat_AS_DOUBLE(oo);
			    }
		    }
		    Msh->NbrVer++;
		    AddVertex(Msh,Msh->NbrVer,crd);
				
		    /* printf("ADD VERTEX %d : %lf %lf %lf\n", Msh->NbrVer, crd[0], crd[1], crd[2]); */
		    /* exit(1); */
		}
	}
	
    if ( PyList_Check(pyMarkers) )
	{
	    for (i=0; i<NbrMarkers; i++){
		PyObject *oo = PyList_GetItem(pyMarkers,i);
		strcpy(Msh->Markers[i], (char*) PyString_AS_STRING(oo));	
	    }
	    Msh->NbrMarkers = NbrMarkers;
	}
	
	
    /* Get Solution size and check it matches the number of vertices */
	
    if ( PyList_Check(pySol) )
	siz = PyList_Size(pySol);
	
    if ( siz > 0 ) {
			
	if ( siz%Msh->NbrVer == 0 ) {
			
	    Msh->SolSiz = siz/Msh->NbrVer;
	    Msh->NbrFld = Msh->SolSiz;
	    Msh->FldTab = (int*) malloc(sizeof(int)*Msh->SolSiz);
	    for (j=0; j<Msh->NbrFld; j++){
		Msh->FldTab[j] = GmfSca;
		sprintf(Msh->SolTag[j], "scalar_%d", j);
	    }
	    Msh->Sol = (double*) malloc(sizeof(double)*(Msh->NbrVer+1)*Msh->SolSiz);
	    memset(Msh->Sol, 0, sizeof(double)*(Msh->NbrVer+1)*Msh->SolSiz);
			
			
	    Msh->Sol[0] = 0.0;
	    for (i=0; i<siz; i++)
		{
		    PyObject *oo = PyList_GetItem(pySol,i);
		    if ( PyFloat_Check(oo) )
			{
			    Msh->Sol[i+Msh->SolSiz] = (double) PyFloat_AS_DOUBLE(oo);
			}
		}
	}
	else {
	    printf("  ## ERROR py_WriteMesh: Inconsistent solution provided. Skip.\n");
	}
		
    }
	
    /* Write Mesh */
	
    int FilTyp = GetInputFileType(MshNam);
    char *ptr = NULL;
    char BasNam[1024], BasNamSol[1024], OutSol[1024];
	
    /* Get BasNam */
	
    strcpy(BasNam,MshNam);

    ptr = strstr(BasNam,".su2");	
    if ( ptr != NULL )
	BasNam[ptr-BasNam]='\0';
    ptr = strstr(BasNam,".meshb");	
    if ( ptr != NULL )
	BasNam[ptr-BasNam]='\0';
	
    strcpy(BasNamSol,SolNam);
	
    ptr = strstr(BasNamSol,".dat");	
    if ( ptr != NULL )
	BasNamSol[ptr-BasNamSol]='\0';
    ptr = strstr(BasNamSol,".solb");	
    if ( ptr != NULL )
	BasNamSol[ptr-BasNamSol]='\0';
	
    if ( FilTyp != FILE_SU2 ) {
	WriteGMFMesh(BasNam, Msh, 1);
	if ( Msh->Sol ) {
	    sprintf(OutSol, "%s.solb", BasNamSol);
	    if ( ! WriteGMFSolutionItf(OutSol, Msh) ) {
		printf("  ## ERROR : Output solution FAILED.\n");
	    }
	}
    }
    else {
	WriteSU2Mesh(BasNam, Msh);
	if ( Msh->Sol ) {		
	    sprintf(OutSol, "%s.dat", BasNamSol);
	    WriteSU2Solution (OutSol, Msh, Msh->Sol, Msh->NbrVer,  Msh->SolSiz, Msh->SolTag);
	}
    }


    return 1;
}
    

void py_WriteSolution__ (char *SolNam, PyObject *pyVer, PyObject *pySol, PyObject *pySolHeader, int NbrVer, int Dim)
{
    int siz, i, j, idx;
    int SolSiz=0, NbrFld=0, NbrTag=0;
    int *FldTab = NULL;

    double  *Sol = NULL;
    double3 *Ver = NULL;

    char SolTag[100][256];

    if ( PyList_Check(pySol) )
	siz = PyList_Size(pySol);

    if ( PyList_Check(pySolHeader) )
	NbrTag = PyList_Size(pySolHeader);

    int FilTyp = GetInputFileType(SolNam);
	
    if ( siz > 0 ) {

	if ( siz%NbrVer == 0 ) {

	    SolSiz = siz/NbrVer;
	    NbrFld = SolSiz;
	    FldTab = (int*) malloc(sizeof(int)*SolSiz);
	    for (j=0; j<NbrFld; j++){
		FldTab[j] = GmfSca;
				
		if ( NbrTag == NbrFld ) {
		    PyObject *oo = PyList_GetItem(pySolHeader,j);
		    if ( PyFloat_Check(oo) )
			{
			    sprintf(SolTag[j], "%s", (char*) PyString_AS_STRING(oo));
			}
		}
		else 
		    sprintf(SolTag[j], "scalar_%d", j);
	    }
			
	    Sol = (double*) malloc(sizeof(double)*(NbrVer+1)*SolSiz);
	    memset(Sol, 0, sizeof(double)*(NbrVer+1)*SolSiz);
			
	    Sol[0] = 0.0;
	    for (i=0; i<siz; i++)
		{
		    PyObject *oo = PyList_GetItem(pySol,i);
		    if ( PyFloat_Check(oo) )
			{
			    Sol[i+SolSiz] = (double) PyFloat_AS_DOUBLE(oo);
			}
		}
	}
	else {
	    printf("  ## ERROR py_WriteSolution: Inconsistent solution provided. Skip.\n");
	    printf("siz %d NbrVer %d -> %d\n", siz, NbrVer, siz%NbrVer);
	    exit(1);
	}
		

	if ( PyList_Check(pyVer) )
	    {
		siz = PyList_Size(pyVer);
				
		if ( NbrVer != siz/3 ) {
		    printf("  ## ERROR py_WriteSolution: Inconsistent number of vertices. Skip.\n");
		    exit(1);
		}

		Ver = (double3*) malloc(sizeof(double3)*(NbrVer+1));
			
		for (i=0; i<siz/3; i++)
		    {
			idx = 3*i;
				
			for (j=0; j<3; j++) {
			    PyObject *oo = PyList_GetItem(pyVer,idx+j);
			    if ( PyFloat_Check(oo) )
				{
				    Ver[i+1][j] = (double) PyFloat_AS_DOUBLE(oo);
				}
						
			}
					
			/* printf("ADD VERTEX %d : %lf %lf %lf\n", Msh->NbrVer, crd[0], crd[1], crd[2]); */
			/* exit(1); */
		    }
	    }
		
	if ( FilTyp == FILE_GMFSOL ) {
	    WriteGMFSolution(SolNam, Sol, SolSiz, NbrVer, Dim, NbrFld, FldTab);
	}
	else {
	    WriteSU2Solution_2 (SolNam, Dim, NbrVer, Ver, Sol, SolSiz, SolTag);
	}

    }	

    if ( Sol )
	free(Sol);

    if ( Ver )
	free(Ver);

}

