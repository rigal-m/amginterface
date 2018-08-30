#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <libmeshb7.h>

#include "amgio/amgio.h"
#include "amgio/amgio_py.h"
#include "amgio/amgio_tools.h"
#include "tools.h"



static PyObject *
pyamg_ConvertSU2toLibmeshb(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"su2_mesh", "libmeshb_mesh", "su2_sol", NULL};
    char *SU2_mesh = NULL;
    char *SU2_sol = NULL;
    char *output = NULL;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|s",
				     kwlist, &SU2_mesh, &output, &SU2_sol))
	return NULL;

    if (!py_ConvertSU2toLibmeshb(SU2_mesh, SU2_sol, output)) return NULL;
    
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_ConvertLibmeshbtoSU2(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"libmeshb_mesh", "su2_mesh", "libmeshb_sol", NULL};
    char *inria_mesh = NULL;
    char *inria_sol = NULL;
    char *output = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss|s",
				     kwlist, &inria_mesh, &output, &inria_sol))
	return NULL;

    if (!py_ConvertLibmeshbtoSU2(inria_mesh, inria_sol, output)) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_SplitSolution(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"solution", "dim", "prefix", "adap_sensor", NULL};
    char *solution = NULL;
    int dim;
    char *prefix = NULL;
    char *adap_sensor = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "siss", kwlist, &solution,
				     &dim, &prefix, &adap_sensor))
	return NULL;

    if (!py_SplitSolution(solution, dim, prefix, adap_sensor)) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_ReadMeshToLists(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
	"mesh_name", "sol_name", "vertices", "triangles", "tetrahedra", "edges",
	"hexahedra", "quadrilaterals", "pyramids", "prisms", "sol", "sol_ref",
	"markers", NULL
    };

    char *MshNam        = NULL;
    char *SolNam        = NULL;

    PyObject *pyVer     = NULL;
    PyObject *pyTri     = NULL;
    PyObject *pyTet     = NULL;
    PyObject *pyEdg     = NULL;
    PyObject *pyHex     = NULL;
    PyObject *pyQua     = NULL;
    PyObject *pyPyr     = NULL;
    PyObject *pyPri     = NULL;
    PyObject *pySol     = NULL;
    PyObject *pySolRef  = NULL;
    PyObject *pyMarkers = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOOOOOOOOO", kwlist,
				     &MshNam, &SolNam, &pyVer, &pyTri,
				     &pyTet, &pyEdg, &pyHex, &pyQua, &pyPyr,
				     &pyPri, &pySol, &pySolRef, &pyMarkers))
	return NULL;

    py_ReadMesh__ (MshNam, SolNam, pyVer, pyTri, pyTet, pyEdg, pyHex, pyQua,
		   pyPyr, pyPri, pySol, pySolRef, pyMarkers);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_WriteMeshFromLists(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
	"mesh_name", "sol_name", "vertices", "triangles", "tetrahedra", "edges",
	"hexahedra", "quadrilaterals", "pyramids", "prisms", "sol", "markers",
	"dim", NULL
    };

    char *MshNam        = NULL;
    char *SolNam        = NULL;

    PyObject *pyVer     = NULL;
    PyObject *pyTri     = NULL;
    PyObject *pyTet     = NULL;
    PyObject *pyEdg     = NULL;
    PyObject *pyHex     = NULL;
    PyObject *pyQua     = NULL;
    PyObject *pyPyr     = NULL;
    PyObject *pyPri     = NULL;
    PyObject *pySol     = NULL;
    PyObject *pyMarkers = NULL;

    int Dim;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ssOOOOOOOOOOi", kwlist,
				     &MshNam, &SolNam, &pyVer, &pyTri,
				     &pyTet, &pyEdg, &pyHex, &pyQua, &pyPyr,
				     &pyPri, &pySol, &pyMarkers, &Dim))
	return NULL;

    py_WriteMesh__ (MshNam, SolNam, pyVer, pyTri, pyTet, pyEdg, pyHex, pyQua,
		    pyPyr, pyPri, pySol, pyMarkers, Dim);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_WriteSolutionFromLists(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {
	"sol_name", "vertices", "sol", "sol_header", "nb_vertices", "dim", NULL
    };

    char *SolNam          = NULL;

    PyObject *pyVer       = NULL;
    PyObject *pySol       = NULL;
    PyObject *pySolHeader = NULL;

    int NbrVer;
    int Dim;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sOOOii", kwlist,
				     &SolNam, &pyVer, &pySol, &pySolHeader,
				     &NbrVer, &Dim))
	return NULL;

    py_WriteSolution__ (SolNam, pyVer, pySol, pySolHeader, NbrVer, Dim);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_GetMeshToDict(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"mesh_name", "sol_name", NULL};

    PyObject *MeshNam = NULL;
    PyObject *SolNam  = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
				     &MeshNam, &SolNam))
	return NULL;

    return py_GetMeshToDict__ (MeshNam, SolNam);
}


static PyObject *
pyamg_WriteMeshFromDict(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"mesh", "mesh_name", "sol_name", NULL};

    PyObject *Mesh = NULL;
    PyObject *MeshNam = NULL;
    PyObject *SolNam  = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
				     &Mesh, &MeshNam, &SolNam))
	return NULL;

    if (!PyDict_Check(Mesh)) return NULL;

    if(!py_WriteMeshFromDict__ (Mesh, MeshNam, SolNam)) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_WriteSolFromDict(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"sol", "sol_name", NULL};

    PyObject *Sol = NULL;
    PyObject *SolNam  = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
				     &Sol, &SolNam))
	return NULL;

    if (!PyDict_Check(Sol)) return NULL;

    if(!py_WriteSolFromDict__ (Sol, SolNam)) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *
pyamg_CreateSensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"sol", "sensor", NULL};

    PyObject *SolDict = NULL;
    PyObject *Sensor = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", kwlist,
				     &SolDict, &Sensor))
	return NULL;

    if (!PyDict_Check(SolDict) || !PyString_Check(Sensor)) return NULL;

    return py_CreateSensor__ (SolDict, Sensor);
}


static PyObject *
pyamg_GetMeshSizes(PyObject *self, PyObject *args)
{
    PyObject *ConfigDict = NULL;

    if (!PyArg_ParseTuple(args, "O", &ConfigDict))
	return NULL;

    if (!PyDict_Check(ConfigDict)) return NULL;

    return get_mesh_sizes__ (ConfigDict);
}


static PyObject *
pyamg_GetSubIterations(PyObject *self, PyObject *args)
{
    PyObject *ConfigDict = NULL;

    if (!PyArg_ParseTuple(args, "O", &ConfigDict))
	return NULL;

    if (!PyDict_Check(ConfigDict)) return NULL;

    return get_sub_iterations__ (ConfigDict);
}


static PyObject *
pyamg_GetResidualReduction(PyObject *self, PyObject *args)
{
    PyObject *ConfigDict = NULL;

    if (!PyArg_ParseTuple(args, "O", &ConfigDict))
	return NULL;

    if (!PyDict_Check(ConfigDict)) return NULL;

    return get_residual_reduction__ (ConfigDict);
}


static PyObject *
pyamg_GetExtIter(PyObject *self, PyObject *args)
{
    PyObject *ConfigDict = NULL;

    if (!PyArg_ParseTuple(args, "O", &ConfigDict))
	return NULL;

    if (!PyDict_Check(ConfigDict)) return NULL;

    return get_ext_iter__ (ConfigDict);
}


static PyObject *
pyamg_PrintAdapOptions(PyObject *self, PyObject *args)
{
    PyObject *ConfigDict = NULL;
    PyObject *KwdsList = NULL;

    if (!PyArg_ParseTuple(args, "OO", &ConfigDict, &KwdsList))
	return NULL;

    if (!PyDict_Check(ConfigDict) || !PyList_Check(KwdsList)) return NULL;

    return print_adap_options__ (ConfigDict, KwdsList);
}


static PyObject *
pyamg_SU2_GetDim(PyObject *self, PyObject *args)
{
    PyObject *Filename = NULL;

    if (!PyArg_ParseTuple(args, "O", &Filename))
	return NULL;

    if (!PyString_Check(Filename)) return NULL;

    return su2_get_dim__ (Filename);
}


static const char help_ConvertSU2toLibmeshb[]="\
Convert an SU2 python mesh file into a libmeshb one and write it in a file\n\
USAGE:\n\
\tamgio.su2_to_libmeshb(su2_mesh, libmeshb_mesh[, su2_sol=sol])";

static const char help_ConvertLibmeshbtoSU2[]="\
Convert a libmeshb python mesh file into a SU2 one\n\
USAGE:\n\
\tamgio.libmeshb_to_su2(libmeshb_mesh, su2_mesh[, libmeshb_sol=sol])";

static const char help_SplitSolution[]="\
Split a given solution\n\
USAGE:\n\
\tamgio.split_solution(solution, dim, prefix, adap_sensor)";

static const char help_ReadMeshToLists[]="\
Read a .mesh(b) file and store the elements into lists given as arguments\n\
USAGE:\n\
\tamgio.read_mesh_to_lists(\n\
\t\tmesh_name, sol_name, vertices, triangles, tetrahedra,\n\
\t\tedges, hexahedra, quadrilaterals, pyramids, prisms,\n\
\t\tsol, sol_ref, markers\n\
\t)";

static const char help_WriteMeshFromLists[]="\
Write a set of lists defining a mesh into a (.mesh(b) | SU2) file\n\
USAGE:\n\
\tamgio.write_mesh_from_lists(\n\
\t\tmesh_name, sol_name, vertices, triangles, tetrahedra,\n\
\t\tedges, hexahedra, quadrilaterals, pyramids, prisms,\n\
\t\tsol, markers, dim\n\
\t)";

static const char help_WriteSolutionFromLists[]="\
Write a set of lists defining a solution into a (.sol(b) | SU2) file\n\
USAGE:\n\
\tamgio.write_sol_from_lists(\n\
\t\tsol_name, vertices, sol, sol_header, nb_vertices, dim\n\
\t)";

static const char help_GetMeshToDict[]="\
Return a dictionary representation of the given mesh file \
(and optional solution)\n\
USAGE:\n\
\tmsh_dict = amgio.get_mesh_to_dict(mesh_name[, sol_name=sol])";

static const char help_WriteMeshFromDict[]="\
Write a dictionary representation of a mesh (and optional solution) to file\n\
USAGE:\n\
\tamgio.write_mesh_from_dict(mesh, mesh_name[, sol_name=sol])";

static const char help_WriteSolFromDict[]="\
Write a dictionary representation of a solution to file\n\
USAGE:\n\
\tamgio.write_sol_from_dict(sol, sol_name)";

static const char help_CreateSensor[]="\
Create a sensor given a solution dict\n\
USAGE:\n\
\tsensor_wrap_dict = amgio.create_sensor(sol, sensor)";

static const char help_GetMeshSizes[]="\
Get the mesh sizes for each AMG iteration from an SU2 config dictionary\n\
USAGE:\n\
\tsize_list = amgio.get_mesh_sizes(config_dict)";

static const char help_GetSubIterations[]="\
Get the number of subiterations required for each AMG iteration from an \
SU2 config dictionary\n\
USAGE:\n\
\tsub_iter_list = amgio.get_sub_iterations(config_dict)";

static const char help_GetResidualReduction[]="\
Get the number of residual reductions required for each AMG iteration \
from an SU2 config dictionary\n\
USAGE:\n\
\tresidual_reduction_list = amgio.get_residual_reduction(config_dict)";

static const char help_GetExtIter[]="\
Get the number of exterior iterations required for each AMG iteration \
from an SU2 config dictionary\n\
USAGE:\n\
\text_iter_list = amgio.get_ext_iter(config_dict)";

static const char help_PrintAdapOptions[]="\
Print options regarding the adaptative process\n\
USAGE:\n\
\tamgio.print_adap_options(config_dict, keword_list)";

static const char help_SU2_GetDim[]="\
Retrieve the dimension from an SU2 mesh file\n\
USAGE:\n\
\tdim = amgio.su2_get_dim(file_name)";


static const char help_amgio[]={"\
Adaptive Mesh Generation IO\n\
(2D | 3D) mesh converter between libmeshb and SU2 formats"};

/* Module method table */
static PyMethodDef amgioMethods[] = {
    
    {"su2_to_libmeshb", (PyCFunction)pyamg_ConvertSU2toLibmeshb,
     METH_VARARGS | METH_KEYWORDS, help_ConvertSU2toLibmeshb},
    
    {"libmeshb_to_su2", (PyCFunction)pyamg_ConvertLibmeshbtoSU2,
     METH_VARARGS | METH_KEYWORDS, help_ConvertLibmeshbtoSU2},
    
    {"split_solution", (PyCFunction)pyamg_SplitSolution,
     METH_VARARGS | METH_KEYWORDS, help_SplitSolution},
    
    {"read_mesh_to_lists", (PyCFunction)pyamg_ReadMeshToLists,
     METH_VARARGS | METH_KEYWORDS, help_ReadMeshToLists},
    
    {"write_mesh_from_lists", (PyCFunction)pyamg_WriteMeshFromLists,
     METH_VARARGS | METH_KEYWORDS, help_WriteMeshFromLists},
    
    {"write_sol_from_lists", (PyCFunction)pyamg_WriteSolutionFromLists,
     METH_VARARGS | METH_KEYWORDS, help_WriteSolutionFromLists},
    
    {"get_mesh_to_dict", (PyCFunction)pyamg_GetMeshToDict,
     METH_VARARGS | METH_KEYWORDS, help_GetMeshToDict},
    
    {"write_mesh_from_dict", (PyCFunction)pyamg_WriteMeshFromDict,
     METH_VARARGS | METH_KEYWORDS, help_WriteMeshFromDict},
    
    {"write_sol_from_dict", (PyCFunction)pyamg_WriteSolFromDict,
     METH_VARARGS | METH_KEYWORDS, help_WriteSolFromDict},
    
    {"create_sensor", (PyCFunction)pyamg_CreateSensor,
     METH_VARARGS | METH_KEYWORDS, help_CreateSensor},

    {"get_mesh_sizes", pyamg_GetMeshSizes,
     METH_VARARGS, help_GetMeshSizes},
    
    {"get_sub_iterations", pyamg_GetSubIterations,
     METH_VARARGS, help_GetSubIterations},
    
    {"get_residual_reduction", pyamg_GetResidualReduction,
     METH_VARARGS, help_GetResidualReduction},
    
    {"get_ext_iter", pyamg_GetExtIter,
     METH_VARARGS, help_GetExtIter},
    
    {"print_adap_options", pyamg_PrintAdapOptions,
     METH_VARARGS, help_PrintAdapOptions},
    
    {"su2_get_dim", pyamg_SU2_GetDim,
     METH_VARARGS, help_SU2_GetDim},
    
    {NULL, NULL, 0, NULL}
};



#ifdef PYTHON_3
static struct PyModuleDef amgiomodule = {
    PyModuleDef_HEAD_INIT,
    "amgio",         /* name of module */
    help_amgio,      /* module documentation */
    -1,              /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
    amgioMethods,
};

PyMODINIT_FUNC
PyInit_amgio(void)
{
    PyObject *m;
    m = PyModule_Create(&amgiomodule);
    import_array();
    if (NULL == m)
	return NULL;

    return m;
}

#else

PyMODINIT_FUNC
initamgio(void)
{
    import_array();
    (void)Py_InitModule3("amgio", amgioMethods, help_amgio);
}
#endif

