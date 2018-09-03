#ifndef AMGIO_PY_HEADER__
#define AMGIO_PY_HEADER__

int py_ConvertSU2toLibmeshb( char *MshNam, char *SolNam, char *OutNam ) ;
int py_ConvertLibmeshbtoSU2( char *MshNam, char *SolNam, char *OutNam ) ;
int py_SplitSolution(char *SolNam, int dim, char *prefix, char *adap_sensor);

int py_ReadMesh__ (char *MshNam, char *SolNam, PyObject *pyVer,
		   PyObject *pyTri, PyObject *pyTet, PyObject *pyEdg,
		   PyObject *pyHex, PyObject *pyQua, PyObject *pyPyr,
		   PyObject *pyPri, PyObject *pySol, PyObject *pySolHeader,
		   PyObject *pyMarkers);
PyObject *py_GetMeshToDict__ (PyObject *mesh_name, PyObject *solution_name);
int py_WriteMeshFromDict__ (PyObject *mesh, PyObject *mesh_name, PyObject *solution_name);
int py_WriteSolFromDict__ (PyObject *sol, PyObject *solution_name);
PyObject *py_CreateSensor__ (PyObject *SolDict, PyObject *sensor);
int py_WriteMesh__ (char *MshNam, char *SolNam, PyObject *pyVer,
		    PyObject *pyTri, PyObject *pyTet, PyObject *pyEdg,
		    PyObject *pyHex, PyObject *pyQua, PyObject *pyPyr,
		    PyObject *pyPri, PyObject *pySol, PyObject *pyMarkers,
		    int Dim);
void py_WriteSolution__ (char *SolNam, PyObject *pyVer, PyObject *pySol, PyObject *pySolHeader, int NbrVer, int Dim);

#endif // AMGIO PY HEADER
