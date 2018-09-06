#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "types.h"
#include "tools.h"


PyObject *flatten_double_PyArray(PyArrayObject *array)
{
    int i;
    int j;
    PyObject *list = NULL;
    PyObject *item = NULL;

    npy_intp *siz = NULL;
    siz = PyArray_DIMS(array);
    list = PyList_New(siz[0]*siz[1]);
    if (NULL == list) goto err;
    for (i = 0 ; i < siz[0] ; ++i) {
	for (j = 0 ; j < siz[1] ; ++j) {
	    double val_ = *((double *)PyArray_GETPTR2(array, i, j));
	    item = PyFloat_FromDouble(val_);
	    if (NULL == item) goto err;
	    if (-1 == PyList_SetItem(list, siz[1]*i+j, item)) goto err;
	}
    }

    return list;


 err:
    Py_XDECREF(list);
    Py_XDECREF(item);
    return NULL;
}


PyObject *flatten_int_PyArray(PyArrayObject *array)
{
    int i;
    int j;
    PyObject *list = NULL;
    PyObject *item = NULL;

    npy_intp *siz = NULL;
    siz = PyArray_DIMS(array);
    list = PyList_New(siz[0]*siz[1]);
    if (NULL == list) goto err;
    for (i = 0 ; i < siz[0] ; ++i) {
	for (j = 0 ; j < siz[1] ; ++j) {
	    int val_ = *((int *)PyArray_GETPTR2(array, i, j));
	    item = PyInt_FromLong(val_);
	    if (NULL == item) goto err;
	    if (-1 == PyList_SetItem(list, siz[1]*i+j, item)) goto err;
	}
    }

    return list;


 err:
    Py_XDECREF(list);
    Py_XDECREF(item);
    return NULL;
}


void *unflat_double_PyList(PyObject *list, npy_intp dim[2])
{
    int i;
    int j;
    PyObject *item = NULL;
    
    double (*EntityPtr)[dim[1]] = NULL;
    EntityPtr = malloc(dim[0] * dim[1] * sizeof(double));

    if (NULL == EntityPtr) {
	fprintf(stderr,
		"unflat_double_PyList: impossible to allocate %lu bytes of "
		"memory\n",
		dim[0] * dim[1] * sizeof(int));
	goto err;
    }

    for (i = 0 ; i < dim[0] ; ++i) {
	for (j = 0 ; j < dim[1] ; ++j) {
	    item = PyList_GetItem(list, dim[1]*i+j);
	    if (NULL == item) goto err;
	    EntityPtr[i][j] = PyFloat_AsDouble(item);;
	}
    }

    return EntityPtr;

 err:
    if (NULL != EntityPtr) {
	free(EntityPtr);
	EntityPtr = NULL;
    }

    return NULL;
}


void *unflat_int_PyList(PyObject *list, npy_intp dim[2])
{
    int i;
    int j;
    PyObject *item = NULL;
    
    int (*EntityPtr)[dim[1]] = NULL;
    EntityPtr = malloc(dim[0] * dim[1] * sizeof(int*));

    if (NULL == EntityPtr) {
	fprintf(stderr,
		"unflat_int_PyList: impossible to allocate %lu bytes of "
		"memory\n",
		dim[0] * dim[1] * sizeof(int*));
	goto err;
    }

    for (i = 0 ; i < dim[0] ; ++i) {
	for (j = 0 ; j < dim[1] ; ++j) {
	    item = PyList_GetItem(list, dim[1]*i+j);
	    if (NULL == item) goto err;
	    EntityPtr[i][j] = PyLong_AsLong(item);
	}
    }

    return EntityPtr;

 err:
    if (NULL != EntityPtr) {
	free(EntityPtr);
	EntityPtr = NULL;
    }

    return NULL;
}
