#ifndef TOOLS_HEADER__
#define TOOLS_HEADER__

#define handle_err_(err_type, msg, stamp) \
    do {				  \
	PyErr_SetString(err_type, msg);	  \
	goto stamp;			  \
    } while (0)

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyObject *flatten_double_PyArray(PyArrayObject *array);
PyObject *flatten_int_PyArray(PyArrayObject *array);
void *unflat_PyList(PyObject *list, npy_intp dim[2], size_t siz);
void *unflat_int_PyList(PyObject *list, npy_intp dim[2]);
void *unflat_double_PyList(PyObject *list, npy_intp dim[2]);

#endif // TOOLS HEADER
