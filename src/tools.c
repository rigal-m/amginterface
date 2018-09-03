#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "types.h"
#include "tools.h"


PyObject *flatten_double_PyArray(PyArrayObject *array)
{
    PyObject *list = NULL;
    PyObject *item = NULL;

    npy_intp *siz = NULL;
    siz = PyArray_DIMS(array);
    list = PyList_New(siz[0]*siz[1]);
    if (NULL == list) goto err;
    for (int i = 0 ; i < siz[0] ; ++i) {
	for (int j = 0 ; j < siz[1] ; ++j) {
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
    PyObject *list = NULL;
    PyObject *item = NULL;

    npy_intp *siz = NULL;
    siz = PyArray_DIMS(array);
    list = PyList_New(siz[0]*siz[1]);
    if (NULL == list) goto err;
    for (int i = 0 ; i < siz[0] ; ++i) {
	for (int j = 0 ; j < siz[1] ; ++j) {
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

    for (int i = 0 ; i < dim[0] ; ++i) {
	for (int j = 0 ; j < dim[1] ; ++j) {
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

    for (int i = 0 ; i < dim[0] ; ++i) {
	for (int j = 0 ; j < dim[1] ; ++j) {
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


void *unflat_PyList(PyObject *list, npy_intp dim[2], size_t siz)
{
    PyObject *item = NULL;
    PyObject *byte_item = NULL;
    byte_t *itemPtr = NULL;

    
    byte_t (*EntityPtr)[dim[1]] = NULL;
    EntityPtr = malloc(dim[0] * dim[1] * siz);

    if (NULL == EntityPtr) {
	fprintf(stderr,
		"unflat_PyList: impossible to allocate %lu bytes of memory\n",
		dim[0] * dim[1] * siz);
	goto err;
    }

    int i, j;
    int i_ = 0;
    int j_ = 0;

    printf("Dim: (%ld, %ld)\n", dim[0], dim[1]);

    for (i = 0 ; i < dim[0] ; i = i + siz) {
	for (j = 0 ; j < dim[1] ; j = j + siz) {
	    item = PyList_GetItem(list, dim[1]*i_+j_);
	    if (NULL == item) goto err;
	    
	    byte_item = PyObject_Bytes(item);
	    if (NULL == byte_item) goto err;

	    itemPtr = PyBytes_AsString(byte_item);
	    printf("Item value: ");
	    PyObject_Print(byte_item, stdout, Py_PRINT_RAW);
	    printf("\n");
	    printf("Byte repr: ");
	    PyObject_Print(byte_item, stdout, Py_PRINT_RAW);
	    printf("\n");
	    printf(">>> %s\n\n", itemPtr);
	    EntityPtr[i][j] = *itemPtr;

	    /* int *test = NULL; */
	    /* test = PyLong_AsLong(item); */
	    /* printf("Test: %d\n", test); */

	    /* for (int k = 0 ; k < siz ; ++k) { */
	    /* 	if (0 == *(itemPtr + k)) */
	    /* 	    *(&(EntityPtr[i][j]) + k) = 0; */
	    /* 	else if (1 == *(itemPtr + k)) */
	    /* 	    *(&(EntityPtr[i][j]) + k) = 1; */
	    /* 	else printf("No: %c\n", *(itemPtr + k)); */
	    /* 	/\* *(&(EntityPtr[i][j]) + k) = *(itemPtr + k); *\/ */
	    /* } */
	    
	    ++j_;
	}
	++i_;
    }

    return EntityPtr;

 err:
    if (NULL != EntityPtr) {
	free(EntityPtr);
	EntityPtr = NULL;
    }

    return NULL;
}


/* void *unflat_PyList(PyObject *list, npy_intp dim[2], size_t siz) */
/* { */
/*     PyObject *item = NULL; */
/*     byte_t *itemPtr = NULL; */

    
/*     byte_t (*EntityPtr)[dim[1]] = NULL; */
/*     EntityPtr = malloc(dim[0] * dim[1] * siz); */

/*     if (NULL == EntityPtr) { */
/* 	fprintf(stderr, */
/* 		"Error: impossible to allocate %lu bytes of memory\n", */
/* 		dim[0] * dim[1] * siz); */
/* 	goto err; */
/*     } */

/*     long i, j; */
/*     long i_ = 0; */
/*     long j_ = 0; */

/*     printf("Dim: (%ld, %ld)\n", dim[0], dim[1]); */
/*     printf("Siz: %ld\n", (long)siz); */

/*     for (i = 0 ; i < dim[0] ; i = i + (long)siz) { */
/* 	for (j = 0 ; j < dim[1] ; j = j + (long)siz) { */
/* 	    item = PyList_GetItem(list, dim[1]*i_+j_); */
/* 	    if (NULL == item) goto err; */
	    
/* 	    itemPtr = PyLong_AsVoidPtr(item); */
/* 	    printf("itemPtr addr: %d\n\n", PyLong_AsVoidPtr(item)); */
/* 	    if (NULL == itemPtr) { */
/* 		fprintf(stderr, "Err: list item is NULL\n"); */
/* 		goto err; */
/* 	    } */
	    
/* 	    /\* for (int k = 0 ; k < siz ; ++k) { *\/ */
/* 	    /\* 	*(&(EntityPtr[i][j]) + k) = *(itemPtr + k); *\/ */
/* 	    /\* } *\/ */
/* 	    EntityPtr[i][j] = itemPtr; */

/* 	    ++j_; */
/* 	} */
/* 	++i_; */
/*     } */

/*     return EntityPtr; */

/*  err: */
/*     if (NULL != EntityPtr) { */
/* 	free(EntityPtr); */
/* 	EntityPtr = NULL; */
/*     } */

/*     return NULL; */
/* } */

