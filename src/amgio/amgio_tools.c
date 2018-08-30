#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include "amgio_tools.h"
#include "../tools.h"

#if (__STDC_VERSION__ >= 199901L)
#include <stdint.h>
#endif


int is_char_in_str(char c, const char *str)
{
    int len = strlen(str);
    for (int i = 0 ; i < len ; ++i) {
	if (str[i] == c) return 1;
    }
    return 0;
}


char *str_strip(char *old, const char *remove_chars)
{
    int len = strlen(old);
    int start = 0;

    while (is_char_in_str(old[len], remove_chars)) --len;
    while (*old && is_char_in_str(*old, remove_chars)) ++old, --len;
    
    return strndup(old, len-1);
}


PyObject *str_split(char *str, char *c)
{
    PyObject *String = NULL;
    PyObject *List = NULL;
    
    List = PyList_New(0);
    if (NULL == List) goto err;

    char *token = NULL;
    char *ptr = str;
    int len = strlen(str);

    int coverd_range = 0;
    while (NULL != (token = strtok(ptr, c)) && coverd_range <= len) {
	String = PyString_FromString(token);
	if (NULL == String) goto err;
	if (-1 == PyList_Append(List, String)) goto err;
	Py_DECREF(String);
	ptr += strlen(token)+1;
	coverd_range += strlen(token)+1;
    }

    return List;


 err:
    Py_XDECREF(String);
    Py_XDECREF(List);
    return NULL;
}


char *repl_str(const char *str, const char *from, const char *to)
{
    /* Adjust each of the below values to suit your needs */

    /* Increment positions cache size initially by this number */
    size_t cache_sz_inc = 16;
    /* Thereafter, each time capacity needs to be increased,
     * multiply the increment by this factor */
    const size_t cache_sz_inc_factor = 3;
    /* But never increment capacity by more than this number */
    const size_t cache_sz_inc_max = 1048576;

    char *pret, *ret = NULL;
    const char *pstr2, *pstr = str;
    size_t i, count = 0;
#if (__STDC_VERSION__ >= 199901L)
    uintptr_t *pos_cache_tmp, *pos_cache = NULL;
#else
    ptrdiff_t *pos_cache_tmp, *pos_cache = NULL;
#endif
    size_t cache_sz = 0;
    size_t cpylen, orglen, retlen, tolen, fromlen = strlen(from);

    /* Find all matches and cache their positions */
    while ((pstr2 = strstr(pstr, from)) != NULL) {
	++count;

	/* Increase the cache size when necessary */
	if (cache_sz < count) {
	    cache_sz += cache_sz_inc;
	    pos_cache_tmp = realloc(pos_cache, sizeof(*pos_cache) * cache_sz);
	    if (pos_cache_tmp == NULL) {
		goto end_repl_str;
	    } else pos_cache = pos_cache_tmp;
	    cache_sz_inc *= cache_sz_inc_factor;
	    if (cache_sz_inc > cache_sz_inc_max) {
		cache_sz_inc = cache_sz_inc_max;
	    }
	}

	pos_cache[count-1] = pstr2 - str;
	pstr = pstr2 + fromlen;
    }

    orglen = pstr - str + strlen(pstr);

    /* Allocate memory for the post-replacement string */
    if (count > 0) {
	tolen = strlen(to);
	retlen = orglen + (tolen - fromlen) * count;
    } else	retlen = orglen;
    ret = malloc(retlen + 1);
    if (ret == NULL) {
	goto end_repl_str;
    }

    if (count == 0) {
	/* If no matches, then just duplicate the string */
	strcpy(ret, str);
    } else {
	/* Otherwise, duplicate the string whilst performing
	 * the replacements using the position cache */
	pret = ret;
	memcpy(pret, str, pos_cache[0]);
	pret += pos_cache[0];
	for (i = 0; i < count; ++i) {
	    memcpy(pret, to, tolen);
	    pret += tolen;
	    pstr = str + pos_cache[i] + fromlen;
	    cpylen = (i == count-1 ? orglen : pos_cache[i+1]) - pos_cache[i] - fromlen;
	    memcpy(pret, pstr, cpylen);
	    pret += cpylen;
	}
	ret[retlen] = '\0';
    }

 end_repl_str:
    /* Free the cache and return the post-replacement string,
     * which will be NULL in the event of an error */
    free(pos_cache);
    return ret;
}



PyObject *get_mesh_sizes__ (PyObject *ConfigDict)
{
    PyObject *item = NULL;
    PyObject *List = NULL;

    item = PyDict_GetItemString(ConfigDict, "ADAP_SIZES");
    if (NULL == item) goto err;

    char *str = PyString_AsString(item);
    if (NULL == str) goto err;

    char *str_ = str_strip(str, "()");
    List = str_split(str_, ",");
    if (NULL == List) goto err;

    return List;


 err:
    Py_XDECREF(item);
    Py_XDECREF(List);
    return NULL;
}


PyObject *get_sub_iterations__ (PyObject *ConfigDict)
{
    PyObject *item = NULL;
    PyObject *List = NULL;

    item = PyDict_GetItemString(ConfigDict, "ADAP_SUBITE");
    if (NULL == item) goto err;

    char *str = PyString_AsString(item);
    if (NULL == str) goto err;

    str = str_strip(str, "()");
    List = str_split(str, ",");
    if (NULL == List) goto err;

    return List;


 err:
    Py_XDECREF(item);
    Py_XDECREF(List);
    return NULL;
}


PyObject *get_residual_reduction__ (PyObject *ConfigDict)
{
    PyObject *item = NULL;
    PyObject *List = NULL;
    PyObject *Res = NULL;
    
    if (PyDict_Contains(ConfigDict,
			Py_BuildValue("s", "ADAP_RESIDUAL_REDUCTION"))) {
	item = PyDict_GetItemString(ConfigDict, "ADAP_SUBITE");
	if (NULL == item) goto err;

	char *str = PyString_AsString(item);
	if (NULL == str) goto err;

	str = str_strip(str, "()");
	List = str_split(str, ",");
	if (NULL == List) goto err;

	return List;
    }

    item = PyDict_GetItemString(ConfigDict, "ADAP_SIZES");
    if (NULL == item) goto err;
    
    char *str = PyString_AsString(item);
    if (NULL == str) goto err;

    str = str_strip(str, "()");
    List = str_split(str, ",");
    Py_DECREF(item);
    if (NULL == List) goto err;

    int length = PyList_Size(List);
    if (-1 == length) goto err;
    Py_DECREF(List);

    Res = PyList_New(0);
    if (NULL == Res) goto err;

    for (int i = 0 ; i < length ; ++i) {
	if (-1 == PyList_Append(Res, PyDict_GetItemString(ConfigDict, "RESIDUAL_REDUCTION")))
	    goto err;
    }

    return Res;
    


 err:
    Py_XDECREF(item);
    Py_XDECREF(List);
    Py_XDECREF(Res);
    return NULL;
}



PyObject *get_ext_iter__ (PyObject *ConfigDict)
{
    PyObject *item = NULL;
    PyObject *List = NULL;
    PyObject *ExtIterList = NULL;

    if (PyDict_Contains(ConfigDict, PyString_FromString("ADAP_EXT_ITER"))) {

	item = PyDict_GetItemString(ConfigDict, "ADAP_EXT_ITER");
	if (NULL == item) goto err;

	char *str = PyString_AsString(item);
	if (NULL == str) goto err;

	str = str_strip(str, "()");
	List = str_split(str, ",");
	if (NULL == List) goto err;

	return List;
    }

    item = PyDict_GetItemString(ConfigDict, "ADAP_SIZES");
    if (NULL == item) goto err;

    char *str = PyString_AsString(item);
    if (NULL == str) goto err;
	
    int nbExtIter = PyList_Size(str_split(str, ","));

    ExtIterList = PyList_New(nbExtIter);
    if (NULL == ExtIterList) goto err;

    for (int i = 0 ; i < nbExtIter ; ++i) {
	if (-1 == PyList_SetItem(ExtIterList, i,
				 PyDict_GetItemString(ConfigDict, "EXT_ITER")))
	    goto err;
    }

    return ExtIterList;


 err:
    Py_XDECREF(item);
    Py_XDECREF(List);
    return NULL;
}


PyObject *print_adap_options__ (PyObject *ConfigDict, PyObject *KwdsList)
{
    PyObject *String = NULL;
    PyObject *NewString = NULL;
    PyObject *Kwd = NULL;
    PyObject *item = NULL;
    PyObject *tmp_str = NULL;

    String = PyString_FromString("\nMesh adaptation options:\n");
    if (NULL == String) goto err;

    int len = PyList_Size(KwdsList);
    if (-1 == len) goto err;

    for (int i = 0 ; i < len ; ++i) {
	Kwd = PyList_GetItem(KwdsList, i);
	if (NULL == Kwd) goto err;
	if (PyDict_Contains(ConfigDict, Kwd)) {
	    item = PyDict_GetItem(ConfigDict, Kwd);
	    if (NULL == item) goto err;
	    NewString = PyString_FromFormat("%s: %s\n", PyString_AsString(Kwd),
					    PyString_AsString(item));
	    /* Py_DECREF(item); */
	    if (NULL == NewString) goto err;
	    PyString_Concat(&String, NewString);
	    Py_DECREF(NewString);
	    if (NULL == String) goto err;
	}
    }

    NewString = PyString_FromString("\n");
    if (NULL == NewString) goto err;
    PyString_Concat(&String, NewString);
    Py_DECREF(NewString);
    if (NULL == String) goto err;

    return String;


 err:
    Py_XDECREF(String);
    Py_XDECREF(NewString);
    Py_XDECREF(Kwd);
    Py_XDECREF(item);
    Py_XDECREF(tmp_str);
    return NULL;
}


static PyObject *su2_mesh_readlines__ (PyObject *args)
{
    PyObject *MeshFile = NULL;
    PyObject *nbLines = NULL;
    PyObject *List = NULL;
    PyObject *Line = NULL;

    nbLines = Py_BuildValue("i", 1);
    if (NULL == nbLines) goto err;

    if (!PyArg_ParseTuple(args, "OO", &MeshFile, &nbLines))
	goto err;

    int read_size;

    List = PyList_New(0);
    if (NULL == List) goto err;

    int nbLines_ = PyLong_AsLong(nbLines);
    if (-1 == nbLines_) goto err;

    for (int i = 0 ; i < nbLines_ ; ++i) {
	Line = PyFile_GetLine(MeshFile, 0);
    	if (NULL == Line) goto err;
    	read_size = PyString_Size(Line);
    	if (-1 == read_size) goto err;
    	else if (0 == read_size) break;
    	if (-1 == PyList_Append(List, Line)) goto err;
    	Py_DECREF(Line);
    }
    
    return List;


 err:
    Py_XDECREF(List);
    Py_XDECREF(Line);
    return NULL;
}


PyObject *su2_get_dim__ (PyObject *Filename)
{
    PyObject *FILE = NULL;
    PyObject *nbLines = NULL;
    PyObject *Lines = NULL;
    PyObject *Line = NULL;
    PyObject *NewLineList = NULL;
    PyObject *NewLine = NULL;
    PyObject *args = NULL;
    PyObject *Dim = NULL;
    
    char *tmp_str = NULL;
    tmp_str = PyString_AsString(Filename);
    if (NULL == tmp_str) goto err;

    /* We open the SU2 mesh file */
    FILE = PyFile_FromString(tmp_str, "r");
    if (NULL == FILE) goto err;

    nbLines = Py_BuildValue("i", 1);
    if (NULL == nbLines) goto err;
    args = Py_BuildValue("(OO)", FILE, nbLines);
    if (NULL == args) goto err;

    int i = 0;

    while (1) {
	/* We extract the next line */
	Lines = su2_mesh_readlines__ (args);
	if (NULL == Lines) goto err;
	Line = PyList_GetItem(Lines, 0);
	Py_DECREF(Lines);
	if (NULL == Line) goto err;
	char *line_ = NULL;
	line_ = PyString_AsString(Line);
	if (NULL == line_) goto err;

	/* We remove special characters */
	char *old_line_ = repl_str(line_, "\t", " ");
	if (NULL == old_line_) goto err;
	char *new_line_ = repl_str(old_line_, "\n", " ");
	free(old_line_);
	if (NULL == new_line_) goto err;

	/* We skip comments */
	if ('%' == *new_line_) {
	    free(new_line_);
	    continue;
	}

	/* We search if "NDIME=" is a subset of the line */
	char *dim_pos_ = strstr(new_line_, "NDIME=");
	if (NULL == dim_pos_) {
	    free(new_line_);
	    continue;
	}

	/* We retrieve the dimension value */
	NewLineList = str_split(new_line_, "=");
	free(new_line_);
	if (NULL == NewLineList) goto err;
	NewLine = PyList_GetItem(NewLineList, 1);
	char *new_line__ = PyString_AsString(NewLine);
	if (NULL == new_line__) goto err;
	Dim = PyLong_FromString(new_line__, NULL, 0);
	Py_DECREF(NewLineList);
	if (NULL == Dim) goto err;
	
	break;
    }

    Py_DECREF(args);

    return Dim;


 err:
    Py_XDECREF(FILE);
    Py_XDECREF(Lines);
    Py_XDECREF(Line);
    Py_XDECREF(NewLineList);
    Py_XDECREF(NewLine);
    Py_XDECREF(args);
    Py_XDECREF(Dim);
    return NULL;
}
