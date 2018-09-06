#ifndef AMGIO_TOOLS__
#define AMGIO_TOOLS__

int is_char_in_str(char c, const char *str);
char *str_strip(char *old, const char *remove_chars);
PyObject *str_split(char *str, char *c);
char *repl_str(const char *str, const char *from, const char *to);
PyObject *get_mesh_sizes__ (PyObject *ConfigDict);
PyObject *get_residual_reduction__ (PyObject *ConfigDict);
PyObject *get_sub_iterations__ (PyObject *ConfigDict);
PyObject *get_ext_iter__ (PyObject *ConfigDict);
PyObject *print_adap_options__ (PyObject *ConfigDict, PyObject *KwdsTuple);
static PyObject *su2_mesh_readlines__ (PyObject *args);
PyObject *su2_get_dim__ (PyObject *Filename);

#endif /* AMGIO TOOLS */
