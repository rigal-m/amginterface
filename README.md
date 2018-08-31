# AMGIO Python module

## Description

The amgio Python module is a library providing methods to convert meshes and
solutions between the following formats:
* libmeshb
* SU2

The goal is to facilitate the interaction between the Python Adaptative Mesh
Generation program [pyamg](https://pyamg.saclay.inria.fr/) and
the [SU2](https://su2code.github.io/) solver.

## Building the wheel archive

To generate the .whl archive, please follow the steps below:
```bash
./compile.sh (linux|mac) -py=(2|3)
(sudo -H pip | pip3) install --upgrade --force-reinstall <PATH_TO_WHEEL_ARCHIVE>
```
Currently, the only supported architecture is linux for Python 2.
The path to the wheel archive is printed at the end of the compilation process.

To uninstall this module and clean the directory, type:
```bash
(sudo -H pip | pip3) uninstall amgio
./compile.sh clean
```

## Module userguide

The amgio module can be imported as follow:
```python
import amgio
```

In this section we give an overview of the different methods offered by the
amgio module:

- `su2_to_libmeshb(su2_mesh, libmeshb_mesh[, su2_sol=sol])`  
Convert an SU2 python mesh file into a libmeshb one and write it in a file.  

- `amgio.libmeshb_to_su2(libmeshb_mesh, su2_mesh[, libmeshb_sol=sol])`  
Convert a libmeshb python mesh file into a SU2 one.  

- `amgio.split_solution(solution, dim, prefix, adap_sensor)`  
Split a given solution.  

- `amgio.read_mesh_to_lists(mesh_name, sol_name, vertices, triangles, tetrahedra, edges, hexahedra, quadrilaterals, pyramids, prisms, sol, sol_ref, markers)`  
Read a .mesh(b) file and store the elements into lists given as arguments.  

- `amgio.write_mesh_from_lists(mesh_name, sol_name, vertices, triangles, tetrahedra, edges, hexahedra, quadrilaterals, pyramids, prisms, sol, markers, dim)`  
Write a set of lists defining a mesh into a (.mesh(b) | SU2) file.  

- `amgio.write_sol_from_lists(sol_name, vertices, sol, sol_header, nb_vertices, dim)`  
Write a set of lists defining a solution into a (.sol(b) | SU2) file.  

- `msh_dict = amgio.get_mesh_to_dict(mesh_name[, sol_name=sol])`  
Return a dictionary representation of the given mesh file (and optional solution).  

- `amgio.write_mesh_from_dict(mesh, mesh_name[, sol_name=sol])`  
Write a dictionary representation of a mesh (and optional solution) to file.  

- `amgio.write_sol_from_dict(sol, sol_name)`  
Write a dictionary representation of a solution to file.  

- `sensor_wrap_dict = amgio.create_sensor(sol, sensor)`  
Create a sensor given a solution dict.  

- `size_list = amgio.get_mesh_sizes(config_dict)`  
Get the mesh sizes for each AMG iteration from an SU2 config dictionary.  

- `sub_iter_list = amgio.get_sub_iterations(config_dict)`  
Get the number of subiterations required for each AMG iteration from an
SU2 config dictionary.  

- `residual_reduction_list = amgio.get_residual_reduction(config_dict)`  
Get the number of residual reductions required for each AMG iteration
from an SU2 config dictionary.  

- `ext_iter_list = amgio.get_ext_iter(config_dict)`  
Get the number of exterior iterations required for each AMG iteration
from an SU2 config dictionary.  

- `amgio.print_adap_options(config_dict, keword_list)`  
Print options regarding the adaptative process.  

- `dim = amgio.su2_get_dim(file_name)`  
Retrieve the dimension from an SU2 mesh file.
