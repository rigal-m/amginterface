# AMGIO Python module

## Description

The amgio Python module is a library providing methods to convert meshes and
solutions between the following formats:
* libmeshb
* SU2

## Building the wheel archive

To generate the .whl archive, please follow the following steps:
```bash
./compile.sh (linux|mac) -py=(2|3)
sudo -H pip install --upgrade --force-reinstall <PATH_TO_WHEEL_ARCHIVE>
```
Currently, the only supported architecture is linux for Python 3.
The path to the wheel archive is printed at the end of the compilation process.

## Module userguide

In the section we give an overview of the different methods offered by the
amgio module. To these functions, one first has to import amgio:
```python
import amgio
```


`su2_to_libmeshb(su2_mesh, libmeshb_mesh[, su2_sol=sol])`python  
Convert an SU2 python mesh file into a libmeshb one and write it in a file.  

`amgio.libmeshb_to_su2(libmeshb_mesh, su2_mesh[, libmeshb_sol=sol])`python  
Convert a libmeshb python mesh file into a SU2 one.  

`amgio.split_solution(solution, dim, prefix, adap_sensor)`python  
Split a given solution.  

`amgio.read_mesh_to_lists(mesh_name, sol_name, vertices, triangles, tetrahedra, edges, hexahedra, quadrilaterals, pyramids, prisms, sol, sol_ref, markers)`python  
Read a .mesh(b) file and store the elements into lists given as arguments.  

`amgio.write_mesh_from_lists(mesh_name, sol_name, vertices, triangles, tetrahedra, edges, hexahedra, quadrilaterals, pyramids, prisms, sol, markers, dim)`python  
Write a set of lists defining a mesh into a (.mesh(b) | SU2) file.  

`amgio.write_sol_from_lists(sol_name, vertices, sol, sol_header, nb_vertices, dim)`python  
Write a set of lists defining a solution into a (.sol(b) | SU2) file.  

`msh_dict = amgio.get_mesh_to_dict(mesh_name[, sol_name=sol])`python  
Return a dictionary representation of the given mesh file (and optional solution).  

`amgio.write_mesh_from_dict(mesh, mesh_name[, sol_name=sol])`python  
Write a dictionary representation of a mesh (and optional solution) to file.  

`amgio.write_sol_from_dict(sol, sol_name)`python  
Write a dictionary representation of a solution to file.  

`sensor_wrap_dict = amgio.create_sensor(sol, sensor)`python  
Create a sensor given a solution dict.  

`size_list = amgio.get_mesh_sizes(config_dict)`python  
Get the mesh sizes for each AMG iteration from an SU2 config dictionary.  

`sub_iter_list = amgio.get_sub_iterations(config_dict)`python  
Get the number of subiterations required for each AMG iteration from an
SU2 config dictionary.  

`residual_reduction_list = amgio.get_residual_reduction(config_dict)`python  
Get the number of residual reductions required for each AMG iteration
from an SU2 config dictionary.  

`ext_iter_list = amgio.get_ext_iter(config_dict)`python  
Get the number of exterior iterations required for each AMG iteration
from an SU2 config dictionary.  

`amgio.print_adap_options(config_dict, keword_list)`python  
Print options regarding the adaptative process.  

`dim = amgio.su2_get_dim(file_name)`python  
Retrieve the dimension from an SU2 mesh file.

