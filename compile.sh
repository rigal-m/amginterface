#!/bin/bash 

nbarg=`expr $# `

if [ $nbarg == 1 ]; then
    case $1 in
	"clean")
	    rm -rfv ./amgio/ ./build/
	    exit
	    shift ;;

    esac
fi

if [ $nbarg -lt 2 ]; then
    echo "Please use one of the following commands:"
    echo "1/  ./compile.sh (linux|mac) -py=(2|3)"
    echo "2/  ------------ clean"
    exit
fi

case $1 in
    "linux")
	nb_cores=$(nproc --all)
	echo "linux architecture ($nb_cores cores detected)"

	if [ ! -d .obj ]; then
	    mkdir .obj
	fi

	if [ ! -d ./amgio/dep/ ]; then
	    mkdir -p ./amgio/dep/
	fi

	case $2 in
	    "-py=2")
		INCPYTHON=`python-config --includes`
		INCNUMPY=`python2 -c "import numpy; print(numpy.get_include())"`
		PYVERSION=-DPYTHON_2

		if [ ! -d ./build/linux64/amgio-linux-python2/ ]; then
		    mkdir -p ./build/linux64/amgio-linux-python2/
		fi

		HEADERS="-I./src -I./src/libmeshb -I./src/feflo.a/core -I./src/feflo.a/util -I./src/feflo.a/libol $INCPYTHON -I$INCNUMPY"
		C_FLAGS="-Di4 -fPIC -ansi"

		# su2 support
		echo "Generating objects"
		echo " -> amgio_py.o"
		gcc -c  src/amgio/amgio_py.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> amgio_tools.o"
		gcc -c  src/amgio/amgio_tools.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> amgiomodule.o"
		gcc -c  src/amgiomodule.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> convert.o"
		gcc -c  src/amgio/convert.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> GMFio.o"
		gcc -c  src/amgio/GMFio.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> libmeshb7.o"
		gcc -c src/libmeshb/libmeshb7.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> mesh.o"
		gcc -c  src/amgio/mesh.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> options.o"
		gcc -c  src/amgio/option.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> SU2io.o"
		gcc -c  src/amgio/SU2io.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> tools.o"
		gcc -c  src/tools.c -I. $HEADERS $C_FLAGS $PYVERSION

		# amgio module
		echo "Generating library amgiomodule.so"
		gcc -shared libmeshb7.o amgio_tools.o tools.o SU2io.o amgio_py.o convert.o GMFio.o mesh.o option.o amgiomodule.o -Lamgio/dep/ -o amgiomodule.so -Wl,-rpath='$ORIGIN'/amgio/dep/

		cp -r setup_linux_python2.py README.md amgiomodule.so amgio/ build/linux64/amgio-linux-python2/

		echo "Entering package directory ./build/linux64/amgio-linux-python2/"
		cd ./build/linux64/amgio-linux-python2/
		echo "Packaging the Wheel archive"
		python2 setup_linux_python2.py bdist_wheel

		echo "Exiting package directory"
		cd ../../../

		echo "Removing unnecessary files"
		rm -f libmeshb7.o amgio_tools.o tools.o GMFio.o mesh.o option.o SU2io.o convert.o mesh_converter.o amgio_py.o

		## remove old amgiomodule
		rm -f amgiomodule.o amgiomodule.so
		rm -r amgio/

		echo ""
		echo "The wheel archive has been generated in build/linux64/amgio-linux-python2/dist/"
		echo "To install it directly from the later directory, please type:"
		echo "sudo -H pip install <WHEEL ARCHIVE>"

		exit
		
		shift ;;

	    "-py=3")
		INCPYTHON=`python3-config --includes`
		INCNUMPY=`python3 -c "from numpy import get_include; print(get_include())"`
		PYVERSION=-DPYTHON_3

		if [ ! -d ./build/linux64/amgio-linux-python3/ ]; then
		    mkdir -p ./build/linux64/amgio-linux-python3/
		fi

		HEADERS="-I./src -I./src/libmeshb -I./src/feflo.a/core -I./src/feflo.a/util -I./src/feflo.a/libol $INCPYTHON -I$INCNUMPY"
		C_FLAGS="-Di4 -fPIC"

		# su2 support
		echo "Generating objects"
		echo " -> amgio_py.o"
		gcc -c  src/amgio/amgio_py.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> amgio_tools.o"
		gcc -c  src/amgio/amgio_tools.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> amgiomodule.o"
		gcc -c  src/amgiomodule.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> convert.o"
		gcc -c  src/amgio/convert.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> GMFio.o"
		gcc -c  src/amgio/GMFio.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> libmeshb7.o"
		gcc -c src/libmeshb/libmeshb7.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> mesh.o"
		gcc -c  src/amgio/mesh.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> options.o"
		gcc -c  src/amgio/option.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> SU2io.o"
		gcc -c  src/amgio/SU2io.c $HEADERS $C_FLAGS $PYVERSION
		echo " -> tools.o"
		gcc -c  src/tools.c -I. $HEADERS $C_FLAGS $PYVERSION

		# amgio module
		echo "Generating library amgiomodule.so"
		gcc -shared libmeshb7.o amgio_tools.o tools.o SU2io.o amgio_py.o convert.o GMFio.o mesh.o option.o amgiomodule.o -Lamgio/dep/ -o amgiomodule.so -Wl,-rpath='$ORIGIN'/amgio/dep/

		cp -r setup_linux_python3.py README.md amgiomodule.so amgio/ build/linux64/amgio-linux-python3/

		echo "Entering package directory ./build/linux64/amgio-linux-python3/"
		cd ./build/linux64/amgio-linux-python3/
		echo "Packaging the Wheel archive"
		python3 setup_linux_python3.py bdist_wheel

		echo "Exiting package directory"
		cd ../../../

		echo "Removing unnecessary files"
		rm -f libmeshb7.o amgio_tools.o tools.o GMFio.o mesh.o option.o SU2io.o convert.o mesh_converter.o amgio_py.o

		## remove old amgiomodule
		rm -f amgiomodule.o amgiomodule.so
		rm -r amgio/

		echo ""
		echo "The wheel archive has been generated in build/linux64/amgio-linux-python3/dist/"
		echo "To install it directly from the later directory, please type:"
		echo "sudo -H pip install <WHEEL ARCHIVE>"

		exit
		
		shift ;;
	    
	esac

	shift ;;

    "mac")
	nb_cores=$(sysctl -n hw.ncpu)
	echo "macintel architecture ($nb_cores cores detected)"

	case $2 in
	    "-py=2")
		echo "Not implemented"
		exit
		shift ;;

	    "-py=3")
		echo "Not implemented"
		exit
		shift ;;
	esac
	shift ;;
esac
