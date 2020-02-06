===============================================================================================
hgmca - A hierarchical component separation algorithm based on sparsity in the wavelet basis.
===============================================================================================

.. image:: https://travis-ci.com/swagnercarena/hgmca.png?branch=master
	:target: https://travis-ci.org/swagnercarena/hgmca

.. image:: https://readthedocs.org/projects/hgmca/badge/?version=latest
	:target: https://hgmca.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

Installation (~15 minutes)
--------------------------

If you intend to use gmca or hgmca together with the scale discretized wavelet
transform, you will need to install ``s2let``. The instructions for installation can be found on the `s2let website <http://astro-informatics.github.io/s2let/scratch_install.html>`. While the installation will take a few minutes, the instructions below should be comprehensive enough 

1. Install CFITSIO using brew (or equivalent). Use brew info to get the path.


.. code-block:: bash

	$ brew install CFITSIO
	$ brew info CFITSIO
	$ export CFITSIO=<CFITSIO path>

2. Install FFTW using brew (or equivalent). Use brew info to get the path.

.. code-block:: bash

	$ brew install FFTW
	$ brew info FFTW
	$ export FFTW=<FFTW path>

3. Download `Healpix <https://sourceforge.net/projects/healpix/>` and then run the following (we will provide excruciating detail here since this can't be done with a simple brew command):

.. code-block:: bash

	$ cd <Healpix path>
	$ ./configure
	$ Enter your choice (configuration of packages can be done in any order) [0]: 2
	$ Should I attempt to create these directories (y|n) [y]? y
	$ enter C compiler you want to use [gcc]: <compiler>
	$ enter options for C compiler [-O2 -Wall]:
	$ enter archive creation (and indexing) command [ar -rsv]: 
	$ do you want the HEALPix/C library to include CFITSIO-related functions ? (y|n) [y]: y
	$ enter full name of cfitsio library [libcfitsio.a]: libcfitsio.a
	$ enter location of cfitsio library [/usr/local/lib]: <CFITSIO path>
	$ enter location of cfitsio header fitsio.h [<CFITSIO path>]:
	$ A static library is produced by default. Do you also want a shared library ? (y|n) [n]: n
	$ Do you want this modification to be done (y|n)? [n]: y
	$ Enter your choice (configuration of packages can be done in any order) [0]: 3
	$ enter C compiler you want to use [gcc]: <compiler>
	$ [-O3 -ffast-math -march=native -fopenmp]: 
	$ enter name of your F90 compiler [gfortran]: <gfortran compiler>
	$ enter suffix for directories []:
	$ Should I attempt to create these directories (y|n) [y]? y
	$ enter compilation flags for gfortran-9 compiler [-I$(F90_INCDIR)]:
	$ enter optimisation flags for gfortran-9 compiler [-O3]:
	$ enter name of your C compiler [<compiler>]:
	$ enter compilation/optimisation flags for C compiler [-O3 -std=c99 -I$(HEALPIX)/include]: 
	$ enter command for library archiving [libtool -static -s -o]:
	$ enter full name of cfitsio library [libcfitsio.a]:
	$ enter location of cfitsio library [/usr/local/lib]: <CFITSIO path>
	$ (this assumes that PGPLOT is already installed on your computer) (y|n) [n]: n
	$ Enter choice                                      [1]: 1
	$ (recommended if the Healpix-F90 library is to be linked to external codes) (y|n) [y]: y
	$ A static library is produced by default. Do you rather want a shared/dynamic library ? (y|n) [n]: n
	$ Enter your choice (configuration of packages can be done in any order) [0]: 0
	$ make
	$ export HEALPIX=<HEALPIX path>

Note that on MAC I used gcc 9.2.0 and gfortran 9.2.0. Feel free to run

.. code-block:: bash

	make test

to make sure everything is installed correctly.

4. Clone the repos for ssht, so3, s2let, and hgmca:

.. code-block:: bash

	$ git clone https://github.com/astro-informatics/ssht
	$ git clone https://github.com/astro-informatics/so3
	$ git clone https://github.com/astro-informatics/s2let
	$ git clone https://github.com/swagnercarena/hgmca

5. Export the path to each of the four new directories

.. code-block:: bash

	$ export SSHT=<SSHT path>
	$ export SO3=<S03 path>
	$ export S2LET=<S2LET path>
	$ export HGMCA=<HGMCA path>

6. Go into the ssht directory and compile the package (note you may want to change the compiler in the makefile if you're on Mac. Get gcc-9 from brew and use that instead of gcc).

.. code-block:: bash

	$ cd $SSHT
	$ make

and once again go ahead and test that SSHT is working:

.. code-block:: bash

	$ ./bin/c/ssht_test 128 0

7. Now it's time to install SO3 (Remember to check your compiler! If you get an error like ``unsupported option '-fopenmp'``, you need to change the compiler):

.. code-block:: bash

	$ cd $SO3
	$ make

and once again test your compilation:

.. code-block:: bash

	$ ./bin/c/so3_test

8. Almost done with all that C compilation! Just s2let left. First we need to copy over our modified s2let files, and then we can make. Don't forget to change the compilers in the makefile if you don't want to use default gcc and gfortran (with Mac you'll want gcc-9 and gfortran-9).

.. code-block:: bash

	$ cd $S2LET
	$ cp $HGMCA/s2let_mods/makefile $S2LET/
	$ cp $HGMCA/s2let_mods/*.c $S2LET/src/main/c/
	$ cp $HGMCA/s2let_mods/*.h $S2LET/include/
	$ make lib
	$ make mw_bin
	$ make hpx_bin

If you want to test that everything went according to plan, run

.. code-block:: bash

	$ ./bin/s2let_test
	$ ./bin/s2let_hpx_test


