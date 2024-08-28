# CabanaDEM

Discrete element method with the Cabana library

## Dependencies
CabanaDEM has the following dependencies:

|Dependency | Version  | Required | Details|
|---------- | -------  |--------  |------- |
|CMake      | 3.11+    | Yes      | Build system
|Cabana     | a73697f  | Yes      | Performance portable particle algorithms

Cabana must be built with the following in order to work with CabanaPD:
|Cabana Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.16+   | Yes      | Build system
|Kokkos     | 3.7.0+  | Yes      | Performance portable on-node parallelism
|HDF5       | master  | Yes       | Particle output
|SILO       | master  | No       | Particle output

The underlying parallel programming models are available on most systems, as is
CMake. Those must be installed first, if not available. Kokkos and Cabana are
available on some systems or can be installed with `spack` (see
https://spack.readthedocs.io/en/latest/getting_started.html):

```
spack install cabana@master+cajita+silo
```

Alternatively, Kokkos can be built locally, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/1-Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana must be
compiled with MPI and the Grid sub-package.

## Obtaining CabanaDEM

Clone the master branch:

```
git clone https://github.com/ExaPhysics/CabanaDEM.git
```

## Build and install
### CPU Build

After building Kokkos and Cabana for CPU, the following script will build and install CabanaDEM:

```
#Change directory as needed
export CABANA_INSTALL_DIR=/home/username/Cabana/build/install

cd ./CabanaDEM
mkdir build
cd build
cmake \
    -D CMAKE_PREFIX_PATH="$CABANA_INSTALL_DIR" \
    -D CMAKE_INSTALL_PREFIX=install \
    .. ;
make install
```

### CUDA Build

After building Kokkos and Cabana for Cuda:
https://github.com/ECP-copa/Cabana/wiki/CUDA-Build

The CUDA build script is identical to that above, but again note that Kokkos
must be compiled with the CUDA backend.

Note that the same compiler should be used for Kokkos, Cabana, and CabanaPD.

### HIP Build

After building Kokkos and Cabana for HIP:
https://github.com/ECP-copa/Cabana/wiki/HIP-and-SYCL-Build#HIP

The HIP build script is identical to that above, except that `hipcc` compiler
must be used:

```
-D CMAKE_CXX_COMPILER=hipcc
```

Note that `hipcc` should be used for Kokkos, Cabana, and CabanaPD.


### More detailed Cabana and Kokkos installation

A more detailed installation can also be found at:
https://gist.github.com/dineshadepu/4313d6a148f7188965b951406e9fb83f


## Features

CabanaDEM currently includes the following:
 - Non-linear force model: Hertz normal contact force model
 - Tangential contact: Mindlin and Deresiewicz
 - Particle boundary conditions
 - CPU, OpenMP, GPU support

## Examples

Once built and installed, CabanaDEM `examples/` can be run. Timing and energy
information is output to file and particle output is written to files (if enabled within Cabana) that can be visualized with Paraview and similar applications.
New examples can be created by using any of the current cases as a template. All inputs are specified in the example JSON files within the relevant `inputs/` subdirectory.
We validate our code using the benchmark tests provided by Chung et al. [1], which are widely used in the DEM community.

### Fundamental benchmarks
The first example is two particles impacting head on [1]. Assuming the build paths above, the example can be run with:

```
./build/examples/01ElasticNormalImpactOfTwoIdenticalParticles ./examples/inputs/01_elastic_normal_impact_of_two_identical_particles.json ./
```
![alt text](https://github.com/ExaPhysics/CabanaDEM/blob/master/doc/images/dem_01_head_on_schematic.png?raw=true)

![alt text](https://github.com/ExaPhysics/CabanaDEM/blob/master/doc/images/fn_overlap.png?raw=true)

The second example is spherical particle impacting a wall normally [1]. Assuming the build paths above, the example can be run with:

```
./build/examples/02ElasticNormalImpactOfParticleWall ./examples/inputs/02_elastic_normal_impact_of_particle_wall.json ./
```

The third example is to test the damping part of the contact force model [1]. Assuming the build paths above, the example can be run with:

```
./build/examples/03NormalParticleWallDifferentCOR ./examples/inputs/03_normal_particle_wall_different_cor.json ./
```

Fourth example is to test the tangential part of the contact force model [1]. Assuming the build paths above, the example can be run with:

```
./build/examples/04ObliqueParticleWallDifferentAngles ./examples/inputs/04_oblique_particle_wall_different_angles.json ./
```

![alt text](https://github.com/ExaPhysics/CabanaDEM/blob/master/doc/images/schematic.png?raw=true)

![alt text](https://github.com/ExaPhysics/CabanaDEM/blob/master/doc/images/angle_vs_ang_vel.png?raw=true)

## References

[1] Chung, Y. C., and J. Y. Ooi. "Benchmark tests for verifying discrete element
modelling codes at particle impact level." Granular Matter 13.5 (2011): 643-656.

## Contributing

We encourage you to contribute to CabanaDEM! Please check the
[guidelines](CONTRIBUTING.md) on how to do so.


<!-- ## License -->

<!-- CabanaDEM is distributed under an [open source 3-clause BSD license](LICENSE). -->
