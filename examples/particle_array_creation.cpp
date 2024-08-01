#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

#include <CabanaDEM_Particles.hpp>

#define DIM 2


template<class MemorySpace, class ExecutionSpace>
void aosoaExample(const ExecutionSpace& exec_space)
{

  auto particles = CabanaLearn::Particles<MemorySpace, DIM>(exec_space, 100);

  /*
    Print the label and size data. In this case we have created an AoSoA
    with 5 tuples. Because a vector length of 4 is used, a total memory
    capacity for 8 tuples will be allocated in 2 SoAs.
  */
  std::cout << "aosoa.label() = " << particles.sliceNoFail().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceDensity().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceDamage().label() << std::endl;
  std::cout << "aosoa.label() = " << particles.sliceVelocity().label() << std::endl;

  auto u = particles.sliceDisplacement();
    //for (int i=0; i < u.size(); i++){
      //std::cout << u (i, 0) << "\n";
    //}
  // std::cout << "aosoa.label() = " <<  << std::endl;
  // std::cout << "aosoa.size() = " << particles.size() << std::endl;
  // std::cout << "aosoa.capacity() = " << particles.capacity() << std::endl;
  // std::cout << "aosoa.numSoA() = " << particles.numSoA() << std::endl;

}



int main( int argc, char* argv[] )
{

  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  // check inputs and write usage
  if ( argc < 1 )
    {
      std::cerr << "Usage: ./TwoBlocksColliding exec_space \n";

      std::cerr << "      exec_space      execute with: serial, openmp, "
	"cuda, hip\n";
      std::cerr << "\nfor example: ./TwoBlocksColliding serial\n";
      Kokkos::finalize();
      MPI_Finalize();
      return 0;
    }

  // execution space
  std::string exec_space( argv[1] );


  if ( 0 == exec_space.compare( "serial" ) ||
       0 == exec_space.compare( "Serial" ) ||
       0 == exec_space.compare( "SERIAL" ) )
    {
#ifdef KOKKOS_ENABLE_SERIAL
      aosoaExample<Kokkos::HostSpace, Kokkos::Serial>(Kokkos::Serial());
#else
      throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "openmp" ) ||
	    0 == exec_space.compare( "OpenMP" ) ||
	    0 == exec_space.compare( "OPENMP" ) )
    {
#ifdef KOKKOS_ENABLE_OPENMP
      aosoaExample<Kokkos::HostSpace, Kokkos::OpenMP>(Kokkos::OpenMP());
#else
      throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
  else if ( 0 == exec_space.compare( "cuda" ) ||
	    0 == exec_space.compare( "Cuda" ) ||
	    0 == exec_space.compare( "CUDA" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
      aosoaExample<Kokkos::CudaSpace, Kokkos::Cuda>(Kokkos::Cuda());
#else
      throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }


  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
