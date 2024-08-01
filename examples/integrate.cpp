#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

#define DIM 2


//---------------------------------------------------------------------------//
// AoSoA example.
//---------------------------------------------------------------------------//
template <class MemorySpace, int Dimension>
class Particles
{
public:
  using memory_space = MemorySpace;
  using execution_space = typename memory_space::execution_space;
  static constexpr int dim = Dimension;

  using vector_type = Cabana::MemberTypes<double[dim]>;
  using aosoa_x_type = Cabana::AoSoA<vector_type, memory_space, 1>;

  // Constructor which initializes particles on regular grid.
  template <class ExecSpace>
  Particles( const ExecSpace& exec_space, std::size_t n )
  {
    resize( n );
    createParticles( exec_space );
  }

  // Constructor which initializes particles on regular grid.
  template <class ExecSpace>
  void createParticles( const ExecSpace& exec_space )
  {
    auto x = slicePosition();

    auto create_particles_func = KOKKOS_LAMBDA( const int i )
      {
	for (int j=0; j < DIM; j++){
	  x( i, j ) = DIM * i + j;
	}
      };
    Kokkos::RangePolicy<ExecSpace> policy( 0, x.size() );
    Kokkos::parallel_for( "create_particles_lambda", policy,
			  create_particles_func );
  }

  auto slicePosition()
  {
    return Cabana::slice<0>( _aosoa_x, "position" );
  }
  auto slicePosition() const
  {
    return Cabana::slice<0>( _aosoa_x, "position" );
  }

  void resize(const std::size_t n)
  {
    _aosoa_x.resize( n );
  }


private:
  aosoa_x_type _aosoa_x;
};


template <class ExecutionSpace>
class Integrator
{
  using exec_space = ExecutionSpace;

  double _dt, _half_dt;
  // Timer _timer;
public:
  Integrator ( double dt )
    : _dt (dt)
  {
    _half_dt = 0.5 * dt;

  }
  ~Integrator() {}

  template <class ParticlesType>
  void initialHalfStep(ParticlesType& p){

    // _time.start();
    auto x = p.slicePosition();
    auto dt = _dt;
    auto half_dt = _half_dt;
    auto init_func = KOKKOS_LAMBDA( const int i )
    {
      x( i, 0 ) += dt * 1.;
      x( i, 1 ) += dt * 1.;
      x( i, 2 ) += dt * 1.;
    };
    Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
    Kokkos::parallel_for( "CabanaPD::Integrator::Initial", policy,
			init_func );

    // _timer.stop();
  }

  double timeInit() { return 0.0; };
  // auto time() { return _timer.time(); };

};

template<class MemorySpace, class ExecutionSpace>
void aosoaExample(const ExecutionSpace& exec_space)
{
  auto particles = std::make_shared<Particles<MemorySpace, DIM>>(exec_space, 10);

  auto x = particles->slicePosition();
  // auto x = particles.slicePosition();
  for (int i=0; i < x.size(); i++){
    std::cout << x (i, 0) << " " << x (i, 1) << "\n";
  }


  auto integrator = Integrator<ExecutionSpace>(1e-4);
  integrator.initialHalfStep(*particles);

  /*
    Print the label and size data. In this case we have created an AoSoA
    with 5 tuples. Because a vector length of 4 is used, a total memory
    capacity for 8 tuples will be allocated in 2 SoAs.
  */
  // std::cout << "aosoa.label() = " << particles.sliceNoFail().label() << std::endl;
  // std::cout << "aosoa.label() = " << particles.sliceDensity().label() << std::endl;
  // std::cout << "aosoa.label() = " << particles.sliceDamage().label() << std::endl;
  // std::cout << "aosoa.label() = " << particles.sliceVelocity().label() << std::endl;

  // auto x = particles->slicePosition();
  // auto x = particles.slicePosition();
  for (int i=0; i < x.size(); i++){
    std::cout << x (i, 0) << " " << x (i, 1) << "\n";
  }
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
