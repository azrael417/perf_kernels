#include "global.hpp"
#include "utils.hpp"
#include <mkl.h>

int main(int argc, char* argv[]){
  //init MPI
  int required=MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc,&argv,required,&provided);
  if(required!=provided){
    std::cerr << "Error, asked for MPI thread level " << required << " but got " << provided << ", abort." << std::endl;
    return EXIT_FAILURE;
  }

  //get my rank:
  int myrank=0, numranks;
  MPI_Comm_size(MPI_COMM_WORLD,&numranks);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  if(myrank==0) std::cout << "--- starting DGEMM minibench ---" << std::endl;

  //set up matrix:
  const int N=8192;
  const int niter=10;
  double *m1 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));
  double *m2 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));
  double *m3 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));

  //set up seed for seeded random numbers:
  if(myrank==0) std::cout << "set up matrix" << std::endl;
#pragma omp parallel firstprivate(N) shared(m1,m2)
  {
    int tid=omp_get_thread_num();
    Ran rng(tid);
#pragma omp for
    for(unsigned int i=0; i<N*N; ++i){
      m1[i]=rng.doub();
      m2[i]=rng.doub();
    }
  }

  //set up BLAS stuff
  char TA='N', TB='N';
  double alpha=1., beta=0.;

  //run and time
  if(myrank==0) std::cout << "run benchmark" << std::endl;
  double ts=MPI_Wtime();
  dgemm(&TA,&TB,&N,&N,&N,&alpha,m1,&N,m2,&N,&beta,m3,&N);
  double t=MPI_Wtime()-ts;
  double tmin=t, tmax=t, tave=t;
  for(unsigned int i=1; i<niter; i++){
    ts=MPI_Wtime();
    dgemm(&TA,&TB,&N,&N,&N,&alpha,m1,&N,m2,&N,&beta,m3,&N);
    t=MPI_Wtime()-ts;
    tmin=(t<tmin ? t : tmin);
    tmax=(t>tmax ? t : tmax);
    tave+=t;
  }
  tave/=double(niter);
  MPI_Barrier(MPI_COMM_WORLD);

  //ordered timings printed
  for(unsigned int rank=0; rank<numranks; rank++){
    if(rank==myrank){
      std::cout << "DGEMM time (size: " << N << "x" << N << ", rank: " << myrank << "): min = " << tmin << ", max = " << tmax << ", mean = " << tave << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //cleaning up
  delete [] m1, m2, m3;

  if(myrank==0) std::cout << "--- finishing DGEMM minibench ---" << std::endl;

  MPI_Finalize();
}
