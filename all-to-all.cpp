#include "global.hpp"
#include "utils.hpp"

void a2a(int n, int nt, int tlim, double *t, double *ts, MPI_Comm comm) {
  int i, nprocs;
  double dt = tlim / (double) nt;
  int udt = (int) (1000000 * dt);

  MPI_Comm_size(comm, &nprocs);
  double *sbuf = reinterpret_cast<double*>(aligned_alloc(64, n*nprocs*sizeof(double)));
  double *rbuf = reinterpret_cast<double*>(aligned_alloc(64, n*nprocs*sizeof(double)));

  for (i=0; i<n*nprocs; i++) {
    sbuf[i] = 1.0;
    rbuf[i] = 0.0;
  }

  MPI_Barrier(comm);
  for (i=0; i<nt; i++) {
    ts[i] = MPI_Wtime();
    int ierr = MPI_Alltoall(sbuf, n, MPI_DOUBLE,
			    rbuf, n, MPI_DOUBLE,
			    comm);
    t[i] = MPI_Wtime() - ts[i];
    usleep(udt);
  }
  free(sbuf);
  free(rbuf);
  return;
}

// get xyz dragonfly coords
void coordinates(MPI_Comm comm, int *coords) {
  int nid=-1;
  int pmi_rank;
  PMI_Get_rank(&pmi_rank);
  pmi_mesh_coord_t xyz;
  PMI_Get_rank(&pmi_rank);
  PMI_Get_nid(pmi_rank, &nid);
  PMI_Get_meshcoord((pmi_nid_t) nid, &xyz);

  coords[0] = (int) xyz.mesh_x;
  coords[1] = (int) xyz.mesh_y;
  coords[2] = (int) xyz.mesh_z;
  return;
}

long random_at_mostL(long max) {
  unsigned long num_bins = (unsigned long) max + 1, num_rand = (unsigned long) RAND_MAX + 1, bin_size =\
    num_rand / num_bins, defect = num_rand % num_bins;
  long x;
  do {
    x = random();
  }
  while (num_rand - defect <= (unsigned long)x);
  return x/bin_size;
}

int mycolor(MPI_Comm comm) {
  return (int) random_at_mostL(1);
}

void pprintI(char* str, int i, MPI_Comm comm) {
  // for debugging output
  int j;
  int rank, nprocs;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &rank);
  for (j=0; j<nprocs; j++) {
    if (rank == j) {
      printf("%i: %-20s = %i\n", rank, str, i);
      fflush(stdout);
    }
    MPI_Barrier(comm);
  }
}

int main(int argc, char** argv) {
  int i;
  int rank, nprocs;

  //init MPI
  int required=MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc,&argv,required,&provided);
  if(required!=provided){
    std::cerr << "Error, asked for MPI thread level " << required << " but got " << provided << ", abort." << std::endl;
    return EXIT_FAILURE;
  }

  //comm stuff
  MPI_Comm comm, comm_s;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  //read parameters
  //init parser
  int error=0;
  int n=8, nt=1000, numcols=1;
  double trun = 20.0;
  if(rank==0){
    std::cout << "reading input parameters." << std::endl;

    InputParser input(argc, argv);

    //parse box size arguments:
    if(input.cmdOptionExists("--buffer_size")){
        // Do stuff
        n = atoi(input.getCmdOption("--buffer_size").c_str());
    }
    if(input.cmdOptionExists("--niter")){
        // Do stuff
        nt = atoi(input.getCmdOption("--niter").c_str());
    }
    if(input.cmdOptionExists("--sleeptime")){
        // Do stuff
        trun = atof(input.getCmdOption("--sleeptime").c_str());
    }
    if(input.cmdOptionExists("--num_colors")){
        // Do stuff
        numcols = atoi(input.getCmdOption("--num_colors").c_str());
    }
  }
  //broadcast
  MPI_Bcast(&n,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&nt,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&trun,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(&numcols,1,MPI_INTEGER,0,MPI_COMM_WORLD);

  //do coloring;
  srand(rank);
  int color = static_cast<int>(random_at_mostL(numcols));
  MPI_Comm_split(comm, color, rank, &comm_s);
  pprintI("color", color, comm);

  double* t1  = new double[nt];
  double* ts1 = new double[nt];
  double* t2  = new double[nt];
  double* ts2 = new double[nt];

  double delay=1./double(numcols);
  printf("Sleeping 0 - %f\n", MPI_Wtime());
  sleep( static_cast<int>(color*delay*trun) );
  printf("Woke 0 - %f\n", MPI_Wtime());
  a2a(n, nt, trun, t1, ts1, comm_s);
  MPI_Barrier(comm);

  for(unsigned int c=0; c<numcols; c++){
    if(c==color){
      output_timing(const_cast<char*>(std::to_string(c).c_str()), t1, ts1, nt, comm_s);
      fflush(stdout);
    }
    MPI_Barrier(comm);
  }

  //clean up
  delete [] t1,ts1,t2,ts2;

  MPI_Finalize();
  return 0;
}


/*
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

*/
