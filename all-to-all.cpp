#include "global.hpp"

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

void output_timing(char* str, double *t, double *ts, int nt, MPI_Comm comm) {
  int i, rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  double tmin[nt],tmax[nt],tavg[nt];
  for (i=0; i<nt; i++) {
    MPI_Reduce(&t[i], &tmin[i], 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t[i], &tmax[i], 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t[i], &tavg[i], 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    tavg[i] /= nprocs;
  }

  if (rank==0) {
    for (i=0; i<nt; i++) {
      printf("%-15s ts,min,max,avg = %f %f %f %f\n", str, ts[i], tmin[i], tmax[i], tavg[i]);
    }
  }

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
  MPI_Init(&argc, &argv);
  MPI_Comm comm, comm_s;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  //int c = color(comm);
  srand(rank);
  int color = (int) random_at_mostL(1);

  MPI_Comm_split(comm, color, rank, &comm_s);

  pprintI("color", color, comm);

  int n=8, nt=1000;
  double trun = 20.0;

  double t1[nt], ts1[nt];
  double t2[nt], ts2[nt];

  if ( color == 0 ) {
    printf("Sleeping 0 - %f\n", MPI_Wtime());
    sleep( (int) (0.5*trun) );
    printf("Woke 0 - %f\n", MPI_Wtime());
    a2a(n, nt, trun, t1, ts1, comm_s);
  } else {
    printf("Woke 1 - %f\n", MPI_Wtime());
    a2a(n, nt, trun, t2, ts2, comm_s);
  }
  MPI_Barrier(comm);

  if (color == 0) {
    output_timing("0", t1, ts1, nt, comm_s);
    fflush(stdout);
  }
  MPI_Barrier(comm);
  if (color == 1) {
    output_timing("1", t2, ts2, nt, comm_s);
    fflush(stdout);
  }

  MPI_Finalize();
  return 0;
}


/*
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

*/
