#include "global.hpp"
#include "input.hpp"

//face packing
void pack_face(int odir, int* lgrid, double* arr, double* sbuf){
  int orient=odir%2;
  int dir=(odir-orient)/2;

  int foff=(odir==1 ? lgrid[dir] : 1);
  switch(dir){
    case 0:
#pragma omp parallel for collapse(2)
      for(unsigned int z=1; z<=lgrid[2]; z++){
        for(unsigned int y=1; y<=lgrid[1]; y++){
          sbuf[(y-1)+lgrid[1]*(z-1)]=arr[foff+(lgrid[0]+2)*(y+(lgrid[1]+2)*z)];
        }
      }
      break;
    case 1:
#pragma omp parallel for collapse(2)
      for(unsigned int z=1; z<=lgrid[2]; z++){
        for(unsigned int x=1; x<=lgrid[0]; x++){
          sbuf[(x-1)+lgrid[0]*(z-1)]=arr[x+(lgrid[0]+2)*(foff+(lgrid[1]+2)*z)];
        }
      }
      break;
    case 2:
#pragma omp parallel for collapse(2)
      for(unsigned int y=1; y<=lgrid[1]; y++){
        for(unsigned int x=1; x<=lgrid[0]; x++){
          sbuf[(x-1)+lgrid[0]*(y-1)]=arr[x+(lgrid[0]+2)*(y+(lgrid[1]+2)*foff)];
        }
      }
      break;
  }
}

//this is some proxy for doing work: sleeptime is in microseconds
void do_some_work(unsigned int sleeptime){
  usleep(sleeptime);
}

int main(int argc, char* argv[]){
  //init MPI
  int required=MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc,&argv,required,&provided);
  if(required!=provided){
    std::cerr << "Error, asked for MPI thread level " << required << " but got " << provided << ", abort." << std::endl;
    return EXIT_FAILURE;
  }

  //get my rank:
  int myrank=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  //init parser
  int error=0;
  int nx=0, ny=0, nz=0;
  int nnx=0, nny=0, nnz=0;
  int maxiter=1;
  unsigned int sleeptime=100;
  if(myrank==0){
    std::cout << "reading input parameters." << std::endl;

    InputParser input(argc, argv);

    //parse box size arguments:
    if(input.cmdOptionExists("--grid")){
        // Do stuff
        std::string gridstring = input.getCmdOption("--grid");
        std::vector<std::string> gridsplit=split(gridstring,'.');
        if(gridsplit.size()!=3){
          std::cerr << "Error, please specify x, y and z extent." << std::endl;
          error=1;
        }
        else{
          nx=atoi(gridsplit[0].c_str());
          ny=atoi(gridsplit[1].c_str());
          nz=atoi(gridsplit[2].c_str());
        }
    }
    else{
      std::cerr << "Please specify a grid with --grid nx.ny.nz" << std::endl;
      error=1;
    }

    //parse numnodes arguments:
    if(input.cmdOptionExists("--mpi")){
        // Do stuff
        std::string gridstring = input.getCmdOption("--mpi");
        std::vector<std::string> gridsplit=split(gridstring,'.');
        if(gridsplit.size()!=3){
          std::cerr << "Error, please specify the number of nodes in x, y and z direction." << std::endl;
          error=1;
        }
        else{
          nnx=atoi(gridsplit[0].c_str());
          nny=atoi(gridsplit[1].c_str());
          nnz=atoi(gridsplit[2].c_str());
          if( (nx%nnx!=0) || (ny%nny!=0) || (nz%nnz!=0)){
            std::cerr << "Error, please make sure that the grid dimensions can be divided by the number of nodes in each direction." << std::endl;
            error=1;
          }
          int commsize=1;
          MPI_Comm_size(MPI_COMM_WORLD, &commsize);
          if(commsize!=nnx*nny*nnz){
            std::cerr << "Error, the total number of ranks should match the mpi decompsition." << std::endl;
            error=1;
          }
        }
    }
    else{
      std::cerr << "Please specify an mpi-grid with --mpi nnx.nny.nnz" << std::endl;
      error=1;
    }

    //parse number of iterations:
    if(input.cmdOptionExists("--niter")){
      maxiter=atoi(input.getCmdOption("--niter").c_str());
    }
    //parse sleeptime
    //default sleeptime is 1 second
    if(input.cmdOptionExists("--sleeptime")){
      sleeptime=static_cast<unsigned int>(atoi(input.getCmdOption("--sleeptime").c_str()));
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&error,1,MPI_INTEGER,0,MPI_COMM_WORLD);

  if(error){
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  //communicate the read-data: inelegant but works
  if(myrank==0){
    std::cout << "broadcasting input parameters." << std::endl;
  }
  MPI_Bcast(&nx,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&ny,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&nz,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&nnx,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&nny,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&nnz,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&maxiter,1,MPI_INTEGER,0,MPI_COMM_WORLD);
  MPI_Bcast(&sleeptime,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);

  //local sizes
  if(myrank==0){
    std::cout << "setting up problem." << std::endl;
  }
  int* lgrid=new int[3];
  lgrid[0]=nx/nnx;
  lgrid[1]=ny/nny;
  lgrid[2]=nz/nnz;
  //create array to store field including faces:
  double* u = new double[(lgrid[0]+2)*(lgrid[1]+2)*(lgrid[2]+2)];
  /*//use another pointer for easier access:
  double**** umat=new double***[lgrid[0]+2];
  for(unsigned int x=0; x<lgrid[0]+2; x++){
    umat[x]=new double**[lgrid[1]+2];
    for(unsigned int y=0; y<lgrid[1]+2; y++){
      umat[x][y]=new double*[lgrid[2]+2];
      for(unsigned int z=0; z<lgrid[2]+2; z++){
        umat[x][y][z]=&u[x+lgrid[0]*(y+lgrid[1]*z)];
      }
    }
  }*/

  //let's fill the array, make x fastest, y second fastest and z slowest:
#pragma omp parallel for
  for(unsigned int x=0; x<((lgrid[0]+2)*(lgrid[1]+2)*(lgrid[2]+2)); x++) u[x]=-1.;
#pragma omp parallel for collapse(3)
  for(unsigned int z=1; z<=lgrid[2]; z++){
    for(unsigned int y=1; y<=lgrid[1]; y++){
      for(unsigned int x=1; x<=lgrid[0]; x++){
        u[x+(lgrid[0]+2)*(y+(lgrid[1]+2)*z)]=double(x+(lgrid[0]+2)*(y+(lgrid[1]+2)*z));
      }
    }
  }

  //create MPI datatypes:
  MPI_Datatype* slices=new MPI_Datatype[3];
  MPI_Type_contiguous(lgrid[1]*lgrid[2],MPI_DOUBLE,&slices[0]);
  MPI_Type_contiguous(lgrid[0]*lgrid[2],MPI_DOUBLE,&slices[1]);
  MPI_Type_contiguous(lgrid[0]*lgrid[1],MPI_DOUBLE,&slices[2]);
  //commit slice-types
  for(unsigned int dir=0; dir<3; dir++) MPI_Type_commit( &slices[dir] );

  /*//in x, slice of size y*z is communicated:
  MPI_Datatype yslice;
  //create vector of ny elements:
  MPI_Type_vector(lgrid[1],1,(lgrid[0]+2),MPI_DOUBLE,&yslice);
  MPI_Type_commit(&yslice);
  //now gather nz elements of ny vectory:
  MPI_Type_vector(lgrid[2],1,(lgrid[0]+2)*(lgrid[1]+2),yslice,&slices[0]);
  //in y, slice of size x*z is communicated:
  MPI_Type_vector(lgrid[2],lgrid[0],(lgrid[0]+2)*(lgrid[1]+2),MPI_DOUBLE,&slices[1]);
  //in z, slice of size x*y is communicated
  MPI_Type_vector(lgrid[1],lgrid[0],2,MPI_DOUBLE,&slices[2]);

  //check extents:
  for(unsigned int dir=0; dir<3; dir++){
    long size=0;
    MPI_Type_extent(slices[dir],&size);
    if(myrank==0) std::cout << "Slice in direction " << dir << " has size " << size;
  }*/

  //create cartesian communicator:
  int* dims=new int[3];
  dims[0]=nnx;
  dims[1]=nny;
  dims[2]=nnz;
  MPI_Dims_create(nnx*nny*nnz, 3, dims);
  MPI_Comm comm3d;
  int* periodic=new int[3];
  periodic[0]=1;
  periodic[1]=1;
  periodic[2]=1;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, 1, &comm3d);
  delete [] periodic, dims;

  //get my rank and that of my neghbours:
  int* nnranks=new int[6];
  MPI_Comm_rank(comm3d,&myrank);
  //store the coords of the neighbors:
  for(unsigned int dir=0; dir<3; dir++){
    MPI_Cart_shift(comm3d, dir, 1, &nnranks[2*dir], &nnranks[2*dir+1]);
  }

  //create some receive buffers:
  double** rbuf=new double*[6];
  rbuf[0]=new double[lgrid[1]*lgrid[2]];
  rbuf[1]=new double[lgrid[1]*lgrid[2]];
  rbuf[2]=new double[lgrid[0]*lgrid[2]];
  rbuf[3]=new double[lgrid[0]*lgrid[2]];
  rbuf[4]=new double[lgrid[0]*lgrid[1]];
  rbuf[5]=new double[lgrid[0]*lgrid[1]];
  double** sbuf=new double*[6];
  sbuf[0]=new double[lgrid[1]*lgrid[2]];
  sbuf[1]=new double[lgrid[1]*lgrid[2]];
  sbuf[2]=new double[lgrid[0]*lgrid[2]];
  sbuf[3]=new double[lgrid[0]*lgrid[2]];
  sbuf[4]=new double[lgrid[0]*lgrid[1]];
  sbuf[5]=new double[lgrid[0]*lgrid[1]];

  /*//y*z slice
  rbuf[0]=umat[0][1][1];
  rbuf[1]=umat[lgrid[0]+1][1][1];
  //x*z slice
  rbuf[2]=umat[1][0][1];
  rbuf[3]=umat[1][lgrid[1]+1][1];
  //x*y slice
  rbuf[4]=umat[1][1][0];
  rbuf[5]=umat[1][1][lgrid[2]+1];
  //now the send buffers
  double** sbuf=new double*[6];
  //x-dir
  sbuf[0]=umat[1][1][1];
  sbuf[1]=umat[lgrid[0]][1][1];
  //y-dir
  sbuf[2]=umat[1][1][1];
  sbuf[3]=umat[1][lgrid[1]][1];
  //z-dir
  sbuf[4]=umat[1][1][1];
  sbuf[5]=umat[1][1][lgrid[2]];*/

  //start the kernel:
  if(myrank==0){
    std::cout << "starting the kernel." << std::endl;
  }
  MPI_Barrier(comm3d);
  for(unsigned int iter=0; iter<maxiter; iter++){
    //comm in directions: issue send-receives
    std::queue<MPI_Request> queue;

    //comms in x:
    for(unsigned int dir=0; dir<3; dir++){

      //define some vars
      MPI_Request req;

      //receives
      //issue receive from -dir:
      MPI_Irecv(rbuf[2*dir], 1, slices[dir], nnranks[2*dir], 0, comm3d, &req);
      queue.push(req);
      //isue receive from +dir:
      MPI_Irecv(rbuf[2*dir+1], 1, slices[dir], nnranks[2*dir+1], 0, comm3d, &req);
      queue.push(req);

      //issue sends:
      //send to +dir:
      pack_face(2*dir+1,lgrid,u,sbuf[2*dir+1]);
      MPI_Isend(sbuf[2*dir+1], 1, slices[dir], nnranks[2*dir+1], 0, comm3d, &req);
      queue.push(req);
      //send to -dir:
      pack_face(2*dir,lgrid,u,sbuf[2*dir]);
      MPI_Isend(sbuf[2*dir], 1, slices[dir], nnranks[2*dir], 0, comm3d, &req);
      queue.push(req);
    }

    //now do some work
    do_some_work(sleeptime);

    //now wait and receive:
    while(!queue.empty()){
      MPI_Request req=queue.front();
      queue.pop();
      int flag=0;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      //MPI_Wait(&req, MPI_STATUS_IGNORE);
      if(!flag){
        queue.push(req);
      }
    }
    //iteration completed
  }
  MPI_Barrier(comm3d);

  if(myrank==0){
    std::cout << "Test done." << std::endl;
  }

  //clean up:
  delete [] u;
  delete [] nnranks;
  for(unsigned int dir=0; dir<6; dir++){
    delete [] rbuf[dir];
    delete [] sbuf[dir];
  }
  delete [] rbuf;
  delete [] sbuf;
  /*for(unsigned int x=0; x<lgrid[0]+2; x++){
    for(unsigned int y=0; y<lgrid[1]+2; y++){
      delete [] umat[x][y];
    }
    delete umat[x];
  }
  delete [] umat;*/
  delete [] lgrid;

  //free types:
  for(unsigned int dir=0; dir<3; dir++) MPI_Type_free(&slices[dir]);
  delete [] slices;
  //MPI_Type_free(&yslice);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
