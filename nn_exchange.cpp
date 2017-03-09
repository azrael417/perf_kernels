#include "global.hpp"
#include "input.hpp"

//this is some proxy for doing work: sleeptime is in miliseconds
void do_some_work(unsigned int sleeptime){
  sleep(sleeptime*1000);
}

int main(int argc, char* argv[]){
  //init MPI
  int required=MPI_THREAD_FUNNELED, provided;
  MPI_Init_thread(&argc,&argv,required,&provided);
  if(required!=provided){
    std::cerr << "Error, asked for MPI thread level " << required << " but got " << provided << ", abort." << std::endl;
    return EXIT_FAILURE;
  }

  //init parser
  InputParser input(argc, argv);

  //parse box size arguments:
  int nx=0, ny=0, nz=0;
  if(input.cmdOptionExists("--grid")){
       // Do stuff
       std::string gridstring = input.getCmdOption("--grid");
       std::vector<std::string> gridsplit=split(gridstring,'.');
       if(gridsplit.size()!=3){
         std::cerr << "Error, please specify x, y and z extent." << std::endl;
         return EXIT_FAILURE;
       }
       nx=atoi(gridsplit[0].c_str());
       ny=atoi(gridsplit[1].c_str());
       nz=atoi(gridsplit[2].c_str());
  }
  else{
    std::cerr << "Please specify a grid with --grid nx.ny.nz" << std::endl;
    return EXIT_FAILURE;
  }

  //parse numnodes arguments:
  int nnx=0, nny=0, nnz=0;
  if(input.cmdOptionExists("--mpi")){
       // Do stuff
       std::string gridstring = input.getCmdOption("--mpi");
       std::vector<std::string> gridsplit=split(gridstring,'.');
       if(gridsplit.size()!=3){
         std::cerr << "Error, please specify the number of nodes in x, y and z direction." << std::endl;
         return EXIT_FAILURE;
       }
       nnx=atoi(gridsplit[0].c_str());
       nny=atoi(gridsplit[1].c_str());
       nnz=atoi(gridsplit[2].c_str());
       if( (nx%nnx!=0) || (ny%nny!=0) || (nz%nnz!=0)){
         std::cerr << "Error, please make sure that the grid dimensions can be divided by the number of nodes in each direction." << std::endl;
         return EXIT_FAILURE;
       }
       int commsize=1;
       MPI_Comm_size(MPI_COMM_WORLD, &commsize);
       if(commsize!=nnx*nny*nnz){
         std::cerr << "Error, the total number of ranks should match the mpi decompsition." << std::endl;
         return EXIT_FAILURE;
       }
  }
  else{
    std::cerr << "Please specify an mpi-grid with --mpi nnx.nny.nnz" << std::endl;
    return EXIT_FAILURE;
  }

  //parse number of iterations:
  int maxiter=1;
  if(input.cmdOptionExists("--niter")){
    maxiter=atoi(input.getCmdOption("--niter").c_str());
  }
  //parse sleeptime
  //default sleeptime is 1 second
  unsigned int sleeptime=1000;
  if(input.cmdOptionExists("--sleeptime")){
    sleeptime=static_cast<unsigned int>(atoi(input.getCmdOption("--sleeptime").c_str()));
  }

  //local sizes
  int* lgrid=new int[3];
  lgrid[0]=nx/nnx;
  lgrid[1]=ny/nny;
  lgrid[2]=nz/nnz;
  //create array to store field:
  double* u = new double[lgrid[0]*lgrid[1]*lgrid[2]];
  //use another pointer for easier access:
  double** umat=new double*[6];
  //plus and minus x
  //x=y=z=0
  umat[0]=&u[0];
  //x=nx-1,y=z=0
  umat[1]=&u[lgrid[0]-1];
  //plus and minus y
  //x=y=z=0
  umat[2]=&u[0];
  //x=z=0,y=lgrid[1]-1
  umat[3]=&u[lgrid[0]*(lgrid[1]-1)];
  //x=y=z=0
  umat[4]=&u[0];
  //x=y=0,z=lgrid[2]-1
  umat[5]=&u[lgrid[0]*lgrid[1]*(lgrid[2]-1)];

  //let's fill the array, make x fastest, y second fastest and z slowest:
  for(unsigned int z=0; z<lgrid[2]; z++){
    for(unsigned int y=0; y<lgrid[1]; y++){
      for(unsigned int x=0; x<lgrid[0]; x++) u[x+lgrid[0]*(y+lgrid[1]*z)]=double(x+lgrid[0]*(y+lgrid[1]*z));
    }
  }

  //create MPI datatypes:
  MPI_Datatype* slices=new MPI_Datatype[3];
  MPI_Type_contiguous(lgrid[0],MPI_DOUBLE,&slices[0]);
  MPI_Type_vector(lgrid[1],1,lgrid[0],MPI_DOUBLE,&slices[1]);
  MPI_Type_vector(lgrid[3],1,lgrid[0]*lgrid[1],MPI_DOUBLE,&slices[2]);
  //commit types
  for(unsigned int dir=0; dir<3; dir++) MPI_Type_commit( &slices[dir] );

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
  int myrank=0;
  int* nnranks=new int[6];
  MPI_Comm_rank(comm3d,&myrank);
  //store the coords of the neighbors:
  for(unsigned int dir=0; dir<3; dir++){
    MPI_Cart_shift(comm3d, dir, 1, &nnranks[2*dir], &nnranks[2*dir+1]);
  }

  //create some receive buffers:
  double** rbuf=new double*[6];
  for(unsigned int dir=0; dir<3; dir++){
    rbuf[2*dir]=new double[lgrid[dir]];
    rbuf[2*dir+1]=new double[lgrid[dir]];
  }

  //start the kernel:
  MPI_Barrier(comm3d);
  for(unsigned int iter=0; iter<maxiter; iter++){
    //comm in directions: issue send-receives
    std::queue<MPI_Request> queue;
    for(unsigned int dir=0; dir<3; dir++){
      //define some vars
      MPI_Request req;

      //+dir direction
      //issue receive from -dir:
      MPI_Irecv(rbuf[2*dir], lgrid[dir], slices[dir], nnranks[2*dir], 0, comm3d, &req);
      queue.push(req);
      //send to +dir:
      MPI_Isend(umat[2*dir+1], lgrid[dir], slices[dir], nnranks[2*dir+1], 0, comm3d, &req);
      queue.push(req);

      //-dir direction
      //issue receive from +dir:
      MPI_Irecv(rbuf[2*dir+1], lgrid[dir], slices[dir], nnranks[2*dir+1], 0, comm3d, &req);
      queue.push(req);
      //send to -dir:
      MPI_Isend(umat[2*dir], lgrid[dir], slices[dir], nnranks[2*dir], 0, comm3d, &req);
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
      if(!flag){
        queue.push(req);
      }
    }

    //iteration completed
  }
  MPI_Barrier(comm3d);

  //clean up:
  delete [] u;
  delete [] slices;
  delete [] nnranks;
  for(unsigned int dir=0; dir<3; dir++){
    delete [] rbuf[2*dir];
    delete [] rbuf[2*dir+1];
  }
  delete [] rbuf;
  delete [] umat;
  delete [] lgrid;

  //MPI_Finalize();
  return EXIT_SUCCESS;
}
