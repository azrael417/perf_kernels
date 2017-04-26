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
  
	//get arguments:
	int N=8192, niter=10;
	int mode=0, detailed=0;
	ssize_t streamsize=10000000;
	if(myrank==0){
		InputParser input(argc, argv);
		if(input.cmdOptionExists("--rank")){
			if(input.cmdOptionExists("--rank")){
				// Do stuff
				N = atoi(input.getCmdOption("--rank").c_str());
			}
		}
		if(input.cmdOptionExists("--niter")){
			if(input.cmdOptionExists("--niter")){
				// Do stuff
				niter = atoi(input.getCmdOption("--niter").c_str());
			}
		}
		if(input.cmdOptionExists("--streamsize")){
			if(input.cmdOptionExists("--streamsize")){
				// Do stuff
				streamsize = atoi(input.getCmdOption("--streamsize").c_str());
			}
		}
		if(input.cmdOptionExists("--csv")) mode=1;
		if(input.cmdOptionExists("--detailed")) detailed=1;
		
		if( (detailed==1) && (mode==0) ){
			std::cout << "Warning, detailed printouts only available in csv mode, please specify --csv. Defaulting to normal printout." << std::endl;
			detailed=0;
		}
	}
	MPI_Bcast(&N,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&niter,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&streamsize,1,MPI_LONG,0,MPI_COMM_WORLD);
	MPI_Bcast(&mode,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	MPI_Bcast(&detailed,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	
	//print what is going on
	if( (myrank==0) && (mode==0) ) std::cout << "--- starting DGEMM minibench ---" << std::endl;

	//set up matrix:
	double *m1 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));
	double *m2 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));
	double *m3 = reinterpret_cast<double*>(aligned_alloc(64, N*N*sizeof(double)));
	
	//gflop for this operation:
	double gflop=2.0*N*N*(N+1)*1E-9;

	//set up seed for seeded random numbers:
	if( (myrank==0) && (mode==0) ) std::cout << "preparing DGEMM matrix" << std::endl;
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
	double ts,t,tmin,tmax,tave, pmin, pmax, pave;
	double alpha=1., beta=0.;
	double* tvals;
	if(detailed) tvals=new double[niter];

	//run and time
	if( (myrank==0) && (mode==0) ) std::cout << "running DGEMM benchmark" << std::endl;
	//exclude the first test
	dgemm(&TA,&TB,&N,&N,&N,&alpha,m1,&N,m2,&N,&beta,m3,&N);
	//now do serious testing
	ts=MPI_Wtime();
	dgemm(&TA,&TB,&N,&N,&N,&alpha,m1,&N,m2,&N,&beta,m3,&N);
	t=MPI_Wtime()-ts;
	tmin=t; tmax=t; tave=t;
	pave=gflop/t;
	for(unsigned int i=1; i<=niter; i++){
		ts=MPI_Wtime();
		dgemm(&TA,&TB,&N,&N,&N,&alpha,m1,&N,m2,&N,&beta,m3,&N);
		t=MPI_Wtime()-ts;
		
		//time
		tmin=(t<tmin ? t : tmin);
		tmax=(t>tmax ? t : tmax);
		tave+=t;
		
		//performance
		pave+=gflop/t;
		
		//if detailed, store t:
		if(detailed) tvals[i-1]=t;
	}
	tave/=double(niter);
	pave/=double(niter);
	MPI_Barrier(MPI_COMM_WORLD);

	//gather data from the nodes:
	//timing
	double* tminv=new double[numranks];
	double* tmaxv=new double[numranks];
	double* tavev=new double[numranks];
	//performance
	double* pavev=new double[numranks];
	//full time results:
	double* tvalsv;
	if(detailed) tvalsv=new double[niter*numranks];
	//some other stuff
	int* hostidv=new int[numranks];
	int hostid=get_hostid();
	MPI_Gather(&tmin,1,MPI_DOUBLE,tminv,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&tmax,1,MPI_DOUBLE,tmaxv,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&tave,1,MPI_DOUBLE,tavev,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&pave,1,MPI_DOUBLE,pavev,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&hostid,1,MPI_INTEGER,hostidv,1,MPI_INTEGER,0,MPI_COMM_WORLD);
	if(detailed) MPI_Gather(tvals,niter,MPI_DOUBLE,tvalsv,niter,MPI_DOUBLE,0,MPI_COMM_WORLD);

	//ordered timings printed
	if(myrank==0){
		switch(mode) {
			case 0:
				for(unsigned int rank=0; rank<numranks; rank++){
					std::cout << "DGEMM time (size: " << N << "x" << N << ", niter: " << niter;
					std::cout << ", rank: " << rank;
					std::cout << ", hostname: " << hostid_to_name(hostidv[rank]);
					std::cout << "): t_min = " << tminv[rank];
					std::cout << ", t_max = " << tmaxv[rank];
					std::cout << ", t_mean = " << tavev[rank] << std::endl;
					std::cout << ", p_min = " << gflop/tmaxv[rank];
					std::cout << ", p_max = " << gflop/tminv[rank];
					std::cout << ", p_mean = " << pavev[rank] << std::endl;
				}
				break;
			case 1:
				if(detailed){
					std::cout << "benchmark,rank,hostname,iter,t,p" << std::endl;
					for(unsigned int rank=0; rank<numranks; rank++){
						for(unsigned int i=0; i<niter; i++){
							std::cout << "DGEMM("<< N << "x" << N <<"),";
							std::cout << rank << ",";
							std::cout << hostid_to_name(hostidv[rank]) << ",";
							std::cout << i << ",";
							//timings
							std::cout << tvalsv[i+rank*niter] << ",";
							//performance
							std::cout << gflop/tvalsv[i+rank*niter];
							std::cout << std::endl;
						}
					}
				}
				else{
					std::cout << "benchmark,niter,rank,hostname,t_min,t_max,t_mean,p_min,p_max,p_mean" << std::endl;
					for(unsigned int rank=0; rank<numranks; rank++){
						std::cout << "DGEMM("<< N << "x" << N <<"),";
						std::cout << niter << ",";
						std::cout << rank << ",";
						std::cout << hostid_to_name(hostidv[rank]) << ",";
						//timings
						std::cout << tminv[rank] << ",";
						std::cout << tmaxv[rank] << ",";
						std::cout << tavev[rank] << ",";
						//performance
						std::cout << gflop/tmaxv[rank] << ",";
						std::cout << gflop/tminv[rank] << ",";
						std::cout << pavev[rank];
						std::cout << std::endl;
					}
				}
				break;
			}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//cleaning up
	delete [] m1, m2, m3;
	
	//print finishing string
	if( (myrank==0) && (mode==0) ) std::cout << "--- finishing DGEMM minibench ---" << std::endl;


	//print stream setup
	if( (myrank==0) && (mode==0) ) std::cout << "--- starting STREAM minibench ---" << std::endl;
	
	//allocate arrays
	m1=new double[streamsize];
	m2=new double[streamsize];
	m3=new double[streamsize];

	//double: GB transferred for this operation
	double gb=streamsize*16./(1024.*1024.*1024.);

	//constant
	const double scalar=3.0;

	//run triad
	if( (myrank==0) && (mode==0) ) std::cout << "running STREAM benchmark" << std::endl;
	//exclude the first test
#pragma omp parallel for
	for (ssize_t j=0; j<streamsize; j++){
		m1[j] = m2[j]+scalar*m3[j];
	}
	//now do serious testing
	ts=MPI_Wtime();
#pragma omp parallel for
	for (ssize_t j=0; j<streamsize; j++){
		m1[j] = m2[j]+scalar*m3[j];
	}
	t=MPI_Wtime()-ts;
	tmin=t; tmax=t; tave=t;
	pave=gb/t;
	for(unsigned int i=1; i<niter; i++){
		ts=MPI_Wtime();
#pragma omp parallel for
		for (ssize_t j=0; j<streamsize; j++){
	    	m1[j] = m2[j]+scalar*m3[j];
		}
		t=MPI_Wtime()-ts;
		
		//time
		tmin=(t<tmin ? t : tmin);
		tmax=(t>tmax ? t : tmax);
		tave+=t;
		
		//bandwidth
		pave+=gb/t;
	}
	tave/=double(niter);
	pave/=double(niter);
	MPI_Barrier(MPI_COMM_WORLD);

	//gather results
	MPI_Gather(&tmin,1,MPI_DOUBLE,tminv,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&tmax,1,MPI_DOUBLE,tmaxv,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&tave,1,MPI_DOUBLE,tavev,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Gather(&pave,1,MPI_DOUBLE,pavev,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

	//ordered timings printed
	if(myrank==0){
		switch(mode) {
			case 0:
				for(unsigned int rank=0; rank<numranks; rank++){
					std::cout << "STREAM-TRIAD time (size: " << streamsize*16./(1024.*1024.) << "MiB, niter: " << niter;
					std::cout << ", rank: " << rank;
					std::cout << ", hostname: " << hostid_to_name(hostidv[rank]);
					std::cout << "): t_min = " << tminv[rank];
					std::cout << ", t_max = " << tmaxv[rank];
					std::cout << ", t_mean = " << tavev[rank];
					std::cout << ", p_min = " << gb/tmaxv[rank];
					std::cout << ", p_max = " << gb/tminv[rank];
					std::cout << ", p_mean = " << pavev[rank];
					std::cout << std::endl;
				}
				break;
			case 1:
				if(detailed){
					for(unsigned int rank=0; rank<numranks; rank++){
						for(unsigned int i=0; i<niter; i++){
							std::cout << "STREAM-TRIAD("<< streamsize*16./(1024.*1024.) <<"MiB),";
							std::cout << rank << ",";
							std::cout << hostid_to_name(hostidv[rank]) << ",";
							std::cout << i << ",";
							//time
							std::cout << tvalsv[i+niter*rank] << ",";
							//bandwidth
							std::cout << gb/tvalsv[i+niter*rank];
							std::cout << std::endl;
						}
					}
				}
				else{
					for(unsigned int rank=0; rank<numranks; rank++){
						std::cout << "STREAM-TRIAD("<< streamsize*16./(1024.*1024.) <<"MiB),";
						std::cout << niter << ",";
						std::cout << rank << ",";
						std::cout << hostid_to_name(hostidv[rank]) << ",";
						//time
						std::cout << tminv[rank] << ",";
						std::cout << tmaxv[rank] << ",";
						std::cout << tavev[rank] << ",";
						//bandwidth
						std::cout << gb/tmaxv[rank] << ",";
						std::cout << gb/tminv[rank] << ",";
						std::cout << pavev[rank];
						std::cout << std::endl;
					}
				}
				break;
			}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	//finishing
	if( (myrank==0) && (mode==0) ) std::cout << "--- finishing STREAM minibench ---" << std::endl;

	//cleaning up
	delete [] tminv, tmaxv, tavev, hostidv;
	if(detailed){
		delete [] tvalsv;
		delete [] tvals;
	}

	MPI_Finalize();
}
