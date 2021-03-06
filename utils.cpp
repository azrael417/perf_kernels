#include "global.hpp"
#include "utils.hpp"

//Input parser
InputParser::InputParser (int &argc, char **argv){
  for (int i=1; i < argc; ++i)
    this->tokens.push_back(std::string(argv[i]));
}

const std::string& InputParser::getCmdOption(const std::string &option) const{
  std::vector<std::string>::const_iterator itr;
  itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
  if (itr != this->tokens.end() && ++itr != this->tokens.end()){
    return *itr;
  }
  static const std::string empty_string("");
  return empty_string;
}

bool InputParser::cmdOptionExists(const std::string &option) const{
  return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
}

//string magic:
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

//output timings
void output_timing(char* str, double *t, double *ts, int nt, MPI_Comm comm) {
  int i, rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  double* tmin=new double[nt];
  double* tmax=new double[nt];
  double* tavg=new double[nt];
  MPI_Reduce(t, tmin, nt, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(t, tmax, nt, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(t, tavg, nt, MPI_DOUBLE, MPI_SUM, 0, comm);
  for(unsigned int i=0; i<nt; i++) tavg[i] /= nprocs;

  if (rank==0) {
    for (i=0; i<nt; i++) {
      printf("%-15s ts,min,max,avg = %f %f %f %f\n", str, ts[i], tmin[i], tmax[i], tavg[i]);
    }
  }
  delete [] tmin,tmax,tavg;
  MPI_Barrier(comm);
  return;
}

int get_hostid(){
	int pmirank=-1;
	int nid=-1;
	PMI_Get_rank(&pmirank);
	PMI_Get_nid(pmirank, &nid);
	return nid;
}

std::string hostid_to_name(const int& hostid, const std::string& prefix){
	std::stringstream stream;
	stream << prefix << std::setfill('0') << std::setw(5) << hostid;
	return stream.str();
}

std::string get_hostname(const std::string& prefix){
	int pmirank=-1;
	int nid=-1;
	PMI_Get_rank(&pmirank);
	PMI_Get_nid(pmirank, &nid);
	std::stringstream stream;
	stream << prefix << std::setfill('0') << std::setw(5) << nid;
	return stream.str();
}