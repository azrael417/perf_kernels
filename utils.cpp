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
