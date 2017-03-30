#ifndef _UTILS_H
#define _UTILS_H

//some types
typedef unsigned long long int Ullong;
typedef unsigned int Uint;

//input parser class:
class InputParser{
  public:
    InputParser (int&, char**);
    const std::string& getCmdOption(const std::string&)const;
    bool cmdOptionExists(const std::string&) const;
  private:
    std::vector <std::string> tokens;
};

//string magic:
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

//string splitting
std::vector<std::string> split(const std::string&,char);

//output time printing
void output_timing(char*,double*,double*,int,MPI_Comm);

//get physical hostname of node
std::string get_hostname(const std::string& prefix="nid");

//serial RNG, same Random numbers on every node. DO NOT USE FOR MC AND NEVER ON 32bit SYSTEMS!! This guy is 64bit only!!!
struct Ran{
  Ullong u,v,w;
Ran(Ullong j=0) : v(4101842887655102017LL), w(1){
  //Default seed:
  if(j==0){
    j=150301;
  }
  u = j ^ v; int64();
  v = u; int64();
  w = v; int64();
}
  inline Ullong int64(){
    u = u*2862933555777941757LL + 7046029254386353087LL;
    v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
    w = 4294957665U*(w & 0xffffffff) + ( w >> 32 );
    Ullong x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
    return (x + v) ^ w;
  }
  inline double doub() { return 5.42101086242752217E-20 * int64(); }
  inline Uint int32() { return static_cast<Uint>(int64()); }

  void getState(std::vector<Ullong>& state){
    state.resize(3);
    state[0]=u;
    state[1]=v;
    state[2]=w;
  }

  void setState(const std::vector<Ullong>& state){
    if(state.size()==3){
      u=state[0];
      v=state[1];
      w=state[2];
    }
  }
};

#endif