#ifndef _INPUT_H
#define _INPUT_H

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

std::vector<std::string> split(const std::string&,char);
#endif
