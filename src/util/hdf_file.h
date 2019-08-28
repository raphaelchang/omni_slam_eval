#ifndef _HDF_FILE_H_
#define _HDF_FILE_H_

#include <H5Cpp.h>
#include <vector>
#include <map>

using namespace H5;

namespace omni_slam
{
namespace util
{

class HDFFile
{
public:
    HDFFile(std::string name);
    bool AddDataset(std::string name, std::vector<std::vector<double>> &data);
    bool AddAttribute(std::string name, double value);
    bool AddAttribute(std::string name, std::string value);
    bool AddAttributes(std::map<std::string, double> attributes);

private:
    H5File file_;
};

}
}

#endif /* _HDF_FILE_H_ */
