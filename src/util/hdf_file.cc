#include "hdf_file.h"

#include <memory>
#include <algorithm>
#include <iostream>

namespace omni_slam
{
namespace util
{

HDFFile::HDFFile(std::string name)
    : file_(H5std_string(name.c_str()), H5F_ACC_TRUNC)
{
    try
    {
        hsize_t dims[1] = {0};
        DataSpace dataspace(1, dims);
        DataSet dataset = file_.createDataSet("attributes", PredType::NATIVE_UCHAR, dataspace);
    }
    catch(FileIException error)
    {
        error.printErrorStack();
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
    }
}

bool HDFFile::AddDataset(std::string name, std::vector<std::vector<double>> &data)
{
    try
    {
        Exception::dontPrint();
        hsize_t dims[2];
        dims[0] = data.size();
        dims[1] = std::max_element(std::begin(data), std::end(data), [](const std::vector<double> &lhs, const std::vector<double> &rhs) { return lhs.size() < rhs.size(); })->size();
        DataSpace dataspace(2, dims);
        DataSet dataset = file_.createDataSet(H5std_string(name.c_str()), PredType::NATIVE_DOUBLE, dataspace);
        std::unique_ptr<double[]> dataBuf;
        dataBuf.reset(new double[dims[0] * dims[1]]);
        for (int i = 0; i < data.size(); i++)
        {
            for (int j = 0; j < data[i].size(); j++)
            {
                dataBuf[i * dims[1] + j] = data[i][j];
            }
        }
        dataset.write(dataBuf.get(), PredType::NATIVE_DOUBLE);
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        return false;
    }
    return true;
}

bool HDFFile::AddAttribute(std::string name, double value)
{
    try
    {
        hsize_t dims[1] = {1};
        DataSpace dataspace(1, dims);
        DataSet dataset = file_.openDataSet("attributes");
        Attribute attr = dataset.createAttribute(H5std_string(name.c_str()), PredType::NATIVE_DOUBLE, dataspace);
        attr.write(PredType::NATIVE_DOUBLE, &value);
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        return false;
    }
    return true;
}

bool HDFFile::AddAttribute(std::string name, std::string value)
{
    try
    {
        DataSpace dataspace(H5S_SCALAR);
        DataSet dataset = file_.openDataSet("attributes");
        StrType strdatatype(PredType::C_S1, value.length());
        Attribute attr = dataset.createAttribute(H5std_string(name.c_str()), strdatatype, dataspace);
        const H5std_string strbuf(value.c_str());
        attr.write(strdatatype, strbuf);
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        return false;
    }
    return true;
}

bool HDFFile::AddAttributes(std::map<std::string, double> attributes)
{
    try
    {
        DataSet dataset = file_.openDataSet("attributes");
        for (auto &attribute : attributes)
        {
            hsize_t dims[1] = {1};
            DataSpace dataspace(1, dims);
            Attribute attr = dataset.createAttribute(H5std_string(attribute.first.c_str()), PredType::NATIVE_DOUBLE, dataspace);
            attr.write(PredType::NATIVE_DOUBLE, &attribute.second);
        }
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        return false;
    }
    return true;
}

bool HDFFile::AddAttributes(std::map<std::string, std::string> attributes)
{
    try
    {
        DataSet dataset = file_.openDataSet("attributes");
        for (auto &attribute : attributes)
        {
            DataSpace dataspace(H5S_SCALAR);
            StrType strdatatype(PredType::C_S1, attribute.second.length());
            Attribute attr = dataset.createAttribute(H5std_string(attribute.first.c_str()), strdatatype, dataspace);
            const H5std_string strbuf(attribute.second.c_str());
            attr.write(strdatatype, strbuf);
        }
    }
    catch(FileIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSetIException error)
    {
        error.printErrorStack();
        return false;
    }
    catch(DataSpaceIException error)
    {
        error.printErrorStack();
        return false;
    }
    return true;
}

}
}
