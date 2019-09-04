#ifndef _RECONSTRUCTION_MODULE_H_
#define _RECONSTRUCTION_MODULE_H_

#include <vector>
#include <memory>

#include "reconstruction/triangulator.h"
#include "data/landmark.h"

namespace omni_slam
{
namespace module
{

class ReconstructionModule
{
public:
    struct Stats
    {
    };

    ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &triangulator);
    ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &&triangulator);

    void Update(std::vector<data::Landmark> &landmarks);

    Stats& GetStats();
    void Visualize();

private:
    class Visualization
    {
    };

    std::shared_ptr<reconstruction::Triangulator> triangulator_;

    Stats stats_;
    Visualization visualization_;
};

}
}

#endif /* _RECONSTRUCTION_MODULE_H_ */
