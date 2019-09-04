#include "reconstruction_module.h"

using namespace std;

namespace omni_slam
{
namespace module
{

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &triangulator)
    : triangulator_(std::move(triangulator))
{
}

ReconstructionModule::ReconstructionModule(std::unique_ptr<reconstruction::Triangulator> &&triangulator)
    : ReconstructionModule(triangulator)
{
}

void ReconstructionModule::Update(std::vector<data::Landmark> &landmarks)
{
    triangulator_->Triangulate(landmarks);
}

ReconstructionModule::Stats& ReconstructionModule::GetStats()
{
    return stats_;
}

void ReconstructionModule::Visualize()
{
}

}
}
