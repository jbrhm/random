///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/************************************************************************************
 ** This sample demonstrates how to use PCL (Point Cloud Library) with the ZED SDK **
 ************************************************************************************/

// TODO: PointXYZRGBNormal

// ZED includes
#include <sl/Camera.hpp>

// PCL includes
// Undef on Win32 min/max for PCL
#ifdef _WIN32
#undef max
#undef min
#endif
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Sample includes
#include <thread>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <string>

using namespace sl;
using namespace std;

sl::Resolution cloud_res(1280, 720);

// Custom std::hash
template<>
struct std::hash<std::pair<int, int>>
{
    std::size_t operator()(const std::pair<int, int>& s) const noexcept
    {
        return s.first * 1000000 * s.second; // or use boost::hash_combine
    }
};

inline float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}

constexpr size_t NORMAL_SPACING = 4;
constexpr size_t DATA_SPACING = 4;
constexpr char const* FILE_NAME = "/home/john/random/zed-pcl/data/test.pcd";

int main(int argc, char **argv) {
    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return -1;
    }

	Camera zed;
	Mat data_cloud;
	Mat normal_cloud;

    // Set configuration parameters
    InitParameters init_params;
    if (argc == 2)
        init_params.input.setFromSVOFile(argv[1]);
    else {
        init_params.camera_resolution = RESOLUTION::HD720;
        init_params.camera_fps = 30;
    }
    init_params.coordinate_units = UNIT::METER;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params.depth_mode = DEPTH_MODE::PERFORMANCE;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	std::cout << "Starting ZED Grab Cycle..." << std::endl;
	while(zed.grab() != ERROR_CODE::SUCCESS){
		// DO nothing while waiting for the ZED to grab
	}
	zed.retrieveMeasure(data_cloud, MEASURE::XYZRGBA, MEM::CPU, cloud_res);
	zed.retrieveMeasure(normal_cloud, MEASURE::NORMALS, MEM::CPU, cloud_res);
	std::cout << "Successfully Grabbed from the ZED..." << std::endl;

	float* p_data_cloud = data_cloud.getPtr<float>();
	float* p_normal_cloud = normal_cloud.getPtr<float>();
	for (std::size_t index = 0; index < cloud_res.area(); ++index) {
		std::cout << index << '\n';
		pcl::PointXYZRGBNormal pt{};
		float X = p_data_cloud[DATA_SPACING * index];
		if (!isValidMeasure(X)) // Checking if it's a valid point
			pt.x = pt.y = pt.z = pt.rgb = pt.a = pt.normal_x = pt.normal_y = pt.normal_z = 0;
		else {
			pt.x = X;
			pt.y = p_data_cloud[DATA_SPACING * index + 1];
			pt.z = p_data_cloud[DATA_SPACING * index + 2];
			pt.rgb = convertColor(p_data_cloud[DATA_SPACING * index + 3]); // Convert a 32bits float into a pcl .rgb format
			pt.normal_x = p_normal_cloud[NORMAL_SPACING * index];
			pt.normal_y = p_normal_cloud[NORMAL_SPACING * index + 1];
			pt.normal_z = p_normal_cloud[NORMAL_SPACING * index + 2];
		}
		p_pcl_point_cloud->push_back(std::move(pt));
	}

	pcl::io::savePCDFile(FILE_NAME, *p_pcl_point_cloud);

	zed.close();

    return 0;
}
