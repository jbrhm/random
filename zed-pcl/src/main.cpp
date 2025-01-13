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

// Sample includes
#include <thread>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <string>

// Namespace
using namespace sl;
using namespace std;

// Global instance (ZED, Mat, callback)
Camera zed;
Mat data_cloud;
Mat normal_cloud;
std::thread zed_callback;
std::mutex mutex_input;
bool stop_signal;
bool has_data;
int PCD_DOWNSAMPLE = 20;

// Sample functions
void startZED();
void run();
void closeZED();
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud);
inline float convertColor(float colorIn);

sl::Resolution cloud_res;

// Custom std::hash
template<>
struct std::hash<std::pair<int, int>>
{
    std::size_t operator()(const std::pair<int, int>& s) const noexcept
    {
        return s.first * 1000000 * s.second; // or use boost::hash_combine
    }
};

// Calculate Sample Statistics
void calcStats(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc){
	float avg_y = 0;
	float avg_normal_y = 0;
	for(auto &pt : pc->points){
		avg_y += pt.y;
	}
	avg_y /= pc->size();
	std::cout << "Average Y: " << avg_y << '\n';
}

// Main process
int main(int argc, char **argv) {

    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return -1;
    }

    // Set configuration parameters
    InitParameters init_params;
    if (argc == 2)
        init_params.input.setFromSVOFile(argv[1]);
    else {
        init_params.camera_resolution = RESOLUTION::HD720;
        init_params.camera_fps = 30;
    }
    init_params.coordinate_units = UNIT::METER;
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_params.depth_mode = DEPTH_MODE::ULTRA;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        return 1;
    }

    cloud_res = sl::Resolution(1280, 720);

    // Allocate PCL point cloud at the resolution
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    p_pcl_point_cloud->points.resize(cloud_res.area() / PCD_DOWNSAMPLE);

    // Create the PCL point cloud visualizer
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = createRGBVisualizer(p_pcl_point_cloud);

    // Start ZED callback
    startZED();

	std::cout << "Created ZED..." << std::endl;

    // Set Viewer initial position
    viewer->setCameraPosition(0, 0, 5,    0, 0, 1,   0, 1, 0);
    viewer->setCameraClipDistances(0.1,1000);

    // Loop until viewer catches the stop signal
    while (!viewer->wasStopped()) {
		std::cout << "In Loop..." << std::endl;

		//Lock to use the point cloud
		mutex_input.lock();
		float *p_data_cloud = data_cloud.getPtr<float>();
		float *p_normal_cloud = data_cloud.getPtr<float>();
		int index = 0;

		// Check and adjust points for PCL format
		std::cout << "Adjusting Points" << std::endl;
		for (auto &it : p_pcl_point_cloud->points) {
			float X = p_data_cloud[index];
			float n_X = p_normal_cloud[index];
			if (!isValidMeasure(X)) // Checking if it's a valid point
				it.x = it.y = it.z = it.rgb = 0;
			else {
				it.x = X;
				it.y = p_data_cloud[index + 1];
				it.z = p_data_cloud[index + 2];
				it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
			}
			index += 4 * PCD_DOWNSAMPLE;
		}
		std::cout << "Done Adjusting Points" << std::endl;

		// Perform the outlier comparisons
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
		int num_neigbor_points = 5;
		double std_multiplier = 1.0;

		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
		sor.setInputCloud (p_pcl_point_cloud);
		sor.setMeanK (num_neigbor_points);
		sor.setStddevMulThresh (std_multiplier);
		sor.filter(*output);

		// Unlock data and update Point cloud
		mutex_input.unlock();
		viewer->updatePointCloud(p_pcl_point_cloud);
		if(!viewer){
			std::cout << "Viewer is invalid...\n";
		}

		// Create bins
		std::cout << "Creating Bins...\n";
		std::unordered_map<std::pair<int, int>, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> bins;
		for (auto &it : output->points) {
			if(bins.find(std::pair<int, int>(it.x, it.z)) == bins.end()){
				bins[std::pair<int, int>(it.x, it.z)] = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
			}
			bins[std::pair<int, int>(it.x, it.z)]->push_back(it);
		}

		std::cout << "Beginning Cycles...\n";
		index = 0;
		auto iter = bins.begin();
		while(true){
			if(iter == bins.end()){
				viewer->updatePointCloud(output);
				iter = bins.begin();
			}else{
				viewer->updatePointCloud(iter->second);
				calcStats(iter->second);
			}
			viewer->spinOnce(100);
			++iter;
		}
    }

    // Close the viewer
	std::cout << "Closed the ZED..." << std::endl;
    viewer->close();

    // Close the zed
    closeZED();

    return 0;
}

/**
 *  This functions start the ZED's thread that grab images and data.
 **/
void startZED() {
    // Start the thread for grabbing ZED data
    stop_signal = false;
    has_data = false;
    zed_callback = std::thread(run);

    //Wait for data to be grabbed
    while (!has_data)
        sleep_ms(1);
}

/**
 *  This function loops to get the point cloud from the ZED. It can be considered as a callback.
 **/
void run() {
    while (!stop_signal) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            mutex_input.lock(); // To prevent from data corruption
            zed.retrieveMeasure(data_cloud, MEASURE::XYZRGBA, MEM::CPU, cloud_res);
            zed.retrieveMeasure(normal_cloud, MEASURE::NORMALS, MEM::CPU, cloud_res);
            mutex_input.unlock();
            has_data = true;
        } else
            sleep_ms(1);
    }
}

/**
 *  This function frees and close the ZED, its callback(thread) and the viewer
 **/
void closeZED() {
    // Stop the thread
    stop_signal = true;
    zed_callback.join();
    zed.close();
}

/**
 *  This function creates a PCL visualizer
 **/
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
    // Open 3D viewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
    viewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

/**
 *  This function convert a RGBA color packed into a packed RGBA PCL compatible format
 **/
inline float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}
