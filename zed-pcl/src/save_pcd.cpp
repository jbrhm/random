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
#include <ctime>
#include <format>

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
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr visualizer_pcd;

// Sample functions
void startZED();
void run();
void closeZED();
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
inline float convertColor(float colorIn);

sl::Resolution cloud_res;

constexpr size_t NORMAL_SPACING = 4;
constexpr size_t DATA_SPACING = 4;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
		void* viewer_void)
{
	pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
	if (event.getKeySym () == "s" && event.keyDown ())
	{
		time_t now = time(0);
		tm* ltm = localtime(&now);

		// Format time and date into a string
		char buffer[80];
		strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", ltm);
		std::string dateTimeString(buffer);

		// Print the formatted string
		std::cout << "Current date and time: " << dateTimeString << std::endl;

		pcl::io::savePCDFile(std::string("/home/john/random/zed-pcl/data/test") + dateTimeString + std::string(".pcd"), *p_pcl_point_cloud);
	}
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr stripNormals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const& cloud){
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pcd (new pcl::PointCloud<pcl::PointXYZRGB>);
	for(auto const& p : cloud->points){
		pcl::PointXYZRGB pt{};
		pt.x = p.x;
		pt.y = p.y;
		pt.z = p.z;
		pt.r = p.r;
		pt.g = p.g;
		pt.b = p.b;
		pt.a = p.a;
		new_pcd->push_back(std::move(pt));
	}

	return new_pcd;
}

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
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_params.depth_mode = DEPTH_MODE::PERFORMANCE;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        return 1;
    }

    cloud_res = sl::Resolution(1280, 720);

    // Allocate PCL point cloud at the resolution
    p_pcl_point_cloud->points.resize(cloud_res.area());
	visualizer_pcd = stripNormals(p_pcl_point_cloud);

    // Create the PCL point cloud visualizer
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = createRGBVisualizer(visualizer_pcd);

    // Start ZED callback
    startZED();

	std::cout << "Created ZED..." << std::endl;

    // Set Viewer initial position
	viewer->setCameraPosition(-5, 0, 1,    1, 0, 0,   0, 0, 1);
	viewer->setCameraClipDistances(0.1,1000);
	viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)viewer.get ());

    // Loop until viewer catches the stop signal
    while (!viewer->wasStopped()) {
		//Lock to use the point cloud
		mutex_input.lock();
		float *p_data_cloud = data_cloud.getPtr<float>();
		float* p_normal_cloud = normal_cloud.getPtr<float>();

		p_pcl_point_cloud->clear();

		for (std::size_t index = 0; index < cloud_res.area(); ++index) {
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

		visualizer_pcd = stripNormals(p_pcl_point_cloud);

		mutex_input.unlock();

		viewer->updatePointCloud(visualizer_pcd);
		if(!viewer){
			std::cout << "Viewer is invalid...\n";
		}
		viewer->spinOnce(10);
    }

	std::cout << "Out of loop..." << std::endl;

    // Close the viewer
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
            mutex_input.unlock();
			zed.retrieveMeasure(data_cloud, MEASURE::XYZRGBA, MEM::CPU, cloud_res);
			zed.retrieveMeasure(normal_cloud, MEASURE::NORMALS, MEM::CPU, cloud_res);
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
