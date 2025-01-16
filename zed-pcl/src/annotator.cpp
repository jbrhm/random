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
shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
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
void calcStats(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc){
	float avg_y = 0;
	float avg_normal_y = 0;
	for(auto &pt : pc->points){
		avg_y += pt.y;
	}
	avg_y /= pc->size();
	std::cout << "Average Y: " << avg_y << '\n';
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

constexpr size_t SPACING = 4;
constexpr char const* FILE_NAME = "/home/john/random/zed-pcl/data/test.pcd";

constexpr size_t GRID_WIDTH = 10;
constexpr size_t GRID_HEIGHT = 10;
constexpr float GRID_RESOLUTION = 0.5;
constexpr float GRID_DENSITY = 0.01;

struct CostMap{
	size_t width;
	size_t height;

	float x;
	float y;

	std::vector<uint8_t> data;
};

pcl::PointCloud<pcl::PointXYZRGB>::Ptr createGridPcd(CostMap const& cm){
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr gridPcd (new pcl::PointCloud<pcl::PointXYZRGB>);
	for(size_t w = 0; w < cm.width; ++w){
		for(size_t h = 0; h < cm.height; ++h){
			float x = w * GRID_RESOLUTION + cm.x - (GRID_WIDTH / 2.0);
			float y = h * GRID_RESOLUTION + cm.y - (GRID_HEIGHT / 2.0);
			for(float dx = 0; dx < GRID_RESOLUTION; dx += GRID_DENSITY){
				for(float dy = 0; dy < GRID_RESOLUTION; dy += GRID_DENSITY){
					gridPcd->push_back(pcl::PointXYZRGB{x + dx, y + dy, 0, static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0)), static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0)), static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0))});
				}
			}
		}
	}
	return gridPcd;
}

int main(int argc, char **argv) {
    if (argc > 2) {
        cout << "Only the path of a SVO can be passed in arg" << endl;
        return -1;
    }

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcd (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (FILE_NAME, *pcd) == -1) {
        std::cerr << "Error loading point cloud file" << std::endl;
    }    

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr visualizer_pcd = stripNormals(pcd);

	shared_ptr<pcl::visualization::PCLVisualizer> viewer = createRGBVisualizer(visualizer_pcd);
	viewer->setCameraPosition(-5, 0, 1,    1, 0, 0,   0, 0, 1);
    viewer->setCameraClipDistances(0.1,1000);

	// Create Fake CostMap
	CostMap cm{};

	cm.width = std::ceil(GRID_WIDTH / GRID_RESOLUTION);
	cm.height = std::ceil(GRID_HEIGHT / GRID_RESOLUTION);

	cm.x = 0;
	cm.y = 0;

	for(size_t w = 0; w < cm.width; ++w){
		for(size_t h = 0; h < cm.height; ++h){
			cm.data.push_back(100 * ((w * cm.width + h) % 2));
		}
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr grid_pcd = createGridPcd(cm);
	viewer->addPointCloud(grid_pcd, "grid");

	while (!viewer->wasStopped()) {
		viewer->updatePointCloud(visualizer_pcd);
		viewer->updatePointCloud(grid_pcd, "grid");
		viewer->spinOnce(100);
	}

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
