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
#include <utility>
#include <string>

// Namespace
using namespace sl;
using namespace std;

Camera zed;
Mat data_cloud;
Mat normal_cloud;
size_t pcd_index = 0;
std::thread input_thread;
std::thread render_thread;
std::mutex mut;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr grid_pcd;
shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pcd;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr visualizer_pcd;
bool populated = false;

shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
inline float convertColor(float colorIn);

sl::Resolution cloud_res;

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

// Constants
constexpr size_t SPACING = 4;
constexpr size_t NUM_PCD = 2;
constexpr char const* FILE_NAME[NUM_PCD] = {"/home/john/random/zed-pcl/data/test.pcd", "/home/john/random/zed-pcl/data/test1.pcd"};
constexpr size_t PC_WIDTH = 1280;
constexpr size_t PC_HEIGHT = 720;
constexpr float GRID_DENSITY = 0.01;
constexpr uint8_t HIGH_COST = 0;
constexpr uint8_t LOW_COST = 100;

// Tunables
constexpr size_t GRID_WIDTH = 10;
constexpr size_t GRID_HEIGHT = 10;
constexpr float GRID_RESOLUTION = 0.5;
constexpr int PCD_DOWNSAMPLE = 20;
constexpr float RIGHT_CLIP = -2.0;
constexpr float LEFT_CLIP = 2.0;
constexpr float FAR_CLIP = 7.0;
constexpr float NEAR_CLIP = 0.5;
constexpr float ROVER_HEIGHT = 1.0;

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
					gridPcd->push_back(pcl::PointXYZRGB{x + dx, y + dy, -ROVER_HEIGHT, static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0)), static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0)), static_cast<uint8_t>(255 * (cm.data[w * cm.width + h] / 100.0))});
				}
			}
		}
	}

	return gridPcd;
}

auto mapToGrid(Eigen::Vector3f const& positionInMap, CostMap const& cm) -> int {
	Eigen::Vector2f origin{cm.x - GRID_WIDTH / 2.0, cm.y - GRID_HEIGHT / 2.0};
	Eigen::Vector2f gridFloat = (Eigen::Vector2f{positionInMap.y(), positionInMap.x()} - origin) / GRID_RESOLUTION;

	int gridX = std::floor(gridFloat.x());
	int gridY = std::floor(gridFloat.y());

	// Clip all of the points based on their relative location to the grid
	if(gridX >= static_cast<int>(cm.width) || gridX < 0 || gridY >= static_cast<int>(cm.height) || gridY < 0){
		return -1;
	}

	return gridY * static_cast<int>(cm.width) + gridX;
}

void fillInCostMap(CostMap& cm, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr const& pcd){
	auto ptr = pcd->points;
	using R2d = Eigen::Vector2d;
    using R3d = Eigen::Vector3d;
    using S3d = Eigen::Quaterniond;

    using R2f = Eigen::Vector2f;
    using R3f = Eigen::Vector3f;
    using S3f = Eigen::Quaternionf;

	struct BinEntry {
		R3f pointInCamera;
		R3f normalInCamera;
		double heightInCamera;
	};

	using Bin = std::vector<BinEntry>;

	std::vector<Bin> bins;
	bins.resize(cm.width * cm.height);

	for (std::size_t r = 0; r < PC_HEIGHT; r += PCD_DOWNSAMPLE) {
		for (std::size_t c = 0; c < PC_WIDTH; c += PCD_DOWNSAMPLE) {
			auto const& pt = ptr[r * PC_WIDTH + c];
			if (!(pt.y > RIGHT_CLIP &&
					pt.y < LEFT_CLIP &&
					pt.x < FAR_CLIP &&
					pt.x > NEAR_CLIP)){
					continue;
			}

			Eigen::Vector3f pointInCamera{pt.x, pt.y, pt.z};
			Eigen::Vector3f normalInCamera{pt.normal_x, pt.normal_y, pt.normal_z};

			int index = mapToGrid(pointInCamera, cm);

			if(index == -1){
				continue;
			}

			bins[index].emplace_back(BinEntry{pointInCamera, normalInCamera, pointInCamera.z()});
		}
	}

	for (std::size_t i = 0; i < cm.width * cm.height; ++i) {
		Bin& bin = bins[i];
		auto& cell = cm.data[i];

		// WRITE ALGORITHM HERE BEGIN
		float avgHeight = 0;
		for(auto const& pt : bin){
			avgHeight += pt.heightInCamera;
		}
		avgHeight /= bin.size();

		uint8_t cost = avgHeight > 0 ? HIGH_COST : LOW_COST;

		// WRITE ALGORITHM HERE END

		cell = cost;
	}
}

int main(int argc, char **argv) {
	if (argc > 2) {
		cout << "Only the path of a SVO can be passed in arg" << endl;
		return -1;
	}

	thread t1([&](){
		while(true){
			mut.lock();
			pcd = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>(*(new pcl::PointCloud<pcl::PointXYZRGBNormal>));

			if (pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (FILE_NAME[pcd_index], *pcd) == -1) {
				std::cerr << "Error loading point cloud file" << std::endl;
			}    

			visualizer_pcd = stripNormals(pcd);

			CostMap cm{};
			cm.width = std::ceil(GRID_WIDTH / GRID_RESOLUTION);
			cm.height = std::ceil(GRID_HEIGHT / GRID_RESOLUTION);
			cm.x = 0;
			cm.y = 0;
			cm.data.resize(cm.width * cm.height);

			std::cout << "Filling in Cost Map...\n";
			fillInCostMap(cm, pcd);
			std::cout << "Done Filling in Cost Map...\n";

			grid_pcd = createGridPcd(cm);

			if(!viewer){
				viewer = createRGBVisualizer(visualizer_pcd);
				viewer->addPointCloud(grid_pcd, "grid");
			}
			viewer->setCameraPosition(-5, 0, 1,    1, 0, 0,   0, 0, 1);
			viewer->setCameraClipDistances(0.1,1000);

			populated = true;
			mut.unlock();
		}
	});

	thread t2([&](){
		while (true) {
			mut.lock();
			if(populated){
				if(viewer->wasStopped()){
					break;
				}
				viewer->updatePointCloud(visualizer_pcd);
				viewer->updatePointCloud(grid_pcd, "grid");
				viewer->spinOnce(100);
			}
			mut.unlock();
		}
	});

	std::cout << "Created Threads...\n";
	t2.join();
	std::cout << "Render Thread Finished...\n";
	t1.join();
	std::cout << "Input Thread Finished...\n";

	return 0;
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
