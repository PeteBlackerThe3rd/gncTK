#ifndef GNCTK_DEM_H_
#define GNCTK_DEM_H_

/*-----------------------------------------------------------\\
||                                                           ||
||                 GNC Toolkit Library                       ||
||               -----------------------                     ||
||                                                           ||
||    Surrey Space Centre - STAR lab                         ||
||    (c) Surrey University 2017                             ||
||    Pete dot Blacker at Gmail dot com                      ||
||                                                           ||
\\-----------------------------------------------------------//

dem.h

Digital Elevation Map processing object
-----------------------------------------

This object provides a set of static methods that operate
on ROS GridMap objects to produce dem's from various sources.
Triangle Mesh's loaded from DEM files.
They can also have their cost-map (goodness) layers calculated
using a variety of algorithms.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include <opencv2/highgui/highgui.hpp>
#include <tf/tf.h>
//#include <dynamic_reconfigure/server.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include "mesh.h"
//#include <smart_fusion_sensor/CostMapGenConfig.h>

namespace gncTK
{
	class DEM;
};

class gncTK::DEM
{
public:
	static grid_map::GridMap* generateFromMesh(Mesh sourceMesh,
											   float gridSpacing = 0.1,
											   float safeZoneDia=0,
											   float safeZoneBlendDia=0,
											   float safeZoneElevation=NAN,
											   grid_map::Position fixedPosition = grid_map::Position(NAN,NAN));

	static void addSafeZone(grid_map::GridMap* map, float safeDia, float gradientDia);

	static grid_map::GridMap* loadFromImage(std::string fileName, float maxHeight = 20.0);

	static gncTK::Mesh* gridMapToMesh(grid_map::GridMap *map, std::string heightLayer, std::string colorLayer);

	static void generateCostMap(grid_map::GridMap* map, std::string layer = "elevation");
	static void generateCostMapFast(grid_map::GridMap* map, std::string layer = "elevation");

	static cv::Mat convertToCvMat(grid_map::GridMap *map, std::string layer = "elevation");

	static void setupReconfigureServer();

	static void cleanupOffscreenGL();

	class CostmapSettingsMER
	{
	public:
		CostmapSettingsMER()
		{
			roverWidth = 0.8;
			clearanceHeight = 0.3;
			maxPitchAngle = 0.5;
			slopeWeight = 1.0;
			roughWeight = 1.0;
			stepWeight = 1.0;
			slopeThreshold = 0.2;
			roughThreshold = 0.5;
			stepThreshold = 1.0;
			roughnessFraction = 0.3;
			gravity << 0, 0, -1;		// ROS default gravity direction
		};

		float roverWidth;
		float clearanceHeight;
		float maxPitchAngle;
		float roughnessFraction;
		float slopeWeight, slopeThreshold;
		float roughWeight, roughThreshold;
		float stepWeight, stepThreshold;
		Eigen::Vector3f gravity;
	};

	static CostmapSettingsMER costmapSettings;

	static void setGravityVector(Eigen::Vector3f newGravity) { costmapSettings.gravity = newGravity; };

private:

	/// dynamic reconfigure server and callback handle
	//static dynamic_reconfigure::Server<smart_fusion_sensor::CostMapGenConfig> *configServer;

	/// Pointer to GLFW window used for OpenGL rendering
	static GLFWwindow* window;
	static bool glContextSetup;
	static GLuint ssFbo, ssColorBuf, ssDepthBuf;
	static int offscreenBufSize;

	// helper functions for costmap generation
	static Eigen::MatrixXf getElevationSamples(grid_map::GridMap* map,
											   Eigen::MatrixXi footprint,
											   Eigen::Vector2d center,
											   std::string layer);
	static float calculateStepCost(Eigen::MatrixXf elevations);
	static Eigen::MatrixXi GenerateRoverMaskMatrix(float gridSize, float roverSize);
	static Eigen::Matrix<float, 3, 2> calculateBestFitPlane(Eigen::MatrixXf elevations, float gridSize);
	static float findMaxPlaneResidual(Eigen::MatrixXf elevations,
									  Eigen::Matrix<float, 3, 2> plane,
									  float gridSize);
	static int countNonNans(Eigen::MatrixXf elevations);

	// helper functions to generate DEM from mesh
	static cv::Mat fillElevationBuffer(gncTK::Mesh sourceMesh,
								Eigen::Array2i start,
								Eigen::Array2i size,
								grid_map::Length mapSize,
								grid_map::Position mapPosition,
								float gridSpacing,
								Eigen::Matrix<float, 3, 2> meshExtents,
								bool findMax = true);
	static cv::Mat fillElevationBuffer2(gncTK::Mesh sourceMesh,
								Eigen::Array2i start,
								Eigen::Array2i size,
								grid_map::Length mapSize,
								grid_map::Position mapPosition,
								float gridSpacing);

	static void setupOffscreenGL();

	//static cv::Vec3b rainbow(float color);
};

#endif /* GNCTK_DEM_H_ */
