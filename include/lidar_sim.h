#ifndef GNCTK_LIDAR_SIM_H_
#define GNCTK_LIDAR_SIM_H_

/*-----------------------------------------------------------\\
||                                                           ||
||                 LIDAR fusion GNC project                  ||
||               ----------------------------                ||
||                                                           ||
||    Surrey Space Centre - STAR lab                         ||
||    (c) Surrey University 2017                             ||
||    Pete dot Blacker at Gmail dot com                      ||
||                                                           ||
\\-----------------------------------------------------------//

lidar_sim.cpp

LIDAR sensor simulator node
----------------------------

This node uses a pair of depth and incident cube map images
to simulate the data from a LIDAR sensor within the
cube maps.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include "opencv2/highgui/highgui.hpp"

namespace gncTK {
class LidarSim;
};

class gncTK::LidarSim
{
public:

	LidarSim();

	struct lidarParams
	{
	  float beamDivergence;
	  // energy to depth error fn
	  // energy to sample probablility fn
	};

	struct cubeSample
	{
	  float depth;
	  float incident;
	  float albedo;
	  float mask;
	};

	Eigen::Vector2f getSampleLocation(Eigen::Vector3f direction);

	cubeSample generateCubeSample(Eigen::Vector3f direction);

	pcl::PointXYZI generateLidarSample(Eigen::Vector3f direction);

	void setResolution(int hRes, int vRes)
	{
		this->hRes = hRes;
		this->vRes = vRes;
	}

	void setFOV(float hFOV, float vFOV)
	{
		this->hFOV = hFOV;
		this->vFOV = vFOV;
	}

	Eigen::Vector3f getDirection(int h, int v)
	{
		float vAngle = (0.5 - ((float)v/vRes)) * vFOV;
		float hAngle = (((float)h/hRes) - 0.5) * hFOV;

		Eigen::Vector3f direction;
		direction[0] = cos(hAngle) * cos(vAngle);
		direction[1] = sin(hAngle);
		direction[2] = sin(0-vAngle);

		return direction;
	}

	pcl::PointCloud<pcl::PointXYZI> generateScan(Eigen::Matrix3f orientation);

	void preCalcScanSamples(Eigen::Matrix3f orientation);
	pcl::PointCloud<pcl::PointXYZI> generatePreCalcScan();

	float calculateFOVSolidAngle();

	void loadCubeMaps(char *prefix);
	void loadCubeMaps(std::string prefix);// { loadCubeMaps(prefix.c_str()); };

	/// Method to return the approximate point density of the last scan generated
	/*
	 * Measured in points per steradian
	 */
	float getPointDensity() { return (hRes*vRes) / FOVSolidAngle; };

	int hRes, vRes;
	float hFOV, vFOV;

	float FOVSolidAngle;
protected:
	bool mapsLoaded;
	cv::Mat depthMap, incidentMap;
	int mapResolution;

	int multisampleCount;
	float maxDepth;

	//float pointDensity;

	//cv::Mat mapCopy;
	//cv::Vec3b debugColor;

};





#endif /* GNCTK_LIDAR_SIM_H_ */
