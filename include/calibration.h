#ifndef GNCTK_CALIBRATION_H_
#define GNCTK_CALIBRATION_H_

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

calibration.h

special calibration fusion object
----------------------------------------------

This class fuses structured lidar point clouds and camera
images into two sets of calibration corners, if they can
be detected. These sets of points are used to calibrate
the fusion function for a specific sensor.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "visualization_msgs/Marker.h"
#include "fusion.h"

namespace gncTK {
	class Calibration;
};

class gncTK::Calibration : public gncTK::Fusion
{
public:

	Calibration();

	Calibration(int targetSizeX, int targetSizeY);

	bool addFusionSample(std::vector<cv::Mat> images,
						 std::vector<tf::Transform> imageTFs,
						 pcl::PointCloud<pcl::PointXYZI> cloud);

	void commitFoundPoints();

	double calibrate(bool useInitialCalibration = false, bool extrinsicOnly = false);


	cv::Mat getIntensityImage() { return intensityImage; };
	cv::Mat getCameraCornersImage() { return cameraCornersImage; };

	/// Method to load a set of image and object points from a csv file
	/*
	 * CSV column order is [imgX, imgY, objX, objY, objZ]
	 */
	double loadPairsFromCSV(std::string fileName, int width, int height);

	pcl::PointCloud<pcl::PointXYZRGB> totalTargetPoints;
	pcl::PointCloud<pcl::PointXYZRGB> foundTargetPoints;

	std::vector<cv::Point2f> cameraCorners;
	std::vector<cv::Point2f> foundCameraCorners;

	// isolate points within calibration pattern
	pcl::PointCloud<pcl::PointXYZI> targetPoints;

	pcl::PointCloud<pcl::PointXYZRGB> debugCloud;

	gncTK::FusionFunction fusionFunction;

	double getRMS() { return rms; };

	bool targetPointsReady() { return newTargetPointsFound; };

	bool newTargetPointsFound;

private:

	bool isValid(pcl::PointXYZI point);
	bool pixelValid(int r, int c);
	void generateIntensityImage(pcl::PointCloud<pcl::PointXYZI> points);
	/// Method to find the plane of best fit to the given point cloud using Singular Value Decomposition
	void findPlaneOfBestFit(pcl::PointCloud<pcl::PointXYZI> cloud,
							Eigen::Vector3f *centroid,
							Eigen::Vector3f *normal);
	Eigen::Vector3f linePlaneIntersect(Eigen::Vector3f lineA,
									   Eigen::Vector3f lineB,
									   Eigen::Vector3f planePoint,
									   Eigen::Vector3f planeNormal);
	bool isOnRight(Eigen::Vector2f Test,
				   Eigen::Vector2f A,
				   Eigen::Vector2f B);
	Eigen::Vector3f interpolatePoint(pcl::PointCloud<pcl::PointXYZI> cloud,
									 float x,
									 float y);
	std::vector<pcl::PointXYZRGB> projectPointsToPlane(std::vector<cv::Point2f> lidarCorners,
													pcl::PointCloud<pcl::PointXYZI> cloud,
													Eigen::Vector3f centroid,
													Eigen::Vector3f normal);
	std::vector<pcl::PointXYZRGB> optimiseLidarCorners(std::vector<cv::Point2f> lidarCorners,
													pcl::PointCloud<pcl::PointXYZI> cloud);
	std::vector<pcl::PointXYZRGB> gridOptimisePoints(std::vector<pcl::PointXYZRGB> points);
	std::vector<cv::Point2f> subpixelCorrectChessboardCorners(std::vector<cv::Point2f> cameraCorners,
												   	          cv::Mat image);

	/// Method to flip the input cloud if the scanning order is different
	void verticallyFlipCloud(pcl::PointCloud<pcl::PointXYZI> *cloud);

	cv::Mat orthoprojectPointCloudToImage(pcl::PointCloud<pcl::PointXYZI> *cloud, Eigen::Vector2f sizeMetres, Eigen::Vector2i sizePixels);

	cv::Mat cameraImage;
	cv::Mat intensityImage;
	cv::Mat cameraCornersImage;
	bool cameraCornersImageSet;
	cv::Size patternsize;


	//std::vector<cv::Point2f> allCameraCorners;
	//std::vector<cv::Point3f> allLidarCorners;

	// cumulative vectors of matched image and object point pairs.
	std::vector<Eigen::Vector2f> imagePoints;
	std::vector<Eigen::Vector3f> objectPoints;

	double rms;
};

#endif /* GNCTK_CALIBRATION_H_ */
