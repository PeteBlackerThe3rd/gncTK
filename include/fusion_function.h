#ifndef GNCTK_FUSION_FUNCTION_H_
#define GNCTK_FUSION_FUNCTION_H_

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

fusion_function.h

class encapsulating the fusion mathematical model
---------------------------------------------------

This class stores, loads, saves and calibrates the lidar
camera fusion function.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "utils.h"

namespace gncTK
{
	class FusionFunction;
};

class gncTK::FusionFunction
{
public:

	FusionFunction()
	{
		cameraMatrix = cv::Mat(3,3,CV_64F, cv::Scalar(0));
		extrinsicTF = cv::Affine3d(cv::Matx<double,4,4>::eye());
		distortionCoeffs = cv::Mat(1,14, CV_64F, cv::Scalar(0));
		reverseProjectionGenerated = false;
	};

	FusionFunction(int width, int height)
	{
		cameraMatrix = cv::Mat(3,3,CV_64F, cv::Scalar(0));
		cameraMatrix.at<double>(0,0) = width/2.0;
		cameraMatrix.at<double>(0,2) = width/2.0;
		cameraMatrix.at<double>(1,1) = height/2.0;
		cameraMatrix.at<double>(1,2) = height/2.0;
		cameraMatrix.at<double>(2,2) = 1;

		extrinsicTF = cv::Affine3d(cv::Matx<double,4,4>::eye());

		distortionCoeffs = cv::Mat(1,14, CV_64F, cv::Scalar(0));
		reverseProjectionGenerated = false;
	};

	FusionFunction(std::string fileName)
	{
		loadConfig(fileName);
		reverseProjectionGenerated = false;
	};

	FusionFunction(const gncTK::FusionFunction& source)
	{
		cameraMatrix = source.cameraMatrix;
		distortionCoeffs = source.distortionCoeffs;
		extrinsicTF = source.extrinsicTF;
		reverseProjection = source.reverseProjection.clone();
		reverseProjectionGenerated = source.reverseProjectionGenerated;
	};

	void setExtrinsicTF(Eigen::Matrix4f extrinsic);

	void setCameraMatrix(Eigen::Matrix3f cameraMat);
	void setDistortionCoeffs(std::vector<double> coeffs);

	cv::Mat getCameraMatrix() { return cameraMatrix; };
	cv::Mat getDistortionCoeffs() { return distortionCoeffs; };
	cv::Affine3d getExtrinsicTF() { return extrinsicTF; };

	tf::Transform getExtrinsicTransform();

	/// Method to return the camera focal point in the LIDAR coordinate system
	Eigen::Vector3f getCameraLocation();

	/// Method to project a point in the ROS coordinate convention of the LIDAR frame to a pixel position
	Eigen::Vector2f projectPoint(Eigen::Vector3f point);

	/// Method to project a vector of points in the ROS coordinate convention of the LIDAR frame to pixel positions
	std::vector<Eigen::Vector2f> projectPoints(std::vector<Eigen::Vector3f> points);

	/// Method to calibrate this fusion function using a matched set of pixel locations and 3D locations
	/**
	 * Returns the mean RMS reprojection error in pixels of this calibration
	 */
	double calibrate(std::vector<Eigen::Vector2f> points2D,
					 std::vector<Eigen::Vector3f> points3D,
					 cv::Size imageSize);

	double optimseExtrinsic(std::vector<Eigen::Vector2f> points2D,
			 	 	 	 	std::vector<Eigen::Vector3f> points3D);

	/// Method to save the configuration of this fusion function
	void saveConfig(std::string fileName);

	/// Method to load the configuration of this fusion function
	bool loadConfig(std::string fileName);

	/// Method to format the calibration parameters into a multi-line string
	std::string configString();

	// Method to generate the reverse projection lookup table
	bool generateReverseProjection(int width, int height, bool useOpticalFrame = false);

	// method to get the reverse projection of the given pixel
	// reports an error if the location is outside the camera frame
	Eigen::Vector3f getReverseProjection(int col, int row);
	Eigen::Vector3f interpolateReverseProjection(float col, float row);

	/// Method to calculate the exact solid angle of the FOV of this camera
	float calculateFOVSolidAngle();

	cv::Mat reverseProjection;

private:

	cv::Mat cameraMatrix;
	cv::Mat distortionCoeffs;
	cv::Affine3d extrinsicTF;
	bool reverseProjectionGenerated;

	Eigen::Vector3f polar2Cart(float az, float el, float d);

};

#endif /* GNCTK_FUSION_FUNCTION_H_ */
