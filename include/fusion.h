#ifndef GNCTK_FUSION_H_
#define GNCTK_FUSION_H_

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

fusion.h

virtual LIDAR camera fusion processing object
----------------------------------------------

This abstract super class defines the interface for all
of the concrete implementations of different LIDAR camera
fusion algorithms in various base classes

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include "opencv2/highgui/highgui.hpp"
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include "visualization_msgs/Marker.h"
#include "fusion_function.h"
#include "mesh.h"

namespace gncTK
{
	class Fusion;
};

class gncTK::Fusion
{
public:

	Fusion() { imageSet = cloudSet = glContextSetupSC = false; debug = true; windowSC = NULL; };

	virtual void setInputCloud(pcl::PointCloud<pcl::PointXYZI> cloud)
	{
		inputCloud = cloud;
		cloudSet = true;
	}

	void setInputImage(cv::Mat image, tf::Transform *imageTF = NULL)
	{
		inputImage = image;
		imageSet = true;

		if (imageTF != NULL)
		{
			inputImages.push_back(image);
			inputImageTFs.push_back(*imageTF);
		}
	}

	void clearInputImages()
	{
		inputImages.clear();
		inputImageTFs.clear();
	}

	void setFusionFunction(gncTK::FusionFunction newFunction)
	{
		fusionFunction = newFunction;
	};

	virtual Mesh generateMesh() {};

	virtual cv::Mat generateDepthImage(int resolutionFactor = 1, int border = 0) {};

	virtual /*std::Vector<[feature_type]>*/ void generateFeatureAtlas() {};

	cv::Mat renderModelToCamera(int resolutionFactor, int border, gncTK::Mesh *model);

	bool debug;

	// additional properties for multi-image scans
	std::vector<cv::Mat> inputImages;
	std::vector<tf::Transform> inputImageTFs;

protected:

	// internal helper functions
	int filterTrianglesOnIncidentAngle(gncTK::Mesh *mesh, float angle);

	cv::Mat inputImage;
	pcl::PointCloud<pcl::PointXYZI> inputCloud;

	bool imageSet, cloudSet;

	gncTK::FusionFunction fusionFunction;


private:

	/// Pointer to GLFW window used for OpenGL rendering
	GLFWwindow* windowSC;
	bool glContextSetupSC;
	GLuint ssFboSC, ssColorBufSC, ssDepthBufSC;

	void setupOffscreenGLBufferSC(int width, int height);
};

#endif /* GNCTK_FUSION_H_ */
