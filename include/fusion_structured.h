#ifndef GNCTK_FUSION_STRUCTURED_H_
#define GNCTK_FUSION_STRUCTURED_H_

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

fusion_structured.h

lidar camera fusion processing object
----------------------------------------------

This class fuses structured point clouds and images
into either textured mesh data or aligned RGB-D images.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include "opencv2/highgui/highgui.hpp"
#include "visualization_msgs/Marker.h"
#include "mesh.h"
#include "fusion.h"
#include "mesh_analysis.h"

namespace gncTK {
	class FusionStructured;
};

class gncTK::FusionStructured : public gncTK::Fusion
{
public:

	FusionStructured();
	~FusionStructured();

	gncTK::Mesh generateMesh(bool sensorOverlapOnly = false);

	cv::Mat generateDepthImage(int resolutionFactor = 1, int border = 0, Eigen::Vector3f gravity = Eigen::Vector3f::Zero());

	void setIncidentAngleThreshold(float angle);

private:

	bool isValid(pcl::PointXYZI point)
	{
		if (point.x == 0 && point.y == 0 && point.z == 0)
			return false;

		return (!std::isnan(point.x) &&
				!std::isnan(point.y) &&
				!std::isnan(point.z));
	}

	void setupOffscreenGLBuffer(int width, int height);

	std::vector<float> calculateGravityAngles(gncTK::Mesh mesh, Eigen::Vector3f gravity);

	std::vector<float> estimateVertexHeights(gncTK::Mesh mesh, float gridSize, Eigen::Vector3f gravity);

	// feature size threshold for mesh simplification
	float featureSize;
	float incidentAngleThreshold;

	/// Pointer to GLFW window used for OpenGL rendering
	GLFWwindow* window;
	bool glContextSetup;
	GLuint ssFbo, ssColorBuf, ssDepthBuf;
};

#endif /* GNCTK_FUSION_STRUCTURED_H_ */
