#ifndef GNCTK_UTILS_H_
#define GNCTK_UTILS_H_

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

utils.h

general helper functions and data structures
---------------------------------------------------



-------------------------------------------------------------*/

#include "cv.h"
#include "ros/ros.h"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>

namespace gncTK
{
	class Utils;
}

class gncTK::Utils
{
public:

	/// Method to aid debugging ROS
	static std::string toString(tf::Transform transform);

	/// Method to generate a point cloud which represents the edges of an orthogonal cuboid, usefull for simple RVIS visualisations
	static pcl::PointCloud<pcl::PointXYZRGB> pointCloudBox(float sX,
														   float sY,
														   float sZ,
														   std::string frameId,
														   float spacing=0.1);

	/// Methods for working with HD matricies
	static Eigen::Matrix4d normalizeDHMatrix(Eigen::Matrix4d matrix);
	static void analyseDHMatrix(Eigen::Matrix4d matrix);

	/// Methods for working with spherical triangles
	static float triangleSolidAngle(Eigen::Vector3f a,
									Eigen::Vector3f b,
									Eigen::Vector3f c, bool debug = false);

	/// Method to calculate the area of a triangle from three side lengths using Herons formula
	static float triangleAreaFromSides(float a, float b, float c);

	/// Templated method to calculate the internal angle of the vectors A->B and B->C
	template <class T>
	static float angleAtB(T a, T b, T c);

	// Method to convert a rotation matrix to a rotation quaternion
	/*
	 * Current assumes that the matrix is a proper rotation.
	 */
	static Eigen::Vector4f quaternionFromMatrix(Eigen::Matrix3f r);

	/// Method to find the average of a set of quaternions using the SVD approach
	/*
	 * The algorithm used in this algorithm is described here:
	 * https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
	 */
	static Eigen::Vector4f quaternionAverage(std::vector<Eigen::Vector4f> quaternions);

	// Methods to load and save CSV files easily
	static int readCSVFile(FILE *file, std::vector<std::vector<std::string> > *content,
						   char separator=',',
						   int skip=0);
	static int readCSVFile(std::string fileName, std::vector<std::vector<std::string> > *content,
						   char separator=',',
						   int skip=0);
	static void writeCSVFile(FILE *file,
							 std::vector<std::vector<std::string> > *content);
	static void writeCSVFile(std::string fileName,
							 std::vector<std::vector<std::string> > *content);
	static void writeCSVFile(FILE *file,
							 std::vector<std::vector<float> > content,
							 std::vector<std::string> header = std::vector<std::string>(0));
	static void writeCSVFile(std::string fileName,
							 std::vector<std::vector<float> > content,
							 std::vector<std::string> header = std::vector<std::string>(0));

	/// method to save floating point cv images as a CSV file
	static void writeCSVImage(std::string fileName,
							  cv::Mat image);

	// Methods to process file name strings
	static std::string fileNameWOPath(std::string fileName);
	static std::string fileNameWOExtension(std::string fileName);

	/// Method to return the HSV rainbow color for the position given by color, values are clamped between 0 and 1.
	static cv::Vec3b rainbow(float color);
	static cv::Mat rainbow(cv::Mat input);

	/// Method to convert a single channel of a floating point CV image as an 8 bit image with a scale bar
	static cv::Mat floatImageTo8Bit(cv::Mat floatImage, int channel=-1, bool useRainbow = true);

	/// Method to convert an Eigen matrix to a CV image as an 8 bit image with a scale bar
	static cv::Mat floatImageTo8Bit(Eigen::MatrixXd matrix, bool useRainbow = true);


	/// Method to fill all nan values in an image with the nearest internal finite value
	static cv::Mat fillEdgeNANs(cv::Mat input);

	/// Method to save a floating point openCV image to a file using custom format
	static void imwriteFloat(FILE *file, cv::Mat image);
	static void imwriteFloat(std::string fileName, cv::Mat image);

	/// Method to load a floating point openCV image to a file using custom format
	static cv::Mat imreadFloat(std::string fileName);

	/// Method to save a floating point openCV image to a pair of 8bit png files
	static void imwriteFloatPNG(std::string fileName, cv::Mat image, float ratio);

	/// Method to save a floating point openCV image to a pair of 8bit png files
	static cv::Mat imreadFloatPNG(std::string fileName, float ratio);

	/// Method to test if two N dimensionsal bounding boxes overlap
	/*
	 * Boxes are stored as Nx2 matrices which are two vectors, the lowest corner and the highest corner
	 */
	inline static bool boxesOverlap(Eigen::MatrixXf boxA, Eigen::MatrixXf boxB)
	{
		/*if (boxA.rows() != boxB.rows())
		{
			ROS_ERROR("Error trying to find overlap of boxes of different dimensionalities, %d and %d.",
					  (int)boxA.rows(),
					  (int)boxB.rows());
			return false;
		}*/
		bool overlap = true;
		for (int d=0; d<boxA.rows(); ++d)
			overlap &= (boxB(d,0) < boxA(d,1) && boxB(d,1) > boxA(d,0));
		return overlap;
	};

	class ErrorStats
	{
	public:
		int n;
		float mean, stdDev, min, max;
		float ratioMean, ratioStdDev, ratioMin, ratioMax;
		std::string toStr();
	};

	/// Method to calculate the error between two cv float mats (assumed to be the same size)
	static ErrorStats calcError(cv::Mat mapA, cv::Mat mapB, bool abs = true);

	struct GLBufferInfo
	{
		void free()
		{
			// release GL buffers
			glDeleteFramebuffers(1,&ssFboSC);

			glDeleteRenderbuffers(1,&ssColorBufSC);
			glDeleteRenderbuffers(1,&ssDepthBufSC);

			// release glfw window
			glfwDestroyWindow(windowSC);
		};

		/// Pointer to GLFW window used for OpenGL rendering
		GLFWwindow* windowSC;
		bool glContextSetupSC;
		GLuint ssFboSC, ssColorBufSC, ssDepthBufSC;
	};

	static GLBufferInfo setupOffscreenGLBuffer(int width, int height, GLint renderBufferType=GL_RGBA, bool quiet = true);

	/// Method to write an Eigen Matrix to a binary file
	template<class Matrix>
	static void writeEigenMatrix(std::string filename, const Matrix& matrix);

	/// Method to write an Eigen Matrix to a csv file
	template<class Matrix>
	static void writeEigenMatrixCSV(std::string filename, const Matrix& matrix, std::string format = "%f", char separator = ',');

	/// Method to read an Eigen Matrix to a binary file
	template<class Matrix>
	static void readEigenMatrix(std::string filename, Matrix& matrix);
};

#endif /* GNCTK_UTILS_H_ */
