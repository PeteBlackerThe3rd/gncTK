#ifndef GNCTK_FUSION_QUAD_TREE_H_
#define GNCTK_FUSION_QUAD_TREE_H_

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

fusion_quad_tree.h

quad tree based camera lidar fusion object
----------------------------------------------

These source files contain two objects;

QuadTreeNode - a recurrsive quad tree object used to store
and manage the saliency quad tree for lidar surface
reconstruction.

FusionQuadTree - a concrete sub-class of the Fusion class
which implements the quad tree heat map method of lidar
camera fusion.

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

namespace gncTK {
	class FusionQuadTree;
	class QuadTreeNode;
};

class gncTK::QuadTreeNode
{
public:

	static int created, destroyed;

	QuadTreeNode(Eigen::Vector2f _topLeft, Eigen::Vector2f _bottomRight)
	{
		created += 1;

		topLeft = _topLeft;
		bottomRight = _bottomRight;
		pointCount = 0;
		meanPoint << 0,0,0;
		meanIntensity = 0.0f;
		pointCount = 0;
		depth = 0;
		isLeaf = true;
		meshVertexIndex = -1;

		sumD = sumD2 = 0.0;
		m2 = m3 = m4 = 0.0;
	};
	~QuadTreeNode()
	{
		destroyed += 1;

		/*ROS_INFO("destructing. d=[%d] tl[%f %f] br[%f %f]",
				 depth,
				 topLeft(0), topLeft(1),
				 bottomRight(0), bottomRight(1));*/
		if (!isLeaf)
		{
			delete tlChild;
			delete trChild;
			delete blChild;
			delete brChild;
		}
	};

	static bool ditheredSplit;

	void splitToCount(float scalingFactor, std::vector<cv::Mat> *heatMaps, int maxDepth);
	void split();
	bool mergeLeaves(bool reccursive = false);
	QuadTreeNode* findNode(Eigen::Vector2f pos);
	int count();
	int countNodes();
	int countNonZero();

	/// Method to check if the given position is within the given level of the heatmap pyramid
	static bool isWithinHeatMap(int row, int col, int level, std::vector<cv::Mat> *heatMaps);

	/// Method to redistribute a residual error to all levels of the given heatmap pyramid
	static void distributeError(int row, int col, int level, std::vector<cv::Mat> *heatMaps, float residualError);

	/// Method to redistribute a residual error down to levels below this of the given heatmap pyramid
	static void distributeErrorDown(int row, int col, int level, std::vector<cv::Mat> *heatMaps, float residualError);

	void filterLeaves(int minPointCount, int minNeighbourCount);

	void addVertices(gncTK::Mesh *mesh, gncTK::FusionFunction *fusionFunction, bool centreMeanPoints = false);

	// Method to populate the neighbour links of all leaves in the tree recursively
	void generateNeighbourLinks();

	// Helper functions for neighbour link generation
	std::vector<QuadTreeNode*> getRightEdgeLeaves();
	std::vector<QuadTreeNode*> getLeftEdgeLeaves();
	std::vector<QuadTreeNode*> getTopEdgeLeaves();
	std::vector<QuadTreeNode*> getBottomEdgeLeaves();

	// Method to triangulate this quad tree into the given mesh
	void generateTriangles(gncTK::Mesh *mesh);

	/// Method to add leaf statistics to the given images
	void addStats(int rows, int cols,
			      cv::Mat *N,
				  cv::Mat *meanImg,
				  cv::Mat *stdDevImg,
				  cv::Mat *skewImg,
				  cv::Mat *kurtImg);

	// method to add this leave and child leaves to a depth image
	void addDepth(int rows, int cols, cv::Mat *Count);

	QuadTreeNode *tlChild, *trChild, *blChild, *brChild;

	// Neighbour links used by meshing algorithm
	std::vector<QuadTreeNode*> topNs, leftNs, bottomNs, rightNs;

	bool isLeaf;
	int depth;

	Eigen::Vector2f topLeft, bottomRight;
	double heatMapValue;
	double meanHeatMapValue;

	Eigen::Vector3f meanPoint;
	double meanIntensity;
	int pointCount;

	// stats accumulators

	double sumD, sumD2;
	double m2, m3, m4;

	int meshVertexIndex;

	std::string frameId;
};

// -------------------------------------------------------------------------------------

class gncTK::FusionQuadTree : public gncTK::Fusion
{
public:

	FusionQuadTree();
	~FusionQuadTree();

	gncTK::Mesh generateMesh(bool centreMeanPoints = false);

	/// overloaded point cloud input method for this fusion methodology
	void setInputCloud(pcl::PointCloud<pcl::PointXYZI> cloud);

	/// Helper function to calcualte the maximum quadtree depth for the given max point density and camera FOV
	static int calculateMaxQuadtreeDepth(float maxPointDensity, float cameraFOV);

	/// Method to populate the lidar decimation quadtree
	/*
	 * This method uses the horizontal planar geometric estimator with the point density map
	 * in points.m^-1
	 * Additional parameters are the rover height, gravity vector in the sensor frame
	 * a dithered quadtree flag and a flag to enable to generation of analysis frames
	 */
	void populateQuadtreePlanar(cv::Mat pointDensityMap,
								float roverHeight,
								float depthWeighting,
								float maxPointDensity,
								int maxQuadtreeDepth,
								Eigen::Vector3f gravity,
								float cameraFOVsolidAngle,
								bool dithered = false,
								bool exportAnalysis = false);

	/// Method to populate the lidar decimation quadtree
	/*
	 * This method uses the spherical geometric estimator with the point density map
	 * in points.m^-1
	 * Additional parameters are the radius, a dithered quadtree flag and a flag to
	 * enable to generation of analysis frames
	 */
	void populateQuadtreeSpherical(cv::Mat pointDensityMap,
								   float radius,
								   float maxPointDensity,
								   int maxQuadtreeDepth,
								   float cameraFOVsolidAngle,
								   bool dithered,
								   bool exportAnalysis);

	/// Method to populate the lidar decimation quadtree
	/*
	 * This method uses a previously calculated geometric estimator map, usually generated
	 * by a previous reconstruction
	 * Additional parameters are the radius, a dithered quadtree flag and a flag to
	 * enable to generation of analysis frames
	 */
	void populateQuadtreeGivenEstimator(cv::Mat pointDensityMap,
									    cv::Mat metersToSrRatio,
									    float maxPointDensity,
									    int maxQuadtreeDepth,
									    float cameraFOVsolidAngle,
									    bool dithered,
									    bool exportAnalysis);

	/// Method to set a target point count
	void setTargetPointCount(int target) { targetPointCount = target; };
	void clearTargetPointCount() { targetPointCount = -1; };
	float getDensityScalingFactorUsed() { return densityScalingFactorUsed; };

	/// Method to create the quad tree from a mono float heat map image
	void setQuadTreeHeatMap(cv::Mat heatMap, int leafCount = 10000, double gamma = 1.0, int maxQTDepth = 8);

	/// Method to create the quad tree from a mono float heat map image
	void setQuadTreeHeatMapV2(cv::Mat pointsPerPixelMap, int maxQTDepth = 8, bool dithered = true);

	/// Method to create a set of images covering the quad tree area showing stats for each leaf
	std::vector<cv::Mat> exportLeafStats();

	/// Method which returns a cv image with the covering the quad tree with the depth of leaves shown
	cv::Mat getLeafDepthImage(int rows, int cols);

	//cv::Mat generateDepthImage(int resolutionFactor = 1, int border = 0, Eigen::Vector3f gravity = Eigen::Vector3f::Zero());

	void setIncidentAngleThreshold(float angle);

	/// method to generate a floating point Mat of the metersPerSr ratio for a planar geometric compensator
	cv::Mat planarGeometricEstimator(Eigen::Vector3f gravity,
									 float roverHeight,
									 float depthWeighting,
									 int camRows,
									 int camCols);

	/// structure to store analysis maps
	struct AnalysisMaps
	{
		bool set;
		cv::Mat metersPerSr, pointsPerSr, pointsPerPixel, qtDepth, qtPointsPerPixel;
		cv::Mat pointsPerPixelDitherMod;
	} analysisMaps;

private:

	bool isValid(pcl::PointXYZI point)
	{
		if (point.x == 0 && point.y == 0 && point.z == 0)
			return false;

		return (!std::isnan(point.x) &&
				!std::isnan(point.y) &&
				!std::isnan(point.z));
	}

	/// Private method used to build the image pyramids (mid-maps) from power of two input image
	std::vector<cv::Mat> buildImagePyramid(cv::Mat input);

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

	std::string frameId;
	Eigen::Vector3f sensorOrigin;

	bool quadTreeSet;
	gncTK::QuadTreeNode *treeRoot;

	float densityScalingFactorUsed;
	int targetPointCount;
};

#endif /* GNCTK_FUSION_QUAD_TREE_H_ */
