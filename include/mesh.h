#ifndef GNCTK_MESH_H_
#define GNCTK_MESH_H_

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

mesh.h

triangular mesh storage object
----------------------------

This object stores various different forms of triangulated
mesh. It can include single or multiple texture information
as well as per vertex color and normal information

Helper functions are provided to load and save this data
in wavefront OBJ and stanford PLY formats.

Functions are also provided to convert the data to PCL
Point cloud objects or ROS mesh marker messages

An efficient geometric deviation calculation is also provided
to calculate the geometric deviations between this mesh and
another.

-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "opencv2/highgui/highgui.hpp"
#include "visualization_msgs/Marker.h"
#include "tf/tf.h"
#include "tf/transform_datatypes.h"
#include <tf/transform_listener.h>
#include "utils.h"
#include "fusion_function.h"
#include "gnc_tool_kit/FusedSurface.h"

namespace gncTK
{
	class Mesh;

	template <class T, int dictSize = 5, int dimensions = 3> class RTree;
};

/// Class used to store, add and search the nodes of an R-Tree (Rectangle Tree)
/*
 * An R-Tree is used to optimise searching for triangles which potentially
 * overlap with a given bounding rectangle search area.
 *
 * This class is N dimensions (defaults to 3)
 * This class can have any size of dictionairy (defaults to 10)
 */
template <class T, int dictSize, int dimensions>
class gncTK::RTree
{
public:

	RTree()
	{
		isLeaf = true;
		parent = NULL;
	};

	RTree(RTree<T, dictSize, dimensions> *parent)
	{
		isLeaf = true;
		this->parent = parent;
	};

	~RTree()
	{
		if (!isLeaf)
			delete childA, childB;
	}

	void bulkLoad(std::vector<T> newValues,
				  std::vector<Eigen::Matrix<float, dimensions, 2> > newExtents);

	std::vector<T> search(Eigen::Matrix<float, dimensions, 2> searchArea);

	int count();

	Eigen::Matrix<float, dimensions, 2> nodeExtents;
	bool isLeaf;
	std::vector<T> values;
	std::vector<Eigen::Matrix<float, dimensions, 2> > extents;
	RTree<T, dictSize, dimensions> *childA;
	RTree<T, dictSize, dimensions> *childB;
	RTree<T, dictSize, dimensions> *parent;

private:

	class SortableElement
	{
	public:
		SortableElement(Eigen::Matrix<float, dimensions, 2> extents, T value, int sortDim)
		{
			this->extents = extents;
			this->value = value;
			this->sortDim = sortDim;
		}

		Eigen::Matrix<float, dimensions, 2> extents;
		T value;
		int sortDim;

		bool operator<(const SortableElement& b)
		{
			return ((extents(sortDim,0)+extents(sortDim,1)) < (b.extents(sortDim,0)+b.extents(sortDim,1)));
		}
	};

	inline Eigen::Matrix<float, dimensions, 2> combine(Eigen::Matrix<float, dimensions, 2> A,
												Eigen::Matrix<float, dimensions, 2> B)
	{
		Eigen::Matrix<float, dimensions, 2> combined;
		for (int d=0; d<dimensions; ++d)
		{
			combined(d,0) = std::min(A(d,0), B(d,0));
			combined(d,1) = std::max(A(d,1), B(d,1));
		}
		return combined;
	};

	inline float size(Eigen::Matrix<float, dimensions, 2> box)
	{
		float size=0;
		for (int d=0; d<dimensions; ++d)
			size += box(d,1) - box(d,0);
		return size;
	};
};

template <class T, int dictSize, int dimensions>
std::vector<T> gncTK::RTree<T, dictSize, dimensions>::search(Eigen::Matrix<float, dimensions, 2> searchArea)
{
	std::vector<T> results;

	// if this is a leaf node then search the values in its dictionary
	if (isLeaf)
	{
		for (int i=0; i<values.size(); ++i)
			if (gncTK::Utils::boxesOverlap(searchArea, extents[i]))
				results.push_back(values[i]);
		return results;
	}
	else // if this is a branch node then check the extents of both children and search accordingly
	{
		bool overlapA = gncTK::Utils::boxesOverlap(searchArea, childA->nodeExtents);
		bool overlapB = gncTK::Utils::boxesOverlap(searchArea, childB->nodeExtents);

		// if neither child node overlaps with the search area return empty list
		if (!overlapA && !overlapB)
			return results;

		// if the just A or B overlap the search area return their respective search results
		if (overlapA && !overlapB)
			return childA->search(searchArea);
		if (!overlapA && overlapB)
			return childB->search(searchArea);

		// if both child regions overlap the search area return both search results concatenated
		if (overlapA && overlapB)
		{
			results = childA->search(searchArea);
			std::vector<T> resultsB = childB->search(searchArea);
			results.insert(results.end(), resultsB.begin(), resultsB.end());
			return results;
		}
	}
}

template <class T, int dictSize, int dimensions>
int gncTK::RTree<T, dictSize, dimensions>::count()
{
	if (isLeaf)
		return values.size();
	else
		return childA->count() + childB->count();
}

template <class T, int dictSize, int dimensions>
void gncTK::RTree<T, dictSize, dimensions>::bulkLoad(std::vector<T> newValues,
													 std::vector<Eigen::Matrix<float, dimensions, 2> > newExtents)
{
	// if there are more than dictionary size elements to be added then find the ideal split
	// and create two new branch nodes
	if (newValues.size() > dictSize)
	{
		isLeaf = false;
		childA = new RTree<T, dictSize, dimensions>(this);
		childB = new RTree<T, dictSize, dimensions>(this);

		// find the optimal split in each dimension
		std::vector<std::vector<SortableElement> > sortedElements;
		std::vector<float> bestSplitValues;
		std::vector<int> bestSplitPositions;
		float bestSplitValue;
		int bestDimension;

		ROS_INFO("RTree finding optimal split of %d rects", (int)newValues.size());

		for (int d=0; d<dimensions; ++d)
		{
			std::vector<SortableElement> sortedDimension;
			for (int i=0; i<newValues.size(); ++i)
				sortedDimension.push_back(SortableElement(newExtents[i], newValues[i], d));
			std::sort(sortedDimension.begin(), sortedDimension.end());
			sortedElements.push_back(sortedDimension);

			// find the ideal split point
			std::vector<float> splitScores;
			splitScores.resize(newValues.size(), 0.0);

			Eigen::Matrix<float, dimensions, 2> bottom = sortedDimension.begin()->extents;
			Eigen::Matrix<float, dimensions, 2> top = sortedDimension.end()->extents;
			splitScores[0] = size(bottom);
			splitScores[splitScores.size()-1] = size(top);

			// Accumulate MBRs from the bottom to the top, to find the best split point
			for (int i=1; i<newValues.size(); ++i)
			{
				bottom = combine(bottom, sortedDimension[i].extents);
				top = combine(top, sortedDimension[sortedDimension.size() - 1 - i].extents);

				splitScores[i] += size(bottom);
				splitScores[splitScores.size() - 1 - i] += size(top);
			}

			float bestSplit = splitScores[0];
			int bestPosition = 0;
			bool first = true;
			for (int i=newValues.size()*0.4; i<newValues.size()*0.6; ++i)
			{
				if (first || splitScores[i] < bestSplit)
				{
					bestSplit = splitScores[i];
					bestPosition = i;
					first = false;
				}
			}

			bestSplitValues.push_back(bestSplit);
			bestSplitPositions.push_back(bestPosition);

			if (d == 0 || bestSplit < bestSplitValue)
			{
				bestSplitValue = bestSplit;
				bestDimension = d;
			}
		}

		// create two child nodes using the ideal split found
		std::vector<T> childAValues, childBValues;
		std::vector<Eigen::Matrix<float, dimensions, 2> > childAExtents, childBExtents;

		for (int i=0; i<bestSplitPositions[bestDimension]; ++i)
		{
			childAValues.push_back(sortedElements[bestDimension][i].value);
			childAExtents.push_back(sortedElements[bestDimension][i].extents);
		}
		for (int i=bestSplitPositions[bestDimension]; i<sortedElements[0].size(); ++i)
		{
			childBValues.push_back(sortedElements[bestDimension][i].value);
			childBExtents.push_back(sortedElements[bestDimension][i].extents);
		}

		childA->bulkLoad(childAValues, childAExtents);
		childB->bulkLoad(childBValues, childBExtents);
		nodeExtents = combine(childA->nodeExtents, childB->nodeExtents);
	}
	else // if there are fewer then simply add them to this leaf node
	{
		values.push_back(newValues[0]);
		extents.push_back(newExtents[0]);
		nodeExtents = newExtents[0];

		for (int i=1; i<newValues.size(); ++i)
		{
			values.push_back(newValues[i]);
			extents.push_back(newExtents[i]);
			nodeExtents = combine(nodeExtents, newExtents[i]);
		}
	}
}

class gncTK::Mesh
{
public:

	Mesh()
	{
		KDTreeCached = false;
		edgesCalculated = false;
	}

	class Triangle
	{
	public:
		Triangle(int _v1, int _v2, int _v3)
		{
			v1 = _v1;
			v2 = _v2;
			v3 = _v3;
			texId = 0;
		}
		Triangle(int _v1, int _v2, int _v3, int _texId)
		{
			v1 = _v1;
			v2 = _v2;
			v3 = _v3;
			texId = _texId;
		}
		Triangle(int _v1, int _v2, int _v3,
				 int _t1, int _t2, int _t3,
				 int _texId)
		{
			v1 = _v1;
			v2 = _v2;
			v3 = _v3;
			t1 = _t1;
			t2 = _t2;
			t3 = _t3;
			texId = _texId;
		}
		Triangle(int _v1, int _v2, int _v3,
							  int _t1, int _t2, int _t3,
							  int _n1, int _n2, int _n3,
							  int _texId)
		{
			v1 = _v1;
			v2 = _v2;
			v3 = _v3;
			t1 = _t1;
			t2 = _t2;
			t3 = _t3;
			n1 = _n1;
			n2 = _n2;
			n3 = _n3;
			texId = _texId;
		}
		void operator=(const gncTK::Mesh::Triangle& b)
		{
			v1 = b.v1;
			v2 = b.v2;
			v3 = b.v3;
			t1 = b.t1;
			t2 = b.t2;
			t3 = b.t3;
			n1 = b.n1;
			n2 = b.n2;
			n3 = b.n3;
			texId = b.texId;
		}

		int v1,v2,v3;
		int t1,t2,t3;
		int n1,n2,n3;
		int texId;
	};

	class Texture
	{
	public:
		cv::Mat texture;
		std::string fileName;
		std::string label;
		unsigned int glId;
	};

	std::vector<Eigen::Vector3f> vertices;
	std::vector<std::vector<int> > vertexTriangleLinks;
	std::vector<bool> vertexEdges;
	std::vector<Eigen::Vector3f> vertexNormals;
	std::vector<cv::Vec3b> vertexColors;
	std::vector<float> vertexIntensities;
	std::vector<Eigen::Vector2f> texCoords;
	std::vector<float> vertexPointDensities;
	std::vector<float> vertexPointPixelDensities;
	std::vector<int> vertexLidarSampleCount;

	std::vector<Eigen::Matrix<float, 3, 2> > triangleBoxes;
	RTree<int, 10> triangleRTree;

	// mesh edge storage, a vector for each vertex in the mesh with a list of
	// vertices of connected edges. Lowest numbered vertex is stored first to avoid duplicates
	std::vector<std::vector<int> > edgeArcs;

	std::vector<Triangle> triangles;
	std::vector<cv::Vec3b> triangleColors;
	std::vector<Texture> textures;
	std::vector<float> triangleProjectedGeometryRatios;
	std::vector<float> triangleIncidentAngles;

	std::string frameId;
	Eigen::Vector3f sensorOrigin;

	// static mesh factory methods
	static Mesh loadOBJ(std::string fileName);
	static Mesh loadPLY(std::string fileName);

	// methods to save this mesh in various formats
	bool saveOBJ(std::string baseName);
	bool savePLY(std::string fileName);

	// static factory methods to create mesh objects from point cloud objects
	static Mesh fromPCL(pcl::PointCloud<pcl::PointXYZ> pointCloud);
	static Mesh fromPCL(pcl::PointCloud<pcl::PointXYZI> pointCloud);
	static Mesh fromPCL(pcl::PointCloud<pcl::PointXYZRGB> pointCloud);

	// methods to generate a PCL point cloud of this mesh
	pcl::PointCloud<pcl::PointXYZ> toPointCloud();
	pcl::PointCloud<pcl::PointXYZRGB> toPointCloudColor();

	// define coloring methods for marker message generation
	static const int ColorAuto = 0;
	static const int ColorVertex = 1;
	static const int ColorTriangle = 2;
	static const int ColorTexture = 3;
	static const int ColorTint = 4;
	static const int ColorDoTint = 0x10;

	// method to generate a ROS marker message of the mesh
	visualization_msgs::Marker toMarkerMsg(int colorMode = ColorAuto, cv::Vec3b tint = cv::Vec3b(50,50,255));

	// method to generate a FusedSurface message of this mesh
	gnc_tool_kit::FusedSurface toFusedSurfaceMsg();

	// factory method to generate a mesh from a FusedSurface message
	static Mesh fromFusedSurfaceMsg(const gnc_tool_kit::FusedSurface &fusedSurfaceMsg);

	// method to create a single texture entry for this mesh and to set it to the given openCV mat
	void setSingleTexture(cv::Mat image);

	// method to get the bounding box of this mesh
	Eigen::Matrix<float, 3, 2> getExtents();

	/// Method to return the centre point of the bounding box of this mesh
	Eigen::Vector3f getCentre();

	static Eigen::Matrix<float, 3, 2> vertexExtents(std::vector<Eigen::Vector3f> vertices);

	// method to re-calculate/create the vertex normals for this mesh
	void calculateNormals();

	/// Method to change the frame id of this mesh and transform all its geometry into this new frame
	void changeCoordinateFrame(std::string newFrame,
							   tf::TransformListener *tfListener,
							   ros::Time tfTime = ros::Time(0));

	// cached KD tree for fast nearest neighbour lookup
	void setUnusedVerticesToNAN();
	bool KDTreeCached;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;

	pcl::KdTreeFLANN<pcl::PointXYZ>* getKdTree(bool debug = false);

	/// method to calculate the vertex to triangle links
	void calculateVertexToTriangleLinks();

	/// method to calculate the triangle bounding boxes
	void calculateTriBoxes();

	/// calculate edge arcs and vertices
	void calculateEdges();

	/// Calculate the point densities of each vertex
	void calculateVertexPointDensities();

	// Method to transfer a vector of numeric values to the color of vertices in this mesh
	/*
	 * Uses the hue rainbow to colorize the values, any numeric c type can be used.
	 * Verifies that the number of elements in values matches the number of vertices before proceeding.
	 */
	template <class elementT>
	void transferNumericValuesToVertexColors(std::vector<elementT> values);

	//void transferPointDensitiesToVertexColors();

	/// Calculate the point pixel densities (points per projected camera image pixel) of each vertex
	void calculateVertexPointPixelDensities();

	/// Method to calculate the projected geometry ratio for each triangle in the mesh given the set sensor origin
	/*
	 * The project geometry ratio is measure in meters squared per steradian, and represents the ratio between
	 * surface area and subtended solid angle at a point on a surface.
	 */
	void calculateTriangleProjectedGeometryRatios();

	/// Method to return a single channel float32 image with the depth of the surface
	/*
	 * The camera image plane locations are defined by the texture UV coordinates. pixels
	 * which do not see any of the surface will be NANs.
	 */
	cv::Mat generateDepthMap();
	/// Related method which projects the 3D positions of the mesh to an image
	/*
	 * The output mat is a 3 channel floating point image where each pixel represents the 3D location
	 * of the point on the mesh which is seen by that camera pixel. Useful for re-projecting visual features
	 * and information back into 3D space.
	 */
	cv::Mat generate3DMap();

	/// Method to project a range of different vertex descriptors into camera space.
	cv::Mat projectToCamera(gncTK::FusionFunction camera, int width, int height, std::string channel);
	cv::Mat projectPointPixelDensityToCamera(int width, int height);

	/// Method which returns true if the given arc is an edge and false otherwise
	/*
	 * The edge arc list must have already been populated using 'calculateEdges()'
	 */
	bool isEdge(int v1, int v2);

private:

	// OBJ specific data structures and methods
	int objLookupMtlName(std::string label);
	void objLoadMaterialLibrary(std::string mtlLibName, std::string basePath);

	/// helper function used by calculateEdges() function
	void processEdge(int v1, int v2);
	bool edgesCalculated;
};

#endif /* GNCTK_MESH_H_ */
