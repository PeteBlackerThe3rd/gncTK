#ifndef GNCTK_MESH_ANALYSIS_H_
#define GNCTK_MESH_ANALYSIS_H_

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

mesh_analysis.h

Object containing a set of mesh analysis functions
---------------------------------------------------

Contains:
 - geometric deviation analysis
 - triangle area analysis
 - vertex spacing analysis


-------------------------------------------------------------*/
#include <stdio.h>
#include "cv.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "stats1d.h"
#include "utils.h"
#include <math.h>
#include "mesh.h"

namespace gncTK
{
	class MeshAnalysis;
}

class gncTK::MeshAnalysis
{
public:

	// method to calculate the geometric deviation between this mesh and another one
	static gncTK::Stats1D geometricDeviation(gncTK::Mesh *meshA,
											 gncTK::Mesh *meshB,
											 tf::StampedTransform transform,
											 int histogramBinCount = 20,
											 bool markVertexColor = true,
											 bool ignoreEdges = false);

	// method to calculate the geometric deviation between this mesh and another one
	static gncTK::Stats1D geometricDeviation(gncTK::Mesh *meshA,
											 gncTK::Mesh *meshB,
											 Eigen::Matrix4f transform,
											 int histogramBinCount = 20,
											 bool markVertexColor = true,
											 bool ignoreEdges = false);

	// method to calculate the area of a single triangle in a mesh
	static double triangleSize(gncTK::Mesh *mesh, int t);

	// method to calculate the area of each triangle and the stats
	static gncTK::Stats1D triangleSizeStats(gncTK::Mesh *mesh,
										    int histogramBinCount = 20,
										    bool markTriangles = true);

	// method to calculate the circumference of each triangle and the stats
	static gncTK::Stats1D triangleCircumference(gncTK::Mesh *mesh,
											    int histogramBinCount = 20,
											    bool markTriangles = true);

	struct ClostestRes
	{
		Eigen::Vector3f point;
		float distance;
	};
	struct ClosestResults
	{
		Eigen::Vector3f cVert, cLine, cTri;
		float distance;
		int type;
	};

//private:

	/// Method to calculate the distance between a point and a line
	static ClostestRes distanceToLine(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f target);

	/// Method to calculate the distance between a point and a triangle
	static ClostestRes distanceToTriangle(Eigen::Vector3f p1, Eigen::Vector3f p2, Eigen::Vector3f p3, Eigen::Vector3f target);

	/// Method to calculate the shortest distance between a point and a mesh
	static ClosestResults distanceToMesh(Eigen::Vector4f point, gncTK::Mesh *mesh, bool *isOnEdge = NULL);

	/// Method to calculate the shortest distance between a point and the vertices of a mesh
	static Eigen::Vector3f distanceToMeshVerts(Eigen::Vector3f point, gncTK::Mesh *mesh);

	static const int ClosestVert = 1;
	static const int ClosestLine = 2;
	static const int ClosestTri = 3;

	static int lastClosestType;
};

#endif /* GNCTK_MESH_ANALYSIS_H_ */
