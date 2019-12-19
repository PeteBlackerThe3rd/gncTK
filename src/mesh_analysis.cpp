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
 - triangle circumference analysis
 - vertex spacing analysis


-------------------------------------------------------------*/
#include <mesh_analysis.h>

int gncTK::MeshAnalysis::lastClosestType = 0;

/// Method to calculate the distance between a point and a line
gncTK::MeshAnalysis::ClostestRes gncTK::MeshAnalysis::distanceToLine(Eigen::Vector3f p1,
																	 Eigen::Vector3f p2,
																	 Eigen::Vector3f target)
{
	ClostestRes res;

	// calculate t factor (position on line which is closest to target
	float p2p1mag = (p2 - p1).norm();
	float t = ((p1 - target).dot(p1 - p2)) / (p2p1mag * p2p1mag);

	// calculate location of closest point on the line
	res.point = p1 + (p2 - p1) * t;

	// if the closest point on the line is not within this line segment
	if (t < 0 || t > 1)
		res.distance = NAN;
	else
		res.distance = (res.point-target).norm();

	// return distance to closest point
	return res;
}

/// Method to calculate the distance between a point and a triangle
gncTK::MeshAnalysis::ClostestRes gncTK::MeshAnalysis::distanceToTriangle(Eigen::Vector3f p1,
																		 Eigen::Vector3f p2,
																		 Eigen::Vector3f p3,
																		 Eigen::Vector3f target)
{
	ClostestRes res;

	// first calculate the normal vector of the triangle
	Eigen::Vector3f normal = (p2-p1).cross(p3-p2);

	// calculate the vector from p1 to the target
	Eigen::Vector3f p1tot = target - p1;

	// calculate the angle between this vector and the normal
	float cosAngle = (p1tot.dot(normal)) / ( p1tot.norm() * normal.norm() );

	// calculate the distance from the plane of the triangle to the target
	res.distance = p1tot.norm() * cosAngle;

	// -----------------------------------------------------------------------------
	// that was the easy bit, now we need to determine if the closest point lies
	// within the triangle or not
	// -----------------------------------------------------------------------------

	// find the target point projected onto the triangle plane
	res.point = target + ((normal / normal.norm()) * (0 - res.distance));

	res.distance = fabs(res.distance);

	// test if the projected point is within the triangle area
	float side1 = ((p1 - res.point).cross( p2 - res.point )).dot(normal);
	float side2 = ((p2 - res.point).cross( p3 - res.point )).dot(normal);
	float side3 = ((p3 - res.point).cross( p1 - res.point )).dot(normal);

	// if the projected point is outside of the triangle then this triangle is not the closest point
	if (side1 <=0 || side2 <= 0 || side3 <= 0)
		res.distance = NAN;

	return res;
}

/// Method to find the vector between the given point and the closest vertex of the given mesh
Eigen::Vector3f gncTK::MeshAnalysis::distanceToMeshVerts(Eigen::Vector3f point, gncTK::Mesh *mesh)
{
	// default not found value
	Eigen::Vector3f res(NAN, NAN, NAN);

	// if the KD tree hasn't been created yet then make it
	if (!mesh->KDTreeCached)
	{
		ROS_INFO("Generating KD Tree.");
		pcl::PointCloud<pcl::PointXYZ> cloud = mesh->toPointCloud();
		mesh->kdTree.setInputCloud(cloud.makeShared());
		mesh->KDTreeCached = true;
		ROS_INFO("finished KD tree.");
	}

	pcl::PointXYZ target(point[0], point[1], point[2]);

	// find closest vertex
	std::vector<int> nearestIndex(1);
	std::vector<float> nearestSqDist(1);

	int error = mesh->kdTree.nearestKSearch(target, 1, nearestIndex, nearestSqDist);

	// if kdTree search found a result
	if (error > 0)
	{
		Eigen::Vector3f closest = mesh->vertices[nearestIndex[0]];
		res = closest - point;
	}

	return res;
}

/// Method to calculate the shortest distance between a point and a mesh
gncTK::MeshAnalysis::ClosestResults gncTK::MeshAnalysis::distanceToMesh(Eigen::Vector4f point,
																	 gncTK::Mesh *mesh,
																	 bool *isOnEdge)
{
	ClosestResults res;
	res.distance = NAN; // <-- default case is no result found
	res.cVert = Eigen::Vector3f::Zero();
	res.cLine = Eigen::Vector3f::Zero();
	res.cTri = Eigen::Vector3f::Zero();

	// if the KD tree hasn't been created yet then make it
	if (!mesh->KDTreeCached)
	{
		ROS_INFO("Generating KD Tree.");
		pcl::PointCloud<pcl::PointXYZ> cloud = mesh->toPointCloud();
		mesh->kdTree.setInputCloud(cloud.makeShared());
		mesh->KDTreeCached = true;
		ROS_INFO("finished KD tree.");
	}

	// if the vertex to triangle links haven't been generated then calculate them
	/*if (mesh->vertices.size() != mesh->vertexTriangleLinks.size())
	{
		ROS_INFO("Generating triangle links.");
		mesh->calculateVertexToTriangleLinks();
		ROS_INFO("finished triangle links");
	}*/

	// of the mesh edge verts and arcs lists have not been calculated
	if (mesh->vertices.size() != mesh->vertexEdges.size())
	{
		mesh->calculateEdges();
	}

	if (mesh->triangleBoxes.size() == 0)
		mesh->calculateTriBoxes();

	pcl::PointXYZ target(point[0], point[1], point[2]);
	Eigen::Vector3f targetE = point.head(3);

	// find closest vertex
	std::vector<int> nearestIndex(1);
    std::vector<float> nearestSqDist(1);

    //ROS_INFO("starting KD search"); fflush(stdout);
    int error = mesh->kdTree.nearestKSearch(target, 1, nearestIndex, nearestSqDist);
    //ROS_INFO("finished KD search. result=%d", error); fflush(stdout);

    /*if (error <= 0)
    {
    	ROS_INFO("KD search didn't find any results!");
    	fflush(stdout);
    }*/

    if (error > 0)
    {
    	//ROS_INFO("KD results: distance=%f   index=%d",
    	//		 sqrt(nearestSqDist[0]),
		//		 nearestIndex[0]); fflush(stdout);

    	lastClosestType = ClosestVert;

    	//ROS_INFO("size of vertex edges array %d.", (int)mesh->vertexEdges.size());
    	//ROS_INFO("size of vertices array %d.", (int)mesh->vertices.size());

    	res.distance = sqrt(nearestSqDist[0]);
    	int closestVertex = nearestIndex[0];
    	res.cVert = mesh->vertices[closestVertex];

    	if (isOnEdge != NULL)
    		*isOnEdge = mesh->vertexEdges[closestVertex];

    	//ROS_INFO("Found closest Vert."); fflush(stdout);

    	if (res.distance < 1.0) // if the closest vertex was within a metre then search for closer edges and faces
    	{
			// calculate bounding box containing all objects which could
			// potentially be closer than the found vertex and find a list
			// of candidate triangles which may intersect this box using the RTree
			Eigen::Matrix<float, 3, 2> closerBox;
			closerBox.block(0,0,3,1) = targetE.array() - res.distance;
			closerBox.block(0,1,3,1) = targetE.array() + res.distance;
			std::vector<int> overlappingTris = mesh->triangleRTree.search(closerBox);

			// test all overlapping triangles to find if their area or any of
			// their edges are closer than the closest vertex
			for (int o=0; o<overlappingTris.size(); ++o)
			{
				int t = overlappingTris[o];

				int v1 = mesh->triangles[t].v1;
				int v2 = mesh->triangles[t].v2;
				int v3 = mesh->triangles[t].v3;

				// test if the area of this triangle is closer and if it isn't
				// then test it's edges
				ClostestRes distRes = distanceToTriangle(mesh->vertices[v1],
														 mesh->vertices[v2],
														 mesh->vertices[v3],
														 targetE);
				if (distRes.distance < res.distance)
				{
					res.distance = distRes.distance;
					res.cTri = distRes.point;
					lastClosestType = ClosestTri;

					if (isOnEdge != NULL)
						 *isOnEdge = false; // A point within a triangle by definition is not on an edge!
				}
				else	// test the edges of the triangle
				{
					std::vector<ClostestRes> edgeResults;
					std::vector<bool> areEdges;
					edgeResults.push_back(distanceToLine(mesh->vertices[v1], mesh->vertices[v2], targetE));
					edgeResults.push_back(distanceToLine(mesh->vertices[v2], mesh->vertices[v3], targetE));
					edgeResults.push_back(distanceToLine(mesh->vertices[v3], mesh->vertices[v2], targetE));
					areEdges.push_back(mesh->isEdge(v1, v2));
					areEdges.push_back(mesh->isEdge(v2, v3));
					areEdges.push_back(mesh->isEdge(v3, v1));

					for (int e=0; e<3; ++e)
					{
						if (edgeResults[e].distance < res.distance)
						{
							res.distance = edgeResults[e].distance;
							res.cLine = edgeResults[e].point;
							lastClosestType = ClosestLine;
							if (isOnEdge != NULL)
								*isOnEdge = areEdges[e];
						}
					}
				}
			}
    	}
    }

    //ROS_INFO("Reached end of closest point method"); fflush(stdout);

    return res;
}

// method to calculate the geometric deviation between this mesh and another one
gncTK::Stats1D gncTK::MeshAnalysis::geometricDeviation(gncTK::Mesh *meshA,
		 	 	 	 	 	 	 	 	 	 	 	   gncTK::Mesh *meshB,
													   tf::StampedTransform transform,
													   int histogramBinCount,
													   bool markVertexColor,
													   bool ignoreEdges)
{
	double tfMatrix[16];
	transform.getOpenGLMatrix(tfMatrix);
	Eigen::Matrix4f EigenTransform;

	for (int r=0; r<4; ++r)
		for (int c=0; c<4; ++c)
			EigenTransform(r,c) = tfMatrix[r+c*4];

	return geometricDeviation(meshA, meshB, EigenTransform, histogramBinCount, markVertexColor, ignoreEdges);
}

// method to calculate the geometric deviation between this mesh and another one
gncTK::Stats1D gncTK::MeshAnalysis::geometricDeviation(gncTK::Mesh *meshA,
		 	 	 	 	 	 	 	 	 	 	 	   gncTK::Mesh *meshB,
													   Eigen::Matrix4f transform,
													   int histogramBinCount,
													   bool markVertexColor,
													   bool ignoreEdges)
{
	std::vector<double> gDevs;
	std::vector<double> gDevsReal;

	// calculate edges for both meshes
	meshA->calculateEdges();
	meshB->calculateEdges();

	printf("Calculating Geometric deviation pointsA=%d pointB=%d \n\n",
			(int)meshA->vertices.size(),
			(int)meshB->vertices.size()); fflush(stdout);

	for (int v=0; v<meshA->vertices.size(); ++v)
	{
		if (v%10 == 0)
			printf("\r%6.2f%% complete (out of %d)",
				   (v*100.0) / meshA->vertices.size(),
				   (int)meshA->vertices.size());

		// if this vertex is a NAN then ignore it
		if (std::isnan(meshA->vertices[v][0]) ||
			std::isnan(meshA->vertices[v][1]) ||
			std::isnan(meshA->vertices[v][2]))
		{
			gDevs.push_back(NAN);
			continue;
		}

		// if this vertex is on the edge then ignore it if requested
		if (ignoreEdges && meshA->vertexEdges[v])
		{
			gDevs.push_back(NAN);
			continue;
		}

		Eigen::Vector4f vertPadded;
		vertPadded << meshA->vertices[v][0],
					  meshA->vertices[v][1],
					  meshA->vertices[v][2],
					  1;

		Eigen::Vector4f vertTransformed = transform.inverse() * vertPadded;

		bool isOnEdge;
		ClosestResults res = distanceToMesh(vertTransformed, meshB, &isOnEdge);
		if (ignoreEdges && isOnEdge) res.distance = NAN;

		gDevs.push_back(res.distance);
		if (!std::isnan(res.distance))
			gDevsReal.push_back(res.distance);
	}
	printf("\nCompleted Geometric deviation\n");

	gncTK::Stats1D stats(gDevsReal, histogramBinCount);

	cv::Vec3b nanColor = cv::Vec3b(0,0,255);

	if (markVertexColor)
	{
		// ensure that the mesh's vertex color vector is filled
		if (meshA->vertexColors.size() != meshA->vertices.size())
			meshA->vertexColors.resize(meshA->vertices.size());

		for (int v=0; v<meshA->vertices.size(); ++v)
		{
			if (std::isnan(gDevs[v]))
				meshA->vertexColors[v] = nanColor;
			else
			{
				float gdNorm = (gDevs[v]-stats.minInlier) / (stats.maxInlier - stats.minInlier);
				if (gdNorm > 1)
					meshA->vertexColors[v] = cv::Vec3b(255,255,255);
				else if (gdNorm < 0)
					meshA->vertexColors[v] = cv::Vec3b(0,0,0);
				else
				{
					meshA->vertexColors[v][0] = gdNorm * 255.0;
					meshA->vertexColors[v][1] = (1.0 - gdNorm) * 255.0;
					meshA->vertexColors[v][2] = 0;
				}
			}
		}
	}

	return stats;
}

double gncTK::MeshAnalysis::triangleSize(gncTK::Mesh *mesh, int t)
{
	double edge1 = (mesh->vertices[mesh->triangles[t].v1] - mesh->vertices[mesh->triangles[t].v2]).norm();
	double edge2 = (mesh->vertices[mesh->triangles[t].v2] - mesh->vertices[mesh->triangles[t].v3]).norm();
	double edge3 = (mesh->vertices[mesh->triangles[t].v3] - mesh->vertices[mesh->triangles[t].v1]).norm();

	double s = (edge1+edge2+edge3) / 2;

	return sqrt( s * (s-edge1) * (s-edge2) * (s-edge3) );
}

gncTK::Stats1D gncTK::MeshAnalysis::triangleSizeStats(gncTK::Mesh *mesh,
													  int histogramBinCount,
													  bool markTriangles)
{
	std::vector<double> sizes;

	for (int t=0; t<mesh->triangles.size(); ++t)
	{
		sizes.push_back(triangleSize(mesh, t));
	}

	gncTK::Stats1D stats(sizes, histogramBinCount);

	if (markTriangles)
	{
		mesh->triangleColors.resize(mesh->triangles.size());

		for (int t=0; t<mesh->triangles.size(); ++t)
		{
			float sizeNorm = (sizes[t]-stats.min) / (stats.max - stats.min);
			if (sizeNorm > 1) sizeNorm = 1;
			if (sizeNorm < 0) sizeNorm = 0;

			mesh->triangleColors[t][0] = sizeNorm * 255;
			mesh->triangleColors[t][1] = (1.0 - sizeNorm) * 255;
			mesh->triangleColors[t][2] = 0;
		}
	}

	return stats;
}

// method to calculate the circumference of each triangle and the stats
gncTK::Stats1D gncTK::MeshAnalysis::triangleCircumference(gncTK::Mesh *mesh,
														  int histogramBinCount,
														  bool markTriangles)
{
	std::vector<double> circs;

	for (int t=0; t<mesh->triangles.size(); ++t)
	{
		double edge1 = (mesh->vertices[mesh->triangles[t].v1] - mesh->vertices[mesh->triangles[t].v2]).norm();
		double edge2 = (mesh->vertices[mesh->triangles[t].v2] - mesh->vertices[mesh->triangles[t].v3]).norm();
		double edge3 = (mesh->vertices[mesh->triangles[t].v3] - mesh->vertices[mesh->triangles[t].v1]).norm();

		circs.push_back((edge1+edge2+edge3) / 2);
	}

	gncTK::Stats1D stats(circs, histogramBinCount);

	if (markTriangles)
	{
		mesh->triangleColors.resize(mesh->triangles.size());

		for (int t=0; t<mesh->triangles.size(); ++t)
		{
			float circNorm = (circs[t]-stats.min) / (stats.max - stats.min);
			if (circNorm > 1) circNorm = 1;
			if (circNorm < 0) circNorm = 0;

			mesh->triangleColors[t][0] = circNorm * 255;
			mesh->triangleColors[t][1] = (1.0 - circNorm) * 255;
			mesh->triangleColors[t][2] = 0;
		}
	}

	return stats;
}


