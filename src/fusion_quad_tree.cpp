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

#include <fusion_quad_tree.h>

bool gncTK::QuadTreeNode::ditheredSplit = false;

int gncTK::QuadTreeNode::created = 0;
int gncTK::QuadTreeNode::destroyed = 0;

bool gncTK::QuadTreeNode::isWithinHeatMap(int row, int col, int level, std::vector<cv::Mat> *heatMaps)
{
	return (row >= 0 &&
			col >= 0 &&
			row < heatMaps->at(level).rows &&
			col < heatMaps->at(level).cols);
}

void gncTK::QuadTreeNode::distributeError(int row, int col, int level, std::vector<cv::Mat> *heatMaps, float residualError)
{
	residualError = 0-residualError;

	// update the levels of the pyramid above this
	int tRow = row, tCol = col;
	for (int l = level+1; l<heatMaps->size(); ++l)
	{
		tRow /= 2;
		tCol /= 2;
		heatMaps->at(l).at<float>(tRow,tCol) += residualError;
	}

	// update this level and all levels beneath it
	distributeErrorDown(row, col, level, heatMaps, residualError);

	// update this level of the pyramid
	/*heatMaps->at(level).at<float>(row,col) += residualError;

	// for now I think this should work just updating the level below this one
	if (level-1 >= 0)
	{
		heatMaps->at(level-1).at<float>(row*2  , col*2  ) += residualError / 4.0;
		heatMaps->at(level-1).at<float>(row*2+1, col*2  ) += residualError / 4.0;
		heatMaps->at(level-1).at<float>(row*2  , col*2+1) += residualError / 4.0;
		heatMaps->at(level-1).at<float>(row*2+1, col*2+1) += residualError / 4.0;
	}*/
}

void gncTK::QuadTreeNode::distributeErrorDown(int row, int col, int level, std::vector<cv::Mat> *heatMaps, float residualError)
{
	// update this level of the pyramid
	heatMaps->at(level).at<float>(row,col) += residualError;

	// if this is not the bottom level of the pyramid then continue down recursively
	if (level-1 >= 0)
	{
		distributeErrorDown(row*2  , col*2  , level-1, heatMaps, residualError / 4.0);
		distributeErrorDown(row*2+1, col*2  , level-1, heatMaps, residualError / 4.0);
		distributeErrorDown(row*2  , col*2+1, level-1, heatMaps, residualError / 4.0);
		distributeErrorDown(row*2+1, col*2+1, level-1, heatMaps, residualError / 4.0);
	}
}

void gncTK::QuadTreeNode::splitToCount(float scalingFactor, std::vector<cv::Mat> *heatMaps, int maxDepth)
{
	// if the maximum tree depth has been exceeded then don't split this node
	if (depth >= maxDepth)
		return;

	int level = heatMaps->size() - 2 - depth; // use the depth within the tree instead of log2 calculations

	if (level < 0)
	{
		ROS_ERROR("split to count encountered a negative level of %d", level);
		return;
	}

	int thisLeft = topLeft[0] * (heatMaps->at(level+1).cols);
	int thisTop = topLeft[1] * (heatMaps->at(level+1).rows);

	float density = heatMaps->at(level+1).at<float>(thisTop, thisLeft);

	// if the density is less than 4 then there will almost certainly be
	// some residual error the needs to be shared between neighbouring cells.
	// the only case where this isn't true is if the density is exactly 1
	if ((density*scalingFactor) < 4.0 &&
		gncTK::QuadTreeNode::ditheredSplit)
	{
		float residualError;
		if ((density*scalingFactor) < (8.0/3.0))
		{
			residualError = (1.0 - (density*scalingFactor)) / scalingFactor;
		}
		else
		{
			residualError = (4.0 - (density*scalingFactor)) / scalingFactor;
		}

		// filter matrix (# = this cell) (- = already processed)
		// --------------
		//  -  #  7
		//  1  5  3

		// find how many of the Floyd Steinberg filter elements are within extents of the current heatMap
		float totalWeights = 0.0;
		bool left = false;
		//bool bRight = false;
		bool bottom = false;
		bool bLeft = false;

		if (isWithinHeatMap(thisTop, thisLeft+1, level+1, heatMaps))
		{
			totalWeights += 7;
			left = true;
		}
		if (isWithinHeatMap(thisTop+1, thisLeft, level+1, heatMaps))
		{
			totalWeights += 5;
			bottom = true;
		}
		if (isWithinHeatMap(thisTop+1, thisLeft+1, level+1, heatMaps))
		{
			totalWeights += 3;
			bLeft = true;
		}

		// only proceed to distribute the residual error if there is at least one cell to put it in!
		if (totalWeights > 0.0)
		{
			if (left)
				distributeError(thisTop, thisLeft+1, level+1, heatMaps, (residualError * 7) / totalWeights);
			if (bottom)
				distributeError(thisTop+1, thisLeft, level+1, heatMaps, (residualError * 5) / totalWeights);
			if (bLeft)
				distributeError(thisTop+1, thisLeft+1, level+1, heatMaps, (residualError * 3) / totalWeights);
		}
	}

	// if the target count is greater than 8/3 then split this node
	if ((density*scalingFactor) > (8.0/3.0))
	//if (density > (8.0/3.0))
	{
		split();

		tlChild->splitToCount(scalingFactor, heatMaps, maxDepth);
		trChild->splitToCount(scalingFactor, heatMaps, maxDepth);
		blChild->splitToCount(scalingFactor, heatMaps, maxDepth);
		brChild->splitToCount(scalingFactor, heatMaps, maxDepth);
	}
}

void gncTK::QuadTreeNode::split()
{
	// idiot check
	if (!isLeaf) return;

	isLeaf = false;

	Eigen::Vector2f mid = (topLeft+bottomRight)/2;

	tlChild = new QuadTreeNode( topLeft, mid );
	trChild = new QuadTreeNode( Eigen::Vector2f(mid[0],topLeft[1]), Eigen::Vector2f(bottomRight[0],mid[1]) );
	blChild = new QuadTreeNode( Eigen::Vector2f(topLeft[0],mid[1]), Eigen::Vector2f(mid[0],bottomRight[1]) );
	brChild = new QuadTreeNode( mid, bottomRight );

	tlChild->depth = trChild->depth = blChild->depth = brChild->depth = depth + 1;
}

bool gncTK::QuadTreeNode::mergeLeaves(bool reccursive)
{
	// idiot checks
	if (isLeaf)
	{
		printf("Error trying to merge a leaf node!\n");
		return false;
	}

	if (!tlChild->isLeaf ||
		!trChild->isLeaf ||
		!blChild->isLeaf ||
		!brChild->isLeaf)
	{
		printf("Error: trying to merge a node which children aren't all leaves.\n");
		return false;
	}

	// merge leaves
	isLeaf = true;

	meanHeatMapValue = (tlChild->meanHeatMapValue +
						trChild->meanHeatMapValue +
						blChild->meanHeatMapValue +
						brChild->meanHeatMapValue) / 4.0;
	//printf("merged heat map value %f\n", meanHeatMapValue);

	delete tlChild;
	delete trChild;
	delete blChild;
	delete brChild;

	return true;
}


gncTK::QuadTreeNode* gncTK::QuadTreeNode::findNode(Eigen::Vector2f pos)
{
	if (isLeaf)
		return this;

	if (pos[0] >= tlChild->bottomRight[0])
	{
		if (pos[1] >= tlChild->bottomRight[1])
			return brChild->findNode(pos);
		else
			return trChild->findNode(pos);
	}
	else
	{
		if (pos[1] >= tlChild->bottomRight[1])
			return blChild->findNode(pos);
		else
			return tlChild->findNode(pos);
	}
}

int gncTK::QuadTreeNode::count()
{
	if (isLeaf)
		return 1;
	else
		return trChild->count() +
			   tlChild->count() +
			   blChild->count() +
			   brChild->count();
}

int gncTK::QuadTreeNode::countNodes()
{
	if (isLeaf)
		return 0;
	else
		return trChild->count() +
			   tlChild->count() +
			   blChild->count() +
			   brChild->count() + 1;
}

int gncTK::QuadTreeNode::countNonZero()
{
	if (isLeaf)
	{
		if (pointCount > 0)
			return 1;
		else
			return 0;
	}
	else
		return trChild->countNonZero() +
			   tlChild->countNonZero() +
			   blChild->countNonZero() +
			   brChild->countNonZero();
}

void gncTK::QuadTreeNode::filterLeaves(int minPointCount, int minNeighbourCount)
{
	// if this is a leaf then check if it needs filtering
	if (isLeaf)
	{
		if (pointCount < minPointCount)
		{
			int neighbourCount = topNs.size() + leftNs.size() + bottomNs.size() + rightNs.size();
			if (neighbourCount < minNeighbourCount)
			{
				pointCount = 0;
			}
		}
	}
	else // if this is a node then filter the children
	{
		tlChild->filterLeaves(minPointCount, minNeighbourCount);
		trChild->filterLeaves(minPointCount, minNeighbourCount);
		blChild->filterLeaves(minPointCount, minNeighbourCount);
		brChild->filterLeaves(minPointCount, minNeighbourCount);
	}
}

void gncTK::QuadTreeNode::addVertices(gncTK::Mesh *mesh, gncTK::FusionFunction *fusionFunction, bool centreMeanPoints)
{
	if (!isLeaf)
	{
		tlChild->addVertices(mesh, fusionFunction, centreMeanPoints);
		trChild->addVertices(mesh, fusionFunction, centreMeanPoints);
		brChild->addVertices(mesh, fusionFunction, centreMeanPoints);
		blChild->addVertices(mesh, fusionFunction, centreMeanPoints);
	}
	else
	{
		if (pointCount > 0)
		{
			if (centreMeanPoints) // aggregate points using mean depth and leaf node centre
			{
				float depth = (meanPoint / pointCount).norm();

				// get pixel position of centre of leaf region
				Eigen::Vector2f centre = (topLeft + bottomRight) / 2;
				centre[0] *= mesh->textures[0].texture.cols;
				centre[1] *= mesh->textures[0].texture.rows;

				meanPoint = fusionFunction->interpolateReverseProjection(centre[0], centre[1]) * depth;
			}
			else // aggregate points using naive mean
			{
				meanPoint /= pointCount;
			}
			meanIntensity /= pointCount;

			//cv::Vec3b vertColor = gncTK::Utils::rainbow(meanHeatMapValue);
			//mesh->vertexColors.push_back(vertColor);

			mesh->vertices.push_back(meanPoint);
			mesh->vertexIntensities.push_back(meanIntensity);
			mesh->vertexLidarSampleCount.push_back(pointCount);
			meshVertexIndex = mesh->vertices.size() - 1;
		}
	}
}

// Method to populate the neighbour links of all leaves in the tree recursively
void gncTK::QuadTreeNode::generateNeighbourLinks()
{
  // if this is a node then create the internal links and split external links if needed
  if (!isLeaf)
  {
	// add internal links
	tlChild->rightNs.push_back(trChild);
	trChild->leftNs.push_back(tlChild);
	blChild->rightNs.push_back(brChild);
	brChild->leftNs.push_back(blChild);

	tlChild->bottomNs.push_back(blChild);
	blChild->topNs.push_back(tlChild);
	trChild->bottomNs.push_back(brChild);
	brChild->topNs.push_back(trChild);

	// for each external neighbour link split it if that neighbour has children or duplicate it if not
	if (leftNs.size() == 1)
	{
	  if (leftNs[0]->isLeaf)
	  {
		tlChild->leftNs.push_back(leftNs[0]);
		blChild->leftNs.push_back(leftNs[0]);
	  }
	  else
	  {
		tlChild->leftNs.push_back(leftNs[0]->trChild);
		blChild->leftNs.push_back(leftNs[0]->brChild);
	  }
	}

    if (rightNs.size() == 1)
    {
      if (rightNs[0]->isLeaf)
      {
    	trChild->rightNs.push_back(rightNs[0]);
    	brChild->rightNs.push_back(rightNs[0]);
      }
      else
      {
      	trChild->rightNs.push_back(rightNs[0]->tlChild);
      	brChild->rightNs.push_back(rightNs[0]->blChild);
      }
    }

    if (topNs.size() == 1)
    {
      if (topNs[0]->isLeaf)
      {
    	tlChild->topNs.push_back(topNs[0]);
    	trChild->topNs.push_back(topNs[0]);
      }
      else
      {
      	tlChild->topNs.push_back(topNs[0]->blChild);
      	trChild->topNs.push_back(topNs[0]->brChild);

      }
    }

    if (bottomNs.size() == 1)
    {
      if (bottomNs[0]->isLeaf)
      {
    	blChild->bottomNs.push_back(bottomNs[0]);
    	brChild->bottomNs.push_back(bottomNs[0]);
      }
      else
      {
      	blChild->bottomNs.push_back(bottomNs[0]->tlChild);
      	brChild->bottomNs.push_back(bottomNs[0]->trChild);
      }
    }

    // now generate neighbour links for each child node.
    tlChild->generateNeighbourLinks();
    trChild->generateNeighbourLinks();
    blChild->generateNeighbourLinks();
    brChild->generateNeighbourLinks();
  }
  else // if this is a leaf node
  {
	// for each external link if it's not the edge of the tree then split it's links
	if (leftNs.size() == 1)
	  leftNs = leftNs[0]->getRightEdgeLeaves();

	if (rightNs.size() == 1)
	  rightNs = rightNs[0]->getLeftEdgeLeaves();

	if (topNs.size() == 1)
	  topNs = topNs[0]->getBottomEdgeLeaves();

	if (bottomNs.size() == 1)
	  bottomNs = bottomNs[0]->getTopEdgeLeaves();
  }
}

std::vector<gncTK::QuadTreeNode*> gncTK::QuadTreeNode::getRightEdgeLeaves()
{
  std::vector<QuadTreeNode*> result;

  if (isLeaf)
	result.push_back(this);
  else
  {
	std::vector<QuadTreeNode*> rightTopEdge = trChild->getRightEdgeLeaves();
    result.insert(result.end(), rightTopEdge.begin(), rightTopEdge.end());

    std::vector<QuadTreeNode*> rightBottomEdge = brChild->getRightEdgeLeaves();
    result.insert(result.end(), rightBottomEdge.begin(), rightBottomEdge.end());
  }

  return result;
}

std::vector<gncTK::QuadTreeNode*> gncTK::QuadTreeNode::getLeftEdgeLeaves()
{
  std::vector<QuadTreeNode*> result;

  if (isLeaf)
	result.push_back(this);
  else
  {
	std::vector<QuadTreeNode*> leftTopEdge = tlChild->getLeftEdgeLeaves();
    result.insert(result.end(), leftTopEdge.begin(), leftTopEdge.end());

    std::vector<QuadTreeNode*> leftBottomEdge = blChild->getLeftEdgeLeaves();
    result.insert(result.end(), leftBottomEdge.begin(), leftBottomEdge.end());
  }

  return result;
}

std::vector<gncTK::QuadTreeNode*> gncTK::QuadTreeNode::getTopEdgeLeaves()
{
  std::vector<QuadTreeNode*> result;

  if (isLeaf)
	result.push_back(this);
  else
  {
	std::vector<QuadTreeNode*> topLeftEdge = tlChild->getTopEdgeLeaves();
    result.insert(result.end(), topLeftEdge.begin(), topLeftEdge.end());

    std::vector<QuadTreeNode*> topRightEdge = trChild->getTopEdgeLeaves();
    result.insert(result.end(), topRightEdge.begin(), topRightEdge.end());
  }

  return result;
}

std::vector<gncTK::QuadTreeNode*> gncTK::QuadTreeNode::getBottomEdgeLeaves()
{
  std::vector<QuadTreeNode*> result;

  if (isLeaf)
	result.push_back(this);
  else
  {
	std::vector<QuadTreeNode*> bottomLeftEdge = blChild->getBottomEdgeLeaves();
    result.insert(result.end(), bottomLeftEdge.begin(), bottomLeftEdge.end());

    std::vector<QuadTreeNode*> bottomRightEdge = brChild->getBottomEdgeLeaves();
    result.insert(result.end(), bottomRightEdge.begin(), bottomRightEdge.end());
  }

  return result;
}

void gncTK::QuadTreeNode::generateTriangles(gncTK::Mesh *mesh)
{
  if (!isLeaf)
  {
	tlChild->generateTriangles(mesh);
	trChild->generateTriangles(mesh);
	blChild->generateTriangles(mesh);
	brChild->generateTriangles(mesh);
  }
  else
  {
	// if this node isn't on the top or left edge then add quad using pair of triangles
	if (leftNs.size() != 0 &&
		topNs.size() != 0 &&
		leftNs[0]->rightNs[0] == this &&
		topNs[0]->bottomNs[0] == this)
	{
		if (meshVertexIndex != -1 && leftNs[0]->meshVertexIndex != -1 && topNs[0]->meshVertexIndex != -1)
			mesh->triangles.push_back(gncTK::Mesh::Triangle(leftNs[0]->meshVertexIndex, meshVertexIndex, topNs[0]->meshVertexIndex));

		int index = topNs[0]->leftNs.size()-1;
		if (topNs[0]->meshVertexIndex != -1 && leftNs[0]->meshVertexIndex != -1 && topNs[0]->leftNs[index]->meshVertexIndex != -1)
		{
			mesh->triangles.push_back(gncTK::Mesh::Triangle(leftNs[0]->meshVertexIndex, topNs[0]->meshVertexIndex, topNs[0]->leftNs[index]->meshVertexIndex));
		}
	}

	// add triangle fans on all four sides as needed
	for (int a=0; a<(int)leftNs.size()-1; ++a)
	{
		if (meshVertexIndex != -1 && leftNs[a+1]->meshVertexIndex != -1 && leftNs[a]->meshVertexIndex != -1)
			mesh->triangles.push_back(gncTK::Mesh::Triangle(leftNs[a+1]->meshVertexIndex, meshVertexIndex, leftNs[a]->meshVertexIndex));
	}
	for (int a=0; a<(int)rightNs.size()-1; ++a)
	{
		if (meshVertexIndex != -1 && rightNs[a]->meshVertexIndex != -1 && rightNs[a+1]->meshVertexIndex != -1)
			mesh->triangles.push_back(gncTK::Mesh::Triangle(rightNs[a]->meshVertexIndex, meshVertexIndex, rightNs[a+1]->meshVertexIndex));
	}
	for (int a=0; a<(int)topNs.size()-1; ++a)
	{
		if (meshVertexIndex != -1 && topNs[a]->meshVertexIndex != -1 && topNs[a+1]->meshVertexIndex != -1)
			mesh->triangles.push_back(gncTK::Mesh::Triangle(topNs[a]->meshVertexIndex, meshVertexIndex, topNs[a+1]->meshVertexIndex));
	}
	for (int a=0; a<(int)bottomNs.size()-1; ++a)
	{
		if (meshVertexIndex != -1 && bottomNs[a+1]->meshVertexIndex != -1 && bottomNs[a]->meshVertexIndex != -1)
			mesh->triangles.push_back(gncTK::Mesh::Triangle(bottomNs[a+1]->meshVertexIndex, meshVertexIndex, bottomNs[a]->meshVertexIndex));
	}
  }
}

/// Method to add leaf statistics to the given images
void gncTK::QuadTreeNode::addStats(int rows, int cols,
								   cv::Mat *N,
								   cv::Mat *meanImg,
								   cv::Mat *stdDevImg,
								   cv::Mat *skewImg,
								   cv::Mat *kurtImg)
{
	if (!isLeaf)
	{
		trChild->addStats(rows,cols, N,meanImg,stdDevImg,skewImg,kurtImg);
		tlChild->addStats(rows,cols, N,meanImg,stdDevImg,skewImg,kurtImg);
		brChild->addStats(rows,cols, N,meanImg,stdDevImg,skewImg,kurtImg);
		blChild->addStats(rows,cols, N,meanImg,stdDevImg,skewImg,kurtImg);
	}
	else
	{
		cv::Point tl(topLeft[0] * cols, topLeft[1] * rows);
		cv::Point br(bottomRight[0] * cols, bottomRight[1] * rows);

		// calculate stats from the accumulated values
		float mean = sumD/pointCount;

		float stdDev = sqrt((sumD2 / pointCount) - (mean * mean));

		float skew = (sqrt(pointCount) * m3) / (pow(m2, 3/2.0));

		float kurtosis = (pointCount*m4) / (m2*m2) - 3;

		cv::rectangle(*N, tl,br, pointCount, -1);
		cv::rectangle(*meanImg, tl,br, mean, -1);
		cv::rectangle(*stdDevImg, tl,br, stdDev, -1);
		cv::rectangle(*skewImg, tl,br, skew, -1);
		cv::rectangle(*kurtImg, tl,br, kurtosis, -1);
	}
}

/// method to add this leave and child leaves to a depth image
void gncTK::QuadTreeNode::addDepth(int rows, int cols, cv::Mat *Count)
{
	if (!isLeaf)
	{
		trChild->addDepth(rows,cols, Count);
		tlChild->addDepth(rows,cols, Count);
		brChild->addDepth(rows,cols, Count);
		blChild->addDepth(rows,cols, Count);
	}
	else
	{
		cv::Point tl(topLeft[0] * cols, topLeft[1] * rows);
		cv::Point br(bottomRight[0] * cols, bottomRight[1] * rows);
		cv::rectangle(*Count, tl,br, depth, -1);
	}
}

// -------------------------------------------------------------------------------------
// FusionQuadTree methods
// -------------------------------------------------------------------------------------


gncTK::FusionQuadTree::FusionQuadTree() : Fusion::Fusion()
{
	featureSize = 0.05;
	setIncidentAngleThreshold(86);
	glContextSetup = false;

	quadTreeSet = false;
	analysisMaps.set = false;

	targetPointCount = -1;
	densityScalingFactorUsed = -1;
}

gncTK::FusionQuadTree::~FusionQuadTree()
{
	// if the offscreen GL context was setup then destroy it
	if (glContextSetup)
	{
		glDeleteFramebuffers(1,&ssFbo);
		glDeleteRenderbuffers(1,&ssColorBuf);
		glDeleteRenderbuffers(1,&ssDepthBuf);
	}

	if (quadTreeSet)
		delete treeRoot;
}

void gncTK::FusionQuadTree::setIncidentAngleThreshold(float angle)
{
	incidentAngleThreshold = angle * (M_PI / 180.0);
}

/// Helper function to calcualte the maximum quadtree depth for the given max point density and camera FOV
int gncTK::FusionQuadTree::calculateMaxQuadtreeDepth(float maxPointDensity, float cameraFOV)
{
	int maxQuadtreeDepth = 0;
	int level = 0;
	float density;
	do
	{
		density = pow(4, level) / cameraFOV;
		if (density < maxPointDensity)
			maxQuadtreeDepth = level;
		++level;
	}
	while(density < maxPointDensity);

	return maxQuadtreeDepth;
}

/// Method to populate the lidar decimation quadtree
/*
 * This method uses the horizontal planar geometric estimator with the point density map
 * in points.m^-1
 * Additional parameters are the rover height, gravity vector in the sensor frame
 * a dithered quadtree flag and a flag to enable to generation of analysis frames
 */
void gncTK::FusionQuadTree::populateQuadtreePlanar(cv::Mat pointDensityMap,
												   float roverHeight,
												   float depthWeighting,
												   float maxPointDensity,
												   int maxQuadtreeDepth,
												   Eigen::Vector3f gravity,
												   float cameraFOVsolidAngle,
												   bool dithered,
												   bool exportAnalysis)
{
	// generate the geometric estimator frame
	cv::Mat metersToSrRatio(pointDensityMap.size(), CV_32F);
	metersToSrRatio = planarGeometricEstimator(gravity,
											   roverHeight,
											   depthWeighting,
											   pointDensityMap.rows,
											   pointDensityMap.cols);

	// calculate the density in points per Sr
	cv::Mat pointsPerSr = metersToSrRatio.mul(pointDensityMap);
	pointsPerSr = cv::min(pointsPerSr, maxPointDensity);

	// convert map of point angular density into map of points per (pow 2 image) pixel
	int newSize = pow(2, floor( log(std::min(pointDensityMap.rows,pointDensityMap.cols)) / log(2) ) );
	int pow2PixelCount = newSize*newSize;
	cv::Mat pointsPerPixel = pointsPerSr * (cameraFOVsolidAngle/pow2PixelCount);

	// create quad tree fused mesh
	setQuadTreeHeatMapV2(pointsPerPixel, maxQuadtreeDepth, dithered);

	analysisMaps.set = exportAnalysis;
	if (exportAnalysis)
	{
		// save maps used to generate points per pixel
		analysisMaps.metersPerSr = metersToSrRatio;
		analysisMaps.pointsPerSr = pointsPerSr;
		analysisMaps.pointsPerPixel = pointsPerPixel;

		// generate a map of the points per pixel that the quadtree represents
		analysisMaps.qtDepth = getLeafDepthImage(pointDensityMap.rows,
				   	   	   	   	   	   	   	     pointDensityMap.cols);
		analysisMaps.qtPointsPerPixel = cv::Mat(analysisMaps.qtDepth.size(), CV_32F);
		for (int i=0; i<analysisMaps.qtDepth.rows*analysisMaps.qtDepth.cols; ++i)
		{
			analysisMaps.qtPointsPerPixel.at<float>(i) = pow(4, analysisMaps.qtDepth.at<float>(i)) / pow2PixelCount;
		}

		// resize the quadtree points per pixel image to the same size as the camera image
		cv::resize(analysisMaps.qtPointsPerPixel,
				   analysisMaps.qtPointsPerPixel,
				   pointDensityMap.size(),
				   0, 0,
				   CV_INTER_LINEAR);
	}
}

/// Method to populate the lidar decimation quadtree
/*
 * This method uses the spherical geometric estimator with the point density map
 * in points.m^-1
 * Additional parameters are the radius, a dithered quadtree flag and a flag to
 * enable to generation of analysis frames
 */
void gncTK::FusionQuadTree::populateQuadtreeSpherical(cv::Mat pointDensityMap,
												      float radius,
													  float maxPointDensity,
													  int maxQuadtreeDepth,
													  float cameraFOVsolidAngle,
													  bool dithered,
													  bool exportAnalysis)
{
	// generate the geometric estimator frame
	cv::Mat metersToSrRatio(pointDensityMap.size(), CV_32F);
	metersToSrRatio = radius*radius;

	// calculate the density in points per Sr
	cv::Mat pointsPerSr = metersToSrRatio.mul(pointDensityMap);
	pointsPerSr = cv::min(pointsPerSr, maxPointDensity);

	// convert map of point angular density into map of points per (pow 2 image) pixel
	int newSize = pow(2, floor( log(std::min(pointDensityMap.rows,pointDensityMap.cols)) / log(2) ) );
	int pow2PixelCount = newSize*newSize;
	cv::Mat pointsPerPixel = pointsPerSr * (cameraFOVsolidAngle/pow2PixelCount);

	// create quad tree fused mesh
	setQuadTreeHeatMapV2(pointsPerPixel, maxQuadtreeDepth, dithered);

	analysisMaps.set = exportAnalysis;
	if (exportAnalysis)
	{
		// save maps used to generate points per pixel
		analysisMaps.metersPerSr = metersToSrRatio;
		analysisMaps.pointsPerSr = pointsPerSr;
		analysisMaps.pointsPerPixel = pointsPerPixel;

		// generate a map of the points per pixel that the quadtree represents
		analysisMaps.qtDepth = getLeafDepthImage(pointDensityMap.rows,
				   	   	   	   	   	   	   	     pointDensityMap.cols);
		analysisMaps.qtPointsPerPixel = cv::Mat(analysisMaps.qtDepth.size(), CV_32F);
		for (int i=0; i<analysisMaps.qtDepth.rows*analysisMaps.qtDepth.cols; ++i)
		{
			analysisMaps.qtPointsPerPixel.at<float>(i) = pow(4, analysisMaps.qtDepth.at<float>(i)) / pow2PixelCount;
		}

		// resize the quadtree points per pixel image to the same size as the camera image
		cv::resize(analysisMaps.qtPointsPerPixel,
				   analysisMaps.qtPointsPerPixel,
				   pointDensityMap.size(),
				   0, 0,
				   CV_INTER_LINEAR);
	}
}

/// Method to populate the lidar decimation quadtree
/*
 * This method uses a previously calculated geometric estimator map, usually generated
 * by a previous reconstruction
 * Additional parameters are the radius, a dithered quadtree flag and a flag to
 * enable to generation of analysis frames
 */
void gncTK::FusionQuadTree::populateQuadtreeGivenEstimator(cv::Mat pointDensityMap,
														   cv::Mat metersToSrRatio,
														   float maxPointDensity,
														   int maxQuadtreeDepth,
														   float cameraFOVsolidAngle,
														   bool dithered,
														   bool exportAnalysis)
{
	// generate the geometric estimator frame
	//cv::Mat metersToSrRatio(pointDensityMap.size(), CV_32F);
	//metersToSrRatio = radius*radius;

	// calculate the density in points per Sr
	cv::Mat pointsPerSr = metersToSrRatio.mul(pointDensityMap);
	pointsPerSr = cv::min(pointsPerSr, maxPointDensity);

	// convert map of point angular density into map of points per (pow 2 image) pixel
	int newSize = pow(2, floor( log(std::min(pointDensityMap.rows,pointDensityMap.cols)) / log(2) ) );
	int pow2PixelCount = newSize*newSize;
	cv::Mat pointsPerPixel = pointsPerSr * (cameraFOVsolidAngle/pow2PixelCount);

	// create quad tree fused mesh
	setQuadTreeHeatMapV2(pointsPerPixel, maxQuadtreeDepth, dithered);

	analysisMaps.set = exportAnalysis;
	if (exportAnalysis)
	{
		// save maps used to generate points per pixel
		analysisMaps.metersPerSr = metersToSrRatio;
		analysisMaps.pointsPerSr = pointsPerSr;
		analysisMaps.pointsPerPixel = pointsPerPixel;

		// generate a map of the points per pixel that the quadtree represents
		analysisMaps.qtDepth = getLeafDepthImage(pointDensityMap.rows,
				   	   	   	   	   	   	   	     pointDensityMap.cols);
		analysisMaps.qtPointsPerPixel = cv::Mat(analysisMaps.qtDepth.size(), CV_32F);
		for (int i=0; i<analysisMaps.qtDepth.rows*analysisMaps.qtDepth.cols; ++i)
		{
			analysisMaps.qtPointsPerPixel.at<float>(i) = pow(4, analysisMaps.qtDepth.at<float>(i)) / pow2PixelCount;
		}

		// resize the quadtree points per pixel image to the same size as the camera image
		cv::resize(analysisMaps.qtPointsPerPixel,
				   analysisMaps.qtPointsPerPixel,
				   pointDensityMap.size(),
				   0, 0,
				   CV_INTER_LINEAR);
	}
}

cv::Mat gncTK::FusionQuadTree::planarGeometricEstimator(Eigen::Vector3f gravity,
														float roverHeight,
														float depthWeighting,
														int camRows,
														int camCols)
{
	cv::Mat metersToSrRatio(camRows, camCols, CV_32F);

	for (int i=0; i<camRows*camCols;++i)
	{
		// find angle between local gravity vector and this pixels projected ray
		cv::Vec4f reprojection = fusionFunction.reverseProjection.at<cv::Vec4f>(i);
		Eigen::Vector3f ray; ray << reprojection[0], reprojection[1], reprojection[2];
		float theta = acos(ray.dot(gravity));

		if (theta > M_PI/2) // if this ray is pointing upwards then sent mPerSr to positive infinity
		{
			metersToSrRatio.at<float>(i) = std::numeric_limits<float>::infinity();
		}
		else // if this ray is facing down then calculate the mPerSr at it's intersection with the ground plane
		{
			metersToSrRatio.at<float>(i) = (roverHeight*roverHeight) * pow(depthWeighting, (1.0/cos(theta)) );
		}
	}

	return metersToSrRatio;
}

std::vector<cv::Mat> gncTK::FusionQuadTree::buildImagePyramid(cv::Mat input)
{
	std::vector<cv::Mat> levels;

	// first level is the input image
	levels.push_back(input);
	cv::Mat *lastImage = &levels[0];

	cv::Mat reducedImage;

	// continually halve the image until it's less than 1 pixel square
	while(lastImage->rows/2.0 >= 1)
	{
		reducedImage = cv::Mat(lastImage->rows/2, lastImage->cols/2, CV_32F);

		for (int r=0; r<reducedImage.rows; ++r)
			for (int c=0; c<reducedImage.cols; ++c)
			{
				reducedImage.at<float>(r,c) =
						(lastImage->at<float>(r+r,   c+c) +
						lastImage->at<float>(r+r+1, c+c) +
						lastImage->at<float>(r+r,   c+c+1) +
						lastImage->at<float>(r+r+1, c+c+1));// / 4.0;
			}

		levels.push_back(reducedImage);
		lastImage = &levels[levels.size()-1];
	}

	return levels;
}

/// Method to create the quad tree from a mono float heat map image
void gncTK::FusionQuadTree::setQuadTreeHeatMap(cv::Mat heatMap, int leafCount, double gamma, int maxQTDepth)
{
	// downscale the input heat map to the closest smaller power of two square size
	int newSize = pow(2, floor( log(std::min(heatMap.rows,heatMap.cols)) / log(2) ) );
	resize(heatMap, heatMap, cv::Size(newSize,newSize), 0, 0, cv::INTER_LINEAR);

	// build image pyramid used for quad-tree construction
	std::vector<cv::Mat> heatMaps = buildImagePyramid(heatMap);

	int totalPyramidPointCount = heatMaps[heatMaps.size()-1].at<float>(0,0);

	// free old quad tree if one exists
	if (quadTreeSet)
		delete treeRoot;

	// create new quad tree with approximately the number of leaves given
	treeRoot = new QuadTreeNode( Eigen::Vector2f(0,0), Eigen::Vector2f(1,1) );
	treeRoot->splitToCount(0, &heatMaps, maxQTDepth);

	printf("Generated a quad tree with a target count of %d and an actual count of %d.\n", totalPyramidPointCount, treeRoot->count());

    quadTreeSet = true;
}

/// Method to create the quad tree from a mono float heat map image
void gncTK::FusionQuadTree::setQuadTreeHeatMapV2(cv::Mat pointsPerPixelMap, int maxQTDepth, bool dithered)
{
	cv::Mat pppMap;// = pointsPerPixelMap.clone();

	// down scale the input heat map to the closest smaller power of two square size
	int newSize = pow(2, floor( log(std::min(pointsPerPixelMap.rows,pointsPerPixelMap.cols)) / log(2) ) );
	resize(pointsPerPixelMap, analysisMaps.pointsPerPixelDitherMod, cv::Size(newSize,newSize), 0, 0, cv::INTER_LINEAR);

	// build image pyramid used for quad-tree construction
	std::vector<cv::Mat> heatMaps = buildImagePyramid(analysisMaps.pointsPerPixelDitherMod);

	int totalPyramidPointCount = heatMaps[heatMaps.size()-1].at<float>(0,0);

	// if a targetPointCount was given then calculate the scaling factor
	float densityScalingFactor = 1.0;
	int targetCount = totalPyramidPointCount;
	if (targetPointCount != -1)
	{
		targetCount = targetPointCount;
		densityScalingFactor = (float)targetPointCount / totalPyramidPointCount;
	}
	densityScalingFactorUsed = densityScalingFactor;

	// free old quad tree if one exists
	if (quadTreeSet)
	{
		//int leafCount = treeRoot->count();
		//int nodeCount = treeRoot->countNodes();
		//ROS_WARN("deleting old quadtree, with [%d] nodes and [%d leaves], total [%d]", nodeCount, leafCount, nodeCount+leafCount);
		delete treeRoot;
	}

	/*ROS_WARN("created = [%12d] destroyed = [%12d] leftover = [%12d]",
			  gncTK::QuadTreeNode::created,
			  gncTK::QuadTreeNode::destroyed,
			  gncTK::QuadTreeNode::created - gncTK::QuadTreeNode::destroyed);*/

	// create new quad tree with approximately the number of leaves given
	gncTK::QuadTreeNode::ditheredSplit = dithered;
	treeRoot = new QuadTreeNode( Eigen::Vector2f(0,0), Eigen::Vector2f(1,1) );
	treeRoot->splitToCount(densityScalingFactor, &heatMaps, maxQTDepth);

	/*ROS_ERROR("created = [%12d] destroyed = [%12d] leftover = [%12d]",
			  gncTK::QuadTreeNode::created,
			  gncTK::QuadTreeNode::destroyed,
			  gncTK::QuadTreeNode::created - gncTK::QuadTreeNode::destroyed);*/



	ROS_INFO("Generated a quad tree with a target count of %d and an actual count of %d.\n", targetCount, treeRoot->count());

    quadTreeSet = true;
}

/// Method to create a set of images covering the quad tree area showing stats for each leaf
std::vector<cv::Mat> gncTK::FusionQuadTree::exportLeafStats()
{
	cv::Mat N(inputImage.size(), CV_32F);
	cv::Mat mean(inputImage.size(), CV_32F);
	cv::Mat stdDev(inputImage.size(), CV_32F);
	cv::Mat skew(inputImage.size(), CV_32F);
	cv::Mat kurtosis(inputImage.size(), CV_32F);

	treeRoot->addStats(inputImage.rows, inputImage.cols,
					   &N,
					   &mean,
					   &stdDev,
					   &skew,
					   &kurtosis);

	cv::Mat logN, logKurt;
	logN = cv::min(N, 10.0);
	cv::log(kurtosis, logKurt);

	cv::Mat oN = gncTK::Utils::floatImageTo8Bit(logN);
	cv::Mat omean = gncTK::Utils::floatImageTo8Bit(mean);
	cv::Mat ostdDev = gncTK::Utils::floatImageTo8Bit(stdDev);
	cv::Mat oskew = gncTK::Utils::floatImageTo8Bit(skew);
	cv::Mat okurtosis = gncTK::Utils::floatImageTo8Bit(logKurt);

	imwrite("stats_N.png", oN);
	imwrite("stats_mean.png", omean);
	imwrite("stats_stdDev.png", ostdDev);
	imwrite("stats_skew.png", oskew);
	imwrite("stats_kurtosis.png", okurtosis);

	std::vector<cv::Mat> stats;
	stats.push_back(N);
	stats.push_back(mean);
	stats.push_back(stdDev);

	return stats;
}

/// Method which returns a cv image with the covering the quad tree with the depth of leaves shown
cv::Mat gncTK::FusionQuadTree::getLeafDepthImage(int rows, int cols)
{
	cv::Mat count(rows, cols, CV_32F);
	treeRoot->addDepth(rows, cols, &count);

	return count;
}

// overloaded point cloud input method for this fusion methodology
void gncTK::FusionQuadTree::setInputCloud(pcl::PointCloud<pcl::PointXYZI> cloud)
{
	cloudSet = true;

	// set the frame id
	frameId = cloud.header.frame_id;

	// copy the sensor origin from the cloud to the new mesh
	sensorOrigin << inputCloud.sensor_origin_[0],
					inputCloud.sensor_origin_[1],
					inputCloud.sensor_origin_[2];

	// create vector of Eigen vectors from point cloud
	Eigen::Vector3f point;
	std::vector<Eigen::Vector3f> points;
	for (int p=0; p<cloud.points.size(); ++p)
	{
		point << cloud.points[p].x,
				 cloud.points[p].y,
				 cloud.points[p].z;
		points.push_back(point);
	}
	//printf("Made Eigen vector.\n"); fflush(stdout);

	std::vector<Eigen::Vector2f> camPoints = fusionFunction.projectPoints(points);
	//printf("Projected Points.\n"); fflush(stdout);

	int addedCount=0, ignoredCount = 0;

	for (int p=0; p<cloud.points.size(); ++p)
	{
		// convert camera image position to UV coordinates
		Eigen::Vector2f cam = camPoints[p];
		cam[0] /= inputImage.cols-1;
		cam[1] /= inputImage.rows-1;

		// if this lidar point is projected to within the camera frame
		if (cam[0] >= 0 && cam[1] >= 0 &&
			cam[0] <= 1 && cam[1] <= 1)
		{
			QuadTreeNode *bin = treeRoot->findNode(cam);
			/*if (bin->pointCount == 0)
				bin->meanPoint = points[p];
			else
			{
				if (points[p].norm() < bin->meanPoint.norm())
					bin->meanPoint = points[p];
			}*/
			bin->meanPoint += points[p];
			bin->meanIntensity += cloud.points[p].intensity;
			++bin->pointCount;
			++addedCount;

			/*double depth = points[p].norm();
			bin->sumD += depth;
			bin->sumD2 += depth*depth;
			//bin->sumD3 += depth*depth*depth;
			//bin->sumD4 += depth*depth*depth*depth;

			int n = bin->pointCount;
			int n1 = n - 1;
			double mean = bin->sumD / n;
			double delta = depth - mean;
			double delta_n = delta / n;
			double delta_n2 = delta_n * delta_n;

			double term1 = delta * delta_n * n1;

			bin->m4 += (term1 * delta_n2 * (n*n - 3*n + 3)) + (6 * delta_n2 * bin->m2) - (4 * delta_n * bin->m3);
			bin->m3 += (term1 * delta_n * (n - 2)) - (3 * delta_n * bin->m2);
			bin->m2 += term1;*/

/*			n1 = n
			n = n + 1
			delta = x - mean
			delta_n = delta / n
			delta_n2 = delta_n * delta_n
			term1 = delta * delta_n * n1
			mean = mean + delta_n
			M4 = M4 + term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
			M3 = M3 + term1 * delta_n * (n - 2) - 3 * delta_n * M2
			M2 = M2 + term1*/
		}
		else
			++ignoredCount;
	}
	//printf("Added points to quad tree.\n"); fflush(stdout);

	/*printf("Added %d verts and ignored %d\n",
		   addedCount, ignoredCount);

	float nonZeroRatio = treeRoot->countNonZero() / (float)treeRoot->count();
	printf("Leaf node non-zero ratio is %f after adding %d points.\n",
		   nonZeroRatio,
		   (int)camPoints.size());*/
}



gncTK::Mesh gncTK::FusionQuadTree::generateMesh(bool centreMeanPoints)
{
	Mesh newMesh;
	newMesh.frameId = frameId;
	// set the input image as the only texture for this mesh
	newMesh.setSingleTexture(inputImage);

	if (!cloudSet)
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : input point cloud not set or zero size.\n");
		return newMesh;
	}

	if (!imageSet)
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : no input image set.\n");
		return newMesh;
	}

	if (!quadTreeSet)
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : quad tree heat map not set.\n");
		return newMesh;
	}

	// calculate lists of leaf neighbours, used for filtering and triangulation
	treeRoot->generateNeighbourLinks();

	// generate vertices for every leaf in the quad tree which includes one or more points
	treeRoot->addVertices(&newMesh, &fusionFunction, centreMeanPoints);

	// for each vertex add a texture coordinate using the fusion function
	newMesh.texCoords = fusionFunction.projectPoints(newMesh.vertices);

	for (int t=0; t<newMesh.texCoords.size(); ++t)
	{
		float U = newMesh.texCoords[t][0] / (inputImage.cols-1);
		float V = 1.0 - (newMesh.texCoords[t][1] / (inputImage.rows-1) );

		if (U < 0) U = 0;
		if (U > 1) U = 1;
		if (V < 0) V = 0;
		if (V > 1) V = 1;

		newMesh.texCoords[t][0] = U;
		newMesh.texCoords[t][1] = V;
	}

	// triangulate the quad tree to the mesh
	treeRoot->generateTriangles(&newMesh);
	for (int t=0; t<newMesh.triangles.size(); ++t)
	{
		newMesh.triangles[t].t1 = newMesh.triangles[t].v1;
		newMesh.triangles[t].t2 = newMesh.triangles[t].v2;
		newMesh.triangles[t].t3 = newMesh.triangles[t].v3;
	}

	// copy the sensor origin from the cloud to the new mesh
	newMesh.sensorOrigin[0] = inputCloud.sensor_origin_[0];
	newMesh.sensorOrigin[1] = inputCloud.sensor_origin_[1];
	newMesh.sensorOrigin[2] = inputCloud.sensor_origin_[2];

	int filtered = filterTrianglesOnIncidentAngle(&newMesh, incidentAngleThreshold);
	//ROS_WARN("GQSR reconstruction complete, filtered %d triangles out due to incient angles.", filtered);

	return newMesh;
}
