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

#include <pcl_ros/transforms.h>
#include <calibration.h>

gncTK::Calibration::Calibration() : Fusion()
{
	patternsize = cv::Size(5,4);
	rms = NAN;
	newTargetPointsFound = false;
	cameraCornersImageSet = false;
}

gncTK::Calibration::Calibration(int targetSizeX, int targetSizeY) : Fusion()
{
	patternsize = cv::Size(targetSizeX, targetSizeY);
	rms = NAN;
	newTargetPointsFound = false;
	cameraCornersImageSet = false;
}

bool gncTK::Calibration::isValid(pcl::PointXYZI point)
{
	if (point.x == 0 && point.y == 0 && point.z == 0)
		return false;

	return (!std::isnan(point.x) &&
			!std::isnan(point.y) &&
			!std::isnan(point.z));
}

bool gncTK::Calibration::pixelValid(int r, int c)
{
	cv::Vec3b pixel = intensityImage.at<cv::Vec3b>(r,c);

	return (pixel[0] == pixel[1] && pixel[1] == pixel[2]);
}

void gncTK::Calibration::generateIntensityImage(pcl::PointCloud<pcl::PointXYZI> points)
{
	// find min and 98th percentile intensity values
	float min=0,max;
	bool first = true;

	// find 98th percentile of intensity values
	std::vector<float> top2Percent;
	for (int i=0; i<points.points.size(); ++i)
		top2Percent.push_back(points.points[i].intensity);
	std::sort(top2Percent.begin(), top2Percent.end());
	max = top2Percent[(int)(top2Percent.size() * 0.99)];

	//printf("98th percentile of intensity is %f.\n", max); fflush(stdout);

	// create intensity image
	intensityImage = cv::Mat(points.height, points.width, CV_8UC3, cv::Scalar(63));

	//printf("created intensity image"); fflush(stdout);

	for (int r=0; r<points.height; ++r)
		for (int c=0; c<points.width; ++c)
		{
			int index = (r*points.width) + (points.width - 1 - c);

			if (index >= points.points.size())
			{
				printf("point index out of range!.\n");
				continue;
			}

			if (isValid(points.points[index]))
			{
				float norm = (points.points[index].intensity - min) / (max-min);
				if (norm > 1) norm = 1;
				intensityImage.at<cv::Vec3b>(r,c) = cv::Vec3b(norm*255,norm*255,norm*255);
			}
			else
				intensityImage.at<cv::Vec3b>(r,c) = cv::Vec3b(0,0,255);
		}


	//printf("filled intensity image"); fflush(stdout);

	// fill small gaps in intensity image
	// this interpolates nearby points to fill the intensity image and the structured cloud itself
	for (int r=1; r<points.height-1; ++r)
		for (int c=1; c<points.width-1; ++c)
		{
			if (!pixelValid(r,c))
			{
				pcl::PointXYZI newPoint;
				int index = (r*points.width) + (points.width - 1 - c);

				if (pixelValid(r-1,c) &&
					pixelValid(r+1,c) &&
					pixelValid(r,c-1) &&
					pixelValid(r,c+1))
				{
					//printf("X");
					intensityImage.at<cv::Vec3b>(r,c) = (intensityImage.at<cv::Vec3b>(r-1,c)/4) +
														(intensityImage.at<cv::Vec3b>(r+1,c)/4) +
														(intensityImage.at<cv::Vec3b>(r,c+1)/4) +
														(intensityImage.at<cv::Vec3b>(r,c-1)/4);

					newPoint.x = (points.points[index+1].x +
								  points.points[index-1].x +
								  points.points[index+points.width].x +
								  points.points[index-points.width].x) / 4.0;
					newPoint.y = (points.points[index+1].y +
								  points.points[index-1].y +
								  points.points[index+points.width].y +
								  points.points[index-points.width].y) / 4.0;
					newPoint.z = (points.points[index+1].z +
								  points.points[index-1].z +
								  points.points[index+points.width].z +
								  points.points[index-points.width].z) / 4.0;

					newPoint.intensity = (points.points[index+1].intensity +
										  points.points[index-1].intensity +
										  points.points[index+points.width].intensity +
										  points.points[index-points.width].intensity) / 4.0;

					points.points[index] = newPoint;
				}
				else if (pixelValid(r-1,c) &&
						 pixelValid(r+1,c))
				{
					//printf("|");
					intensityImage.at<cv::Vec3b>(r,c) = (intensityImage.at<cv::Vec3b>(r-1,c)/2) +
														(intensityImage.at<cv::Vec3b>(r+1,c)/2);

					newPoint.x = (points.points[index+points.width].x +
								  points.points[index-points.width].x) / 2;
					newPoint.y = (points.points[index+points.width].y +
								  points.points[index-points.width].y) / 2;
					newPoint.z = (points.points[index+points.width].z +
								  points.points[index-points.width].z) / 2;

					newPoint.intensity = (points.points[index+points.width].intensity +
										  points.points[index-points.width].intensity) / 2;

					points.points[index] = newPoint;
				}
				else if (pixelValid(r,c-1) &&
						 pixelValid(r,c+1))
				{
					//printf("-");
					intensityImage.at<cv::Vec3b>(r,c) = (intensityImage.at<cv::Vec3b>(r,c+1)/2) +
														(intensityImage.at<cv::Vec3b>(r,c-1)/2);

					newPoint.x = (points.points[index+1].x +
								  points.points[index-1].x) / 2;
					newPoint.y = (points.points[index+1].y +
								  points.points[index-1].y) / 2;
					newPoint.z = (points.points[index+1].z +
								  points.points[index-1].z) / 2;

					newPoint.intensity = (points.points[index+1].intensity +
										  points.points[index-1].intensity) / 2;

					points.points[index] = newPoint;
				}
			}
		}
}

/// Method to find the plane of best fit to the given point cloud using Singular Value Decomposition
void gncTK::Calibration::findPlaneOfBestFit(pcl::PointCloud<pcl::PointXYZI> inputCloud,
											Eigen::Vector3f *centroid,
											Eigen::Vector3f *normal)
{

	// Limit the size of the input cloud to 65535 to stop the Eigen SVD solver from crashing!
	pcl::PointCloud<pcl::PointXYZI> cloud;
	if (inputCloud.points.size() > 65535)
	{
		for (int i=0; i<65536; ++i)
		{
			int refIndex = ((double)i/65536.0) * inputCloud.points.size();
			cloud.points.push_back(inputCloud.points[refIndex]);
		}
	}
	else
		cloud = inputCloud;

  // find the centroid of the point cloud
  *centroid << 0, 0, 0;
  for (int p=0; p<cloud.points.size(); ++p)
  {
	(*centroid)(0) += cloud.points[p].x;
	(*centroid)(1) += cloud.points[p].y;
	(*centroid)(2) += cloud.points[p].z;
  }
  *centroid /= cloud.points.size();

  // create 3xN matrix of points with centroid subtracted
  Eigen::Matrix3Xf pointsMatrix(3, (int)cloud.points.size());
  for (int p=0; p<cloud.points.size(); ++p)
  {
	pointsMatrix(0,p) = cloud.points[p].x - (*centroid)(0);
	pointsMatrix(1,p) = cloud.points[p].y - (*centroid)(1);
	pointsMatrix(2,p) = cloud.points[p].z - (*centroid)(2);
  }

  // setup and compute a jacobian estimator of the SVD solution
  Eigen::BDCSVD<Eigen::MatrixXf> jacobiSolver(cloud.points.size(), 3, Eigen::ComputeFullU | Eigen::ComputeThinV);
  Eigen::BDCSVD<Eigen::MatrixXf> result = jacobiSolver.compute(pointsMatrix, Eigen::ComputeFullU | Eigen::ComputeThinV);

  // extract the singular vector and U matrix from the result
  Eigen::VectorXf singularValues = result.singularValues();
  Eigen::MatrixXf uMatrix = result.matrixU();

  // find the column vector of the U matrix which corresponds to the lowest singular value
  // this is the normal vector of the best fit plane
  int column = uMatrix.cols() - 1;
  *normal = uMatrix.block(0,column,3,1);
}

/// Method to calculate the intersection of a plane and a line using vector algebra
Eigen::Vector3f gncTK::Calibration::linePlaneIntersect(Eigen::Vector3f lineA,
													   Eigen::Vector3f lineB,
													   Eigen::Vector3f planePoint,
													   Eigen::Vector3f planeNormal)
{
	// find vector along line
	Eigen::Vector3f l = lineB - lineA;

	// find if the line and plane are parallel
	if (l.dot(planeNormal) == 0)
		return (Eigen::Vector3f::Zero());

	// now a specific solution must exist
	float d = ((planePoint - lineA).dot(planeNormal)) / l.dot(planeNormal);
	return lineA + l*d;
}

/// Method to return true if the point Test is on the right of the line from A to B as seen from A looking at B
bool gncTK::Calibration::isOnRight(Eigen::Vector2f Test, Eigen::Vector2f A, Eigen::Vector2f B)
{
    float side = (B[0] - A[0]) * (Test[1] - A[1]) - (B[1] - A[1]) * (Test[0] - A[0]);
    return (side > 0);
}

Eigen::Vector3f gncTK::Calibration::interpolatePoint(pcl::PointCloud<pcl::PointXYZI> cloud, float x, float y)
{
	// check this point is within the structured cloud
	if (x < 0  || y < 0 ||
		x >= cloud.width || y >= cloud.height)
		return Eigen::Vector3f::Zero();

	int xI = x;
	int yI = y;
	int i = xI + (yI*cloud.width);
	float xR = x-xI;
	float yR = y-yI;

	Eigen::Vector3f TL; TL << cloud.points[i].x, cloud.points[i].y, cloud.points[i].z;
	Eigen::Vector3f TR; TR << cloud.points[i+1].x, cloud.points[i+1].y, cloud.points[i+1].z;
	Eigen::Vector3f BL; BL << cloud.points[i+cloud.width].x, cloud.points[i+cloud.width].y, cloud.points[i+cloud.width].z;
	Eigen::Vector3f BR; BR << cloud.points[i+cloud.width+1].x, cloud.points[i+cloud.width+1].y, cloud.points[i+cloud.width+1].z;

	float wTL = (1-xR)*(1-yR);
	float wTR = xR*(1-yR);
	float wBL = (1-xR)*(yR);
	float wBR = xR*(yR);

	Eigen::Vector3f total; total << 0,0,0;
	float totalW = 0;

	if (!std::isnan(TL[0]))
	{
		total += wTL*TL;
		totalW += wTL;
	}
	if (!std::isnan(TR[0]))
	{
		total += wTR*TR;
		totalW += wTR;
	}
	if (!std::isnan(BL[0]))
	{
		total += wBL*BL;
		totalW += wBL;
	}
	if (!std::isnan(BR[0]))
	{
		total += wBR*BR;
		totalW += wBR;
	}

	/*if (std::isnan(TL[0])) printf("TL nan\n");
	if (std::isnan(BL[0])) printf("BL nan\n");
	if (std::isnan(TR[0])) printf("TR nan\n");
	if (std::isnan(BR[0])) printf("BR nan\n");*/

	return total / totalW;
}

std::vector<pcl::PointXYZRGB> gncTK::Calibration::projectPointsToPlane(std::vector<cv::Point2f> lidarCorners,
																	   pcl::PointCloud<pcl::PointXYZI> cloud,
																	   Eigen::Vector3f centroid,
																	   Eigen::Vector3f normal)
{
	std::vector<pcl::PointXYZRGB> corners;

	for (int p=0; p<lidarCorners.size(); ++p)
	{
		// get an interpolated 3D position at the given
		// sub sample accurate point in the structured point cloud
		Eigen::Vector3f ray = interpolatePoint(cloud, (cloud.width - 1) - lidarCorners[p].x, lidarCorners[p].y);

		if (std::isnan(ray[0]))
		{
			printf("2D was [%f %f] then 3D was [%f %f %f]\n",
				   lidarCorners[p].x, lidarCorners[p].y,
				   ray[0], ray[1], ray[2]);
		}

		// intersect ray with best fit plane
		Eigen::Vector3f corner3D = linePlaneIntersect(Eigen::Vector3f::Zero(), ray, centroid, normal);

		float n = p / (float)lidarCorners.size();

		pcl::PointXYZRGB newPoint;
		newPoint.x = corner3D(0);
		newPoint.y = corner3D(1);
		newPoint.z = corner3D(2);
		newPoint.r = 255*(1-n);		//  <--- color the lidar points to match the
		newPoint.g = 255*n;			//       open cv function 'drawChessboardCorners'
		newPoint.b = 0;

		corners.push_back(newPoint);
	}

	return corners;
}

std::vector<pcl::PointXYZRGB> gncTK::Calibration::gridOptimisePoints(std::vector<pcl::PointXYZRGB> points)
{
	debugCloud.points.clear();

	// first find the mean point to determine the centre of the target
	Eigen::Vector3f centre;
	for (int p=0; p<points.size(); ++p)
	{
		centre(0) += points[p].x;
		centre(1) += points[p].y;
		centre(2) += points[p].z;
	}
	centre /= points.size();

	// calculate arrays of point positions for each calibration target point
	std::vector<Eigen::Vector2f> gridPoints;
	for (int y=0; y<patternsize.height; ++y)
		for (int x=0; x<patternsize.width; ++x)
		{
			Eigen::Vector2f point;
			point(0) = x - ((patternsize.width-1) / 2.0);
			point(1) = y - ((patternsize.height-1) / 2.0);
			gridPoints.push_back(point);
		}

	// create coefficient matrix and vector of constants to solve the best grid fit
	Eigen::MatrixXf coefficients(points.size() * 3, 6);
	Eigen::VectorXf constants(points.size() * 3);

	// for each point add the three linear equations it defines
	for (int p=0; p<points.size(); ++p)
	{
		int e = p*3;
		constants(e) = points[p].x - centre(0);
		coefficients(e, 0) = gridPoints[p](0);
		coefficients(e, 3) = gridPoints[p](1);

		constants(e+1) = points[p].y - centre(1);
		coefficients(e+1, 1) = gridPoints[p](0);
		coefficients(e+1, 4) = gridPoints[p](1);

		constants(e+2) = points[p].z - centre(2);
		coefficients(e+2, 2) = gridPoints[p](0);
		coefficients(e+2, 5) = gridPoints[p](1);
	}


	Eigen::FullPivLU<Eigen::MatrixXf> decomp(coefficients);
	int rank = decomp.rank();
	printf("Rank of the coefficients matrix is %d\n", rank);

	printf(" coefficients have %d rows %d cols\n constants have %d rows\n",
		   (int)coefficients.rows(), (int) coefficients.cols(),
		   (int)constants.rows());

	Eigen::VectorXf solution = coefficients.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(constants);

	// create debug cloud showing two reference x and y vectors found
	Eigen::Vector3f refX = solution.block(0,0,3,1);
	Eigen::Vector3f refY = solution.block(3,0,3,1);

	float angle = acos(refX.dot(refY) / (refX.norm() * refY.norm()));
	printf("Angle between reference vectors is %f degrees.\n", angle * (180/M_PI));

	for (int g=0; g<gridPoints.size(); ++g)
	{
		Eigen::Vector3f p = centre + (gridPoints[g](0) + 7) * refX + gridPoints[g](1) * refY;
		float n = g / (float)gridPoints.size();

		pcl::PointXYZRGB newPoint;
		newPoint.x = p(0);
		newPoint.y = p(1);
		newPoint.z = p(2);
		newPoint.r = 255*n;
		newPoint.g = 255*(1-n);
		newPoint.b = 0;

		debugCloud.push_back(newPoint);
	}

	std::cout << "grid solution was :\n" << solution << "\n-----------\n";

	return points;
}

std::vector<pcl::PointXYZRGB> gncTK::Calibration::optimiseLidarCorners(std::vector<cv::Point2f> lidarCorners,
																	   pcl::PointCloud<pcl::PointXYZI> cloud)
{
	cv::Mat intensityImageGrey;
	cvtColor(intensityImage, intensityImageGrey, CV_RGB2GRAY);
	lidarCorners = subpixelCorrectChessboardCorners(lidarCorners, intensityImageGrey);

	// isolate points within calibration pattern
	targetPoints.points.clear();

	// find sub set of point cloud that is within the calibration target
	Eigen::Vector2f a,b,c,d;
	cv::Point2f aP = lidarCorners[0];
	a << aP.x, aP.y;
	cv::Point2f bP = lidarCorners[patternsize.width - 1];
	b << bP.x, bP.y;
	cv::Point2f cP = lidarCorners[patternsize.width*patternsize.height - 1];
	c << cP.x, cP.y;
	cv::Point2f dP = lidarCorners[patternsize.width*patternsize.height - patternsize.width];
	d << dP.x, dP.y;

	// test each 'pixel' to see if it's within the abcd quadrilateral defined above
	Eigen::Vector2f p;
	for (p(0)=0; p(0)<cloud.width; ++p(0))
		for (p(1)=0; p(1)<cloud.height; ++p(1))
		{
			if (isOnRight(p, a, b) && isOnRight(p, b, c) &&
				isOnRight(p, c, d) && isOnRight(p, d, a))
			{
				pcl::PointXYZI point = cloud.points[(((int)p(1))*cloud.width)+cloud.width-1-((int)p(0))];
				if (!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z))
					targetPoints.points.push_back(point);
		    }
		}

	// find plane of best fit for these points
	Eigen::Vector3f centroid, normal;
	findPlaneOfBestFit(targetPoints, &centroid, &normal);

	// Now the 3D positions of the LIDAR target corners are found by
	// intersecting the projected rays of the 2D corners with the plane of the calibration target
	std::vector<pcl::PointXYZRGB> projectedPoints =  projectPointsToPlane(lidarCorners,
																	   cloud,
																	   centroid,
																	   normal);

	// refine the points by finding the best fit to a regular grid of points of the target size
	//projectedPoints = gridOptimisePoints(projectedPoints);

	return projectedPoints;
}

std::vector<cv::Point2f> gncTK::Calibration::subpixelCorrectChessboardCorners(std::vector<cv::Point2f> cameraCorners, cv::Mat image)
{
	std::vector <cv::Point2f> refinedCorners;

	// for each corner find the smallest connecting edge
	for (int p=0; p<cameraCorners.size(); ++p)
	{
		// get list of connected points
		std::vector<cv::Point2f> connected;
		cv::Point2i pos( p%patternsize.width,  p/patternsize.width );
		if (pos.x > 0)
			connected.push_back(cameraCorners[ pos.y*patternsize.width + pos.x - 1 ]);
		if (pos.x <= patternsize.width-1)
			connected.push_back(cameraCorners[ pos.y*patternsize.width + pos.x + 1 ]);
		if (pos.y > 0)
			connected.push_back(cameraCorners[ (pos.y-1)*patternsize.width + pos.x ]);
		if (pos.y <= patternsize.height-1)
			connected.push_back(cameraCorners[ (pos.y+1)*patternsize.width + pos.x ]);

		// find shortest connecting edge
		float shortestEdge = cv::norm(connected[0] - cameraCorners[p]);
		for (int e=1; e<connected.size(); ++e)
		{
			float edgeLength = cv::norm(connected[e] - cameraCorners[p]);
			if (edgeLength < shortestEdge)
				shortestEdge = edgeLength;
		}

		// calculate largest safe sub-pixel correction window size
		float windowSize = 2 * sqrt( (shortestEdge*shortestEdge) / 2 );
		int halfWinSize = floor( (windowSize-1) / 2 );

		// if the window size is one or less (i.e. 3x3 or less) don't bother with refinement
		if (halfWinSize <= 1)
		{
			refinedCorners.push_back(cameraCorners[p]);
			continue;
		}

		// refine point position
		std::vector<cv::Point2f> singleCorner;
		singleCorner.push_back(cameraCorners[p]);
		cv::TermCriteria criteria = cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

		cv::cornerSubPix(image, singleCorner,
						 cv::Size(halfWinSize,halfWinSize),
						 cv::Size(-1,-1),
						 criteria );

		cv::Point2f residual = singleCorner[0] - cameraCorners[p];
		refinedCorners.push_back(singleCorner[0]);
	}

	return refinedCorners;
}

double gncTK::Calibration::loadPairsFromCSV(std::string fileName, int width, int height)
{
	std::vector<std::vector<std::string> > values;
	gncTK::Utils::readCSVFile(fileName, &values);

	for (int l=0; l<values.size(); ++l)
	{
		if (values[l].size() >= 5)
		{
			Eigen::Vector2f img;
			img[0] = atof(values[l][0].c_str());
			img[1] = atof(values[l][1].c_str());
			imagePoints.push_back(img);

			Eigen::Vector3f obj;
			obj[0] = atof(values[l][2].c_str());
			obj[1] = atof(values[l][3].c_str());
			obj[2] = atof(values[l][4].c_str());
			objectPoints.push_back(obj);
		}
	}

	// attempt calibration of fusion function
	fusionFunction = gncTK::FusionFunction(width, height);
	return fusionFunction.calibrate(imagePoints, objectPoints, cv::Size(width, height));
}

void gncTK::Calibration::verticallyFlipCloud(pcl::PointCloud<pcl::PointXYZI> *cloud)
{
	std::vector<pcl::PointXYZI, Eigen::aligned_allocator< pcl::PointXYZI> > originalPoints = cloud->points;

	// swap whole lines from the top to the bottom
	for (int r=0; r<cloud->height; ++r)
	{
		for (int c=0; c<cloud->width; ++c)
		{
			int originalIndex = c + r*cloud->width;
			int newIndex = c + ((cloud->height-r-1) * cloud->width);

			cloud->points[newIndex] = originalPoints[originalIndex];
		}
	}
}

cv::Mat gncTK::Calibration::orthoprojectPointCloudToImage(pcl::PointCloud<pcl::PointXYZI> *cloud,
														  Eigen::Vector2f sizeMetres,
														  Eigen::Vector2i sizePixels)
{
	// find min and 98th percentile intensity values
	float min=0,max;
	bool first = true;

	// find 98th percentile of intensity values
	std::vector<float> top2Percent;
	for (int i=0; i<cloud->points.size(); ++i)
		top2Percent.push_back(cloud->points[i].intensity);
	std::sort(top2Percent.begin(), top2Percent.end());
	max = top2Percent[(int)(top2Percent.size() * 0.99)];

	cv::Mat oriImage(sizePixels(1), sizePixels(0), CV_8UC3, cv::Scalar(255,0,0));

	for (int p=0; p<cloud->points.size(); ++p)
	{
		// transform 3D point to 2D ORI point.
		// projection using standard optical frame convention

		Eigen::Vector2f point(cloud->points[p].x,
							  cloud->points[p].y);

		point = (point.array() / sizeMetres.array()) + Eigen::Array2f(0.5, 0.5);

		Eigen::Vector2i pixel;
		pixel(0) = point(0) * sizePixels(0);
		pixel(1) = point(1) * sizePixels(1);

		if (pixel(0) < 0 || pixel(0) >= sizePixels(0) ||
			pixel(1) < 0 || pixel(1) >= sizePixels(1))
			continue;

		// get normalized intensity value
		float norm = (cloud->points[p].intensity - min) / (max-min);
		if (norm > 1)
			norm = 1;

		int x = pixel(0);
		int y = pixel(1);
		oriImage.at<cv::Vec3b>(y,x) = cv::Vec3b(norm*255, norm*255, norm*255);
	}

	return oriImage;
}

bool gncTK::Calibration::addFusionSample(std::vector<cv::Mat> images,
										 std::vector<tf::Transform> imageTFs,
										 pcl::PointCloud<pcl::PointXYZI> cloud)
{
	cameraImage = images[0];
	newTargetPointsFound = false;

	inputImages = images;
	inputImageTFs = imageTFs;

	if (debug)
	{
		ROS_INFO("Starting calibration capture of scan with %d images and %d image transforms.",
				 (int)images.size(),
				 (int)imageTFs.size());
		ROS_INFO("generating lidar intensity image.");
	}

	// flip the structured point cloud top to bottom if it was scanned from bottom to top
	if (true) //(scannedBottomUp)
	{
		verticallyFlipCloud(&cloud);
	}

	generateIntensityImage(cloud);

	imwrite("intensity.png", intensityImage);

	if (debug)
		ROS_INFO("Done. looking for target corners in lidar image.");

	// set the debugging image to the default (not found version)
	/*cameraCornersImage = images[0].clone();
	cv::line(cameraCornersImage,
			 cv::Point(0,0),
			 cv::Point(images[0].cols, images[0].rows),
			 cv::Scalar(0,0,255),
			 images[0].cols/50.0);
	cv::line(cameraCornersImage,
			 cv::Point(0, images[0].rows),
			 cv::Point(images[0].cols, 0),
			 cv::Scalar(0,0,255),
			 images[0].cols/50.0);*/

	// look for chess board in LIDAR image
	std::vector<cv::Point2f> lidarCorners;
	bool lidarPatternFound = cv::findChessboardCorners(intensityImage,
													   patternsize,
													   lidarCorners,
													   cv::CALIB_CB_ADAPTIVE_THRESH);// + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

	if (!lidarPatternFound)
	{
		if (debug)
			ROS_INFO("Pattern not found in lidar data. Skipping this scan.");

		return false;
	}

	//lidarCorners[0].x += 30;

	drawChessboardCorners(intensityImage, patternsize, lidarCorners, true);

	imwrite("intensity_with_pattern.png", intensityImage);

	// Bodge for the new upsidown sensor, reverse the order  of points in the found lidar corners
	// This is because the old system scanned top to bottom in the same raster pattern as a computer
	// screen, whereas the new one scans up. This means the chequer boards are mirrored resulting in
	// their point numbering order being reversed so no good matches are
	/*std::vector<cv::Point2f> reversedLidarCorners;
	for (int c=lidarCorners.size()-1; c>=0; --c)
		reversedLidarCorners.push_back(lidarCorners[c]);
	lidarCorners = reversedLidarCorners;*/

	// refine found lidar corners
	std::vector<pcl::PointXYZRGB> lidarCorners3D = optimiseLidarCorners(lidarCorners, cloud);

	if (debug)
		ROS_INFO("Pattern found in lidar data okay. Looking for target corners in camera images.");

	// look for chess board in camera images
	foundTargetPoints.points.clear();
	foundCameraCorners.clear();
	if (!cameraCornersImageSet)
	{
		cameraCornersImage = cv::Mat(images[0].size(), CV_8UC3, cv::Scalar(127,127,127));
		cameraCornersImageSet = true;
	}

	int chosenInputImage=0;
	if (images.size() > 0)
		chosenInputImage = inputImages.size() / 2;

	//for (int c=0; c<images.size(); ++c)
	for (int c=chosenInputImage; c<=chosenInputImage; ++c)
	{
		cv::Mat cameraImageGrey;
		cvtColor(images[c], cameraImageGrey, CV_RGB2GRAY);
		bool cameraPatternFound = cv::findChessboardCorners(cameraImageGrey,
															patternsize,
															cameraCorners,
															cv::CALIB_CB_ADAPTIVE_THRESH);// + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
		if (!cameraPatternFound)
		{
			if (debug)
				ROS_WARN(" pattern not found in image %d.", c);
			continue;
		}

		//cameraCorners[0].x += 30;

		//cameraCorners.pop_back();

		if (debug)
			printf("detected %d corners in camera image %d and LIDAR intensity image.\n", (int)cameraCorners.size(), c);

		// refine found camera corners and generate debugging image
		cameraCorners = subpixelCorrectChessboardCorners(cameraCorners, cameraImageGrey);
		drawChessboardCorners(cameraCornersImage, patternsize, cameraCorners, true);

		// Transform lidar corners into image frame and add to debugging point cloud
		tf::Transform tiltToCamera = imageTFs[c];

		std::vector<cv::Point2f> transformedLidarCorners;
		Eigen::Vector2f orthoLidarSize(3,3);
		Eigen::Vector2i orthoLidarPixels(400,400);

		for (int p=0; p<lidarCorners3D.size(); ++p)
		{
			tf::Vector3 point(lidarCorners3D[p].x,
						  lidarCorners3D[p].y,
						  lidarCorners3D[p].z);
			point = tiltToCamera * point;

			pcl::PointXYZRGB transformedPoint = lidarCorners3D[p];
			transformedPoint.x = point[0];
			transformedPoint.y = point[1];
			transformedPoint.z = point[2];

			foundTargetPoints.points.push_back(transformedPoint);

			Eigen::Vector2f lidarPoint(transformedPoint.x,
								  	   transformedPoint.y);

			lidarPoint = (lidarPoint.array() / orthoLidarSize.array()) + Eigen::Array2f(0.5, 0.5);

			cv::Point2f pixel(lidarPoint(0) * orthoLidarPixels(0),
					lidarPoint(1) * orthoLidarPixels(1));
			transformedLidarCorners.push_back(pixel);
		}

		// add the detected camera corners to the list of points found from this scan
		foundCameraCorners.insert(foundCameraCorners.begin(),
								  cameraCorners.begin(),
								  cameraCorners.end());


		pcl::PointCloud<pcl::PointXYZI> transformedCloud;
		pcl_ros::transformPointCloud(cloud, transformedCloud, tiltToCamera);

		cv::Mat debugCameraImage = images[c].clone();
		drawChessboardCorners(debugCameraImage, patternsize, cameraCorners, true);

		cv::Mat debugLidarImage = orthoprojectPointCloudToImage(&transformedCloud, orthoLidarSize, orthoLidarPixels);
		drawChessboardCorners(debugLidarImage, patternsize, transformedLidarCorners, true);

		std::string debugCameraFilename = "debug_camera_" + std::to_string(c) + ".png";
		std::string debugLidarFilename = "debug_lidar_" + std::to_string(c) + ".png";
		imwrite(debugCameraFilename, debugCameraImage);
		imwrite(debugLidarFilename, debugLidarImage);
	}

	// reshape the found lidar and camera points so the orientation and order is apparent
	/*int patternLength = patternsize.width * patternsize.height;
	std::vector<int> indicesToRemove;

	for (int w = 1; w<patternsize.width; ++w)
		for (int h=1; h<patternsize.height; ++h)
		{
			indicesToRemove.push_back( w+(h*patternsize.width) );
		}
	for (int w = patternsize.width/2; w<patternsize.width; ++w)
		for (int h=patternsize.height/2; h<patternsize.height; ++h)
			indicesToRemove.push_back( (patternLength * (images.size()-1)) + w + (h*patternsize.width) );

	for (int r=indicesToRemove.size()-1; r>=0; --r)
	{
		foundCameraCorners.erase(foundCameraCorners.begin() + indicesToRemove[r]);
		foundTargetPoints.points.erase(foundTargetPoints.points.begin() + indicesToRemove[r]);
	}*/

	// update the frame_id of the debugging point cloud
	foundTargetPoints.header.frame_id = cloud.header.frame_id;

	newTargetPointsFound = true;
	return true;
}

void gncTK::Calibration::commitFoundPoints()
{
	ROS_INFO("Commiting new target & camera points");
	ROS_INFO("lidar point count %d, camera point count %d",
			 (int)totalTargetPoints.size(),
			 (int)imagePoints.size());
	ROS_INFO("Commiting %d lidar points and %d camera points.",
			 (int)foundTargetPoints.size(),
			 (int)foundCameraCorners.size());

	if (newTargetPointsFound)
	{
		// add image points and object points to cumulative vectors
		for (int p=0; p<foundTargetPoints.size(); ++p)
		{
			// add found image points to total
			Eigen::Vector2f newImagePoint; newImagePoint << foundCameraCorners[p].x, foundCameraCorners[p].y;
			imagePoints.push_back(newImagePoint);

			// add found object points to total
			Eigen::Vector3f newObjectPoint; newObjectPoint << foundTargetPoints.points[p].x,
															  foundTargetPoints.points[p].y,
															  foundTargetPoints.points[p].z;
			objectPoints.push_back(newObjectPoint);

			// add found object points to total points debug cloud
			totalTargetPoints.header.frame_id = foundTargetPoints.header.frame_id;
			totalTargetPoints.points.push_back(foundTargetPoints[p]);
			int end = totalTargetPoints.points.size()-1;
			//totalTargetPoints.points[end].r = 255;
			//totalTargetPoints.points[end].g = 255;
			//totalTargetPoints.points[end].b = 255;
		}
	}

	ROS_INFO("New lidar point count %d, new camera point count %d",
			 (int)totalTargetPoints.size(),
			 (int)imagePoints.size());
}

double gncTK::Calibration::calibrate(bool useInitialCalibration, bool extrinsicOnly)
{
	// reset fusion function if needed
	if (!useInitialCalibration)
		fusionFunction = gncTK::FusionFunction(cameraImage.size().width, cameraImage.size().height);

	if (extrinsicOnly)
	{
		rms = fusionFunction.optimseExtrinsic(imagePoints, objectPoints);
	}
	else
	{
		// attempt calibration of fusion function
		rms = fusionFunction.calibrate(imagePoints, objectPoints, cameraImage.size());
		std::cout << "calibration RMS (pixels) was " << rms << "\n";
	}

	return rms;
}
