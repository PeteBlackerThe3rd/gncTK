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

utils.cpp

general helper functions and data structures
---------------------------------------------------



-------------------------------------------------------------*/
#include <string>
#include <pcl_ros/point_cloud.h>
#include <Eigen/Dense>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/SVD>

#include <utils.h>

/// Method to aid debugging ROS
std::string gncTK::Utils::toString(tf::Transform transform)
{
	std::string result;
	char buf[256];

	sprintf(buf, "Transform [%f %f %f]\n",
			transform.getOrigin().getX(),
			transform.getOrigin().getY(),
			transform.getOrigin().getZ());
	result = std::string(buf);

	sprintf(buf, "Rotation Quaternion [%f %f %f %f] (x y z w)\n",
			transform.getRotation().getAxis().getX(),
			transform.getRotation().getAxis().getY(),
			transform.getRotation().getAxis().getZ(),
			transform.getRotation().getW());
	result += std::string(buf);

	tf::Matrix3x3 dcm(transform.getRotation());
	tfScalar r,p,y;
	dcm.getRPY(r,p,y);
	sprintf(buf, "Rotation RPY [%f %f %f]\n",
			r, p, y);
	result += std::string(buf);

	return result;
}


/// Method to produce a point cloud with points describing a box of the given size.
/*
 * X axis lines are red, y axis lines are green and z axis lines are blue.
 * frameId is a string specifying the coordinate frame label
 * The point spacing defaults to 0.1 but can be set to anything you want.
 */
pcl::PointCloud<pcl::PointXYZRGB> gncTK::Utils::pointCloudBox(float sX, float sY, float sZ, std::string frameId, float spacing)
{
	pcl::PointCloud<pcl::PointXYZRGB> cloud;
	cloud.header.frame_id = frameId;

	for (float x=0; x<sX; x+=spacing)
	{
		pcl::PointXYZRGB point;
		point.r = 255;
		point.g = 0;
		point.b = 0;

		point.x = x;
		point.y = 0;
		point.z = 0;

		cloud.points.push_back(point);
		point.y = sY;
		cloud.points.push_back(point);
		point.z = sZ;
		cloud.points.push_back(point);
		point.y = 0;
		cloud.points.push_back(point);
	}

	for (float y=0; y<sY; y+=spacing)
	{
		pcl::PointXYZRGB point;
		point.r = 0;
		point.g = 255;
		point.b = 0;

		point.x = 0;
		point.y = y;
		point.z = 0;

		cloud.points.push_back(point);
		point.x = sX;
		cloud.points.push_back(point);
		point.z = sZ;
		cloud.points.push_back(point);
		point.x = 0;
		cloud.points.push_back(point);
	}

	for (float z=0; z<sZ; z+=spacing)
	{
		pcl::PointXYZRGB point;
		point.r = 0;
		point.g = 0;
		point.b = 255;

		point.x = 0;
		point.y = 0;
		point.z = z;

		cloud.points.push_back(point);
		point.x = sX;
		cloud.points.push_back(point);
		point.y = sY;
		cloud.points.push_back(point);
		point.x = 0;
		cloud.points.push_back(point);
	}

	return cloud;
}

/// Method to normalize a DH matrix
/*
 * The magnitudes of the three axis vectors are normalized to 1
 * The angles between the axis vectors are normalised using cross producs
 * The constant values are updated
 */
Eigen::Matrix4d gncTK::Utils::normalizeDHMatrix(Eigen::Matrix4d matrix)
{
	Eigen::Vector3d axisX, axisY, axisZ;
	axisX = matrix.block(0,0,3,1);
	axisY = matrix.block(0,1,3,1);
	axisZ = matrix.block(0,2,3,1);

	axisX = axisX / axisX.squaredNorm();
	axisY = axisY / axisY.squaredNorm();
	axisZ = axisZ / axisZ.squaredNorm();

	axisY = axisZ.cross(axisX);
	axisZ = axisX.cross(axisY);

	axisX = axisX / axisX.squaredNorm();
	axisY = axisY / axisY.squaredNorm();
	axisZ = axisZ / axisZ.squaredNorm();

	matrix.block(0,0,3,1) = axisX;
	matrix.block(0,1,3,1) = axisY;
	matrix.block(0,2,3,1) = axisZ;

	matrix(0,3) = 0.0;
	matrix(1,3) = 0.0;
	matrix(2,3) = 0.0;
	matrix(3,3) = 1.0;

	return matrix;
}

/// This method prints out an analysis of the DH matrix to stdout
/*
 * The magnitude of the three axis vectors of the rotation are checked, as well as
 * the angles between them.
 * The rotation matrix is checked to make sure it's not improper (a reflection)
 * The translation component is reported and the zero and one elements checked.
 */
void gncTK::Utils::analyseDHMatrix(Eigen::Matrix4d matrix)
{
	std::cout << "Matrix is:\n-------------------------------------------\n" << matrix << "\n-------------------------------------------\n";

	Eigen::Vector3d axisX, axisY, axisZ, translation;
	Eigen::Vector4d constants;

	axisX = matrix.block(0,0,3,1);
	axisY = matrix.block(0,1,3,1);
	axisZ = matrix.block(0,2,3,1);
	translation = matrix.block(3,0,1,3).transpose();
	constants = matrix.block(0,3,4,1);
	double det = matrix.block(0,0,3,3).determinant();

	printf("X axis magnitude : %f\n", axisX.norm());
	printf("Y axis magnitude : %f\n", axisY.norm());
	printf("Z axis magnitude : %f\n", axisZ.norm());

	printf("X-Y angle : %f degrees\n", acos(axisX.dot(axisY) / (axisX.norm()*axisY.norm())) * (180/M_PI));
	printf("Y-Z angle : %f degrees\n", acos(axisY.dot(axisZ) / (axisY.norm()*axisZ.norm())) * (180/M_PI));
	printf("Z-X angle : %f degrees\n", acos(axisZ.dot(axisX) / (axisZ.norm()*axisX.norm())) * (180/M_PI));

	printf("Translational components are [%f, %f, %f]\n",
		   translation(0),
		   translation(1),
		   translation(2));

	if (constants(0) == 0 && constants(1) == 0 && constants(2) == 0 && constants(3) == 1)
		printf("Constants OK.\n");
	else
		printf("constants [%f, %f, %f, %f] are invalid!\n",
			   constants(0),
			   constants(1),
			   constants(2),
			   constants(3));

	printf("determinant of rotation is [%f]\n", det);
	if (det >= 0.999)
		printf("This is a proper rotation matrix.\n");
	else if (det <= -0.999)
		printf("This is an improper rotation matrix (mirror).\n");
	else
	{
		Eigen::FullPivLU<Eigen::Matrix3d> decomp(matrix.block(0,0,3,3));
		int rank = decomp.rank();
		printf("This is not a rotation matrix and its rank is %d\n", rank);
	}
	printf("-------------------------------------------\n");
}

/// Methods for working with spherical triangles
float gncTK::Utils::triangleSolidAngle(Eigen::Vector3f a,
									   Eigen::Vector3f b,
									   Eigen::Vector3f c, bool debug)
{
	// calculate the three normals
	Eigen::Vector3f normAB = (a.cross(b-a)).normalized();
	Eigen::Vector3f normBC = (b.cross(c-b)).normalized();
	Eigen::Vector3f normCA = (c.cross(a-c)).normalized();

	// calculate the thee dot products
	float dotA = normAB.dot(normCA);
	float dotB = normBC.dot(normAB);
	float dotC = normCA.dot(normBC);

	// correct for out of range (-1 to 2) values caused by floating point rounding errors on small triangles
	if (dotA < -1) dotA = -1;
	if (dotA > 1 ) dotA = 1 ;
	if (dotB < -1) dotB = -1;
	if (dotB > 1 ) dotB = 1 ;
	if (dotC < -1) dotC = -1;
	if (dotC > 1 ) dotC = 1 ;

	// calculate the three angles
	float angleA = M_PI - acos(dotA);
	float angleB = M_PI - acos(dotB);
	float angleC = M_PI - acos(dotC);

	if (debug)
	{
		printf(" a = %f %f %f\n", a[0],a[1],a[2]);
		printf(" b = %f %f %f\n", b[0],b[1],b[2]);
		printf(" c = %f %f %f\n", c[0],c[1],c[2]);

		printf("A dot (%f), B dot (%f), C dot (%f)\n", dotA, dotB, dotC);

		printf("internal angles are %f, %f, %f SA is %f\n", angleA, angleB, angleC,
				angleA+angleB+angleC-M_PI);
	}

	// because the angle between two vectors can never be greater than pi (180 degrees)
	// determine which side of the triangle the origin is in order to detect triangles with
	// angles greater than 180 degree. The winding order decides if they're 'inside' or 'outside'
	// spherical triangles.

	// TO DO! currently assumes all triangles are 'inside'

	float solidAngle = angleA+angleB+angleC - M_PI;

	return solidAngle;
}

/// Method to calculate the area of a triangle from three side lengths using Herons formula
float gncTK::Utils::triangleAreaFromSides(float a, float b, float c)
{
	float p = (a+b+c) / 2.0;

	return sqrt( p * (p-a) * (p-b) * (p-c) );
}

/// Templated method to calculate the internal angle of the vectors A->B and B->C
template <class T>
float gncTK::Utils::angleAtB(T a, T b, T c)
{
	// find magnitudes of vectors
	float magAB = (a-b).norm();
	float magBC = (c-b).norm();

	float dotProd = (a-b).dot(c-b);

	return acos( dotProd / (magAB*magBC) );
}
// explicit instantiations of this templated method so that they are available in the shared object of the gnc_tool_kit
template float gncTK::Utils::angleAtB(Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f);
template float gncTK::Utils::angleAtB(Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d);
template float gncTK::Utils::angleAtB(Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f);
template float gncTK::Utils::angleAtB(Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d);

/// Method to convert a 3x3 rotation matrix to a rotation quaternion
/*
 * The method used is described here:
 * http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 * Currently the input rotationm matrix isn't checked to ensure it's a proper
 * rotation matrix.
 */
Eigen::Vector4f gncTK::Utils::quaternionFromMatrix(Eigen::Matrix3f r)
{
	float tr = r.trace();//m00 + m11 + m22
	Eigen::Vector4f q;

	if (tr > 0)
	{
	  float S = sqrt(tr+1.0) * 2; // S=4*qw
	  q(0) = (r(2,1) - r(1,2)) / S;
	  q(1) = (r(0,2) - r(2,0)) / S;
	  q(2) = (r(1,0) - r(0,1)) / S;
	  q(3) = 0.25 * S;
	}
	else if ((r(0,0) > r(1,1))&(r(0,0) > r(2,2)))
	{
	  float S = sqrt(1.0 + r(0,0) - r(1,1) - r(2,2)) * 2; // S=4*qx
	  q(0) = 0.25 * S;
	  q(1) = (r(0,1) + r(1,0)) / S;
	  q(2) = (r(0,2) + r(2,0)) / S;
	  q(3) = (r(2,1) - r(1,2)) / S;
	}
	else if (r(1,1) > r(2,2))
	{
	  float S = sqrt(1.0 + r(1,1) - r(0,0) - r(2,2)) * 2; // S=4*qy
	  q(0) = (r(0,1) + r(1,0)) / S;
	  q(1) = 0.25 * S;
	  q(2) = (r(1,2) + r(2,1)) / S;
	  q(3) = (r(0,2) - r(2,0)) / S;
	}
	else
	{
	  float S = sqrt(1.0 + r(2,2) - r(0,0) - r(1,1)) * 2; // S=4*qz
	  q(0) = (r(0,2) + r(2,0)) / S;
	  q(1) = (r(1,2) + r(2,1)) / S;
	  q(2) = 0.25 * S;
	  q(3) = (r(1,0) - r(0,1)) / S;
	}

	return q.normalized();
}

/// Method to find the average of a set of quaternions using the SVD approach
/*
 * The algorithm used in this algorithm is described here:
 *
 */
Eigen::Vector4f gncTK::Utils::quaternionAverage(std::vector<Eigen::Vector4f> quaternions)
{
	if (quaternions.size() == 0)
	{
		ROS_ERROR("Error trying to find the average quaternion from an empty set!");
		return Eigen::Vector4f::Zero();
	}

	// first build a 4x4 matrix which is the elementwise sum of the product of each quaternion with itself
	Eigen::Matrix4f A = Eigen::Matrix4f::Zero();

	for (int q=0; q<quaternions.size(); ++q)
	{
		A += quaternions[q] * quaternions[q].transpose();
	}

	// normalise with the number of quaternions
	A /= quaternions.size();

	// Compute the SVD of this 4x4 matrix
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

	Eigen::VectorXf singularValues = svd.singularValues();
	Eigen::MatrixXf U = svd.matrixU();

	// find the eigen vector corresponding to the largest eigen value
	int largestEigenValueIndex;
	float largestEigenValue;
	bool first = true;

	for (int i=0; i<singularValues.rows(); ++i)
	{
		if (first)
		{
			largestEigenValue = singularValues(i);
			largestEigenValueIndex = i;
			first = false;
		}
		else if (singularValues(i) > largestEigenValue)
		{
			largestEigenValue = singularValues(i);
			largestEigenValueIndex = i;
		}
	}

	Eigen::Vector4f average;
	average(0) = U(0, largestEigenValueIndex);
	average(1) = U(1, largestEigenValueIndex);
	average(2) = U(2, largestEigenValueIndex);
	average(3) = U(3, largestEigenValueIndex);

	return average;
}

/// Method to read a CSV file into a grid of strings
int gncTK::Utils::readCSVFile(FILE *file,
									 std::vector<std::vector<std::string> > *content,
									 char separator,
									 int skip)
{
  char sepStr[] = ",";
  sepStr[0] = separator;

  // clear output vector
  content->empty();
  int lineCount = 0;

  // read each line of the CSV file
  char *line;
  size_t len = 0;

  while(getline(&line, &len, file) != -1)
  {
	if (skip-- > 0) { continue; printf("Skipped a line.\n"); }

	// new vector for tokens on this line
	std::vector<std::string> tokens;
	char *token;

	while (token = strsep(&line, sepStr))
	{
	  if (token != NULL && strcmp(token, "\n") != 0)
	  {
		// if the last character of this string is a carriage return, then trim it
		if (strlen(token) > 0 && token[strlen(token)-1] == '\n')
		  token[strlen(token)-1] = '\0';

		if (strlen(token) > 0)
		  tokens.push_back(std::string(token));
	  }
	}

	// add the new line of tokens to the output
	content->push_back(tokens);
	++lineCount;
  }

  return lineCount;
}

int gncTK::Utils::readCSVFile(std::string fileName,
									 std::vector<std::vector<std::string> > *content,
									 char separator,
									 int skip)
{
  FILE *file = fopen(fileName.c_str(), "r");
  if (ferror(file) != 0)
	  return -1;
  int lineCount = gncTK::Utils::readCSVFile(file, content, separator, skip);
  fclose(file);
  return lineCount;
}

/// Method to write a CSV file
void gncTK::Utils::writeCSVFile(FILE *file,
									   std::vector<std::vector<std::string> > *content)
{
  for (int l=0; l<content->size(); ++l)
  {
	for (int e=0; e<content->at(l).size(); ++e)
	{
	  if (e != 0)
		  fprintf(file, ", ");

	  fprintf(file, "%s", content->at(l).at(e).c_str());
	}

	fprintf(file, "\n");
  }
}

void gncTK::Utils::writeCSVFile(std::string fileName,
									   std::vector<std::vector<std::string> > *content)
{
  FILE *file = fopen(fileName.c_str(), "w");
  gncTK::Utils::writeCSVFile(file, content);
  fclose(file);
}

/// Method to write a CSV file
void gncTK::Utils::writeCSVFile(FILE *file,
								std::vector<std::vector<float> > content,
								std::vector<std::string> header)
{
	if (header.size() > 0)
	{
		for (int e=0; e<header.size(); ++e)
		{
		  if (e != 0)
			fprintf(file, ", ");

		  fprintf(file, "%s", header.at(e).c_str());
		}

		fprintf(file, "\n");
	}

  for (int l=0; l<content.size(); ++l)
  {
	for (int e=0; e<content.at(l).size(); ++e)
	{
	  if (e != 0)
		fprintf(file, ", ");

	  if (content.at(l).at(e) < 1.0)
		  fprintf(file, "%e", content.at(l).at(e));
	  else
		  fprintf(file, "%f", content.at(l).at(e));
	}

	fprintf(file, "\n");
  }
}

void gncTK::Utils::writeCSVFile(std::string fileName,
							    std::vector<std::vector<float> > content,
								 std::vector<std::string> header)
{
  FILE *file = fopen(fileName.c_str(), "w");
  gncTK::Utils::writeCSVFile(file, content, header);
  fclose(file);
}

/// method to save floating point cv images as a CSV file
void gncTK::Utils::writeCSVImage(std::string fileName,
						  	     cv::Mat image)
{
    // create the file stream
    FILE *file = fopen(fileName.c_str(), "w");
	if (ferror(file))
	{
		printf("Error: Failed to open file \"%s\" for writing while saving floating point cv image.\n",
			   fileName.c_str());
		return;
	}

	int width = image.cols;
	int height = image.rows;
	int type = image.type();

	// get the pixel size of this image type and check it's supported
	int pixelBytes=0;
	switch(type)
	{
		case CV_8U :	pixelBytes = 1;		break;
		case CV_32F: 	pixelBytes = 4;	 	break;
		case CV_64F: 	pixelBytes = 8;	 	break;
		default:
			printf("Error: pixel type '%d' no supported by gncTK::Utils::writeCSVImage function.", type);
			fclose(file);
			return;
	}

	for (int r=0; r<height; ++r)
	{
		if (type == CV_8U)
		{
			for(int c=0; c<width; ++c)
			{
				if (c!=0)
					fprintf(file, ",");

				fprintf(file, "%d", image.at<unsigned char>(r,c));
			}
		}
		else if (type == CV_32F)
		{
			for(int c=0; c<width; ++c)
			{
				if (c!=0)
					fprintf(file, ",");

				fprintf(file, "%f", image.at<float>(r,c));
			}
		}
		else if (type == CV_64F)
		{
			for(int c=0; c<width; ++c)
			{
				if (c!=0)
					fprintf(file, ",");

				fprintf(file, "%f", image.at<double>(r,c));
			}
		}

		fprintf(file, "\n");
	}

    //close file
    fclose(file);
}

// -------------- Methods to process file name strings -------------------------------------------

std::string gncTK::Utils::fileNameWOPath(std::string fileName)
{
	int lastSlash = fileName.rfind("/");
	return fileName.substr(lastSlash+1);
}


std::string gncTK::Utils::fileNameWOExtension(std::string fileName)
{
	int lastDot = fileName.rfind(".");
	return fileName.substr(0, lastDot);
}

// -------------- Methods to process cv Images ---------------------------------------------------

cv::Vec3b gncTK::Utils::rainbow(float color)
{
	static bool generated = false;
	static std::vector<cv::Vec3b> colorBar;
	cv::Vec3b nanColor;
	nanColor[0] = nanColor[1] = nanColor[2] = 0;

	if (!generated)
	{
		for (int a=0; a<256; ++a)
		{
			cv::Vec3b color;

			float pos = (a/256.0) * 0.8 * 6;

			if (pos <= 1)
			{
				color[0] = 255;
				color[1] = pos * 255;
				color[2] = 0;
			}
			else if (pos <= 2)
			{
				color[0] = (2-pos) * 255;
				color[1] = 255;
				color[2] = 0;
			}
			else if (pos <= 3)
			{
				color[0] = 0;
				color[1] = 255;
				color[2] = (pos-2) * 255;
			}
			else if (pos <= 4)
			{
				color[0] = 0;
				color[1] = (4-pos) * 255;
				color[2] = 255;
			}
			else if (pos <= 5)
			{
				color[0] = (pos-4) * 255;
				color[1] = 0;
				color[2] = 255;
			}
			else
			{
				color[0] = 255;
				color[1] = 0;
				color[2] = (6-pos) * 255;
			}
			colorBar.push_back(color);
		}

		generated = true;
	}

	if (!std::isfinite(color))
		return nanColor;

	if (color < 0) color = 0;
	if (color > 1) color = 1;

	return colorBar[(int)((1-color)*255)];
}

/// Method to convert a single channel floating point image to an RGB rainbow representation
cv::Mat gncTK::Utils::rainbow(cv::Mat input)
{
	// test for the edge case where all the pixels are the same value
	// must be caught here otherwise normalization produces all NAN values!
	bool uniform = true;
	if (input.type() == CV_32F)
	{
		float value = input.at<float>(0);
		for (int i=0; i<input.rows*input.cols; ++i)
			uniform  = uniform && (value == input.at<float>(i));
	}
	else if (input.type() == CV_64F)
	{
		double value = input.at<double>(0);
		for (int i=0; i<input.rows*input.cols; ++i)
			uniform  = uniform && (value == input.at<double>(i));
	}

	if (uniform)
	{
		cv::Mat output(input.size(), CV_8UC3, gncTK::Utils::rainbow(0.0));
		return output;
	}

	// clone image (to avoid modifying the original) and normalise it
	cv::Mat imgCopy = input.clone();
	normalize(imgCopy, imgCopy, 1.0, 0.0, cv::NORM_MINMAX);
	cv::Mat output(imgCopy.size(), CV_8UC3);

	if (imgCopy.type() == CV_32F)
		for (int i=0; i<imgCopy.rows*imgCopy.cols; ++i)
			output.at<cv::Vec3b>(i) = gncTK::Utils::rainbow(imgCopy.at<float>(i));

	if (imgCopy.type() == CV_64F)
		for (int i=0; i<imgCopy.rows*imgCopy.cols; ++i)
			output.at<cv::Vec3b>(i) = gncTK::Utils::rainbow(imgCopy.at<double>(i));

	return output;
}

/// Method to convert a single channel of a floating point CV image as an 8 bit image with a scale bar
cv::Mat gncTK::Utils::floatImageTo8Bit(cv::Mat floatImage, int channel, bool useRainbow)
{
	int border = 10;
	int scaleWidth = 50;
	int scaleHeight = floatImage.rows - border*2;
	cv::Vec3b NANColor;
	if (useRainbow)
		NANColor[0] = NANColor[1] = NANColor[2] = 0;
	else
	{
		NANColor[0] = 255;
		NANColor[1] = NANColor[2] = 0;
	}

	// validate channel number
	if (floatImage.channels() != 1 && (channel < 0 || channel >= floatImage.channels()))
	{
		printf("Error trying to generate 8 bit image from float image, channel %d doesn't exist in image.\n", channel);
		return cv::Mat(10,10, CV_8UC3);
	}

	// find the minimum and maximum values
	float minValue = NAN, maxValue = NAN;
	bool first = true;
	bool NANsPresent = false;

	for (int r=0; r<floatImage.rows; ++r)
		for (int c=0; c<floatImage.cols; ++c)
		{
			float value;
			if (floatImage.channels() == 1)
				value = floatImage.at<float>(r,c);
			else
				value = floatImage.at<cv::Vec4f>(r,c)[channel];

			if (std::isnan(value))
				NANsPresent = true;
			else
			{
				if (first)
				{
					minValue = maxValue = value;
					first = false;
				}
				else
				{
					if (value < minValue) minValue = value;
					if (value > maxValue) maxValue = value;
				}
			}
		}

	// find the size of the text labels
	char maxLabel[255], minLabel[255];
	sprintf(maxLabel, " %5f", maxValue);
	sprintf(minLabel, " %5f", minValue);

	int textBaseline=0;
	double textScale = 1.3;
	int textThickness = 2;
	cv::Size maxSize = cv::getTextSize(std::string(maxLabel),
								   cv::FONT_HERSHEY_COMPLEX,
								   textScale, textThickness, &textBaseline);
	cv::Size minSize = cv::getTextSize(std::string(minLabel),
								   cv::FONT_HERSHEY_COMPLEX,
								   textScale, textThickness, &textBaseline);

	if (NANsPresent)
		scaleHeight -= border + minSize.height*2;

	// create output image
	cv::Mat output(floatImage.rows + border*2,
				   floatImage.cols + border*3 + scaleWidth + std::max(maxSize.width,minSize.width),
				   CV_8UC3,
				   cv::Scalar(255,255,255));

	// add float image
	for (int r=0; r<floatImage.rows; ++r)
		for (int c=0; c<floatImage.cols; ++c)
		{
			float value;
			if (floatImage.channels() == 1)
				value = floatImage.at<float>(r,c);
			else
				value = floatImage.at<cv::Vec4f>(r,c)[channel];

			if (std::isnan(value))
			{
				output.at<cv::Vec3b>(r+border, c+border) = NANColor;
			}
			else
			{
				if (maxValue != minValue)
					value = (value - minValue) / (maxValue-minValue);
				else
					value = 0.5;

				cv::Vec3b color;
				if (useRainbow)
					color = rainbow(value);
				else
					color[2] = color[1] = color[0] = value * 255;

				output.at<cv::Vec3b>(r+border, c+border) = color;
			}
		}
	cv::rectangle(output,
				  cv::Point(border-2,border-2),
				  cv::Point(floatImage.cols + border+1, floatImage.rows + border+1),
				  cv::Scalar(0,0,0),
				  2);

	// add color/grey scale
	for (int r = border*2; r < scaleHeight+(border*2); ++r)
	{
		cv::Vec3b color;
		float value = (r - border*2) / (float)scaleHeight;
		value = 1 - value;
		if (useRainbow)
			color = rainbow(value);
		else
			color[2] = color[1] = color[0] = value * 255;

		for (int c = floatImage.cols + border*2; c < floatImage.cols + border*2 + scaleWidth; ++c)
			output.at<cv::Vec3b>(r,c) = color;
	}
	cv::rectangle(output,
			  	  cv::Point(floatImage.cols+border*2-2,border*2-2),
				  cv::Point(floatImage.cols + scaleWidth + border*2+1, border*2 + scaleHeight + 1),
				  cv::Scalar(0,0,0),
				  2);

	// add NAN color label if needed
	if (NANsPresent)
	{
		cv::rectangle(output,
				  	  cv::Point(floatImage.cols+border*2-1,border*4 + scaleHeight-1),
					  cv::Point(floatImage.cols + scaleWidth + border*2, output.rows - border*2),
					  NANColor,
					  -1);

		cv::rectangle(output,
				  	  cv::Point(floatImage.cols+border*2-2,border*4 + scaleHeight-2),
					  cv::Point(floatImage.cols + scaleWidth + border*2+1, output.rows - border*2 + 1),
					  cv::Scalar(0,0,0),
					  1);

		cv::putText(output,
					" NAN",
					cv::Point(floatImage.cols + border*2 + scaleWidth,
							  output.rows - border*2),
					cv::FONT_HERSHEY_COMPLEX,
					textScale,
					cv::Scalar(0,0,0),
					textThickness,
					8);
	}

	// add min and max text labels
	cv::putText(output,
				minLabel,
				cv::Point(floatImage.cols + border*2 + scaleWidth,
						  border + scaleHeight + minSize.height/2),
				cv::FONT_HERSHEY_COMPLEX,
				textScale,
				cv::Scalar(0,0,0),
				textThickness,
				8);

	cv::putText(output,
				maxLabel,
				cv::Point(floatImage.cols + border*2 + scaleWidth,
						  border*2 + maxSize.height/2),
				cv::FONT_HERSHEY_COMPLEX,
				textScale,
				cv::Scalar(0,0,0),
				textThickness,
				8);

	return output;
}

/// Method to convert an Eigen matrix to a CV image as an 8 bit image with a scale bar
cv::Mat gncTK::Utils::floatImageTo8Bit(Eigen::MatrixXd matrix, bool useRainbow)
{
	cv::Mat image(matrix.rows(),
				  matrix.cols(),
				  CV_32F);

	for (int r=0; r<matrix.rows(); ++r)
		for (int c=0; c<matrix.cols(); ++c)
		{
			image.at<float>(r,c) = matrix(r,c);
		}

	return(floatImageTo8Bit(image, -1, useRainbow) );
}

/// Method to fill all nan values in an image with the nearest internal finite value
cv::Mat gncTK::Utils::fillEdgeNANs(cv::Mat input)
{
	cv::Mat result = input.clone();
	Eigen::Vector2f centre; centre << (result.cols/2) , (result.rows/2);
	Eigen::Vector2f pos;

	for (pos(1)=0; pos(1)<result.rows; ++pos(1))
		for (pos(0)=0; pos(0)<result.cols; ++pos(0))
		{
			// if this pixel needs to be filled
			if (!std::isfinite(result.at<float>((int)pos(1), (int)pos(0))))
			{
				// find nearest edge pixel

				Eigen::Vector2f dir = pos - centre;
				float len = dir.norm();

				// dirty line interpolation
				bool finiteFound = false;
				float finiteValue;
				for (float x = len; x >=0; x -= 1.0)
				{
					Eigen::Vector2f testPos = centre + (dir * (x/len));
					float testValue = result.at<float>((int)testPos(1), (int)testPos(0));

					if (std::isfinite(testValue))
					{
						finiteValue = testValue;
						finiteFound = true;
						break;
					}
				}

				if (finiteFound)
					result.at<float>((int)pos(1), (int)pos(0)) = finiteValue;
			}
		}

	return result;
}

// -------------- Methods to load and save floating point cv Images --------------------------

void gncTK::Utils::imwriteFloat(FILE *file, cv::Mat image)
{
    int width = image.cols;
    int height = image.rows;
    int type = image.type();
    char header[] = "CV_FLOAT_IMAGE";

    // get the pixel size of this image type and check it's supported
    int pixelBytes=0;
    switch(type)
    {
		case CV_32F: 	pixelBytes = 4;	 	break;
		case CV_32FC2:	pixelBytes = 8; 	break;
		case CV_32FC3:	pixelBytes = 12;	break;
		case CV_32FC4:	pixelBytes = 16;	break;
		case CV_64F: 	pixelBytes = 8;	 	break;
		case CV_64FC2:	pixelBytes = 16; 	break;
		case CV_64FC3:	pixelBytes = 24;	break;
		case CV_64FC4:	pixelBytes = 32;	break;
		default:
			printf("Error: pixel type '%d' no supported by gncTK::Utils::imwriteFloat function.", type);
			return;
    }

    // write type and size of the matrix first
    fwrite((const char*) header, sizeof(char), strlen(header), file);
    fwrite((const char*) &type, sizeof(type), 1, file);
    fwrite((const char*) &width, sizeof(width), 1, file);
    fwrite((const char*) &height, sizeof(height), 1, file);

    // write data
    fwrite((const char*)image.ptr(0), pixelBytes, width*height, file);
}

void gncTK::Utils::imwriteFloat(std::string fileName, cv::Mat image)
{
    // create the file stream
    FILE *file = fopen(fileName.c_str(), "w");
	if (ferror(file))
	{
		printf("Error: Failed to open file \"%s\" for writing while saving floating point cv image.\n",
			   fileName.c_str());
		return;
	}

	gncTK::Utils::imwriteFloat(file, image);

    //close file
    fclose(file);
}

cv::Mat gncTK::Utils::imreadFloat(std::string fileName)
{
	int width, height, type;
	char header[] = "CV_FLOAT_IMAGE";
	char headerFound[14];
	cv::Mat error(10,10,CV_8UC3);

	// open the file stream
	FILE *file = fopen(fileName.c_str(), "r");
	if (ferror(file))
	{
		printf("Error: Failed to open file \"%s\" for reading while loading floating point cv image.\n",
			   fileName.c_str());
		fflush(stdout);
		return error;
	}

	// check header
	fread(headerFound, sizeof(char), 14, file);

	if (strncmp(header, headerFound,14) != 0)
	{
		printf("Error: unrecognised file header of \"%s\" found when attempting to open float image \"%s\".\n",
			   headerFound, fileName.c_str());
		fflush(stdout);
		return error;
	}

	// read type and size of image
	fread(&type, sizeof(type), 1, file);
	fread(&width, sizeof(width), 1, file);
	fread(&height, sizeof(height), 1, file);

	// get the pixel size of this image type and check it's supported
	int pixelBytes=0;
	switch(type)
	{
		case CV_32F: 	pixelBytes = 4;	 	break;
		case CV_32FC2:	pixelBytes = 8; 	break;
		case CV_32FC3:	pixelBytes = 12;	break;
		case CV_32FC4:	pixelBytes = 16;	break;
		case CV_64F: 	pixelBytes = 8;	 	break;
		case CV_64FC2:	pixelBytes = 16; 	break;
		case CV_64FC3:	pixelBytes = 24;	break;
		case CV_64FC4:	pixelBytes = 32;	break;
		default:
			printf("Error: pixel type '%d' no supported by gncTK::Utils::imreadFloat function.", type);
			fflush(stdout);
			return error;
	}

	/*printf("reading cv float image size (%d x %d) type is '%d' and pixel depth %d bytes.\n",
		   width,height,
		   type,
		   pixelBytes);
	fflush(stdout);*/

	// construct openCV image
	cv::Mat image(height, width, type);

	// read data
	fread(image.ptr(0), pixelBytes, width*height, file);

	//close file
	fclose(file);

	return image;
}

/// Method to save a floating point openCV image to a pair of 8bit png files
void gncTK::Utils::imwriteFloatPNG(std::string fileName, cv::Mat image, float ratio)
{
	cv::Mat img8l(image.size(), CV_8UC1);
	cv::Mat img8h(image.size(), CV_8UC1);

	for (int r=0; r<image.rows; ++r)
		for (int c=0; c<image.cols; ++c)
		{
			unsigned int value = image.at<float>(r,c) * ratio;

			if (value > 65535) value = 65535;

			unsigned char low = value & 0xff;
			unsigned char high = (value >> 8) & 0xff;

			img8l.at<unsigned char>(r,c) = low;
			img8h.at<unsigned char>(r,c) = high;
		}

	imwrite((fileName + ".l.png").c_str(), img8l);
	imwrite((fileName + ".h.png").c_str(), img8h);
}

/// Method to save a floating point openCV image to a pair of 8bit png files
cv::Mat gncTK::Utils::imreadFloatPNG(std::string fileName, float ratio)
{
	printf("Reading packed float image from pair of pngs\n(%s)\n(%s)\n",
		   (fileName + ".l.png").c_str(),
		   (fileName + ".h.png").c_str());

	cv::Mat img8l = cv::imread((fileName + ".l.png").c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img8h = cv::imread((fileName + ".h.png").c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (img8l.data == NULL)
	{
		ROS_ERROR("Failed to read png file \"%s\"", (fileName + ".l.png").c_str());
		return img8l;
	}

	if (img8h.data == NULL)
	{
		ROS_ERROR("Failed to read png file \"%s\"", (fileName + ".h.png").c_str());
		return img8h;
	}

	printf("loaded two part images (%d %d) and (%d %d)\n",
			img8l.rows, img8l.cols,
			img8h.rows, img8h.cols);

	cv::Mat image(img8l.size(), CV_32F);

	for (int r=0; r<image.rows; ++r)
		for (int c=0; c<image.cols; ++c)
		{
			unsigned char low = img8l.at<unsigned char>(r,c);
			unsigned char high = img8h.at<unsigned char>(r,c);

			unsigned int intValue = low || (high << 8);

			image.at<float>(r,c) = intValue / ratio;
		}

	printf("completed loading packed float image\n");

	return image;
}

/// Method to test if two bounding boxes overlap
/*
 * Boxes are stored as 3x2 matrices which are two vectors, the lowest corner and the highest corner
 */
/*bool gncTK::Utils::boxesOverlap(Eigen::MatrixXf boxA, Eigen::MatrixXf boxB)
{
	bool overlap = true;
	for (int d=0; d<3; ++d)
		overlap &= (boxB(d,0) < boxA(d,1) && boxB(d,1) > boxA(d,0));
	return overlap;
}*/

std::string gncTK::Utils::ErrorStats::toStr()
{
	return "error n=" + std::to_string(n) +
		   "\n\t value[ mean=" + std::to_string(mean) +
		   ", std dev=" + std::to_string(stdDev) +
		   ", min=" + std::to_string(min) +
		   ", max=" + std::to_string(max) +
		   " ]\n\t ratio[ mean=" + std::to_string(ratioMean) +
		   ", std dev=" + std::to_string(ratioStdDev) +
		   ", min=" + std::to_string(ratioMin) +
		   ", max=" + std::to_string(ratioMax) +
		   " ]";
}

/* Old Version of this function with incorrect ratio stats calculation
 *
gncTK::Utils::ErrorStats gncTK::Utils::calcError(cv::Mat mapA, cv::Mat mapB, bool abs)
{
	ErrorStats result;
	result.n = 0;
	float sumX = 0, sumX2 = 0;
	float minX, maxX;
	bool first = true;

	float ratioSumX = 0, ratioSumX2 = 0;
	float ratioMinX, ratioMaxX;

	// iterate over the intersection of the two maps
	int rows = std::min<int>(mapA.rows, mapB.rows);
	int cols = std::min<int>(mapA.cols, mapB.cols);
	for (int r=0; r<rows; ++r)
		for (int c=0; c<cols; ++c)
		{
			float error, ratioError;
			if (abs)
			{
				error = fabs(mapB.at<float>(r,c) - mapA.at<float>(r,c));
				ratioError = fabs(mapB.at<float>(r,c) / mapA.at<float>(r,c));
			}
			else
			{
				error = mapB.at<float>(r,c) - mapA.at<float>(r,c);
				ratioError = mapB.at<float>(r,c) / mapA.at<float>(r,c);
			}

			if (std::isfinite(error))
			{
				sumX += error;
				sumX2 += error*error;
				ratioSumX += ratioError;
				ratioSumX2 += ratioError*ratioError;
				++result.n;

				if (first)
				{
					minX = maxX = error;
					ratioMinX = ratioMaxX = ratioError;
					first = false;
				}
				else
				{
					minX = std::min<float>(minX, error);
					ratioMinX = std::min<float>(ratioMinX, ratioError);
					maxX = std::max<float>(maxX, error);
					ratioMaxX = std::max<float>(ratioMaxX, ratioError);
				}
			}
		}

	// calculate mean and standard deviation if possible
	if (result.n >= 2)
	{
		result.mean = sumX / result.n;
		float variance = ( (1.0/(result.n-1)) * sumX2 ) - ( (result.n / (result.n-1.0)) * result.mean * result.mean );
		result.stdDev = sqrt(variance);
		result.min = minX;
		result.max = maxX;

		result.ratioMean = ratioSumX / result.n;
		float ratioVariance = ( (1.0/(result.n-1)) * ratioSumX2 ) -
							   ( (result.n / (result.n-1.0)) * result.ratioMean * result.ratioMean );
		result.ratioStdDev = sqrt(ratioVariance);
		result.ratioMin = ratioMinX;
		result.ratioMax = ratioMaxX;
	}

	return result;
}*/

/// Method to calculate the error between two cv float mats (assumed to be the same size)
gncTK::Utils::ErrorStats gncTK::Utils::calcError(cv::Mat mapA, cv::Mat mapB, bool abs)
{
	ErrorStats result;
	result.n = 0;
	float sumX = 0, sumX2 = 0;
	float minX, maxX;
	bool first = true;

	// The mean and variance of the ratio are calulcated using the method described in
	// this post:
	// https://www.researchgate.net/post/How_do_I_calculate_the_variance_of_the_ratio_of_two_independent_variables
	// I need to check this is a valid statistical approach to use.

	float sumB2divA2 = 0;
	float sumB = 0;
	float sumOnedivA = 0;

	float ratioSumX = 0, ratioSumX2 = 0;
	float ratioMinX, ratioMaxX;
	float ratioSumLn = 0;

	// iterate over the intersection of the two maps
	int rows = std::min<int>(mapA.rows, mapB.rows);
	int cols = std::min<int>(mapA.cols, mapB.cols);
	for (int r=0; r<rows; ++r)
		for (int c=0; c<cols; ++c)
		{
			//float error, ratioError;

			//error = mapB.at<float>(r,c) - mapA.at<float>(r,c);
			//ratioError = mapB.at<float>(r,c) / mapA.at<float>(r,c);

			float A = mapA.at<float>(r,c);
			float B = mapB.at<float>(r,c);

			if (std::isfinite(A) && std::isfinite(B))
			{
				float error = B - A;
				float ratioError = B / A;

				// update error totals
				sumX += B-A;
				sumX2 += (B-A) * (B-A);

				// update ratio totals
				sumB2divA2 += (B*B) / (A*A);
				sumB += B;
				sumOnedivA += 1.0 / A;

				ratioSumLn += std::log(B / A);

				//ratioSumX += ratioError;
				//ratioSumX2 += ratioError*ratioError;
				++result.n;

				if (first)
				{
					minX = maxX = error;
					ratioMinX = ratioMaxX = ratioError;
					first = false;
				}
				else
				{
					minX = std::min<float>(minX, error);
					ratioMinX = std::min<float>(ratioMinX, ratioError);
					maxX = std::max<float>(maxX, error);
					ratioMaxX = std::max<float>(ratioMaxX, ratioError);
				}
			}
		}

	// calculate mean and standard deviation if possible
	if (result.n >= 2)
	{
		result.mean = sumX / result.n;
		float variance = ( (1.0/(result.n-1)) * sumX2 ) - ( (result.n / (result.n-1.0)) * result.mean * result.mean );
		result.stdDev = sqrt(variance);
		result.min = minX;
		result.max = maxX;

		//result.ratioMean = ratioSumX / result.n;
		//float ratioVariance = ( (1.0/(result.n-1)) * ratioSumX2 ) -
		//					   ( (result.n / (result.n-1.0)) * result.ratioMean * result.ratioMean );
		//result.ratioStdDev = sqrt(ratioVariance);

		result.ratioMean = std::exp(ratioSumLn / result.n);

		//float ratioVariance = (sumB2divA2 / result.n) - (result.ratioMean * result.ratioMean);
		result.ratioStdDev = 0;//sqrt(ratioVariance);

		result.ratioMin = ratioMinX;
		result.ratioMax = ratioMaxX;
	}

	return result;
}

gncTK::Utils::GLBufferInfo gncTK::Utils::setupOffscreenGLBuffer(int width, int height, GLint renderBufferType, bool quiet)
{
	GLBufferInfo bufferInfo;
	//static bool glContextSetup = false;

	//if (glContextSetup)
	//	return;

	if (!glfwInit())
	{
		ROS_ERROR("Failed to Init GLFW!\n");
		ROS_ERROR("---[ Setup Failed ]---\n");
		return bufferInfo;
	}

	glfwWindowHint(GLFW_VISIBLE, 0);
	bufferInfo.windowSC = glfwCreateWindow(width, height, "Hidden window", NULL, NULL);
	if (!bufferInfo.windowSC)
	{
		glfwTerminate();
		ROS_ERROR("Failed to create hidden GLFW window\n");
		ROS_ERROR("---[ Setup Failed ]---\n");
		return bufferInfo;
	}
	glfwMakeContextCurrent(bufferInfo.windowSC);

	// load dynamic OpenGL functions using GLEW
	GLenum glErr;
	glewExperimental = true;
	GLenum err;
	if((err=glewInit()) != GLEW_OK)
	{
		ROS_ERROR("Failed to init GLEW! : %s\n", glewGetErrorString(err));
		ROS_ERROR("---[ Setup Failed ]---\n");
		return bufferInfo;
	}

	if (!quiet)
	{
		ROS_INFO("OpenGL Context Created OK.\n");
		ROS_INFO("GL device : %s\n", glGetString(GL_RENDERER));
		ROS_INFO("GL device vendor : %s\n", glGetString(GL_VENDOR));
		ROS_INFO("OpenGL version : %s\n", glGetString(GL_VERSION));
	}

	// create frame buffer with single sampled color and depth attached render buffers
	glGenFramebuffers(1, &bufferInfo.ssFboSC);
	glBindFramebuffer(GL_FRAMEBUFFER, bufferInfo.ssFboSC);

	glGenRenderbuffers(1, &bufferInfo.ssColorBufSC);
	glBindRenderbuffer(GL_RENDERBUFFER, bufferInfo.ssColorBufSC);
	glRenderbufferStorage(GL_RENDERBUFFER, renderBufferType, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, bufferInfo.ssColorBufSC);

	glGenRenderbuffers(1, &bufferInfo.ssDepthBufSC);
	glBindRenderbuffer(GL_RENDERBUFFER, bufferInfo.ssDepthBufSC);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, bufferInfo.ssDepthBufSC);

	// setup projection with 1:1 mapping between geometry and buffer pixels
	glViewport(0,0, width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glOrtho(0, width, height, 0, 1.0,-1.0);

	return bufferInfo;
}

/* Method to write an Eigen matrix to a file in binary format.
 *
 * The data type of the matrix is not encoded in this version! Will need adding
 * also the row-column order is not checked.
 *
 * TODO add datatype to file header
 * TODO ensure matrix is written in column major order (convert row major matrices before writing)
 */
template<class Matrix>
void gncTK::Utils::writeEigenMatrix(std::string filename, const Matrix& matrix)
{
    std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}

template void gncTK::Utils::writeEigenMatrix<Eigen::MatrixXf>(std::string filename, const Eigen::MatrixXf& matrix);
template void gncTK::Utils::writeEigenMatrix<Eigen::MatrixXd>(std::string filename, const Eigen::MatrixXd& matrix);
template void gncTK::Utils::writeEigenMatrix<Eigen::MatrixXi>(std::string filename, const Eigen::MatrixXi& matrix);

/// Method to write an Eigen Matrix to a csv file
template<class Matrix>
void gncTK::Utils::writeEigenMatrixCSV(std::string filename, const Matrix& matrix, std::string format, char separator)
{
	std::ofstream ofs (filename, std::ofstream::out);

	for (int r=0; r<matrix.rows(); ++r)
	{
		char cell[256];

		for (int c=0; c<matrix.cols(); ++c)
		{
			sprintf(cell, format.c_str(), matrix(r,c));
			if (c != 0)
				ofs << separator << " ";

			ofs << std::string(cell);
		}

		ofs << std::endl;
	}
	ofs.close();
}

template void gncTK::Utils::writeEigenMatrixCSV<Eigen::MatrixXf>(std::string filename, const Eigen::MatrixXf& matrix, std::string format, char separator);
template void gncTK::Utils::writeEigenMatrixCSV<Eigen::MatrixXd>(std::string filename, const Eigen::MatrixXd& matrix, std::string format, char separator);
template void gncTK::Utils::writeEigenMatrixCSV<Eigen::MatrixXi>(std::string filename, const Eigen::MatrixXi& matrix, std::string format, char separator);

template<class Matrix>
void gncTK::Utils::readEigenMatrix(std::string filename, Matrix& matrix)
{
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}

template void gncTK::Utils::readEigenMatrix<Eigen::MatrixXf>(std::string filename, Eigen::MatrixXf& matrix);
template void gncTK::Utils::readEigenMatrix<Eigen::MatrixXd>(std::string filename, Eigen::MatrixXd& matrix);
template void gncTK::Utils::readEigenMatrix<Eigen::MatrixXi>(std::string filename, Eigen::MatrixXi& matrix);


