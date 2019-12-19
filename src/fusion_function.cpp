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

fusion_function.h

class encapsulating the fusion mathematical model
---------------------------------------------------

This class stores, loads, saves and calibrates the lidar
camera fusion function.

-------------------------------------------------------------*/

#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <fusion_function.h>

void gncTK::FusionFunction::setExtrinsicTF(Eigen::Matrix4f extrinsic)
{
	cv::Mat exMatrix(4,4,CV_64F);
	for (int r=0; r<4; ++r)
		for (int c=0; c<4; ++c)
			exMatrix.at<double>(r,c) = extrinsic(r,c);

	extrinsicTF = cv::Affine3d(exMatrix);
}

void gncTK::FusionFunction::setCameraMatrix(Eigen::Matrix3f cameraMat)
{
	cameraMatrix = cv::Mat(3,3,CV_64F, cv::Scalar(0));
	for (int r=0; r<3; ++r)
		for (int c=0; c<3; ++c)
			cameraMatrix.at<double>(r,c) = cameraMat(r,c);
}

void gncTK::FusionFunction::setDistortionCoeffs(std::vector<double> coeffs)
{
	for (int d=0; d<coeffs.size() && d < 14; ++d)
		distortionCoeffs.at<double>(d) = coeffs[d];
}


tf::Transform gncTK::FusionFunction::getExtrinsicTransform()
{
	tf::Vector3 translation(extrinsicTF.translation()(0),
							extrinsicTF.translation()(1),
							extrinsicTF.translation()(2));

	cv::Matx33d cvR = extrinsicTF.rotation();

	tf::Matrix3x3 tfRot(cvR(0,0), cvR(1,0), cvR(2,0),
						cvR(0,1), cvR(1,1), cvR(2,1),
						cvR(0,2), cvR(1,2), cvR(2,2));

	return tf::Transform(tfRot, translation);
}

/// Method to return the camera focal point in the LIDAR coordinate system
Eigen::Vector3f  gncTK::FusionFunction::getCameraLocation()
{
	cv::Vec3f translation = extrinsicTF.translation();
	Eigen::Vector3f trans; trans << translation[0], translation[1], translation[2];

	return Eigen::Vector3f::Zero() - trans;
}

/// Method to project a point in the ROS coordinate convention of the LIDAR frame to a pixel position
Eigen::Vector2f gncTK::FusionFunction::projectPoint(Eigen::Vector3f point)
{
	std::vector<cv::Point3f> cvPoints;
	cvPoints.push_back(cv::Point3f(point[1], 0-point[2], point[0])); // <-- includes ROS to optical frame conversion

	std::vector<cv::Point2f> imagePoints;

	cv::projectPoints(cvPoints,
					  extrinsicTF.rvec(),
					  extrinsicTF.translation(),
					  cameraMatrix,
					  distortionCoeffs,
					  imagePoints);

	Eigen::Vector2f result; result << imagePoints[0].x, imagePoints[0].y;

	return result;
}

/// Method to project a vector of points in the ROS coordinate convention of the LIDAR frame to pixel positions
std::vector<Eigen::Vector2f> gncTK::FusionFunction::projectPoints(std::vector<Eigen::Vector3f> points)
{
	std::vector<cv::Point3f> cvPoints;
	for (int p=0; p<points.size(); ++p)
		cvPoints.push_back(cv::Point3f(points[p][1], 0-points[p][2], points[p][0])); // <-- includes ROS to optical frame conversion

	std::vector<cv::Point2f> imagePoints;

	cv::projectPoints(cvPoints,
					  extrinsicTF.rvec(),
					  extrinsicTF.translation(),
					  cameraMatrix,
					  distortionCoeffs,
					  imagePoints);

	std::vector<Eigen::Vector2f> results;
	for (int p=0; p<imagePoints.size(); ++p)
	{
		Eigen::Vector2f resultPoint; resultPoint << imagePoints[p].x, imagePoints[p].y;
		results.push_back(resultPoint);
	}

	return results;
}

/// Method to calibrate this fusion function using a matched set of pixel locations and 3D locations
/**
 * Returns the mean RMS reprojection error in pixels of this calibration
 */
double gncTK::FusionFunction::calibrate(std::vector<Eigen::Vector2f> points2D,
										std::vector<Eigen::Vector3f> points3D,
										cv::Size imageSize)
{
	printf("calibrating fusion function.\n");

	std::vector<std::vector<cv::Point2f> > imagePoints;
	std::vector<std::vector<cv::Point3f> > objectPoints;

	if (points2D.size() != points3D.size())
	{
		printf("Error trying to calibrate the fusion function with different numbers of image and object points.\n");
		return 0;
	}
	if (points2D.size() < 3)
	{
		printf("Error trying to calibrate the fusion function with fewer than 3 points.\n");
		return 0;
	}

	std::vector<cv::Point2f> imagePointsVec;
	std::vector<cv::Point3f> objectPointsVec;
	for (int p=0; p<points2D.size(); ++p)
	{
		imagePointsVec.push_back(cv::Point2f(points2D[p][0], points2D[p][1]) );
		objectPointsVec.push_back(cv::Point3f(points3D[p][1], 0-points3D[p][2], points3D[p][0]));
	}
	imagePoints.push_back(imagePointsVec);
	objectPoints.push_back(objectPointsVec);

	std::vector<cv::Mat> rvec;
	std::vector<cv::Mat> tvec;
	double rms;

	try
	{
		rms = cv::calibrateCamera(objectPoints,
								imagePoints,
								imageSize,
								cameraMatrix,
								distortionCoeffs,
								rvec, tvec,
								CV_CALIB_RATIONAL_MODEL | CV_CALIB_USE_INTRINSIC_GUESS,
								cv::TermCriteria(3, 20, 1e-6));

		extrinsicTF = cv::Affine3d(rvec[0], tvec[0]);
	}
	catch(cv::Exception& e)
	{
		ROS_ERROR("OpenCV error while attempting to calibrate fusion sensor. \"%s\".", e.what());
	}
	return rms;
}


double gncTK::FusionFunction::optimseExtrinsic(std::vector<Eigen::Vector2f> points2D,
		 	 	 	 						   std::vector<Eigen::Vector3f> points3D)
{
	printf("optimising extrinsics.\n");

	//std::vector<std::vector<cv::Point2f> > imagePoints;
	//std::vector<std::vector<cv::Point3f> > objectPoints;

	if (points2D.size() != points3D.size())
	{
		printf("Error trying to calibrate the fusion function with different numbers of image and object points.\n");
		return 0;
	}
	if (points2D.size() < 3)
	{
		printf("Error trying to calibrate the fusion function with fewer than 3 points.\n");
		return 0;
	}

	std::vector<cv::Point2f> imagePointsVec;
	std::vector<cv::Point3f> objectPointsVec;
	for (int p=0; p<points2D.size(); ++p)
	{
		imagePointsVec.push_back(cv::Point2f(points2D[p][0], points2D[p][1]) );
		objectPointsVec.push_back(cv::Point3f(points3D[p][1], 0-points3D[p][2], points3D[p][0]));
	}
	//imagePoints.push_back(imagePointsVec);
	//objectPoints.push_back(objectPointsVec);

	cv::Affine3<double>::Vec3 rvec = extrinsicTF.rvec();

	cv::Affine3<double>::Vec3 tvec = extrinsicTF.translation();

	solvePnP(objectPointsVec,
			 imagePointsVec,
			 cameraMatrix,
			 distortionCoeffs,
			 rvec, tvec,
			 1);

	double rms = 1; // need to calculate this ourselves here since soldPnP doesn't return it!

	extrinsicTF = cv::Affine3d(rvec, tvec);

	return rms;

}

/// Method to save the configuration of this fusion function
void gncTK::FusionFunction::saveConfig(std::string fileName)
{
	char buf[64];
	std::vector<std::vector<std::string> > values;

	values.push_back( std::vector<std::string>(1, "gncTK Camera Calibration Configuration") );
	values.push_back( std::vector<std::string>(1, "Extrinsic Matrix") );

	for (int r=0; r<4; ++r)
	{
		std::vector<std::string> row;

		for (int c=0; c<4; ++c)
		{
			sprintf(buf, "%.12f", extrinsicTF.matrix(r,c) );
			row.push_back( std::string(buf) );
		}

		values.push_back(row);
	}

	values.push_back( std::vector<std::string>(1, "Camera Matrix") );

	for (int r=0; r<3; ++r)
	{
		std::vector<std::string> row;

		for (int c=0; c<3; ++c)
		{
			sprintf(buf, "%.12f",cameraMatrix.at<double>(r,c)  );
			row.push_back( std::string(buf) );
		}

		values.push_back(row);
	}

	values.push_back( std::vector<std::string>(1, "Distortion Coefficients") );

	std::vector<std::string> row;
	for (int c=0; c<distortionCoeffs.cols; ++c)
	{
		sprintf(buf, "%.12f",distortionCoeffs.at<double>(0,c)  );
		row.push_back( std::string(buf) );
	}
	values.push_back(row);

	gncTK::Utils::writeCSVFile(fileName, &values);
}

/// Method to load the configuration of this fusion function
bool gncTK::FusionFunction::loadConfig(std::string fileName)
{
	std::vector<std::vector<std::string> > content;
	if (gncTK::Utils::readCSVFile(fileName, &content, ',', 2) == -1)
	{
		ROS_ERROR("Failed to open config file \"%s\".", fileName.c_str());
		return false;
	}

	// verify config table is the right size
	if (content.size() != 10)
	{
		ROS_ERROR("Error: fusion config file doesn't have 12 rows.\n");
		return false;
	}
	if (content[0].size() != 4 ||
		content[1].size() != 4 ||
		content[2].size() != 4 ||
		content[3].size() != 4)
	{
		ROS_ERROR("Error: fusion config file doesn't have the right sized extrinsic matrix.\n");
		return false;
	}
	if (content[5].size() != 3 ||
		content[6].size() != 3 ||
		content[7].size() != 3)
	{
		ROS_ERROR("Error: fusion config file doesn't have the right sized camera matrix.\n");
		return false;
	}
	if (content[9].size() != 14)
	{
		ROS_ERROR("Error: fusion config file doesn't have the right number of distortion coefficients.\n");
		return false;
	}

	// extract extrinsic matrix
	cv::Mat extrinsicMatrix = cv::Mat(4,4,CV_64F, cv::Scalar(0));
	for (int r=0; r<4; ++r)
		for (int c=0; c<4; ++c)
			extrinsicMatrix.at<double>(r,c) = std::stod( content[r][c] );
	extrinsicTF = cv::Affine3d(extrinsicMatrix);

	// extract camera matrix
	cameraMatrix = cv::Mat(3,3,CV_64F, cv::Scalar(0));
	for (int r=0; r<3; ++r)
		for (int c=0; c<3; ++c)
			cameraMatrix.at<double>(r,c) = std::stod( content[5+r][c] );

	// extract distortion coeffs
	distortionCoeffs = cv::Mat(1,14, CV_64F, cv::Scalar(0));
	for (int d=0; d<14; ++d)
		distortionCoeffs.at<double>(d) = std::stod( content[9][d] );

	return true;
}

/// Method to format the calibration parameters into a multi-line string
std::string gncTK::FusionFunction::configString()
{
	std::stringstream output;
	output << "Fusion Function configuration\n-----------------\n";

	output << "Camera Matrix\n" << cameraMatrix << "\n";
	output << "Distortion Coefficients\n" << distortionCoeffs << "\n";
	output << "Extrinsic Matrix\n" << extrinsicTF.matrix << "\n----------------\n";

	return output.str();
}

Eigen::Vector3f gncTK::FusionFunction::polar2Cart(float az, float el, float d)
{
	Eigen::Vector3f result;

	result[0] = sin(az) * cos(el) * d;
	result[1] = cos(az) * cos(el) * d;
	result[2] = sin(el) * d;

	return result;
}

// Method to generate the reverse projection lookup table
bool gncTK::FusionFunction::generateReverseProjection(int width, int height, bool useOpticalFrame)
{
	/// Pointer to GLFW window used for OpenGL rendering
	GLFWwindow* window;
	GLuint ssFbo, ssColorBuf;

	if (reverseProjectionGenerated)
	{
		ROS_ERROR("Error trying to generate the fusion fuction reverse projection twice!");
		return false;
	}

	// setup offscreen GL rendering context
	if (!glfwInit())
	{
		ROS_ERROR("Failed to Init GLFW!\n");
		return false;
	}

	GLenum errCode;
	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(width, height, "Hidden window", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		ROS_ERROR("Failed to create hidden GLFW window\n");
		return false;
	}
	glfwMakeContextCurrent(window);

	// load dynamic OpenGL functions using GLEW
	GLenum glErr;
	glewExperimental = true;
	GLenum err;
	if((err=glewInit()) != GLEW_OK)
	{
		ROS_ERROR("Failed to init GLEW! : %s\n", glewGetErrorString(err));
		return false;
	}

	//printf("OpenGL Context Created OK.\n");
	//printf("GL device : %s\n", glGetString(GL_RENDERER));
	//printf("GL device vendor : %s\n", glGetString(GL_VENDOR));
	//printf("OpenGL version : %s\n", glGetString(GL_VERSION));

	// create frame buffer with single sampled color and depth attached render buffers
	glGenFramebuffers(1, &ssFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, ssFbo);

	glGenRenderbuffers(1, &ssColorBuf);
	glBindRenderbuffer(GL_RENDERBUFFER, ssColorBuf);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ssColorBuf);

	// setup projection with 1:1 mapping between geometry and buffer pixels
	glViewport(0,0, width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glOrtho(0, width, height, 0, 1.0,-1.0);

	// disable clamping to 0-1 so that full range of positions can be stored in color buffer
	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	glClearColor(-20.0f, -20.0f, -20.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw a unit sphere with the camera coordinates as x&y positions and 3D optical coordinates as float color
	glBegin(GL_TRIANGLES);
	float interval = (2*M_PI) / 100;
	float d = 10;
	for (float x=-20; x < 20; x+=0.1)
		for (float y=-20; y<20; y+=0.1)
		{
			Eigen::Vector3f tl; tl << -10, x,     y;
			Eigen::Vector3f tr; tr << -10, x,     y+0.1;
			Eigen::Vector3f bl; bl << -10, x+0.1, y;
			Eigen::Vector3f br; br << -10, x+0.1, y+0.1;

			Eigen::Vector3f ptl = tl;
			Eigen::Vector3f ptr = tr;
			Eigen::Vector3f pbl = bl;
			Eigen::Vector3f pbr = br;

			// if the calibration was conducted in the optical frame
			if (useOpticalFrame)
			{
				ptl << x,     y,     -10;
				ptr << x,     y+0.1, -10;
				pbl << x+0.1, y,     -10;
				pbr << x+0.1, y+0.1, -10;
			}

			// project each point into the camera frame
			Eigen::Vector2f tlImg = projectPoint(ptl);
			Eigen::Vector2f trImg = projectPoint(ptr);
			Eigen::Vector2f blImg = projectPoint(pbl);
			Eigen::Vector2f brImg = projectPoint(pbr);

			// NEED TO CHECK may not need this with optical calibrations!
			if (!useOpticalFrame)
			{
				tlImg[1] = height - 1 - tlImg[1];  // vertical flip
				trImg[1] = height - 1 - trImg[1];
				blImg[1] = height - 1 - blImg[1];
				brImg[1] = height - 1 - brImg[1];
			}

			// draw 1st triangle
			glColor4f(tl[0], tl[1], tl[2], 1.0);
			glVertex2f(tlImg[0], tlImg[1]);
			glColor4f(tr[0],tr[1],tr[2], 1.0);
			glVertex2f(trImg[0], trImg[1]);
			glColor4f(bl[0],bl[1],bl[2], 1.0);
			glVertex2f(blImg[0], blImg[1]);

			// draw 2nd triangle
			glColor4f(tr[0],tr[1],tr[2], 1.0);
			glVertex2f(trImg[0], trImg[1]);
			glColor4f(br[0],br[1],br[2], 1.0);
			glVertex2f(brImg[0], brImg[1]);
			glColor4f(bl[0],bl[1],bl[2], 1.0);
			glVertex2f(blImg[0], blImg[1]);
		}
	glEnd();

	// read completed buffer out
	reverseProjection = cv::Mat(height, width, CV_32FC4);
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,reverseProjection.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
	{
		ROS_ERROR("Error on reading spatial values from color buffer. %s\n", gluErrorString(errCode));
		return false;
	}

	// convert from optical frame to ROS frame (reverse of project points) and normalise
	for (int i=0; i<reverseProjection.rows*reverseProjection.cols; ++i)
	{
		cv::Vec4f newP, oldP = reverseProjection.at<cv::Vec4f>(i);
		newP[0] = 0-oldP[1];
		newP[1] = oldP[2];
		newP[2] = 0-oldP[0];
		float l2 = sqrt(newP[0]*newP[0]+newP[1]*newP[1]+newP[2]*newP[2]);
		reverseProjection.at<cv::Vec4f>(i) = newP / l2;
	}

	// cleanup offscreen GL rendering context
	glDeleteFramebuffers(1,&ssFbo);
	glDeleteRenderbuffers(1,&ssColorBuf);

	reverseProjectionGenerated = true;
	return true;
}

Eigen::Vector3f gncTK::FusionFunction::getReverseProjection(int col, int row)
{
	if (col < 0 || col >= reverseProjection.cols ||
		row < 0 || row >= reverseProjection.rows)
	{
		ROS_ERROR("Attempting to lookup projection position (%d x %d) is outside of matrix size (%d x %d)",
				  col, row,
				  reverseProjection.cols, reverseProjection.rows);

		return Eigen::Vector3f::Zero();
	}

	Eigen::Vector3f projection;
	cv::Vec4f ray = reverseProjection.at<cv::Vec4f>(row, col);
	projection[0] = ray[0];
	projection[1] = ray[1];
	projection[2] = ray[2];
	return projection;
}

Eigen::Vector3f gncTK::FusionFunction::interpolateReverseProjection(float col, float row)
{
	if (col < 0.0 || col >= reverseProjection.cols ||
		row < 0.0 || row >= reverseProjection.rows)
	{
		ROS_ERROR("Attempting to lookup projection position (%f x %f) is outside of matrix size (%d x %d)",
				  col, row,
				  reverseProjection.cols, reverseProjection.rows);

		return Eigen::Vector3f::Zero();
	}

	int colInt = col;
	int rowInt = row;
	float colRes = col - colInt;
	float rowRes = row - rowInt;

	float wTL = colRes     * rowRes;
	float wTR = (1-colRes) * rowRes;
	float wBL = colRes     * (1-rowRes);
	float wBR = (1-colRes) * (1-rowRes);

	cv::Vec4f ray = reverseProjection.at<cv::Vec4f>(rowInt, colInt) * wTL;

	if (colInt + 1 < reverseProjection.cols)
		ray += reverseProjection.at<cv::Vec4f>(rowInt  , colInt+1) * wTR;
	if (rowInt + 1 < reverseProjection.rows)
		ray += reverseProjection.at<cv::Vec4f>(rowInt+1, colInt  ) * wBL;
	if (colInt + 1 < reverseProjection.cols && rowInt + 1 < reverseProjection.rows)
		ray += reverseProjection.at<cv::Vec4f>(rowInt+1, colInt+1) * wBR;

	Eigen::Vector3f projection;
	//cv::Vec4f ray = reverseProjection.at<cv::Vec4f>(row, col);
	projection[0] = ray[0];
	projection[1] = ray[1];
	projection[2] = ray[2];
	return projection;
}

/// Method to calculate the exact solid angle of the FOV of this camera
float gncTK::FusionFunction::calculateFOVSolidAngle()
{
	if (!reverseProjectionGenerated)
	{
		ROS_ERROR("Attempting to calculate the FOV solid angle of a fusion function without a reverse projection matrix");
		return 0;
	}

	float FOVSolidAngle = 0;

	int step = 2;

	for (int v=0; v<reverseProjection.rows-1; v+=step)
		for (int h=0; h<reverseProjection.cols-1; h+=step)
		{
			Eigen::Vector3f tl,tr,bl,br;

			int vPlus = v + step, hPlus = h + step;
			if (vPlus >= reverseProjection.rows) vPlus = reverseProjection.rows-1;
			if (hPlus >= reverseProjection.cols) hPlus = reverseProjection.cols-1;

			tl = getReverseProjection(h,     v    );
			tr = getReverseProjection(hPlus, v    );
			bl = getReverseProjection(h,     vPlus);
			br = getReverseProjection(hPlus, vPlus);

			float t1 = gncTK::Utils::triangleSolidAngle(tl,tr,bl);
			float t2 = gncTK::Utils::triangleSolidAngle(tr,br,bl);

			if (std::isnan(t1)) t1 = 0;
			if (std::isnan(t2)) t2 = 0;

			FOVSolidAngle += t1+t2;
		}

	return FOVSolidAngle;
}
