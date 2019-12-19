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

lidar_sim.cpp

LIDAR sensor simulator node
----------------------------

This node uses a pair of depth and incident cube map images
to simulate the data from a LIDAR sensor within the
cube maps.

-------------------------------------------------------------*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <utils.h>
#include <lidar_sim.h>

gncTK::LidarSim::LidarSim()
{
    mapsLoaded = false;
    maxDepth = 50.0;

	hRes = 200;
	vRes = 200;

	hFOV = 90.0 * (M_PI / 180.0);
	vFOV = 90.0 * (M_PI / 180.0);
}

Eigen::Vector2f gncTK::LidarSim::getSampleLocation(Eigen::Vector3f direction)
{
	// normalise direction in X Y and Z directions
	Eigen::Vector3f pX = direction / fabs(direction[0]);
	Eigen::Vector3f pY = direction / fabs(direction[1]);
	Eigen::Vector3f pZ = direction / fabs(direction[2]);

	float X,Y;

	if (pX[1] >= -1 && pX[1] <= 1 && pX[2] >= -1 && pX[2] <= 1)
	{
		if (pX[0] > 0) // front cube face
		{
			X = 1.5 + (pX[1] / 2);
			Y = 1.5 - (pX[2] / 2);
		}
		else // back cube face
		{
			X = 3.5 - (pX[1] / 2);
			Y = 1.5 - (pX[2] / 2);
		}
	}
	else if (pY[0] >= -1 && pY[0] <= 1 && pY[2] >= -1 && pY[2] <= 1)
	{
		if (pY[1] > 0)  // left cube face
		{
			X = 2.5 - (pY[0] / 2);
			Y = 1.5 - (pY[2] / 2);
		}
		else // right cube face
		{
			X = 0.5 + (pY[0] / 2);
			Y = 1.5 - (pY[2] / 2);
		}
	}
	else
	{
		if (pZ[2] > 0) // top cube face
		{
			X = 1.5 + (pZ[1] / 2);
			Y = 0.5 + (pZ[0] / 2);
		}
		else // bottom cube face
		{
			X = 1.5 + (pZ[1] / 2);
			Y = 2.5 - (pZ[0] / 2);
		}
	}

	Eigen::Vector2f res;
	res[0] = X;
	res[1] = Y;

	return res;
}

gncTK::LidarSim::cubeSample gncTK::LidarSim::generateCubeSample(Eigen::Vector3f direction)
{
	Eigen::Vector2f loc = getSampleLocation(direction) * mapResolution;
	gncTK::LidarSim::cubeSample res;

	//mapCopy.at<cv::Vec3b>(loc[1], loc[0]) = debugColor;

	// get pixel coordinates and residuals
	int x = loc[0];
	int y = loc[1];
	float xRes = loc[0] - x;
	float yRes = loc[1] - y;
	float xResN = 1 - xRes;
	float yResN = 1 - yRes;

	float wTL = xResN * yResN;
	float wTR = xRes * yResN;
	float wBL = xResN * yRes;
	float wBR = xRes * yRes;

	bool locOK = true;

	if (x > depthMap.cols - 2 || y > depthMap.rows - 2 || x < 0 || y < 0)
	{
	  locOK = false;
	}

	if (loc[0] != loc[0] || loc[1] != loc[1])
	{
		printf("NAN location generated for vector (%f, %f, %f)\n", direction[0], direction[1], direction[2]);
		locOK = false;
	}

	if (locOK)
	{
		// get depth values
		float depthTL = depthMap.at<float>(y, x);
		float depthTR = depthMap.at<float>(y, x+1);
		float depthBL = depthMap.at<float>(y+1, x);
		float depthBR = depthMap.at<float>(y+1, x+1);

		res.depth = (depthTL * wTL) +
					(depthTR * wTR) +
					(depthBL * wBL) +
					(depthBR * wBR);

		// incident values
		cv::Vec3b incTL = incidentMap.at<cv::Vec3b>(y, x);
		cv::Vec3b incTR = incidentMap.at<cv::Vec3b>(y, x+1);
		cv::Vec3b incBL = incidentMap.at<cv::Vec3b>(y+1, x);
		cv::Vec3b incBR = incidentMap.at<cv::Vec3b>(y+1, x+1);

		res.albedo = (incTL[0] * wTL) +
					 (incTR[0] * wTR) +
					 (incBL[0] * wBL) +
					 (incBR[0] * wBR);
		res.incident = (incTL[1] * wTL) +
					   (incTR[1] * wTR) +
					   (incBL[1] * wBL) +
					   (incBR[1] * wBR);
		res.mask = (incTL[2] * wTL) +
				   (incTR[2] * wTR) +
				   (incBL[2] * wBL) +
				   (incBR[2] * wBR);
	}


	return res;
}

pcl::PointXYZI gncTK::LidarSim::generateLidarSample(Eigen::Vector3f direction)
{
	//direction /= direction.squaredNorm();
	direction.normalize();
	gncTK::LidarSim::cubeSample sample = generateCubeSample(direction);

	pcl::PointXYZI res;

	// add error to depth measurement TODO

	// if the sample exists in the cube map and is within the depth range of the LIDAR
	if (sample.mask > 254 && sample.depth < maxDepth)
	{
		Eigen::Vector3f pos = direction * sample.depth;
		res.x = pos[1];
		res.y = 0-pos[2];
		res.z = pos[0];
		res.intensity = sample.albedo * 255;
	}
	else // no sample
	{
		res.x = NAN;
		res.y = NAN;
		res.z = NAN;
	}

	return res;
}

pcl::PointCloud<pcl::PointXYZI> gncTK::LidarSim::generateScan(Eigen::Matrix3f orientation)
{
	printf("Generating lidar scan.\n"); fflush(stdout);

	pcl::PointCloud<pcl::PointXYZI> res;
	res.width = hRes;
	res.height = vRes;
	res.points.resize(hRes*vRes);
	res.is_dense = true;

	for (int v=0; v<vRes; ++v)
		for (int h=0; h<hRes; ++h)
		{
			res.points[h + (v*hRes)] = generateLidarSample(getDirection(h,v));
		}

	printf("Finished generating lidar scan.\n"); fflush(stdout);

	return res;
}

float gncTK::LidarSim::calculateFOVSolidAngle()
{
	FOVSolidAngle = 0;

	for (int v=0; v<vRes-1; ++v)
		for (int h=0; h<hRes-1; ++h)
		{
			Eigen::Vector3f tl,tr,bl,br;

			tl = getDirection(h  ,v  );
			tr = getDirection(h+1,v  );
			bl = getDirection(h  ,v+1);
			br = getDirection(h+1,v+1);

			float t1 = gncTK::Utils::triangleSolidAngle(tl,tr,bl);//, ((v*h)%456 == 0));
			float t2 = gncTK::Utils::triangleSolidAngle(tr,br,bl);

			if ((v*h)%456 == 0)
			{
			//	printf("quad area is %f\n", t1+t2);
			}

			FOVSolidAngle += t1+t2;
		}

	return FOVSolidAngle;
}

void gncTK::LidarSim::loadCubeMaps(char *prefix)
{
	printf("loading cubemaps c style (%s)\n", prefix);

	loadCubeMaps(std::string(prefix));
}

void gncTK::LidarSim::loadCubeMaps(std::string prefix)
{
	printf("loading cubemaps stdlib style [%s]\n", prefix.c_str());

	//depthMap = gncTK::Utils::imreadFloat(prefix + "_depth.float");
	depthMap = gncTK::Utils::imreadFloatPNG(prefix + "_depth", 1000.0);

	incidentMap = cv::imread((prefix + "_albedo_incident.png").c_str());

	if (incidentMap.data == NULL)
	{
		ROS_ERROR("Failed to load incident & albedo cube map [%s]\n", (prefix + "%s_albedo_incident.png").c_str());
	}

	mapResolution = depthMap.rows / 3;

	printf("Completed loading cube maps. resolution=%d\n", mapResolution);
	printf("depth size (%d %d)\nincident size (%d %d)\n",
			depthMap.rows, depthMap.cols,
			incidentMap.rows, incidentMap.cols);
}

