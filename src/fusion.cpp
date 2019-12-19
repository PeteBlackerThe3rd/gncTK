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

fusion.h

virtual LIDAR camera fusion processing object
----------------------------------------------

This abstract super class defines the interface for all
of the concrete implementations of different LIDAR camera
fusion algorithms in various base classes

-------------------------------------------------------------*/

#include <fusion.h>
#include <boost/math/constants/constants.hpp>

/// Filter triangles that are above the incident angle threshold (To remove jump faces)
/**
 * Takes the threshold angle in radians and returns the number of triangles that
 * were removed from the mesh.
 */
int gncTK::Fusion::filterTrianglesOnIncidentAngle(gncTK::Mesh *mesh, float thresholdAngle)
{
	int filtered = 0;

	for (int t=0; t<mesh->triangles.size(); ++t)
	{
		// calculate triangle normal
		Eigen::Vector3f v1 = mesh->vertices[mesh->triangles[t].v1];
		Eigen::Vector3f v2 = mesh->vertices[mesh->triangles[t].v2];
		Eigen::Vector3f v3 = mesh->vertices[mesh->triangles[t].v3];
		Eigen::Vector3f normal = (v2-v1).cross(v3-v1);

		// calculate angle between normal and vector from sensor origin to triangle centroid
		Eigen::Vector3f sensorRay = mesh->sensorOrigin - ((v1+v2+v3) / 3);
		float angle = acos( (normal.dot(sensorRay)) / (normal.norm() * sensorRay.norm()) );

		// reverse angle if its greater than 90 degrees, make this process triangle winding agnostic
		if (angle > boost::math::constants::pi<float>()/2)
			angle = boost::math::constants::pi<float>() - angle;

		if (angle >= thresholdAngle)
		{
			mesh->triangles[t] = mesh->triangles[mesh->triangles.size()-1];
			mesh->triangles.pop_back();
			--t;
			++filtered;
		}
	}

	return filtered;
}

void gncTK::Fusion::setupOffscreenGLBufferSC(int width, int height)
{
	if (glContextSetupSC)
		return;

	if (!glfwInit())
	{
		printf("Failed to Init GLFW!\n");
		printf("---[ Setup Failed ]---\n");
		return;
	}

	glfwWindowHint(GLFW_VISIBLE, 0);
	windowSC = glfwCreateWindow(width, height, "Hidden window", NULL, NULL);
	if (!windowSC)
	{
		glfwTerminate();
		printf("Failed to create hidden GLFW window\n");
		printf("---[ Setup Failed ]---\n");
		return;
	}
	glfwMakeContextCurrent(windowSC);

	// load dynamic OpenGL functions using GLEW
	GLenum glErr;
	glewExperimental = true;
	GLenum err;
	if((err=glewInit()) != GLEW_OK)
	{
		printf("Failed to init GLEW! : %s\n", glewGetErrorString(err));
		printf("---[ Setup Failed ]---\n");
		return;
	}

	printf("OpenGL Context Created OK.\n");
	printf("GL device : %s\n", glGetString(GL_RENDERER));
	printf("GL device vendor : %s\n", glGetString(GL_VENDOR));
	printf("OpenGL version : %s\n", glGetString(GL_VERSION));

	// create frame buffer with single sampled color and depth attached render buffers
	glGenFramebuffers(1, &ssFboSC);
	glBindFramebuffer(GL_FRAMEBUFFER, ssFboSC);

	glGenRenderbuffers(1, &ssColorBufSC);
	glBindRenderbuffer(GL_RENDERBUFFER, ssColorBufSC);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ssColorBufSC);

	glGenRenderbuffers(1, &ssDepthBufSC);
	glBindRenderbuffer(GL_RENDERBUFFER, ssDepthBufSC);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ssDepthBufSC);

	// setup projection with 1:1 mapping between geometry and buffer pixels
	glViewport(0,0, width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glOrtho(0, width, height, 0, 1.0,-1.0);

	glContextSetupSC = true;
}

cv::Mat gncTK::Fusion::renderModelToCamera(int resolutionFactor, int border, gncTK::Mesh *model)
{
	GLenum errCode;

	// setup GL rendering context
	int width = (inputImage.cols / resolutionFactor) + border;
	int height = (inputImage.rows / resolutionFactor) + border;
	setupOffscreenGLBufferSC(width, height);

	cv::Mat result(height,width, CV_8UC4, cv::Scalar(0,0,0,0));

	// render mesh into buffer
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	//glClearDepthf(1.0f);

	// calculate vertex depth values and find maximum depth
	float maxDepth = 0;
	Eigen::Vector3f cameraPos = fusionFunction.getCameraLocation();
	std::vector<float> vertexDepths;
	for (int v=0; v<model->vertices.size(); ++v)
	{
		float depth = (cameraPos-model->vertices[v]).norm();
		vertexDepths.push_back(depth);
		if (depth > maxDepth)
			maxDepth = depth;
	}

	// pre-calculate UV scaling factors (converts camera image coords to reduced image frame coords)
	float uScale = (1.0/resolutionFactor) + border;
	float vScale = (1.0/resolutionFactor) + border;
	int camImageHeight = inputImage.rows - 1;

	// render depth, height and angle geometry to frame buffer
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<model->triangles.size(); ++t)
	{
		Eigen::Vector3f v1 = model->vertices[model->triangles[t].v1];
		Eigen::Vector3f v2 = model->vertices[model->triangles[t].v2];
		Eigen::Vector3f v3 = model->vertices[model->triangles[t].v3];
		cv::Vec3b c1 = model->vertexColors[model->triangles[t].v1];
		cv::Vec3b c2 = model->vertexColors[model->triangles[t].v2];
		cv::Vec3b c3 = model->vertexColors[model->triangles[t].v3];
		float d1 = vertexDepths[model->triangles[t].v1];
		float d2 = vertexDepths[model->triangles[t].v2];
		float d3 = vertexDepths[model->triangles[t].v3];

		Eigen::Vector2f t1 = fusionFunction.projectPoint(v1);
		Eigen::Vector2f t2 = fusionFunction.projectPoint(v2);
		Eigen::Vector2f t3 = fusionFunction.projectPoint(v3);

		glColor4f(c1[0]/255.0, c1[1]/255.0, c1[2]/255.0, 1.0);
		glVertex3f((t1[0]*uScale)+border, ((camImageHeight-t1[1])*vScale)+border, 0-(d1/maxDepth));

		glColor4f(c2[0]/255.0, c2[1]/255.0, c2[2]/255.0, 1.0);
		glVertex3f((t2[0]*uScale)+border, ((camImageHeight-t2[1])*vScale)+border, 0-(d2/maxDepth));

		glColor4f(c3[0]/255.0, c3[1]/255.0, c3[2]/255.0, 1.0);
		glVertex3f((t3[0]*uScale)+border, ((camImageHeight-t3[1])*vScale)+border, 0-(d3/maxDepth));
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error brefore reading color buffer. %s\n", gluErrorString(errCode));

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_UNSIGNED_BYTE, width*height*4,result.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on reading color buffer. %s\n", gluErrorString(errCode));

	printf("finished reading buffer into cv image.\n"); fflush(stdout);

	cv::Mat rgbResult(height, width, CV_8UC3);

	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			cv::Vec4b srcPix = result.at<cv::Vec4b>(r,c);

			printf("src pixel (%d,%d,%d,%d)\n", srcPix[0], srcPix[1], srcPix[2], srcPix[3]);

			cv::Vec3b dstPix;
			dstPix[0] = srcPix[0];
			dstPix[1] = srcPix[1];
			dstPix[2] = srcPix[2];
			rgbResult.at<cv::Vec3b>(r,c) = dstPix;
		}

	return rgbResult;
}
