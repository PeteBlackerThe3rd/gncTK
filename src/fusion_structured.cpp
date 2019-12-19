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

fusion_structured.h

lidar camera fusion processing object
----------------------------------------------

This class fuses structured point clouds and images
into either textured mesh data or aligned RGB-D images.

-------------------------------------------------------------*/

#include <fusion_structured.h>
#include <pcl_ros/transforms.h>


gncTK::FusionStructured::FusionStructured()
{
	featureSize = 0.05;
	setIncidentAngleThreshold(86);
	glContextSetup = false;
}

gncTK::FusionStructured::~FusionStructured()
{
	// if the offscreen GL context was setup then destroy it
	if (glContextSetup)
	{
		glDeleteFramebuffers(1,&ssFbo);
		glDeleteRenderbuffers(1,&ssColorBuf);
		glDeleteRenderbuffers(1,&ssDepthBuf);
	}
}

void gncTK::FusionStructured::setIncidentAngleThreshold(float angle)
{
	incidentAngleThreshold = angle * (M_PI / 180.0);
}

gncTK::Mesh gncTK::FusionStructured::generateMesh(bool sensorOverlapOnly)
{
	Mesh newMesh;

	int chosenInputImage;
	if (inputImages.size() > 0)
		chosenInputImage = inputImages.size() / 2;
	int inputImageRows = inputImages[chosenInputImage].rows;
	int inputImageCols = inputImages[chosenInputImage].cols;

	if (inputCloud.points.size() == 0)
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : input point cloud not set or zero size.\n");
		return newMesh;
	}

	if (inputCloud.height == 1)
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : input point cloud isn't structured.\n");
		return newMesh;
	}

	if (inputImage.empty())
	{
		ROS_ERROR("gncTK [FusionStructured] Error trying to produce fusion mesh : no input image set.\n");
		return newMesh;
	}

	// find dynamic range of lidar intenisty values
	float intMin, intMax;
	intMin = intMax = inputCloud.points[0].intensity;
	for (int p=0; p<inputCloud.points.size(); ++p)
	{
		float intensity = inputCloud.points[p].intensity;
		if (intensity > intMax) intMax = intensity;
		if (intensity < intMin) intMin = intensity;
	}
	float intRange = intMax - intMin;

	// points transformed into the chosen camera's frame
	std::vector<Eigen::Vector3f> transformedVertices;
	transformedVertices.reserve(inputCloud.height * inputCloud.width);
	pcl::PointCloud<pcl::PointXYZI> transformedCloud;
	pcl_ros::transformPointCloud(inputCloud, transformedCloud, inputImageTFs[chosenInputImage]);

	// first add points from cloud to mesh as vertices
	newMesh.vertices.reserve(inputCloud.height * inputCloud.width);
	newMesh.vertexColors.reserve(inputCloud.height * inputCloud.width);
	for (int h=0; h<inputCloud.height; ++h)
		for (int w=0; w<inputCloud.width; ++w)
		{
			Eigen::Vector3f vert;
			vert << inputCloud.points[(h*inputCloud.width) + w].x,
					inputCloud.points[(h*inputCloud.width) + w].y,
					inputCloud.points[(h*inputCloud.width) + w].z;
			newMesh.vertices.push_back(vert);

			Eigen::Vector3f tVert;
			tVert << transformedCloud.points[(h*transformedCloud.width) + w].x,
					 transformedCloud.points[(h*transformedCloud.width) + w].y,
					 transformedCloud.points[(h*transformedCloud.width) + w].z;
			transformedVertices.push_back(tVert);

			// set the raw IR instensity value
			newMesh.vertexIntensities.push_back( inputCloud.points[(h*inputCloud.width) + w].intensity );

			// set the mesh intensity as the vertex color (greyscale)
			cv::Vec3b color;
			color[0] = ((inputCloud.points[(h*inputCloud.width) + w].intensity - intMin) / intRange) * 255.0;
			color[1] = color[0];
			color[2] = color[0];
			newMesh.vertexColors.push_back(color);
		}

	// for each vertex add a texture coordinate using the fusion function
	newMesh.texCoords = fusionFunction.projectPoints(transformedVertices);
	for (int t=0; t<newMesh.texCoords.size(); ++t)
	{
		float U = newMesh.texCoords[t][0] / (inputImageCols-1);
		float V = 1.0 - (newMesh.texCoords[t][1] / (inputImageRows-1) );

		// if a sensor overlap reconstruction is requested then invalidate this point if it
		// lies outside of the frame of the camera
		if (sensorOverlapOnly)
		{
			if (U < 0 || U > 1 ||
				V < 0 || V > 1)
			{
				newMesh.vertices[t][0] = 0;
				newMesh.vertices[t][1] = 0;
				newMesh.vertices[t][2] = 0;
				continue;
			}
		}

		if (U < 0) U = 0;
		if (U >= 1) U = 0.999;
		if (V < 0) V = 0;
		if (V >= 1) V = 0.999;

		newMesh.texCoords[t][0] = U;
		newMesh.texCoords[t][1] = V;
	}

	// store the mean triangle size per row
	std::vector<double> meanSizes;

	if (debug)
		printf("Created vertex, texturecoord and vertex color vectors.\n");

	// iterate through all sets of 4 adjacent vertices and add triangles where there are enough real vertices
	newMesh.triangles.reserve(2 * (inputCloud.width-1) * (inputCloud.height-1));
	for (int h=0; h<inputCloud.height-1; ++h)
	{
		double sizeTotal = 0;
		double triCount = 0;

		for (int w=0; w<inputCloud.width-1; ++w)
		{
			int index = (h*inputCloud.width) + w;
			bool tlV = isValid(inputCloud.points[index]);
			bool trV = isValid(inputCloud.points[index + 1]);
			bool blV = isValid(inputCloud.points[index + inputCloud.width]);
			bool brV = isValid(inputCloud.points[index + inputCloud.width + 1]);

			// add default triangles if enough vertices are valid
			if (tlV && trV && blV)
			{
				newMesh.triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1,
																  index + inputCloud.width,
																  index,
																  index+1,
																  index + inputCloud.width,
																  0));
				sizeTotal += gncTK::MeshAnalysis::triangleSize(&newMesh, newMesh.triangles.size()-1);
				++triCount;
			}
			if (trV && blV & brV)
			{
				newMesh.triangles.push_back(gncTK::Mesh::Triangle(index+1,
																  index+1+inputCloud.width,
																  index+inputCloud.width,
																  index+1,
																  index+1+inputCloud.width,
																  index+inputCloud.width,
																  0));
				sizeTotal += gncTK::MeshAnalysis::triangleSize(&newMesh, newMesh.triangles.size()-1);
				++triCount;
			}

			// add alternative triangles if only three vertices are valid
			if (tlV && trV && !blV && brV)
			{
				newMesh.triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1,
																  index+1+inputCloud.width,
																  index,
																  index+1,
																  index+1+inputCloud.width,
																  0));
				sizeTotal += gncTK::MeshAnalysis::triangleSize(&newMesh, newMesh.triangles.size()-1);
				++triCount;
			}
			if (tlV && !trV && blV && brV)
			{
				newMesh.triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1+inputCloud.width,
																  index+inputCloud.width,
																  index,
																  index+1+inputCloud.width,
																  index+inputCloud.width,
																  0));
				sizeTotal += gncTK::MeshAnalysis::triangleSize(&newMesh, newMesh.triangles.size()-1);
				++triCount;
			}
		}

		if (triCount > 0)
			meanSizes.push_back(sizeTotal / triCount);
		else
			meanSizes.push_back(NAN);
	}

	// set the input image as the only texture for this mesh
	if (inputImages.size() == 0)
	{
		newMesh.setSingleTexture(inputImage);
		ROS_WARN("Adding single camera image to mesh texture");
	}
	else
	{
		newMesh.setSingleTexture(inputImages[chosenInputImage]);
		ROS_WARN("Adding multiple camera images (%d) (single) to mesh", (int)inputImages.size());
	}

	// set the frame id
	newMesh.frameId = inputCloud.header.frame_id;

	// copy the sensor origin from the cloud to the new mesh
	newMesh.sensorOrigin[0] = inputCloud.sensor_origin_[0];
	newMesh.sensorOrigin[1] = inputCloud.sensor_origin_[1];
	newMesh.sensorOrigin[2] = inputCloud.sensor_origin_[2];

	int filtered = filterTrianglesOnIncidentAngle(&newMesh, incidentAngleThreshold);

	if (debug)
	{
		printf("Sensor origin was [%f %f %f]\n", newMesh.sensorOrigin[0], newMesh.sensorOrigin[1], newMesh.sensorOrigin[2]);
		printf("filtered %d triangles. %d remaining\n", filtered, (int)newMesh.triangles.size());
	}

	return newMesh;
}

void gncTK::FusionStructured::setupOffscreenGLBuffer(int width, int height)
{
	if (!glfwInit())
	{
		printf("Failed to Init GLFW!\n");
		printf("---[ Setup Failed ]---\n");
		return;
	}

	glfwWindowHint(GLFW_VISIBLE, 0);
	window = glfwCreateWindow(width, height, "Hidden window", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		printf("Failed to create hidden GLFW window\n");
		printf("---[ Setup Failed ]---\n");
		return;
	}
	glfwMakeContextCurrent(window);

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
	glGenFramebuffers(1, &ssFbo);
	glBindFramebuffer(GL_FRAMEBUFFER, ssFbo);

	glGenRenderbuffers(1, &ssColorBuf);
	glBindRenderbuffer(GL_RENDERBUFFER, ssColorBuf);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ssColorBuf);

	glGenRenderbuffers(1, &ssDepthBuf);
	glBindRenderbuffer(GL_RENDERBUFFER, ssDepthBuf);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ssDepthBuf);

	// setup projection with 1:1 mapping between geometry and buffer pixels
	glViewport(0,0, width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glOrtho(0, width, height, 0, 1.0,-1.0);

	glContextSetup = true;
}

std::vector<float> gncTK::FusionStructured::calculateGravityAngles(gncTK::Mesh mesh, Eigen::Vector3f gravity)
{
	gravity.normalize();
	std::vector<float> angles;

	for (int v=0; v<mesh.vertices.size(); ++v)
	{
		angles.push_back( acos(mesh.vertexNormals[v].dot(gravity)) );
	}

	return angles;
}

std::vector<Eigen::Vector3f> orientMeshVertices(gncTK::Mesh mesh, Eigen::Vector3f gravity)
{
	Eigen::Vector3f xAxis; xAxis << 1,0,0;
	Eigen::Vector3f yAxis; yAxis << 0,1,0;

	// construct rotation matrix with z axis pointing opposite to the gravity vector
	Eigen::Vector3f zVector = Eigen::Vector3f::Zero() - gravity;
	Eigen::Vector3f yVector = zVector.cross(xAxis);
	if (yVector == Eigen::Vector3f::Zero())
		yVector = zVector.cross(yAxis);
	Eigen::Vector3f xVector = zVector.cross(yVector);
	Eigen::Matrix3f gravityMatrix;
	gravityMatrix.block(0,0,3,1) = xVector;
	gravityMatrix.block(0,1,3,1) = yVector;
	gravityMatrix.block(0,2,3,1) = zVector;

	// rotate vertices into gravity aligned frame
	std::vector<Eigen::Vector3f> orthoVertices;
	for (int v=0; v<mesh.vertices.size(); ++v)
		orthoVertices.push_back( gravityMatrix.transpose() * mesh.vertices[v] );

	return orthoVertices;
}

std::vector<float> gncTK::FusionStructured::estimateVertexHeights(gncTK::Mesh mesh, float gridSize, Eigen::Vector3f gravity)
{
	std::vector<Eigen::Vector3f> orientedVerts = orientMeshVertices(mesh, gravity);

	std::vector<float> vertexHeights;
	Eigen::Matrix<float, 3, 2> extents = gncTK::Mesh::vertexExtents(orientedVerts);

	printf("got mesh extents.\n"); fflush(stdout);

	int gridRows = ceil((extents(0,1) - extents(0,0)) / gridSize);
	int gridCols = ceil((extents(1,1) - extents(1,0)) / gridSize);
	float gridRowOffset = extents(0,0);
	float gridColOffset = extents(1,0);

	printf("got grid size and offset.\n"); fflush(stdout);

	Eigen::MatrixXf gridHeights(gridRows, gridCols);
	float top = extents(2,1);
	gridHeights = Eigen::MatrixXf::Ones(gridRows, gridCols) * top;

	printf("created heights matrix.\n"); fflush(stdout);

	//Eigen::Matrix<int, gridRows, gridCels> gridCounts = Eigen::Dynamic2D::Zeros(gridRowsm gridCols);

	for (int v=0; v<orientedVerts.size(); ++v)
	{
		if (std::isnan(orientedVerts[v](0)))
			continue;

		int row = (orientedVerts[v](0) - gridRowOffset) / gridSize;
		int col = (orientedVerts[v](1) - gridColOffset) / gridSize;
		if (row >= gridRows || col >= gridCols || row<0 || col<0)
		{
			printf("Vert %d has a cell of [%d, %d] from pos (%f %f %f)\n", v, row,col,
					orientedVerts[v](0),
					orientedVerts[v](1),
					orientedVerts[v](2));
			continue;
		}

		if (gridHeights(row,col) > orientedVerts[v](2))
			gridHeights(row,col) = orientedVerts[v](2);
	}

	printf("got min heights.\n"); fflush(stdout);

	int s=3;
	Eigen::MatrixXf gridMeanHeights(gridRows, gridCols);
	for (int r=0; r<gridRows; ++r)
		for (int c=0; c<gridCols; ++c)
		{
			float sum=0;
			int count=0;

			for (int ro=0-s; ro<=s; ++ro)
				for (int co=0-s; co<=s; ++co)
				{
					int row = r+ro;
					int col = c+co;
					if (row>=0 && row<gridRows &&
						col>=0 && col<gridCols &&
						gridHeights(row,col) < top)
					{
						sum += gridHeights(row,col);
						++count;
					}
				}

			//if (count > 0)
				gridMeanHeights(r,c) = sum/count;
			//else
			//	gridMeanHeights(r,c) = NAN;
		}

	printf("got average heights.\n"); fflush(stdout);

	for (int v=0; v<mesh.vertices.size(); ++v)
	{
		if (std::isnan(orientedVerts[v](0)))
		{
			vertexHeights.push_back(0);
			continue;
		}

		float posRow = ((orientedVerts[v](0) - gridRowOffset) / gridSize) - 0.5;
		float posCol = ((orientedVerts[v](1) - gridColOffset) / gridSize) - 0.5;

		int row = posRow;
		int col = posCol;

		float rowResidual = posRow - row;
		float colResidual = posCol - col;

		float crWeight = (1-rowResidual)*(1-colResidual);
		float c1rWeight = (1-rowResidual)*(colResidual);
		float cr1Weight = (rowResidual)*(1-colResidual);
		float c1r1Weight = (rowResidual)*(colResidual);

		float weightSum = 0;
		float heightSum = 0;
		if (row < gridRows && col < gridCols && row>=0 && col>=0 && !std::isnan(gridMeanHeights(row,col)))
		{
			weightSum += crWeight;
			heightSum += crWeight * gridMeanHeights(row,col);
		}
		if (row < gridRows && (col+1) < gridCols && row>=0 && (col+1)>=0 && !std::isnan(gridMeanHeights(row,col+1)))
		{
			weightSum += c1rWeight;
			heightSum += c1rWeight * gridMeanHeights(row,col+1);
		}
		if ((row+1) < gridRows && col < gridCols && (row+1)>=0 && col>=0 && !std::isnan(gridMeanHeights(row+1,col)))
		{
			weightSum += cr1Weight;
			heightSum += cr1Weight * gridMeanHeights(row+1,col);
		}
		if ((row+1) < gridRows && (col+1) < gridCols && (row+1)>=0 && (col+1)>=0 && !std::isnan(gridMeanHeights(row+1,col+1)))
		{
			weightSum += c1r1Weight;
			heightSum += c1r1Weight * gridMeanHeights(row+1,col+1);
		}
		float meanHeight = heightSum / weightSum;

		/*if (row >= gridRows || col >= gridCols || row<0 || col<0)
		{
			printf("Vert %d has a cell of [%d, %d] from pos (%f %f %f)\n", v, row,col,
				   mesh.vertices[v](0),
				   mesh.vertices[v](1),
				   mesh.vertices[v](2));
			vertexHeights.push_back(0);
			continue;
		}*/

		if (std::isnan(meanHeight))
		{
			vertexHeights.push_back(0);
		}
		else
		{
			//vertexHeights.push_back(meanHeight);

			float height = orientedVerts[v](2) - meanHeight;
			if (height > 1) height = 1;

			vertexHeights.push_back(height);
		}
	}

	printf("got vert heights.\n"); fflush(stdout);

	return vertexHeights;
}

cv::Mat gncTK::FusionStructured::generateDepthImage(int resolutionFactor, int border, Eigen::Vector3f gravity)
{
	GLenum errCode;

	// first generate triangular mesh
	gncTK::Mesh mesh = generateMesh();

	// then generate surface normals
	mesh.calculateNormals();

	// detect default gravity vector value
	if (gravity(0) == 0 && gravity(1) == 0 && gravity(2) == 0)
		gravity << 0,0,-1;

	// generate gravity angle values for each vertex
	std::vector<float> angles = calculateGravityAngles(mesh, gravity);

	// calculate vertex depth values and find maximum depth
	float maxDepth = 0;
	Eigen::Vector3f cameraPos = fusionFunction.getCameraLocation();
	std::vector<float> vertexDepths;
	for (int v=0; v<mesh.vertices.size(); ++v)
	{
		float depth = (cameraPos-mesh.vertices[v]).norm();
		vertexDepths.push_back(depth);
		if (depth > maxDepth)
			maxDepth = depth;
	}

	// estimate vertex heights
	std::vector<float> vertexHeights = estimateVertexHeights(mesh, 0.5, gravity);

	printf("got %d height values and %d verts.\n", (int)vertexHeights.size(), (int)mesh.vertices.size());
	fflush(stdout);

	// calculate IR albedo values from intensity and geometry
	std::vector<float> albedos;
	for (int v=0; v<mesh.vertices.size(); ++v)
	{
		Eigen::Vector3f lightDir = mesh.vertices[v];
		Eigen::Vector3f normal = mesh.vertexNormals[v];

		float albedo = mesh.vertexIntensities[v];// * vertexDepths[v] * vertexDepths[v];
		albedo /= lightDir.dot(normal) / lightDir.norm();

		albedos.push_back(albedo);
	}

	// remove outliers from albedo
	float alMean=0;
	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(albedos[v]))
			alMean += albedos[v];
	alMean /= albedos.size();
	float alStdDev = 0;
	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(albedos[v]))
			alStdDev += fabs(albedos[v] - alMean);
	alStdDev /= albedos.size();

	printf("albedo mean=[%f] stdDev=[%f]\n", alMean, alStdDev);

	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(albedos[v]))
		{
			if (albedos[v] - alMean > alStdDev*3)
				albedos[v] = alMean + alStdDev*3;
			if (alMean - albedos[v] > alStdDev*3)
				albedos[v] = alMean - alStdDev*3;
		}

	// remove outliers from raw intensity values (specular highlights)
	float iMean=0;
	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(mesh.vertexIntensities[v]))
			iMean += mesh.vertexIntensities[v];
	iMean /= mesh.vertexIntensities.size();
	float iStdDev = 0;
	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(mesh.vertexIntensities[v]))
			iStdDev += fabs(mesh.vertexIntensities[v] - iMean);
	iStdDev /= mesh.vertexIntensities.size();

	printf("intensity mean=[%f] stdDev=[%f]\n", iMean, iStdDev);

	for (int v=0; v<mesh.vertices.size(); ++v)
		if (!std::isnan(mesh.vertexIntensities[v]))
		{
			if (mesh.vertexIntensities[v] - iMean > iStdDev*3)
				mesh.vertexIntensities[v] = iMean + iStdDev*3;
			if (iMean - mesh.vertexIntensities[v] > iStdDev*3)
				mesh.vertexIntensities[v] = iMean - iStdDev*3;
		}

	// setup GL rendering context
	int width = (inputImage.cols / resolutionFactor) + border;
	int height = (inputImage.rows / resolutionFactor) + border;
	setupOffscreenGLBuffer(width, height);

	cv::Mat partialResult(height,width, CV_32FC4, cv::Scalar(0,1,0,0));
	cv::Mat result(height,width, CV_32FC4, cv::Scalar(0,1,0,0));

	// render mesh into buffer
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepthf(1.0f);
	//glDepthRange(1,0); // reverse depth buffer

	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	// pre-calculate UV scaling factors
	float uScale = inputImage.cols/resolutionFactor;
	float vScale = inputImage.rows/resolutionFactor;

	// render depth, height and angle geometry to frame buffer
	glClearColor(-1.0, -1000.0, 10.0, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<mesh.triangles.size(); ++t)
	{
		float d1 = vertexDepths[mesh.triangles[t].v1];
		float d2 = vertexDepths[mesh.triangles[t].v2];
		float d3 = vertexDepths[mesh.triangles[t].v3];
		float a1 = angles[mesh.triangles[t].v1];
		float a2 = angles[mesh.triangles[t].v2];
		float a3 = angles[mesh.triangles[t].v3];
		float h1 = vertexHeights[mesh.triangles[t].v1];
		float h2 = vertexHeights[mesh.triangles[t].v2];
		float h3 = vertexHeights[mesh.triangles[t].v3];

		Eigen::Vector2f t1 = mesh.texCoords[mesh.triangles[t].t1];
		Eigen::Vector2f t2 = mesh.texCoords[mesh.triangles[t].t2];
		Eigen::Vector2f t3 = mesh.texCoords[mesh.triangles[t].t3];

		glColor4f(d1, h1, a1, 1.0);
		glVertex3f((t1[0]*uScale)+border, (t1[1]*vScale)+border, d1/maxDepth);

		glColor4f(d2, h2, a2, 1.0);
		glVertex3f((t2[0]*uScale)+border, (t2[1]*vScale)+border, d2/maxDepth);

		glColor4f(d3, h3, a3, 1.0);
		glVertex3f((t3[0]*uScale)+border, (t3[1]*vScale)+border, d3/maxDepth);
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error brefore reading color buffer. %s\n", gluErrorString(errCode));

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,partialResult.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on reading color buffer. %s\n", gluErrorString(errCode));

	// copy channels into final output
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			float depth = partialResult.at<cv::Vec4f>(r,c)[0];
			if (depth == -1.0) depth = NAN;
			result.at<cv::Vec4f>(r,c)[0] = depth;

			float height = partialResult.at<cv::Vec4f>(r,c)[1];
			if (height == -1000) height = NAN;
			result.at<cv::Vec4f>(r,c)[1] = height;

			float angle = partialResult.at<cv::Vec4f>(r,c)[2];
			if (angle == 10) angle = NAN;
			result.at<cv::Vec4f>(r,c)[2] = angle;
		}

	// render intensity and normalised intensity geometry to frame buffer
	glClearColor(-1.0, -10.0, -10.0, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<mesh.triangles.size(); ++t)
	{
		float d1 = vertexDepths[mesh.triangles[t].v1];
		float d2 = vertexDepths[mesh.triangles[t].v2];
		float d3 = vertexDepths[mesh.triangles[t].v3];
		float i1 = mesh.vertexIntensities[mesh.triangles[t].v1];
		float i2 = mesh.vertexIntensities[mesh.triangles[t].v2];
		float i3 = mesh.vertexIntensities[mesh.triangles[t].v3];
		//float al1 = albedos[mesh.triangles[t].v1];
		//float al2 = albedos[mesh.triangles[t].v2];
		//float al3 = albedos[mesh.triangles[t].v3];
		float az1 = 0;//azimuth[mesh.triangles[t].v1];
		float az2 = 0;//azimuth[mesh.triangles[t].v2];
		float az3 = 0;//azimuth[mesh.triangles[t].v3];
		float el1 = 0;//elevation[mesh.triangles[t].v1];
		float el2 = 0;//[mesh.triangles[t].v2];
		float el3 = 0;//elevation[mesh.triangles[t].v3];

		Eigen::Vector2f t1 = mesh.texCoords[mesh.triangles[t].t1];
		Eigen::Vector2f t2 = mesh.texCoords[mesh.triangles[t].t2];
		Eigen::Vector2f t3 = mesh.texCoords[mesh.triangles[t].t3];

		glColor4f(i1, az1, el1, 1.0);
		glVertex3f((t1[0]*uScale)+border, (t1[1]*vScale)+border, d1/maxDepth);

		glColor4f(i2, az2, el2, 1.0);
		glVertex3f((t2[0]*uScale)+border, (t2[1]*vScale)+border, d2/maxDepth);

		glColor4f(i3, az3, el3, 1.0);
		glVertex3f((t3[0]*uScale)+border, (t3[1]*vScale)+border, d3/maxDepth);
	}
	glEnd();

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,partialResult.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on reading color buffer. %s\n", gluErrorString(errCode));

	// copy channels into final output
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			float intensity = partialResult.at<cv::Vec4f>(r,c)[0];
			if (intensity == -1) intensity = NAN;
			result.at<cv::Vec4f>(r,c)[3] = intensity;

			//result.at<cv::Vec4f>(r,c)[4] = partialResult.at<cv::Vec4f>(r,c)[1];

			//result.at<cv::Vec4f>(r,c)[5] = partialResult.at<cv::Vec4f>(r,c)[2];
		}

	printf("finished reading buffer into cv image.\n"); fflush(stdout);

	return result;
}
