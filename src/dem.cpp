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

dem.h

Digital Elevation Map processing object
-----------------------------------------

This object provides a set of static methods that operate
on ROS GridMap objects to produce dem's from various sources.
Triangle Mesh's loaded from DEM files.
They can also have their cost-map (goodness) layers calculated
using a variety of algorithms.

-------------------------------------------------------------*/
#include <utils.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <dem.h>

// Initialise static members
//dynamic_reconfigure::Server<smart_fusion_sensor::CostMapGenConfig> *gncTK::DEM::configServer = NULL;
gncTK::DEM::CostmapSettingsMER gncTK::DEM::costmapSettings;

GLFWwindow* gncTK::DEM::window = NULL;
bool gncTK::DEM::glContextSetup = false;
GLuint gncTK::DEM::ssFbo=0, gncTK::DEM::ssColorBuf=0, gncTK::DEM::ssDepthBuf=0;
int gncTK::DEM::offscreenBufSize = 200;

void glfw_error_callback(int error, const char* description)
{
  printf("GLFW error: %s\n", description);
}

void gncTK::DEM::setupOffscreenGL()
{
	if (!glContextSetup)
	{
		if (!glfwInit())
		{
			printf("Failed to Init GLFW!\n");
			printf("---[ Setup Failed ]---\n");
			return;
		}

		glfwWindowHint(GLFW_VISIBLE, 0);
		window = glfwCreateWindow(1024, 1024, "Hidden window", NULL, NULL);
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

		//printf("OpenGL Context Created OK.\n");
		//printf("GL device : %s\n", glGetString(GL_RENDERER));
		//printf("GL device vendor : %s\n", glGetString(GL_VENDOR));
		//printf("OpenGL version : %s\n", glGetString(GL_VERSION));

		// create frame buffer with single sampled color and depth attached render buffers
		glGenFramebuffers(1, &ssFbo);
		glBindFramebuffer(GL_FRAMEBUFFER, ssFbo);

		glGenRenderbuffers(1, &ssColorBuf);
		glBindRenderbuffer(GL_RENDERBUFFER, ssColorBuf);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, offscreenBufSize, offscreenBufSize);
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ssColorBuf);

		glGenRenderbuffers(1, &ssDepthBuf);
		glBindRenderbuffer(GL_RENDERBUFFER, ssDepthBuf);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, offscreenBufSize, offscreenBufSize);
		glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ssDepthBuf);

		// setup projection with 1:1 mapping between geometry and buffer pixels
		glViewport(0,0, offscreenBufSize,offscreenBufSize);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glOrtho(0, offscreenBufSize, offscreenBufSize, 0, 1.0,-1.0);

		glContextSetup = true;
	}
}


void gncTK::DEM::cleanupOffscreenGL()
{
	if (glContextSetup)
	{
		glDeleteFramebuffers(1,&ssFbo);
		glDeleteRenderbuffers(1,&ssColorBuf);
		glDeleteRenderbuffers(1,&ssDepthBuf);
	}
}

cv::Mat gncTK::DEM::fillElevationBuffer2(gncTK::Mesh sourceMesh,
										Eigen::Array2i start,
										Eigen::Array2i size,
										grid_map::Length mapSize,
										grid_map::Position mapPosition,
										float gridSpacing)
{
	cv::Mat buf(size(0), size(1), CV_32F);

	for (int row=0; row<size(0); ++row)
		for (int col=0; col<size(1); ++col)
			buf.at<float>(row,col) = (0 - (start(0) + start(1)*2)) / 100;//NAN;

	for (int v=0; v<sourceMesh.vertices.size(); ++v)
	{
		Eigen::Vector2i cell;

		cell(0) = ((mapPosition(0) - sourceMesh.vertices[v](0) + (mapSize(0)/2)) / gridSpacing);
		cell(1) = ((mapPosition(1) - sourceMesh.vertices[v](1) + (mapSize(1)/2)) / gridSpacing);

		if (cell(0) >= start(0) && cell(0) < start(0)+size(0) &&
			cell(1) >= start(1) && cell(1) < start(1)+size(1))
			buf.at<float>(cell(0) - start(0), cell(1) - start(1)) = sourceMesh.vertices[v](2);
	}

	return buf;
}

cv::Mat gncTK::DEM::fillElevationBuffer(gncTK::Mesh sourceMesh,
										Eigen::Array2i start,
										Eigen::Array2i size,
										grid_map::Length mapSize,
										grid_map::Position mapPosition,
										float gridSpacing,
										Eigen::Matrix<float, 3, 2> meshExtents,
										bool findMax)
{
	cv::Mat buf(offscreenBufSize, offscreenBufSize, CV_32F);
	setupOffscreenGL();
	GLenum errCode;

	float zMin = meshExtents(2,0);
	float zMax = meshExtents(2,1);

	// setup depth buffering
	glEnable(GL_DEPTH_TEST);
	if (!findMax)
	{
		glDepthFunc(GL_GREATER);
		glClearDepthf(0.0f);
	}
	else
	{
		glDepthFunc(GL_LESS);
		glClearDepthf(1.0f);
	}
	glDepthRange(1,0); // reverse depth buffer

	glClearColor(-100.0, -100.0, -100.0, 1.0f);
	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	// reset off screen buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on clearing buffers. %s\n", gluErrorString(errCode));

	// render geometry to frame buffers
	glBegin(GL_TRIANGLES);
	for (int v=0; v<sourceMesh.triangles.size(); ++v)
	{
		Eigen::Vector3f v1 = sourceMesh.vertices[sourceMesh.triangles[v].v1];
		Eigen::Vector3f v2 = sourceMesh.vertices[sourceMesh.triangles[v].v2];
		Eigen::Vector3f v3 = sourceMesh.vertices[sourceMesh.triangles[v].v3];

		Eigen::Vector3f v1t,v2t,v3t;

		v1t(0) = ((mapPosition(0) - v1(0) + (mapSize(0)/2)) / gridSpacing) - start(0);
		v1t(1) = ((mapPosition(1) - v1(1) + (mapSize(1)/2)) / gridSpacing) - start(1);
		v1t(2) = (v1(2) - zMin) / (zMax-zMin);
		v2t(0) = ((mapPosition(0) - v2(0) + (mapSize(0)/2)) / gridSpacing) - start(0);
		v2t(1) = ((mapPosition(1) - v2(1) + (mapSize(1)/2)) / gridSpacing) - start(1);
		v2t(2) = (v2(2) - zMin) / (zMax-zMin);
		v3t(0) = ((mapPosition(0) - v3(0) + (mapSize(0)/2)) / gridSpacing) - start(0);
		v3t(1) = ((mapPosition(1) - v3(1) + (mapSize(1)/2)) / gridSpacing) - start(1);
		v3t(2) = (v3(2) - zMin) / (zMax-zMin);

		glColor4f(v1(2), v1(2), v1(2), 1.0);
		glVertex3f(v1t(0), v1t(1), v1t(2));

		glColor4f(v2(2), v2(2), v2(2), 1.0);
		glVertex3f(v2t(0), v2t(1), v2t(2));

		glColor4f(v3(2), v3(2), v3(2), 1.0);
		glVertex3f(v3t(0), v3t(1), v3t(2));
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error brefore reading color buffer. %s\n", gluErrorString(errCode));

	// read buffer out
	glReadnPixels(0,0,offscreenBufSize,offscreenBufSize,GL_RED,GL_FLOAT,offscreenBufSize*offscreenBufSize*4,buf.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on reading color buffer. %s\n", gluErrorString(errCode));

	cv::Mat bufOut(offscreenBufSize, offscreenBufSize, CV_32F);

	for (int r=0; r<buf.rows; ++r)
		for (int c=0; c<buf.cols; ++c)
		{
			float e = buf.at<float>(r,c);
			if (e > -99.0)
				bufOut.at<float>(c,offscreenBufSize - r - 1) = e;  // rows and cols are rotated due to CV and GL having different screen coordinate systems!
			else
				bufOut.at<float>(c,offscreenBufSize - r - 1) = NAN;
		}

	return bufOut;
}

grid_map::GridMap* gncTK::DEM::generateFromMesh(gncTK::Mesh sourceMesh,
												float gridSpacing,
												float safeZoneDia,
												float safeZoneBlendDia,
												float safeZoneElevation,
												grid_map::Position fixedPosition)
{
	//printf("starting to generate DEM.\n");
	//fflush(stdout);

	//ROS_WARN("Creating DEM from Mesh with sensor origin of (%f %f %f)",
	//		 sourceMesh.sensorOrigin[0],
	//		 sourceMesh.sensorOrigin[1],
	//		 sourceMesh.sensorOrigin[2]);

	Eigen::Matrix<float, 3, 2> extents = sourceMesh.getExtents();

	// add the range of the safe area into the extents
	for (int d=0; d<2; ++d)
	{
		extents(d,0) -= safeZoneDia * 2.0;
		extents(d,1) += safeZoneDia * 2.0;

		/*if (sourceMesh.sensorOrigin[d] + safeZoneDia > extents(d,1))
			extents(d,1) = sourceMesh.sensorOrigin[d] + safeZoneDia;

		if (sourceMesh.sensorOrigin[d] - safeZoneDia < extents(d,0))
			extents(d,0) = sourceMesh.sensorOrigin[d] - safeZoneDia;*/
	}

	float sizeX = fabs(extents(0,1) - extents(0,0));
	float sizeY = fabs(extents(1,1) - extents(1,0));
	float sizeZ = fabs(extents(2,1) - extents(2,0));

	//ROS_WARN("Mesh bounding box size (%f %f %f)",
	//		 sizeX, sizeY, sizeZ);

	// if the mesh is over a km wide then don't attempt to generate the DEM
	if (sizeX > 1000 || sizeY > 1000 || sizeZ > 1000)
	{
		ROS_ERROR("Cannot generate DEM from Mesh. Size of (%f x %f x %f) km is too large!",
				  sizeX/1000,
				  sizeY/1000,
				  sizeX/1000);
		return NULL;
	}

	//grid_map::Length mapLength(extents(0,1) - extents(0,0),
	//						   extents(1,1) - extents(1,0));
	grid_map::Length mapLength(sizeX,
							   sizeY);

	//std::cout << "mesh extents are : \n" << extents << "\n-----\n";
	//ROS_WARN("Extents elements are: [%f %f %f] - [%f %f %f]",
	//		 extents(0,0), extents(1,0), extents(2,0),
	//		 extents(0,1), extents(1,1), extents(2,1));
	//std::cout << "grid spacing is : " << gridSpacing << "\n";
	//fflush(stdout);

	grid_map::GridMap *DTM = new grid_map::GridMap({"elevation", "elevationMax", "elevationMin", "confidence"});

	// if not fixed centre point was given then dynamically centre the map on the mesh given
	if (std::isnan(fixedPosition[0]))
	{
		DTM->setGeometry(mapLength,
						 gridSpacing,
						 grid_map::Position(((extents(0,0)+extents(0,1))/2) ,
											((extents(1,0)+extents(1,1))/2) ));
	}
	else // fixed position given, calculate required size and setup geometry
	{
		//grid_map::Length requiredSize;
		//printf("fixed map center set : (%f, %f)",
		//	   fixedPosition[0], fixedPosition[1]);
		//printf("Updating map length from : (%f, %f)\n",
		//	   mapLength[0], mapLength[1]);

		mapLength[0] = std::max( fabs(fixedPosition[0] - extents(0,0)),
							   	    fabs(fixedPosition[0] - extents(0,1)) ) * 2;
		mapLength[1] = std::max( fabs(fixedPosition[1] - extents(1,0)),
							   	    fabs(fixedPosition[1] - extents(1,1)) ) * 2;


		//printf("New map length : (%f, %f)\n",
		//	   mapLength[0], mapLength[1]);
		//fflush(stdout);

		DTM->setGeometry(mapLength,
						 gridSpacing,
						 fixedPosition);
	}
	DTM->setFrameId(sourceMesh.frameId);
	DTM->setTimestamp(ros::Time::now().toNSec());

	//printf("Created gridMap object.\n");fflush(stdout);
	//fflush(stdout);

	// loop through each 1000x1000 cell block of the gridmap adding triangles using openGL
	grid_map::Size mapSize = DTM->getSize();

	//printf("adding triangles into map blocks:\n");fflush(stdout);
	int blockCount = 0;

	for (int row=0; row<mapSize(0); row += offscreenBufSize)
		for (int col=0; col<mapSize(1); col += offscreenBufSize)
		{
			//printf("Adding Block %d\n", blockCount++); fflush(stdout);

			Eigen::Array2i start; start << row, col;
			Eigen::Array2i size; size << std::min(row+offscreenBufSize, mapSize(0)) - row,
										 std::min(col+offscreenBufSize, mapSize(1)) - col;

			cv::Mat elevationMax = fillElevationBuffer(sourceMesh,
													   start,
													   size,
													   mapLength,
													   DTM->getPosition(),
													   gridSpacing,
													   extents,
													   true);
			cv::Mat elevationMin = fillElevationBuffer(sourceMesh,
													   start,
													   size,
													   mapLength,
													   DTM->getPosition(),
													   gridSpacing,
													   extents,
													   false);

			grid_map::Index pos;
			for (pos[0]=0; pos[0]<size(0); ++pos[0])
				for (pos[1]=0; pos[1]<size(1); ++pos[1])
				{
					float max = elevationMax.at<float>(pos(0), pos(1));
					float min = elevationMin.at<float>(pos(0), pos(1));

					if (!std::isnan(max) && !std::isnan(min))
					{
						DTM->at("elevationMax", pos + start) = max;
						DTM->at("elevationMin", pos + start) = min;
						DTM->at("elevation", pos + start) = (min + max) / 2;
					}
				}
		}

	// if the safe zone diameter is greater than zero then add the safe zone
	if (safeZoneDia > 0.0)
	{
		// if the safe zone elevation was set to auto (NAN) then find it
		if (std::isnan(safeZoneElevation))
		{
			ROS_WARN("Automatically finding elevation of safe zone");

			safeZoneElevation = 0;
			int overlapCount = 0;

			// calculate the automatic safe zone elevation by finding the average
			// elevation of cells which overlap the save zone
			for (grid_map::GridMapIterator it(*DTM); !it.isPastEnd(); ++it)
			{
				// calculate the distance from the sensor origin (int 2D)
				Eigen::Vector2d location;
				DTM->getPosition(*it, location);
				float distance = (location.cast<float>() - sourceMesh.sensorOrigin.head(2) ).norm();

				// if this location is within the blend area
				if (distance > safeZoneDia && distance <= safeZoneBlendDia)
				{
					float elevation = DTM->at("elevation", *it);
					if (!std::isnan(elevation))
					{
						safeZoneElevation += elevation;
						++overlapCount;
					}
				}
			}

			if (overlapCount > 0)
				safeZoneElevation /= overlapCount;
			else
				printf("Warning no valid elevation measurements in safe zone overlap!\n");
		}

		ROS_WARN("Adding safe zone to gridmap with an elevation of %f m", safeZoneElevation);

		for (grid_map::GridMapIterator it(*DTM); !it.isPastEnd(); ++it)
		{
			try
			{
				// calculate the distance from the sensor origin (int 2D)
				Eigen::Vector2d location;
				DTM->getPosition(*it, location);
				float distance = (location.cast<float>() - sourceMesh.sensorOrigin.head(2) ).norm();

				if (distance <= safeZoneDia)
				{
					DTM->at("elevation", *it) = safeZoneElevation;
					DTM->at("elevationMin", *it) = safeZoneElevation;
					DTM->at("elevationMax", *it) = safeZoneElevation;
				}
				else if (distance <= safeZoneBlendDia)
				{
					float ratio = (distance-safeZoneDia) / (safeZoneBlendDia-safeZoneDia);
					float elevation = DTM->at("elevation", *it);
					float elevationMin = DTM->at("elevationMin", *it);
					float elevationMax = DTM->at("elevationMax", *it);

					if (!std::isnan(elevation))
						DTM->at("elevation", *it) = elevation*ratio + safeZoneElevation*(1-ratio);
					if (!std::isnan(elevationMin))
						DTM->at("elevationMin", *it) = elevationMin*ratio + safeZoneElevation*(1-ratio);
					if (!std::isnan(elevationMax))
						DTM->at("elevationMax", *it) = elevationMax*ratio + safeZoneElevation*(1-ratio);
				}
			}
			catch (std::bad_alloc e)
			{
				printf("bad alloc caught.\n");
			}
		}
	}

	return DTM;
}

gncTK::Mesh* gncTK::DEM::gridMapToMesh(grid_map::GridMap *map,
									  std::string heightLayer,
									  std::string colorLayer)
{
	gncTK::Mesh *newMesh = new gncTK::Mesh();
	newMesh->frameId = map->getFrameId();

	Eigen::Vector3f nanVert; nanVert << NAN,NAN,NAN;
	float min=-666,max=-666;
	bool firstColor = true;

	// create grid of vertices in mesh
	grid_map::Size mapSize = map->getSize();
	Eigen::Array2i pos;
	for (pos(0)=0; pos(0)<mapSize(0); ++pos(0))
		for (pos(1)=0; pos(1)<mapSize(1); ++pos(1))
		{
			Eigen::Vector3d vertPos;

			if (map->getPosition3(heightLayer, pos, vertPos))
			{
				newMesh->vertices.push_back(vertPos.cast<float>());
				float color = map->at(colorLayer, pos);

				if (firstColor && std::isfinite(color))
				{
					min = max = color;
					firstColor = false;
				}
				else
				{
					if (color < min) min = color;
					if (color > max) max = color;
				}
			}
			else
			{
				newMesh->vertices.push_back(nanVert);
			}
		}

	ROS_ERROR("gridMapToMesh color range is %f - %f", min, max);

	// add vertex colors to the mesh
	for (pos(0)=0; pos(0)<mapSize(0); ++pos(0))
		for (pos(1)=0; pos(1)<mapSize(1); ++pos(1))
		{
			Eigen::Vector3d vertPos;

			if (map->getPosition3(heightLayer, pos, vertPos))
			{
				float color = map->at(colorLayer, pos);
				newMesh->vertexColors.push_back(gncTK::Utils::rainbow( (color-min)/(max-min) ));
			}
			else
				newMesh->vertexColors.push_back(cv::Vec3b(0,0,0));
		}

	// add triangle lattice over valid vertices
	newMesh->triangles.reserve(2 * (mapSize(0)-1) * (mapSize(1)-1));
	for (int h=0; h<mapSize(0)-1; ++h)
	{
		for (int w=0; w<mapSize(1)-1; ++w)
		{
			int index = (h*mapSize(1)) + w;
			bool tlV = !std::isnan(newMesh->vertices[index][0]);
			bool trV = !std::isnan(newMesh->vertices[index + 1][0]);
			bool blV = !std::isnan(newMesh->vertices[index + mapSize(1)][0]);
			bool brV = !std::isnan(newMesh->vertices[index + mapSize(1) +1][0]);

			// add default triangles if enough vertices are valid
			if (tlV && trV && blV)
			{
				newMesh->triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1,
																  index+mapSize(1),
																  0));

			}
			if (trV && blV && brV)
			{
				newMesh->triangles.push_back(gncTK::Mesh::Triangle(index+1,
																  index+1+mapSize(1),
																  index+mapSize(1),
																  0));
			}

			// add alternative triangles if only three vertices are valid
			if (tlV && trV && !blV && brV)
			{
				newMesh->triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1,
																  index+1+mapSize(1),
																  0));
			}
			if (tlV && !trV && blV && brV)
			{
				newMesh->triangles.push_back(gncTK::Mesh::Triangle(index,
																  index+1+mapSize(1),
																  index+mapSize(1),
																  0));
			}
		}
	}

	return newMesh;
}

Eigen::MatrixXi gncTK::DEM::GenerateRoverMaskMatrix(float gridSize, float roverSize)
{
	int matrixSize = floor((roverSize/gridSize)/2)*2 + 3;
	Eigen::MatrixXi mask(matrixSize,matrixSize);

	for (int r=0; r<matrixSize; ++r)
		for (int c=0; c<matrixSize; ++c)
		{
			float dX = ((r+0.5) - (matrixSize/2.0)) * gridSize;
			float dY = ((c+0.5) - (matrixSize/2.0)) * gridSize;
			float d = sqrt(dX*dX + dY*dY);

			if (d <= roverSize/2)
				mask(r,c) = 1;
			else
				mask(r,c) = 0;
		}

	return mask;
}

Eigen::MatrixXf gncTK::DEM::getElevationSamples(grid_map::GridMap* map,
												Eigen::MatrixXi footprint,
												Eigen::Vector2d center,
												std::string layer)
{
	int offset = (footprint.rows()-1) / 2;
	Eigen::MatrixXf elevations(footprint.rows(), footprint.cols());

	for (int r=0; r<footprint.rows(); ++r)
		for (int c=0; c<footprint.cols(); ++c)
		{
			if (footprint(r,c) == 1)
			{
				Eigen::Vector2d position = center;
				position(0) += (c-offset) * map->getResolution();
				position(1) += (r-offset) * map->getResolution();

				try
				{
					elevations(r,c) = map->atPosition(layer, position);
				}
				catch (std::out_of_range e)
				{
					elevations(r,c) = NAN;
				}
			}
			else
			{
				elevations(r,c) = NAN;
			}
		}

	return elevations;
}

float gncTK::DEM::calculateStepCost(Eigen::MatrixXf elevations)
{
	float maxStep = 0;

	for (int r=1; r<elevations.rows()-1; ++r)
		for (int c=1; c<elevations.cols()-1; ++c)
		{
			// compare with neighbouring cells
			for (int dr=-1; dr<=1; ++dr)
				for (int dc=-1; dc<=1; ++dc)
					if (dr != 0 && dc != 0)
					{
						float delta = fabs(elevations(r+dr, c+dc) - elevations(r, c));
						if (delta > maxStep) maxStep = delta;

						/*if (!std::isnan(elevations(r+dr, c+dc, 0)) && !std::isnan(elevations(r, c, 1)))
						{
							float delta = fabs(elevations(r+dr, c+dc, 0) - elevations(r, c, 1));
							if (delta > maxStep) maxStep = delta;
						}
						if (!std::isnan(elevations(r+dr, c+dc, 1)) && !std::isnan(elevations(r, c, 0)))
						{
							float delta = fabs(elevations(r+dr, c+dc, 1) - elevations(r, c, 0));
							if (delta > maxStep) maxStep = delta;
						}*/
					}
		}

	return maxStep;
}

Eigen::Matrix<float, 3, 2> gncTK::DEM::calculateBestFitPlane(Eigen::MatrixXf elevations, float gridSize)
{
	Eigen::Matrix<float, 3, 2> plane;
	plane << 0,0, 0,0, 1,1;

	// generate array of 3D points
	Eigen::MatrixXf points(3, elevations.rows() * elevations.cols());
	int pointCount = 0;
	for (int r=0; r<elevations.rows(); ++r)
		for (int c=0 ;c<elevations.cols(); ++c)
		{
			if (!std::isnan(elevations(r,c)))
			{
				points(0, pointCount) = r * gridSize;
				points(1, pointCount) = c * gridSize;
				points(2, pointCount) = elevations(r,c);
				++pointCount;
			}
		}
	if (pointCount == 0)
	{
		return plane;
	}
	points.conservativeResize(3,pointCount);


	// find mean and subtract from data
	// find the centroid of the point cloud
	Eigen::Vector3f centroid; centroid << 0, 0, 0;
	for (int p=0; p<pointCount; ++p)
	{
		centroid += points.block(0,p,3,1);
	}
	centroid /= pointCount;
	//return plane;
	for (int p=0; p<pointCount; ++p)
	{
		points.block(0,p,3,1) -= centroid;
	}

	// setup and compute a jacobian estimator of the SVD solution
	Eigen::JacobiSVD<Eigen::MatrixXf> jacobiSolver(pointCount, 3, Eigen::ComputeFullU | Eigen::ComputeThinV);
	Eigen::JacobiSVD<Eigen::MatrixXf> result = jacobiSolver.compute(points, Eigen::ComputeFullU | Eigen::ComputeThinV);

	// extract the singular vector and U matrix from the result
	Eigen::VectorXf singularValues = result.singularValues();
	Eigen::MatrixXf uMatrix = result.matrixU();

	// find the column vector of the U matrix which corresponds to the lowest singular value
	// this is the normal vector of the best fit plane
	int column = uMatrix.cols() - 1;
	Eigen::Vector3f normal = uMatrix.block(0,column,3,1);

	// if normal is pointing down reverse it
	if (normal(2) < 0)
		normal = Eigen::Vector3f::Zero() - normal;

	// return a 3x2 matrix containing the normal and point in plane
	plane.block(0,0,3,1) = normal;
	plane.block(0,1,3,1) = centroid;
	return plane;
}

float gncTK::DEM::findMaxPlaneResidual(Eigen::MatrixXf elevations, Eigen::Matrix<float, 3, 2> plane, float gridSize)
{
	float maxR = 0;
	Eigen::Vector3f normal = plane.block(0,0,3,1);
	Eigen::Vector3f centroid = plane.block(0,1,3,1);

	for (int r=1; r<elevations.rows()-1; ++r)
		for (int c=1; c<elevations.cols()-1; ++c)
			if (!std::isnan(elevations(r,c)))
			{
				Eigen::Vector3f p;
				p << r * gridSize, c * gridSize, elevations(r,c);
				float dist = fabs((p-centroid).dot(normal));
				if (dist > maxR) maxR = dist;
			}

	return maxR;
}

int gncTK::DEM::countNonNans(Eigen::MatrixXf elevations)
{
	int count = 0;
	for (int r=1; r<elevations.rows()-1; ++r)
			for (int c=1; c<elevations.cols()-1; ++c)
				if (!std::isnan(elevations(r,c)))
					++count;
	return count;
}

void gncTK::DEM::generateCostMapFast(grid_map::GridMap* map, std::string layer)
{
	// add costmap layers
	map->add("cost");
	map->add("slope");
	map->add("roughness");
	map->add("step");
	map->add("border");

	// generate mask matrix for rover footprint
	Eigen::MatrixXf footprint = GenerateRoverMaskMatrix(map->getResolution(), costmapSettings.roverWidth).cast<float>();
	int footprintArea = footprint.sum();
	int border = (footprint.rows()/2) + 1;

	// replace zeros with NANs in footprint
	for (int r=0; r<footprint.rows(); ++r)
		for (int c=0; c<footprint.cols(); ++c)
			if (footprint(r,c) == 0)
				footprint(r,c) = NAN;

	// define vertical direction as opposite of local gravity vector
	Eigen::Vector3f vertical = Eigen::Vector3f::Zero() - costmapSettings.gravity; //vertical << 0, 0, 1;

	// copy grid map elevation values into single eigen matrix
	Eigen::Array2i mapSize = map->getSize();
	Eigen::MatrixXf elevationMap(mapSize(0) + 2*border, mapSize(1) + 2*border );
	//Eigen::MatrixXf costMap     (mapSize(0) + 2*border, mapSize(1) + 2*border );
	elevationMap *= NAN;
	//costMap = 0.0;

	printf("Storing elevation map in an eigen matrix."); fflush(stdout);
	Eigen::Array2i mapIndex;
	for (mapIndex(0) = 0; mapIndex(0) < mapSize(0); ++mapIndex(0))
		for (mapIndex(1) = 0; mapIndex(1) < mapSize(1); ++mapIndex(1))
		{
			try
			{
				elevationMap( mapIndex(0)+border, mapIndex(1)+border ) = map->at(layer, mapIndex);
			}
			catch (std::out_of_range e) { }
		}
	printf("Done.\n"); fflush(stdout);

	//int d=0;

	//float maxStep = 0;
	//float maxSlope = 0;
	//float maxRoughness = 0;
	//float maxBorder = 0;

	float borderThreshold = 0.8;

	printf("Calculating cost map values:\nProgress   0.00%%");

	Eigen::Matrix<float, 3, 2> planeT;
	planeT << 0,0, 0,0, 0,0;
	planeT.block(0,0,3,1) = vertical;

	int n=0;

	for (int r = border; r < elevationMap.rows()-border; ++r)
		for (int c = border; c < elevationMap.cols()-border; ++c)
		{

			if (n++ % 500 == 0)
			{
				float progress = n / (float)(mapSize(0)*mapSize(1));
				printf("\rProgress %6.2f%%", progress*100.0); fflush(stdout);
			}
			Eigen::Array2i mapIndex(r-border, c-border);
			map->at("cost", mapIndex) = 0;

			if ( !std::isfinite(elevationMap(r,c)) )
				continue;

			// extract cells within rover footprint
			Eigen::MatrixXf elevations = elevationMap.block(r-border, c-border,
															footprint.rows(), footprint.cols());
			elevations.array() *= footprint.array();

			// first check the border cost and if it's below the threshold don't calculate any others
			float borderCost = (countNonNans(elevations) / (float)footprintArea);
			//if (borderCost > maxBorder)
			//	maxBorder = borderCost;
			borderCost = std::max((double)0.0, (borderCost-borderThreshold) / (1.0-borderThreshold));
			map->at("border", mapIndex) = borderCost;
			if (borderCost == 0)
			{
				continue;
			}

			float stepCost = calculateStepCost(elevations);

			Eigen::Matrix<float, 3, 2> plane = calculateBestFitPlane(elevations, map->getResolution());
			Eigen::Vector3f normal = plane.block(0,0,3,1);
			//normal /= normal.norm();
			normal.normalize();

			float slopeCost = acos(normal.dot(vertical));// / (normal.norm()));

			float roughnessCost = findMaxPlaneResidual(elevations, plane, map->getResolution());

			//if (stepCost > maxStep) maxStep = stepCost;
			//if (slopeCost > maxSlope) maxSlope = slopeCost;
			//if (roughnessCost > maxRoughness) maxRoughness = roughnessCost;

			stepCost = 0.5 * (1.0 - std::min( (float)1.0, stepCost/costmapSettings.clearanceHeight ));

			slopeCost = 1 - std::min( (float)1.0, slopeCost/costmapSettings.maxPitchAngle );

			roughnessCost = 1 - std::min( (float)1.0, costmapSettings.roughnessFraction*(roughnessCost/costmapSettings.clearanceHeight) );

			map->at("step", mapIndex) = stepCost;
			map->at("slope", mapIndex) = slopeCost;
			map->at("roughness", mapIndex) = roughnessCost;

			// calculate cost
			float cost = ((stepCost + slopeCost + roughnessCost) / 3) * borderCost;

			if (cost <= 0.6) cost = 0;
			else
				cost = (cost-0.6) / 0.4;

			map->at("cost", mapIndex) = cost;
		}

	printf("\rProgress 100.00%% Complete\n"); fflush(stdout);
}

void gncTK::DEM::generateCostMap(grid_map::GridMap* map, std::string layer)
{
	// add costmap layers
	map->add("cost");
	map->add("slope");
	map->add("roughness");
	map->add("step");
	map->add("border");

	// generate mask matrix for rover footprint
	Eigen::MatrixXi footprint = GenerateRoverMaskMatrix(map->getResolution(), costmapSettings.roverWidth);
	int footprintSize = footprint.sum();

	Eigen::Vector3f vertical = Eigen::Vector3f::Zero() - costmapSettings.gravity; //vertical << 0, 0, 1;

	//std::cout << "rover footprint is\n" << footprint << "\n-----\n";

	int d=0;

	float maxStep = 0;
	float maxSlope = 0;
	float maxRoughness = 0;
	float maxBorder = 0;

	float borderThreshold = 0.8;

	printf("Calculating cost map values:\nProgress   0.00%%");

	for (grid_map::GridMapIterator it(*map); !it.isPastEnd(); ++it)
	{
		int index = it.getLinearIndex();
		if (index % 500 == 0)
		{
			float progress = index / (float)it.end().getLinearIndex();
			printf("\rProgress %6.2f%%", progress*100.0); fflush(stdout);
		}

		// default all cost map values to zero
		map->at("cost", *it) = 0;

		if ( !std::isfinite(map->at(layer, *it)) )
			continue;

		// collect elevation values for a circle of cells around this one
		grid_map::Position center;
		map->getPosition(*it, center);
		Eigen::MatrixXf elevations = getElevationSamples(map, footprint, center, layer);

		// first check the border cost and if it's below the threshold don't calculate any others
		float borderCost = (countNonNans(elevations) / (float)footprintSize);
		if (borderCost > maxBorder)
			maxBorder = borderCost;
		borderCost = std::max((double)0.0, (borderCost-borderThreshold) / (1.0-borderThreshold));
		map->at("border", *it) = borderCost;
		if (borderCost == 0)
		{
			//map->at("cost", *it) = 0;
			continue;
		}

		float stepCost = calculateStepCost(elevations);

		Eigen::Matrix<float, 3, 2> plane = calculateBestFitPlane(elevations, map->getResolution());
		Eigen::Vector3f normal = plane.block(0,0,3,1);
		normal /= normal.norm();

		float slopeCost = acos(normal.dot(vertical) / (normal.norm()));

		float roughnessCost = findMaxPlaneResidual(elevations, plane, map->getResolution());

		if (stepCost > maxStep) maxStep = stepCost;
		if (slopeCost > maxSlope) maxSlope = slopeCost;
		if (roughnessCost > maxRoughness) maxRoughness = roughnessCost;

		stepCost = 0.5 * (1.0 - std::min( (float)1.0, stepCost/costmapSettings.clearanceHeight ));

		slopeCost = 1 - std::min( (float)1.0, slopeCost/costmapSettings.maxPitchAngle );

		roughnessCost = 1 - std::min( (float)1.0, costmapSettings.roughnessFraction*(roughnessCost/costmapSettings.clearanceHeight) );

		map->at("step", *it) = stepCost;
		map->at("slope", *it) = slopeCost;
		map->at("roughness", *it) = roughnessCost;

		// calculate cost
		float cost = ((stepCost + slopeCost + roughnessCost) / 3) * borderCost;

		if (cost <= 0.6) cost = 0;
		else
			cost = (cost-0.6) / 0.4;

		map->at("cost", *it) = cost;
	}

	printf("\rProgress 100.00%% Complete\n"); fflush(stdout);
}

/// Method to create an openCV Mat of the given layer of this gridmap
cv::Mat gncTK::DEM::convertToCvMat(grid_map::GridMap *map, std::string layer)
{
	// get size and create cv mat image
	Eigen::Array2i mapSize = map->getSize();
	cv::Mat image(mapSize[0], mapSize[1], CV_32F, NAN);

	// copy layer values
	Eigen::Array2i mapIndex;
	for (mapIndex(0) = 0; mapIndex(0) < mapSize(0); ++mapIndex(0))
		for (mapIndex(1) = 0; mapIndex(1) < mapSize(1); ++mapIndex(1))
		{
			try
			{
				image.at<float>( mapIndex(0), mapIndex(1) ) = map->at(layer, mapIndex);
			}
			catch (std::out_of_range e) { }
		}

	return image;
}

// setup dynamic reconfigure server
void gncTK::DEM::setupReconfigureServer()
{
	//dynamic_reconfigure::Server<smart_fusion_sensor::CostMapGenConfig>::CallbackType f;
	//f = boost::bind(&configCallBack, _1, _2);
	//configServer = new dynamic_reconfigure::Server<smart_fusion_sensor::CostMapGenConfig>();
	//configServer->setCallback(f);
}
