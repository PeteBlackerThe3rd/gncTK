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

mesh.cpp

triangular mesh storage object
----------------------------

This object stores various different forms of triangulated
mesh. It can include single or multiple texture information
as well as per vertex color and normal information

Helper functions are provided to load and save this data
in wavefront OBJ and stanford PLY formats.

Functions are also provided to convert the data to PCL
Point cloud objects or ROS mesh marker messages

An efficient geometric deviation calculation is also provided
to calculate the geometric deviations between this mesh and
another.

-------------------------------------------------------------*/
#include <Eigen/Dense>
#include <stdio.h>
#include <regex.h>
#include <pcl_ros/point_cloud.h>
#include <math.h>
//#include "cv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "visualization_msgs/Marker.h"
#include <stats1d.h>
#include <mesh.h>
#include <utils.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

// method to generate a ROS marker message of the mesh
visualization_msgs::Marker gncTK::Mesh::toMarkerMsg(int colorMode, cv::Vec3b tint)
{
	bool doTint = (colorMode & ColorDoTint);
	colorMode &= ~ColorDoTint;

	// create Mesh message
	visualization_msgs::Marker meshMarker;

	// setup Triangle_List marker message
	meshMarker.header.frame_id = frameId;
	meshMarker.header.stamp = ros::Time();
	meshMarker.ns = "lidar_scan_mesh";
	meshMarker.id = 0;
	meshMarker.type = visualization_msgs::Marker::TRIANGLE_LIST;
	meshMarker.action = visualization_msgs::Marker::ADD;
	meshMarker.pose.position.x = 0;
	meshMarker.pose.position.y = 0;
	meshMarker.pose.position.z = 0;
	meshMarker.pose.orientation.x = 0.0;
	meshMarker.pose.orientation.y = 0.0;
	meshMarker.pose.orientation.z = 0.0;
	meshMarker.pose.orientation.w = 1.0;
	meshMarker.scale.x = 1.0;
	meshMarker.scale.y = 1.0;
	meshMarker.scale.z = 1.0;
	meshMarker.color.a = 1.0;

	// create temp vertex and color objects.
	geometry_msgs::Point vertex;
	std_msgs::ColorRGBA	meshColor;

	// determine and check color mode
	if (colorMode == ColorAuto)
	{
		if (triangleColors.size() == triangles.size())
			colorMode = ColorTriangle;
		else if (texCoords.size() == vertices.size())
			colorMode = ColorTexture;
		else if (vertexColors.size() == vertices.size())
			colorMode = ColorVertex;
		else
			colorMode = ColorTint;
	}
	if (colorMode == ColorTriangle && triangleColors.size() == 0)
	{
		ROS_ERROR("Trying to generate marker msg from mesh with triangle colors where none exist. Using tint instead.");
		colorMode = ColorTint;
	}
	if (colorMode == ColorTexture && texCoords.size() == 0)
	{
		ROS_ERROR("Trying to generate marker msg from mesh with texture colors where no UV coordinates exist. Using tint instead.");
		colorMode = ColorTint;
	}
	if (colorMode == ColorVertex && vertexColors.size() == 0)
	{
		ROS_ERROR("Trying to generate marker msg from mesh with vertex colors where none exist. Using tint instead.");
		colorMode = ColorTint;
	}
	if (colorMode == ColorTint)
	{
		meshColor.r = tint[0] / 256.0;
		meshColor.g = tint[1] / 256.0;
		meshColor.b = tint[2] / 256.0;
	}

	// add triangles
	for (int f=0; f<triangles.size(); ++f)
	{
		switch(colorMode)
		{
		case ColorTriangle:
			{
				meshColor.r = triangleColors[f][0] / 256.0;
				meshColor.g = triangleColors[f][1] / 256.0;
				meshColor.b = triangleColors[f][2] / 256.0;
				break;
			}
		case ColorTexture:
			{
				if (triangles[f].texId >= textures.size())
				{
					printf("texture id [%d] is out of range. %d textures available in object.",
							triangles[f].texId,
							(int)textures.size());
				}
				//cv::Mat texture = textures[triangles[f].texId].texture;
				int cols = textures[triangles[f].texId].texture.cols;
				int rows = textures[triangles[f].texId].texture.rows;
				Eigen::Vector2f coords = (texCoords[triangles[f].t1] +
										  texCoords[triangles[f].t2] +
										  texCoords[triangles[f].t3]) / 3;

				coords[1] = 1 - coords[1];

				int U = (coords[0] - (int)coords[0]) * (cols-1);
				int V = (coords[1] - (int)coords[1]) * (rows-1);

				cv::Vec3b color = textures[triangles[f].texId].texture.at<cv::Vec3b>(V,U);
				meshColor.r = color[2] / 256.0;
				meshColor.g = color[1] / 256.0;
				meshColor.b = color[0] / 256.0;
				break;
			}
		case ColorVertex:
			{
				cv::Vec3b p1;
				cv::Vec3b p2;
				cv::Vec3b p3;

				p1 = vertexColors[triangles[f].v1];
				p2 = vertexColors[triangles[f].v2];
				p3 = vertexColors[triangles[f].v3];

				meshColor.r = (p1[0]+p2[0]+p3[0]) / (256.0*3);
				meshColor.g = (p1[1]+p2[1]+p3[1]) / (256.0*3);
				meshColor.b = (p1[2]+p2[2]+p3[2]) / (256.0*3);
				break;
			}
		default:
			{
				meshColor.r = 1.0;
				meshColor.g = 1.0;
				meshColor.b = 1.0;
			}
		}

		if (doTint)
		{
			meshColor.r *= (tint[0] / 256.0);
			meshColor.g *= (tint[1] / 256.0);
			meshColor.b *= (tint[2] / 256.0);
		}

		meshMarker.colors.push_back(meshColor);

		// add triangle to mesh
		vertex.x = vertices[triangles[f].v1][0];
		vertex.y = vertices[triangles[f].v1][1];
		vertex.z = vertices[triangles[f].v1][2];
		meshMarker.points.push_back(vertex);

		vertex.x = vertices[triangles[f].v2][0];
		vertex.y = vertices[triangles[f].v2][1];
		vertex.z = vertices[triangles[f].v2][2];
		meshMarker.points.push_back(vertex);

		vertex.x = vertices[triangles[f].v3][0];
		vertex.y = vertices[triangles[f].v3][1];
		vertex.z = vertices[triangles[f].v3][2];
		meshMarker.points.push_back(vertex);
	}

	return meshMarker;
}

// method to generate a FusedSurface message of this mesh
gnc_tool_kit::FusedSurface gncTK::Mesh::toFusedSurfaceMsg()
{
	gnc_tool_kit::FusedSurface msg;
	msg.header.frame_id = frameId;

	if (textures.size() != 1)
	{
		ROS_ERROR("Error cannot produce a FusedSurface message from a mesh that doesn't have a single texture.");
		return msg;
	}

	// add vertices
	for (int v=0; v<vertices.size(); ++v)
	{
		gnc_tool_kit::Vertex newVertex;
		newVertex.x = vertices[v][0];
		newVertex.y = vertices[v][1];
		newVertex.z = vertices[v][2];
		msg.vertices.push_back(newVertex);
	}

	// add texture coordinates
	for (int c=0; c<texCoords.size(); ++c)
	{
		gnc_tool_kit::TextureCoordinate newTexCoord;
		newTexCoord.u = texCoords[c][0];
		newTexCoord.v = texCoords[c][1];
		msg.texCoords.push_back(newTexCoord);
	}

	// add triangles
	for (int t=0; t<triangles.size(); ++t)
	{
		gnc_tool_kit::Triangle newTriangle;
		newTriangle.v1 = triangles[t].v1;
		newTriangle.v2 = triangles[t].v2;
		newTriangle.v3 = triangles[t].v3;
		newTriangle.t1 = triangles[t].t1;
		newTriangle.t2 = triangles[t].t2;
		newTriangle.t3 = triangles[t].t3;
		msg.triangles.push_back(newTriangle);
	}

	// add sensor origin
	msg.sensor_origin_x = sensorOrigin[0];
	msg.sensor_origin_y = sensorOrigin[1];
	msg.sensor_origin_z = sensorOrigin[2];

	msg.cameraImage = *cv_bridge::CvImage(std_msgs::Header(), "bgr8", textures[0].texture).toImageMsg();

	return msg;
}

// factory method to generate a mesh from a FusedSurface message
gncTK::Mesh gncTK::Mesh::fromFusedSurfaceMsg(const gnc_tool_kit::FusedSurface &fusedSurfaceMsg)
{
	Mesh newMesh;
	newMesh.frameId = fusedSurfaceMsg.header.frame_id;

	// add vertices
	for (int v=0; v<fusedSurfaceMsg.vertices.size(); ++v)
	{
		Eigen::Vector3f newVertex;
		newVertex << fusedSurfaceMsg.vertices[v].x,
					 fusedSurfaceMsg.vertices[v].y,
					 fusedSurfaceMsg.vertices[v].z;
		newMesh.vertices.push_back(newVertex);
	}

	// add texture coordinates
	for (int c=0; c<fusedSurfaceMsg.texCoords.size(); ++c)
	{
		Eigen::Vector2f newTexCoord;
		newTexCoord << fusedSurfaceMsg.texCoords[c].u,
					   fusedSurfaceMsg.texCoords[c].v;
		newMesh.texCoords.push_back(newTexCoord);
	}

	// add triangles
	for (int t=0; t<fusedSurfaceMsg.triangles.size(); ++t)
	{
		Triangle newTriangle(fusedSurfaceMsg.triangles[t].v1,
							 fusedSurfaceMsg.triangles[t].v2,
							 fusedSurfaceMsg.triangles[t].v3,
							 fusedSurfaceMsg.triangles[t].t1,
							 fusedSurfaceMsg.triangles[t].t2,
							 fusedSurfaceMsg.triangles[t].t3,
							 0 );
		newMesh.triangles.push_back(newTriangle);
	}

	// add sensor origin
	newMesh.sensorOrigin[0] = fusedSurfaceMsg.sensor_origin_x;
	newMesh.sensorOrigin[1] = fusedSurfaceMsg.sensor_origin_y;
	newMesh.sensorOrigin[2] = fusedSurfaceMsg.sensor_origin_z;

	// get camera image
	cv_bridge::CvImagePtr cvImg = cv_bridge::toCvCopy(fusedSurfaceMsg.cameraImage, "bgr8");
	newMesh.setSingleTexture(cvImg->image);

	return newMesh;
}

void gncTK::Mesh::setSingleTexture(cv::Mat image)
{
	textures.empty();

	Texture newTexture;
	newTexture.texture = image;

	textures.push_back(newTexture);
}

// method to get the bounding box of this mesh
Eigen::Matrix<float, 3, 2> gncTK::Mesh::vertexExtents(std::vector<Eigen::Vector3f> vertices)
{
	Eigen::Vector3f extentsMin, extentsMax;
	extentsMin << 0,0,0;
	extentsMax << 0,0,0;
	Eigen::Matrix<float, 3, 2> extents;
	bool firstVert = true;

	for (int v=0; v<vertices.size(); ++v)
	{
		bool wasNaN = std::isnan(vertices[v][0]) || std::isnan(vertices[v][1]) || std::isnan(vertices[v][2]);
		if (wasNaN)
			continue;

		if (firstVert)
		{
			for (int d=0; d<3; ++d)
				extentsMin(d) = extentsMax(d) = vertices[v](d);
			firstVert = false;
		}
		else
		{
			for (int d=0; d<3; ++d)
			{
				if (vertices[v](d) < extentsMin(d)) extentsMin(d) = vertices[v](d);
				if (vertices[v](d) > extentsMax(d)) extentsMax(d) = vertices[v](d);
			}
		}
	}

	extents.block(0,0,3,1) = extentsMin;
	extents.block(0,1,3,1) = extentsMax;
	return extents;
}

// method to get the bounding box of this mesh
Eigen::Matrix<float, 3, 2> gncTK::Mesh::getExtents()
{
	return vertexExtents(vertices);
}

/// Method to return the centre point of the bounding box of this mesh
Eigen::Vector3f gncTK::Mesh::getCentre()
{
	Eigen::Matrix<float, 3, 2> extents = getExtents();

	return (extents.block(0,0,3,1) + extents.block(0,1,3,1)) / 2.0;
}

// method to re-calculate/create the vertex normals for this mesh
void gncTK::Mesh::calculateNormals()
{
	// setup
	vertexNormals.clear();
	vertexNormals.insert(vertexNormals.begin(), vertices.size(), Eigen::Vector3f::Zero());
	std::vector<int> vertNormalCount;
	vertNormalCount.insert(vertNormalCount.begin(), vertices.size(), 0);

	// calculate triangle normals
	for (int t=0; t<triangles.size(); ++t)
	{
		Eigen::Vector3f edge1 = vertices[triangles[t].v3] - vertices[triangles[t].v1];
		Eigen::Vector3f edge2 = vertices[triangles[t].v1] - vertices[triangles[t].v2];
		Eigen::Vector3f normal = edge1.cross(edge2).normalized();

		vertexNormals[triangles[t].v1] += normal;
		vertNormalCount[triangles[t].v1]++;
		vertexNormals[triangles[t].v2] += normal;
		vertNormalCount[triangles[t].v2]++;
		vertexNormals[triangles[t].v3] += normal;
		vertNormalCount[triangles[t].v3]++;
	}

	// average normals around vertices
	for (int v=0; v<vertexNormals.size(); ++v)
	{
		if (vertNormalCount[v] == 0)
		{
			vertexNormals[v][0] = 1;
		}
		else
		{
			vertexNormals[v] /= vertNormalCount[v];
		}
	}
}

/// Method to change the frame id of this mesh and transform all its geometry into this new frame
void gncTK::Mesh::changeCoordinateFrame(std::string newFrame, tf::TransformListener *tfListener, ros::Time tfTime)
{
	// lookup transform from current frame to new frame
	//ros::Time tfTime = ros::Time(0);
	std::string tfError;
	if (!tfListener->canTransform(newFrame, frameId, tfTime, &tfError))
	{
		ROS_ERROR("Error [%s] cannot lookup transform from [%s] to [%s] while changing mesh coordinate frame.",
				  tfError.c_str(),
				  frameId.c_str(),
				  newFrame.c_str());
		return;
	}

	tf::StampedTransform transform;
	tfListener->lookupTransform(newFrame, frameId, tfTime, transform);

	// transform vertices
	for (int v=0; v<vertices.size(); ++v)
	{
		tf::Vector3 point(vertices[v][0], vertices[v][1], vertices[v][2]);

		tf::Vector3 newPoint = transform * point;

		vertices[v][0] = newPoint.getX();
		vertices[v][1] = newPoint.getY();
		vertices[v][2] = newPoint.getZ();
	}

	// transform the sensor origin
	tf::Vector3 point(sensorOrigin[0],sensorOrigin[1], sensorOrigin[2]);
	tf::Vector3 newPoint = transform * point;
	sensorOrigin[0] = newPoint.getX();
	sensorOrigin[1] = newPoint.getY();
	sensorOrigin[2] = newPoint.getZ();

	//ROS_WARN("transformed sensor location is (%f %f %f)",
	//		 sensorOrigin[0],
	//		 sensorOrigin[1],
	//		 sensorOrigin[2]);

	calculateNormals();

	// update the frame id string
	frameId = newFrame;
}

//---- OBJ Methods ----------------------------------------------

struct triangle
{
  int v1,v2,v3;
  int t1,t2,t3;
  int n1,n2,n3;
  int texId;

  triangle(int _v1, int _v2, int _v3,
		   int _t1, int _t2, int _t3,
		   int _n1, int _n2, int _n3,
		   int _texId)
  {
	v1 = _v1;
	v2 = _v2;
	v3 = _v3;

	t1 = _t1;
	t2 = _t2;
	t3 = _t3;

	n1 = _n1;
	n2 = _n2;
	n3 = _n3;

	texId = _texId;
  };
};

int gncTK::Mesh::objLookupMtlName(std::string label)
{
  for (int m=0; m<textures.size(); ++m)
	if (label == textures[m].label)
	{
	  return m;
	}

  printf("Warning : Couldn't find texture [%s] in material library.\n", label.c_str());
  return -1;
}

void gncTK::Mesh::objLoadMaterialLibrary(std::string mtlLibName, std::string basePath)
{
  printf("Loading material library [%s]\n", mtlLibName.c_str());
  FILE *mtlFile = fopen(mtlLibName.c_str(), "r");
  if (mtlFile == NULL)
  {
	printf("Failed to open mtl file [%s]\n", mtlLibName.c_str());
	return;
  }

  regex_t labelPattern, filePattern;
  regcomp(&labelPattern, "newmtl ([^[:space:]]+)", REG_EXTENDED);
  regcomp(&filePattern, "map_Kd ([^[:space:]]+)", REG_EXTENDED);
  regmatch_t matches[3];

  char materialLabel[256] = "";

  char line[256];
  while (fgets(line, 256, mtlFile) != NULL)
  {
	// if this line matches the label pattern
	if (regexec(&labelPattern, line, 2, matches, 0) == 0)
	{
	  int len = matches[1].rm_eo - matches[1].rm_so;
	  strncpy(materialLabel,
			  line + matches[1].rm_so,
			  len);
	  materialLabel[len] = '\0';
	}

	// if this line matches the file name pattern
	else if (regexec(&filePattern, line, 2, matches, 0) == 0)
	{
	  char texFile[256];
	  int len = matches[1].rm_eo - matches[1].rm_so;
	  strncpy(texFile,
			  line + matches[1].rm_so,
			  len);
	  texFile[len] = '\0';

	  ROS_WARN("Reading OBJ texture file [%s]", (basePath + "/" + std::string(texFile)).c_str());

	  Texture newTexture;
	  newTexture.label = std::string(materialLabel);
	  newTexture.fileName = std::string(texFile);
	  newTexture.texture = cv::imread(basePath + "/" + std::string(texFile));
	  textures.push_back( newTexture );
	}
  }

  fclose(mtlFile);
}

gncTK::Mesh gncTK::Mesh::loadOBJ(std::string fileName)
{
  gncTK::Mesh newMesh;

  printf("Loading  mesh from obj file [%s].\n", fileName.c_str());

  boost::filesystem::path p(fileName);
  std::string basePath = p.parent_path().string();

  ROS_WARN("Base path of obj file is : [%s]\n", basePath.c_str());

  // read the obj file itself
  FILE *objFile = fopen(fileName.c_str(), "r");

  if (objFile == NULL)
  {
	printf("Failed to open obj file [%s]\n", fileName.c_str());
	return newMesh;
  }

  // compile regex patterns to match types of line in obj file
  regex_t vertexPattern, normalPattern, texturePattern, facePatternFull, facePatternTex, mtlPattern, mtlLibPattern;

  regcomp(&vertexPattern, "[vV] ([+-]?[0-9]*\\.?[0-9]+) ([+-]?[0-9]*\\.?[0-9]+) ([+-]?[0-9]*\\.?[0-9]+)", REG_EXTENDED);
  regcomp(&normalPattern, "[vV][nN] ([+-]?[0-9]*\\.?[0-9]+) ([+-]?[0-9]*\\.?[0-9]+) ([+-]?[0-9]*\\.?[0-9]+)", REG_EXTENDED);
  regcomp(&texturePattern, "[vV][tT] ([+-]?[0-9]*\\.?[0-9]+) ([+-]?[0-9]*\\.?[0-9]+)", REG_EXTENDED);
  regcomp(&facePatternFull, "[fF] ([0-9]+)\\/([0-9]*)\\/([0-9]+) ([0-9]+)\\/([0-9]*)\\/([0-9]+) ([0-9]+)\\/([0-9]*)\\/([0-9]+)", REG_EXTENDED);
  regcomp(&facePatternTex, "[fF] ([0-9]+)\\/([0-9]+) ([0-9]+)\\/([0-9]+) ([0-9]+)\\/([0-9]+)", REG_EXTENDED);
  regcomp(&mtlPattern, "usemtl ([^[:space:]]+)", REG_EXTENDED);
  regcomp(&mtlLibPattern, "mtllib ([^[:space:]]+)\\.mtl", REG_EXTENDED);
  regmatch_t matches[12];

  int lineCount = 0;

  // read successive lines from the obj file
  char line[256];
  int currentTexId = -1;
  while (fgets(line, 256, objFile) != NULL)
  {
	++lineCount;

	if (lineCount % 100000 == 0)
	{
	  printf("#");
	  fflush(stdout);
	}

	// shorten line by 1 character (remove carriage return)
	if (strlen(line)!=0)
	  line[strlen(line)-1] = 0;

	// if this line matches the vertex pattern
	if (regexec(&vertexPattern, line, 12, matches, 0) == 0)
	{
		// match 1,2 and 3 will contain the vertex coordinates now
		Eigen::Vector3f newVert;
		newVert[0] = atof(&line[matches[1].rm_so]);
	  	newVert[1] = atof(&line[matches[2].rm_so]);
	  	newVert[2] = atof(&line[matches[3].rm_so]);

	  	newMesh.vertices.push_back(newVert);
	}

	// if this line matches the normal pattern
	else if (regexec(&normalPattern, line, 12, matches, 0) == 0)
	{
		// match 1,2 and 3 will contain the vertex coordinates now
		Eigen::Vector3f newNormal;
		newNormal[0] = atof(&line[matches[1].rm_so]);
		newNormal[1] = atof(&line[matches[2].rm_so]);
		newNormal[2] = atof(&line[matches[3].rm_so]);

		newMesh.vertexNormals.push_back(newNormal);
	}

	// if this line matches the texture coordinate pattern
	else if (regexec(&texturePattern, line, 7, matches, 0) == 0)
	{
		// match 1 and 2 will contain the texture U V coordinates now
		Eigen::Vector2f newTexCoord;
		newTexCoord[0] = atof(&line[matches[1].rm_so]);
		newTexCoord[1] = atof(&line[matches[2].rm_so]);

		newMesh.texCoords.push_back(newTexCoord);
	}

	// if this line matches the face pattern with verts, texcoords and normals
	else if (regexec(&facePatternFull, line, 12, matches, 0) == 0)
	{
	  // get triangle vertex indices (note obj file indices are 1,2,3.... model indices are 0,1,2.... )
	  int v1 = atoi(&line[matches[1].rm_so]) - 1;
	  int v2 = atoi(&line[matches[4].rm_so]) - 1;
	  int v3 = atoi(&line[matches[7].rm_so]) - 1;

	  int t1 = atoi(&line[matches[2].rm_so]) - 1;
	  int t2 = atoi(&line[matches[5].rm_so]) - 1;
	  int t3 = atoi(&line[matches[8].rm_so]) - 1;

	  int n1 = atoi(&line[matches[3].rm_so]) - 1;
	  int n2 = atoi(&line[matches[6].rm_so]) - 1;
	  int n3 = atoi(&line[matches[9].rm_so]) - 1;

	  newMesh.triangles.push_back(gncTK::Mesh::Triangle(v1,v2,v3,
										   t1,t2,t3,
										   n1,n2,n3,
										   currentTexId));
	}

	// if this line matches the face pattern with verts and texcoords only
	else if (regexec(&facePatternTex, line, 12, matches, 0) == 0)
	{
	  // get triangle vertex indices (note obj file indices are 1,2,3.... model indices are 0,1,2.... )
	  int v1 = atoi(&line[matches[1].rm_so]) - 1;
	  int v2 = atoi(&line[matches[3].rm_so]) - 1;
	  int v3 = atoi(&line[matches[5].rm_so]) - 1;

	  int t1 = atoi(&line[matches[2].rm_so]) - 1;
	  int t2 = atoi(&line[matches[4].rm_so]) - 1;
	  int t3 = atoi(&line[matches[6].rm_so]) - 1;

	  ROS_INFO("read tex coords [%d %d %d]", t1, t2, t3);

	  newMesh.triangles.push_back(gncTK::Mesh::Triangle(v1,v2,v3,
										   t1,t2,t3,
										   currentTexId));
	}

	// if this line matches the material pattern
	else if (regexec(&mtlPattern, line, 12, matches, 0) == 0)
	{
	  char mtlLabel[256];
	  int len = matches[1].rm_eo - matches[1].rm_so;
	  strncpy(mtlLabel,
			  line + matches[1].rm_so,
			  len);
	  mtlLabel[len] = '\0';

	  currentTexId = newMesh.objLookupMtlName(mtlLabel);
	}

	// if this line matches the material lib pattern
	else if (regexec(&mtlLibPattern, line, 12, matches, 0) == 0)
	{
	  char mtlLibName[256], name[256];
	  int len = matches[1].rm_eo - matches[1].rm_so;
	  strncpy(name,
			  line + matches[1].rm_so,
			  len);
	  name[len] = '\0';

	  sprintf(mtlLibName, "%s/%s.mtl", basePath.c_str(), name);
	  newMesh.objLoadMaterialLibrary(mtlLibName, basePath);
	}
  }

  fclose(objFile);

  regfree(&vertexPattern);
  regfree(&normalPattern);
  regfree(&facePatternFull);
  regfree(&facePatternTex);
  regfree(&texturePattern);
  regfree(&mtlPattern);
  regfree(&mtlLibPattern);

  printf("\nCompleted reading obj file.\n%d vertices and %d triangles read.\n",
		 (int)newMesh.vertices.size(), (int)newMesh.triangles.size());

  return newMesh;
}

bool gncTK::Mesh::saveOBJ(std::string baseName)
{
	// find basename without path for relative file names
	std::string baseNameWOPath = gncTK::Utils::fileNameWOPath(baseName);

	// set names of texture files and save
	for (int t=0; t<textures.size(); ++t)
	{
		char absPath[1024];
		sprintf(absPath, "%s_%05d.jpg", baseName.c_str(), t);
		imwrite(absPath, textures[t].texture);

		char relPath[256];
		sprintf(relPath, "%s_%05d.jpg", baseNameWOPath.c_str(), t);
		textures[t].fileName = std::string(relPath);
	}

	// write material file
	FILE *mtlFile = fopen((baseName + ".mtl").c_str(), "w");
	fprintf(mtlFile, "# Material file generated by the GNC Toolkit [ (c) Surrey University ]\n#\n");

	for (int t=0; t<textures.size(); ++t)
	{
		fprintf(mtlFile, "newmtl texture_%05d\n", t);
		fprintf(mtlFile, "Ka 1.0 1.0 1.0\n");
		fprintf(mtlFile, "Kd 1.0 1.0 1.0\n");
		fprintf(mtlFile, "Ks 0.0 0.0 0.0\n");
		fprintf(mtlFile, "illum 1\n");
		fprintf(mtlFile, "Ns 50\n");
		fprintf(mtlFile, "map_Ka %s\n", textures[t].fileName.c_str());
		fprintf(mtlFile, "map_Kd %s\n", textures[t].fileName.c_str());
	}
	fclose(mtlFile);

	// create obj file itself
	FILE *objFile = fopen((baseName + ".obj").c_str(), "w");
	fprintf(objFile, "# Wavefront OBJ file generated by the GNC Toolkit [ (c) Surrey University ]\n#\n");

	fprintf(objFile, "mtllib %s\n#\n", (baseNameWOPath + ".mtl").c_str());

	// write vertex, normal and texture coordinate lists
	for (int v=0; v<vertices.size(); ++v)
	{
		// if any of the dimensions are NAN then write a zero vertex for compatibility with viewers.
		if (std::isnan(vertices[v][0]) ||
			std::isnan(vertices[v][0]) ||
			std::isnan(vertices[v][0]))
			fprintf(objFile, "v 0.0 0.0 0.0\n");
		else
			fprintf(objFile, "v %f %f %f\n", vertices[v][0], vertices[v][1], vertices[v][2]);
	}

	for (int c=0; c<texCoords.size(); ++c)
		fprintf(objFile, "vt %f %f\n", texCoords[c][0], texCoords[c][1]);

	for (int n=0; n<vertexNormals.size(); ++n)
		fprintf(objFile, "vn %f %f %f\n", vertexNormals[n][0], vertexNormals[n][1], vertexNormals[n][2]);

	// write triangle data
	int lastTexId = -1;
	for (int t=0; t<triangles.size(); ++t)
	{
		// if the texture id is different from the last then specify the new material
		if (triangles[t].texId != lastTexId)
		{
			lastTexId = triangles[t].texId;
			fprintf(objFile, "usemtl texture_%05d\n", lastTexId);
		}

		// texture coordinates but no normals
		if (vertexNormals.size() == 0 && texCoords.size() > 0)
			fprintf(objFile, "f %d/%d %d/%d %d/%d\n",
					triangles[t].v1+1, triangles[t].t1+1,
					triangles[t].v2+1, triangles[t].t2+1,
					triangles[t].v3+1, triangles[t].t3+1);

		// normals but no texture coordinates
		if (vertexNormals.size() > 0 && texCoords.size() == 0)
			fprintf(objFile, "f %d//%d %d//%d %d//%d\n",
					triangles[t].v1+1, triangles[t].n1+1,
					triangles[t].v2+1, triangles[t].n2+1,
					triangles[t].v3+1, triangles[t].n3+1);

		// normals and texture coordinates
		if (vertexNormals.size() > 0 && texCoords.size() > 0)
			fprintf(objFile, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
					triangles[t].v1+1, triangles[t].t1+1, triangles[t].n1+1,
					triangles[t].v2+1, triangles[t].t2+1, triangles[t].n2+1,
					triangles[t].v3+1, triangles[t].t3+1, triangles[t].n3+1);
	}

	fclose(objFile);

	return true;
}


gncTK::Mesh gncTK::Mesh::fromPCL(pcl::PointCloud<pcl::PointXYZ> pointCloud)
{
	Mesh newMesh;
	Eigen::Vector3f point;
	newMesh.frameId = pointCloud.header.frame_id;

	for (int i=0; i<pointCloud.points.size(); ++i)
	{
		point << pointCloud.points[i].x, pointCloud.points[i].y, pointCloud.points[i].z;
		newMesh.vertices.push_back(point);
	}

	return newMesh;
}
gncTK::Mesh gncTK::Mesh::fromPCL(pcl::PointCloud<pcl::PointXYZI> pointCloud)
{

}
gncTK::Mesh gncTK::Mesh::fromPCL(pcl::PointCloud<pcl::PointXYZRGB> pointCloud)
{
	Mesh newMesh;
	Eigen::Vector3f point;
	cv::Vec3b color;
	newMesh.frameId = pointCloud.header.frame_id;

	for (int i=0; i<pointCloud.points.size(); ++i)
	{
		point << pointCloud.points[i].x, pointCloud.points[i].y, pointCloud.points[i].z;
		color[0] = pointCloud.points[i].r;
		color[1] = pointCloud.points[i].g;
		color[2] = pointCloud.points[i].b;

		newMesh.vertices.push_back(point);
		newMesh.vertexColors.push_back(color);
	}

	return newMesh;
}

bool gncTK::Mesh::savePLY(std::string fileName)
{
	FILE *plyFile = fopen(fileName.c_str(), "w");
	if (ferror(plyFile))
	{
		printf("Error: failed to create ply file [%s]\n", fileName.c_str());
		return false;
	}

	// write PLY header
	fprintf(plyFile, "ply\nformat ascii 1.0\ncomment generated by the GNC Toolkit (c) Surrey University\n");
	fprintf(plyFile, "element vertex %d\n", (int)vertices.size());
	fprintf(plyFile, "property float x\nproperty float y\nproperty float z\n");
	if (vertices.size() == vertexColors.size())
		fprintf(plyFile, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
	fprintf(plyFile, "element face %d\n", (int)triangles.size());
	fprintf(plyFile, "property list uchar int vertex_index\n");
	fprintf(plyFile, "end_header\n");

	// write vertices
	for (int v=0; v<vertices.size(); ++v)
	{
		if (vertices.size() == vertexColors.size())
		{
			if (std::isnan(vertices[v][0]))
				fprintf(plyFile,"0 0 0 0 0 0\n");
			else
				fprintf(plyFile,"%f %f %f %d %d %d\n",
						vertices[v][0],
						vertices[v][1],
						vertices[v][2],
						vertexColors[v][0],
						vertexColors[v][1],
						vertexColors[v][2]);
		}
		else
			fprintf(plyFile,"%f %f %f\n",
					vertices[v][0],
					vertices[v][1],
					vertices[v][2]);
	}

	// write triangles
	for (int t=0; t<triangles.size(); ++t)
	{
		fprintf(plyFile, "3 %d %d %d\n",
				triangles[t].v1,
				triangles[t].v2,
				triangles[t].v3);
	}

	fclose(plyFile);
}

//---- Point Cloud Conversion Methods -----------------------------------------

// methods to generate a PCL point cloud of this mesh
pcl::PointCloud<pcl::PointXYZ> gncTK::Mesh::toPointCloud()
{
	// mark unused vertices
	setUnusedVerticesToNAN();

	pcl::PointCloud<pcl::PointXYZ> cloud;

	for (int v=0; v<vertices.size(); ++v)
		//if (vertices[v][0] != NAN && vertices[v][1] != NAN && vertices[v][2] != NAN)
			cloud.points.push_back(pcl::PointXYZ(vertices[v][0],
											     vertices[v][1],
											     vertices[v][2]));

	//printf("Created point cloud with %d points.\n", (int)cloud.points.size());

	return cloud;
}

void gncTK::Mesh::setUnusedVerticesToNAN()
{
	std::vector<unsigned int> useCount;

	for (int v=0; v<vertices.size(); ++v)
		useCount.push_back(0);

	for (int t=0; t<triangles.size(); ++t)
	{
		++useCount[triangles[t].v1];
		++useCount[triangles[t].v2];
		++useCount[triangles[t].v3];
	}

	for (int v=0; v<vertices.size(); ++v)
		if (useCount[v] == 0)
		{
			vertices[v][0] = NAN;
			vertices[v][1] = NAN;
			vertices[v][2] = NAN;
		}
}

//---- Mesh Analysis Methods -----------------------------------------

pcl::KdTreeFLANN<pcl::PointXYZ>* gncTK::Mesh::getKdTree(bool debug)
{
	// if the KD tree hasn't been created yet then make it
	if (!KDTreeCached)
	{
		if (debug)
			ROS_INFO("Generating KD Tree.");
		pcl::PointCloud<pcl::PointXYZ> cloud = toPointCloud();
		kdTree.setInputCloud(cloud.makeShared());
		KDTreeCached = true;
		if (debug)
			ROS_INFO("finished KD tree.");
	}

	return &kdTree;
}

/// method to calculate the vertex to triangle links
void gncTK::Mesh::calculateVertexToTriangleLinks()
{
	// clear old links
	vertexTriangleLinks.clear();
	vertexTriangleLinks.resize(vertices.size());

	// add references to each triangle
	for (int t=0; t<triangles.size(); ++t)
	{
		vertexTriangleLinks[triangles[t].v1].push_back(t);
		vertexTriangleLinks[triangles[t].v2].push_back(t);
		vertexTriangleLinks[triangles[t].v3].push_back(t);
	}
}

bool gncTK::Mesh::isEdge(int v1, int v2)
{
	// ensure v1 is less than v2;
	if (v1 > v2)
	{
		int t = v1;
		v1 = v2;
		v2 = t;
	}

	// find if this edge exists
	for (int e=0; e<edgeArcs[v1].size(); ++e)
		if (edgeArcs[v1][e] == v2)
			return true;

	return false;
}

void gncTK::Mesh::processEdge(int v1, int v2)
{
	// ensure v1 is less than v2;
	if (v1 > v2)
	{
		int t = v1;
		v1 = v2;
		v2 = t;
	}

	// find if this edge already exists then remove it
	bool exists = false;
	for (int e=0; e<edgeArcs[v1].size(); ++e)
	{
		if (edgeArcs[v1][e] == v2)
		{
			exists = true;
			edgeArcs[v1].erase(edgeArcs[v1].begin() + e);
			break;
		}
	}

	// if this edge didn't exist then add it
	if (!exists)
		edgeArcs[v1].push_back(v2);
}

/// calculate edge arcs and vertices
void gncTK::Mesh::calculateEdges()
{
	if (edgesCalculated)
		return;

	// clear edge verts and arc lists
	vertexEdges.clear();
	edgeArcs.clear();

	// create empty edge arc list
	edgeArcs.resize(vertices.size());

	// the edge detection algorithm works like this
	// add every edge of every triangle, if the edge is not in the list then add it
	// if the edge is already in the list remove it.
	// Therefore any remaining edges are singular and therefore an edge (for non-manifold mesh's)
	for (int t=0; t<triangles.size(); ++t)
	{
		processEdge(triangles[t].v1, triangles[t].v2);
		processEdge(triangles[t].v2, triangles[t].v3);
		processEdge(triangles[t].v3, triangles[t].v1);
	}

	// set every vertex on an edge arc to an edge vertex
	//vertexEdges.resize(vertices.size(), false);
	vertexEdges.insert(vertexEdges.begin(), vertices.size(), false);
	for (int e=0; e<edgeArcs.size(); ++e)
	{
		if (edgeArcs[e].size() > 0)
			vertexEdges[e] = true;

		for (int ev=0; ev<edgeArcs[e].size(); ++ev)
			vertexEdges[edgeArcs[e][ev]] = true;
	}

	edgesCalculated = true;
}

/// Calculate the point densities of each vertex
void gncTK::Mesh::calculateVertexPointDensities()
{
	// first initialise the vertex point density vector to zeros
	vertexPointDensities.clear();
	vertexPointDensities.resize(vertices.size(), 0);

	std::vector<int> triCounts;
	triCounts.resize(vertices.size(), 0);

	// initialise a vector of empty vectors that will hold the indices of vertices
	// attached to every vertex via a triangle
	std::vector<std::vector<int> > attachedVerts;
	attachedVerts.resize( vertices.size(), std::vector<int>(0) );

	float totalTriangleArea = 0.0;
	int validVertexCount = 0;

	float distanceThreshold = 5.0;

	// for each triangle add it's area to the vertex point density value
	for (int t=0; t<triangles.size(); ++t)
	{
		Eigen::Vector3f v1 = vertices[triangles[t].v1];
		Eigen::Vector3f v2 = vertices[triangles[t].v2];
		Eigen::Vector3f v3 = vertices[triangles[t].v3];

		// add references to attached vertices for each vertex of this triangle
		attachedVerts[triangles[t].v1].push_back( triangles[t].v2 );
		attachedVerts[triangles[t].v1].push_back( triangles[t].v3 );
		attachedVerts[triangles[t].v2].push_back( triangles[t].v1 );
		attachedVerts[triangles[t].v2].push_back( triangles[t].v3 );
		attachedVerts[triangles[t].v3].push_back( triangles[t].v1 );
		attachedVerts[triangles[t].v3].push_back( triangles[t].v2 );

		float sideA = (v2-v1).norm();
		float sideB = (v3-v2).norm();
		float sideC = (v1-v3).norm();

		float area = gncTK::Utils::triangleAreaFromSides(sideA, sideB, sideC);

		vertexPointDensities[triangles[t].v1] += area;
		vertexPointDensities[triangles[t].v2] += area;
		vertexPointDensities[triangles[t].v3] += area;
		++triCounts[triangles[t].v1];
		++triCounts[triangles[t].v2];
		++triCounts[triangles[t].v3];

		// if this is within 5 meters add it to the basic test density
		Eigen::Vector3f centroid = (v1+v2+v3) / 3.0;
		if (centroid.norm() < distanceThreshold)
		{
			totalTriangleArea += area;
		}
	}

	// invert and multiply by three all non zero values
	for (int v=0; v<vertexPointDensities.size(); ++v)
	{
		if (vertexPointDensities[v] != 0)
		{
			// Find if this vertex is completely surrounded by triangles
			// this is done by checking the list of attached triangle vertices
			// contains exactly two copies of each vertex in it.
			std::sort(attachedVerts[v].begin(), attachedVerts[v].end());
			bool okay = true;
			for (int t=0; t<attachedVerts[v].size(); t+=2)
				if (attachedVerts[v][t] != attachedVerts[v][t+1])
					okay = false;

			if (okay)
			{
				// calculate point density using DTFE equation.
				//vertexPointDensities[v] = ((1.0 + triCounts[v]) / vertexPointDensities[v]) / 2.0;
				vertexPointDensities[v] = 3.0 / vertexPointDensities[v];

				if (vertices[v].norm() < distanceThreshold)
					++validVertexCount;
			}
			else
				vertexPointDensities[v] = NAN;
		}
		else
			vertexPointDensities[v] = NAN;
	}

	ROS_WARN("Calulcated mesh spatial point density within %fm, approximate mean is %f points per m^2",
			 distanceThreshold,
			 validVertexCount / totalTriangleArea);

	// the vertex point density values are now calculated
}

// Method to transfer a vector of numeric values to the color of vertices in this mesh
/*
 * Uses the hue rainbow to colorize the values, any numeric c type can be used.
 * Verifies that the number of elements in values matches the number of vertices before proceeding.
 */
template <class elementT>
void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<elementT> values)
{
	// test size of values
	if (values.size() != vertices.size())
	{
		ROS_ERROR("Error: size mismatch, transferring numeric values to vertex colors, %d values and %d vertices,",
				  (int)values.size(), (int)vertices.size());
	}

	// find range of vertex point densities
	float min, max;
	bool first = true;
	for (int v=0; v<vertices.size(); ++v)
	{
		if (std::isfinite(values[v]))
		{
			if (first)
			{
				min = max = values[v];
				first = false;
			}
			else
			{
				if (values[v] < min)
					min = values[v];
				if (values[v] > max)
					max = values[v];
			}
		}
	}

	vertexColors.clear();

	for (int v=0; v< vertices.size(); ++v)
	{
		float scaled = (values[v] - min) / (max-min);
		vertexColors.push_back(gncTK::Utils::rainbow(scaled));
	}
}
// explicit instantiations of this function made available in the shared object
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<int> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<unsigned int> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<long> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<unsigned long> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<float> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<double> values);
template void gncTK::Mesh::transferNumericValuesToVertexColors(std::vector<long double> values);


/*void gncTK::Mesh::transferPointDensitiesToVertexColors()
{
	// find range of vertex point densities
	float min, max;
	bool first = true;
	for (int v=0; v<vertices.size(); ++v)
	{
		if (std::isfinite(vertexPointDensities[v]))
		{
			if (first)
			{
				min = max = vertexPointDensities[v];
				first = false;
			}
			else
			{
				if (vertexPointDensities[v] < min)
					min = vertexPointDensities[v];
				if (vertexPointDensities[v] > max)
					max = vertexPointDensities[v];
			}
		}
	}

	vertexColors.clear();

	for (int v=0; v< vertices.size(); ++v)
	{
		float scaled = (vertexPointDensities[v] - min) / (max-min);

		vertexColors.push_back(gncTK::Utils::rainbow(scaled));
	}
}*/

/// Calculate the point pixel densities (points per projected camera image pixel) of each vertex
void gncTK::Mesh::calculateVertexPointPixelDensities()
{
	// first initialise the vertex point density vector to zeros
	vertexPointPixelDensities.clear();
	vertexPointPixelDensities.resize(vertices.size(), 0);

	// initialise a vector counting the number of triangles connected to each vertex
	std::vector<int> triCounts;
	triCounts.resize(vertices.size(), 0);

	// initialise a vector of empty vectors that will hold the indices of verticies
	// attached to every vertex via a triangle
	std::vector<std::vector<int> > attachedVerts;
	attachedVerts.resize( vertices.size(), std::vector<int>(0) );

	// for each triangle add it's area to the vertex point density value
	for (int t=0; t<triangles.size(); ++t)
	{
		Eigen::Vector2f v1 = texCoords[triangles[t].v1];
		Eigen::Vector2f v2 = texCoords[triangles[t].v2];
		Eigen::Vector2f v3 = texCoords[triangles[t].v3];

		// add references to attached vertices for each vertex of this triangle
		attachedVerts[triangles[t].v1].push_back( triangles[t].v2 );
		attachedVerts[triangles[t].v1].push_back( triangles[t].v3 );
		attachedVerts[triangles[t].v2].push_back( triangles[t].v1 );
		attachedVerts[triangles[t].v2].push_back( triangles[t].v3 );
		attachedVerts[triangles[t].v3].push_back( triangles[t].v1 );
		attachedVerts[triangles[t].v3].push_back( triangles[t].v2 );

		// scale the UV texture coordinates upto pixel coordinates
		Texture *texture = &textures[triangles[t].texId];
		int rows = texture->texture.rows;
		int cols = texture->texture.cols;
		v1[0] *= cols;	v1[1] *= rows;
		v2[0] *= cols;	v2[1] *= rows;
		v3[0] *= cols;	v3[1] *= rows;

		float sideA = (v2-v1).norm();
		float sideB = (v3-v2).norm();
		float sideC = (v1-v3).norm();

		float area = gncTK::Utils::triangleAreaFromSides(sideA, sideB, sideC);

		vertexPointPixelDensities[triangles[t].v1] += area;
		vertexPointPixelDensities[triangles[t].v2] += area;
		vertexPointPixelDensities[triangles[t].v3] += area;

		++triCounts[triangles[t].v1];
		++triCounts[triangles[t].v2];
		++triCounts[triangles[t].v3];
	}

	// calculate the point densities of vertices using DTFE algorithm
	// density = (M + 1) / Sum of areas of connected triangles.
	for (int v=0; v<vertexPointDensities.size(); ++v)
	{
		if (vertexPointPixelDensities[v] != 0)
		{
			// Find if this vertex is completely surrounded by triangles
			// this is done by checking the list of attached triangle vertices
			// contains exactly two copies of each vertex in it.
			std::sort(attachedVerts[v].begin(), attachedVerts[v].end());
			bool okay = true;
			for (int t=0; t<attachedVerts[v].size(); t+=2)
				if (attachedVerts[v][t] != attachedVerts[v][t+1])
					okay = false;

			if (okay)
				// calculate point density using DTFE equation.
				//vertexPointPixelDensities[v] = ((1.0 + triCounts[v]) / vertexPointPixelDensities[v]) / 2.0;
				vertexPointPixelDensities[v] = 3.0 / vertexPointPixelDensities[v];
			else
				vertexPointPixelDensities[v] = NAN;
		}
		else
			vertexPointPixelDensities[v] = NAN;
	}

	// the vertex point density values are now calculated
}

/// Method to calculate the projected geometry ratio for each triangle in the mesh given the set sensor origin
/*
 * The project geometry ratio is measure in meters squared per steradian, and represents the ratio between
 * surface area and subtended solid angle at a point on a surface.
 */
void gncTK::Mesh::calculateTriangleProjectedGeometryRatios()
{
	triangleProjectedGeometryRatios.clear();
	triangleIncidentAngles.clear();

	for (int t=0; t<triangles.size(); ++t)
	{
		// first calculate the mean distance and incident angle of this triangle from the point of view of the sensor
		Eigen::Vector3f centroid = (vertices[triangles[t].v1] +
								    vertices[triangles[t].v2] +
								    vertices[triangles[t].v3]   ) / 3.0;

		// calculate triangle normal
		Eigen::Vector3f v1 = vertices[triangles[t].v1];
		Eigen::Vector3f v2 = vertices[triangles[t].v2];
		Eigen::Vector3f v3 = vertices[triangles[t].v3];
		Eigen::Vector3f normal = (v2-v1).cross(v3-v1);
		normal.normalize();

		// calculate angle between normal and vector from sensor origin to triangle centroid
		Eigen::Vector3f sensorRay = sensorOrigin - centroid;//((v1+v2+v3) / 3);
		float angle = acos( (normal.dot(sensorRay)) / (normal.norm() * sensorRay.norm()) );

		//Eigen::Vector3f point = centroid + normal;

		float distance = (centroid - sensorOrigin).norm();

		float incidentAngle;
		if (angle <= M_PI/2)
			incidentAngle = angle;
		else
			incidentAngle = M_PI - angle;

		//= fabs(angle - (M_PI/2) );

		//float ratio = cos(incidentAngle) * (distance*distance);// * 4.0 * M_PI; <<<< WTF!!!

		float ratio = (distance*distance) / cos(incidentAngle);

		/*if (t % 1000 == 0)
		{
			ROS_WARN("proj geom: distance=%f, angle=%f, incidentAngle=%f, ratio=%f,  centroid=[%f %f %f]",
					 distance, angle,
					 incidentAngle, ratio,
					 centroid(0), centroid(1), centroid(2));
			ROS_WARN("sensor=[%f %f %f], normal=[%f %f %f]",
					sensorOrigin(0), sensorOrigin(1), sensorOrigin(2),
					 normal(0), normal(1), normal(2));
			ROS_WARN("------");
		}*/

		triangleProjectedGeometryRatios.push_back(ratio);
		triangleIncidentAngles.push_back(incidentAngle * (180 / M_PI));
	}
}

/// method to calculate the triangle bounding boxes
void gncTK::Mesh::calculateTriBoxes()
{
	if (triangles.size() == triangleBoxes.size())
		return;

	Eigen::Vector3f extentsMin, extentsMax;
	Eigen::Matrix<float, 3, 2> extents;
	std::vector<int> triangleRefs;

	triangleBoxes.clear();

	for (int t=0; t<triangles.size(); ++t)
	{
		for (int d=0; d<3; ++d)
		{
			extentsMin[d] = vertices[triangles[t].v1][d];
			extentsMax[d] = vertices[triangles[t].v1][d];

			if (vertices[triangles[t].v2][d] < extentsMin[d])
				extentsMin[d] = vertices[triangles[t].v2][d];
			if (vertices[triangles[t].v3][d] < extentsMin[d])
				extentsMin[d] = vertices[triangles[t].v3][d];

			if (vertices[triangles[t].v2][d] > extentsMax[d])
				extentsMax[d] = vertices[triangles[t].v2][d];
			if (vertices[triangles[t].v3][d] > extentsMax[d])
				extentsMax[d] = vertices[triangles[t].v3][d];
		}
		extents.block(0,0,3,1) = extentsMin;
		extents.block(0,1,3,1) = extentsMax;
		triangleBoxes.push_back(extents);

		triangleRefs.push_back(t);
	}

	// populate RTree for rapid searching of triangle bounding boxes
	ROS_INFO("Starting to populate RTree");
	triangleRTree.bulkLoad(triangleRefs, triangleBoxes);
	ROS_INFO("Finished populating RTree");

	printf("Calculate %d triangle boxes and made a tree with %d triangle boxes",
		   (int)triangleBoxes.size(),
		   triangleRTree.count());

}

/// Method to return a single channel float32 image with the depth of the surface
/*
 * The camera image plane locations are defined by the texture UV coordinates. pixels
 * which do not see any of the surface will be NANs.
 */
cv::Mat gncTK::Mesh::generateDepthMap()
{
	GLenum errCode;

	if (textures.size() != 1)
	{
		ROS_ERROR("Error can only generate depth maps from mesh's that contain exactly one texture. This has %d.",
				  (int)textures.size());

		cv::Mat nothing(10,10,CV_32F);
		return nothing;
	}

	// calculate vertex depth values and find maximum depth
	float maxDepth = 0;
	//Eigen::Vector3f cameraPos = fusionFunction.getCameraLocation();
	std::vector<float> vertexDepths;
	for (int v=0; v<vertices.size(); ++v)
	{
		float depth = vertices[v].norm();
		vertexDepths.push_back(depth);
		if (depth > maxDepth)
			maxDepth = depth;
	}

	//ROS_WARN("depth map max depth is [%f]", maxDepth);

	// setup GL rendering context
	int width = textures[0].texture.cols;
	int height = textures[0].texture.rows;
	gncTK::Utils::GLBufferInfo GLBuff = gncTK::Utils::setupOffscreenGLBuffer(width, height);

	cv::Mat partialResult(height,width, CV_32FC4, cv::Scalar(0,1,0,0));
	cv::Mat result(height,width, CV_32F, 0.0);

	// render mesh into buffer
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepthf(1.0f);
	//glDepthRange(1,0); // reverse depth buffer

	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	// render depth, height and angle geometry to frame buffer
	glClearColor(-1.0, 0.0-maxDepth, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<triangles.size(); ++t)
	{
		float d1 = vertexDepths[triangles[t].v1];
		float d2 = vertexDepths[triangles[t].v2];
		float d3 = vertexDepths[triangles[t].v3];

		Eigen::Vector2f t1 = texCoords[triangles[t].t1];
		Eigen::Vector2f t2 = texCoords[triangles[t].t2];
		Eigen::Vector2f t3 = texCoords[triangles[t].t3];

		glColor4f(0, d1/maxDepth, 0.0, 1.0);
		glVertex3f(t1[0]*width, t1[1]*height, d1/maxDepth);

		glColor4f(0, d2/maxDepth, 0.0, 1.0);
		glVertex3f(t2[0]*width, t2[1]*height, d2/maxDepth);

		glColor4f(0, d3/maxDepth, 0.0, 1.0);
		glVertex3f(t3[0]*width, t3[1]*height, d3/maxDepth);
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		ROS_ERROR("Error brefore reading color buffer. %s\n", gluErrorString(errCode));

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,partialResult.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		ROS_ERROR("Error on reading color buffer. %s\n", gluErrorString(errCode));

	GLBuff.free();

	// copy channels into final output
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			float depth = partialResult.at<cv::Vec4f>(r,c)[1];

			if (depth == 0.0)
			{
				result.at<float>(r,c) = NAN;
			}
			else
			{
				result.at<float>(r,c) = depth * maxDepth;
			}
		}

	return result;
}

/// Method to return a three channel float32 image with the 3D location of the surface
/*
 * The camera image plane locations are defined by the texture UV coordinates. pixels
 * which do not see any of the surface will be NANs.
 */
cv::Mat gncTK::Mesh::generate3DMap()
{
	GLenum errCode;

	if (textures.size() != 1)
	{
		ROS_ERROR("Error can only generate depth maps from mesh's that contain exactly one texture. This has %d.",
				  (int)textures.size());

		cv::Mat nothing(10,10,CV_32F);
		return nothing;
	}

	// calculate vertex depth values and find maximum depth
	float maxDepth = 0;
	//Eigen::Vector3f cameraPos = fusionFunction.getCameraLocation();
	std::vector<float> vertexDepths;
	for (int v=0; v<vertices.size(); ++v)
	{
		float depth = vertices[v].norm();
		vertexDepths.push_back(depth);
		if (depth > maxDepth)
			maxDepth = depth;
	}

	//ROS_WARN("depth map max depth is [%f]", maxDepth);

	// setup GL rendering context
	int width = textures[0].texture.cols;
	int height = textures[0].texture.rows;
	gncTK::Utils::GLBufferInfo GLBuff = gncTK::Utils::setupOffscreenGLBuffer(width, height, GL_RGBA32F);

	cv::Mat partialResult(height,width, CV_32FC4, cv::Scalar(0,0,0,0));
	cv::Mat result(height,width, CV_32F, 0.0);

	// render mesh into buffer
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepthf(1.0f);
	//glDepthRange(1,0); // reverse depth buffer

	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	// render depth, height and angle geometry to frame buffer
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<triangles.size(); ++t)
	{
		float d1 = vertexDepths[triangles[t].v1];
		float d2 = vertexDepths[triangles[t].v2];
		float d3 = vertexDepths[triangles[t].v3];

		Eigen::Vector3f v1 = vertices[triangles[t].v1];
		Eigen::Vector3f v2 = vertices[triangles[t].v2];
		Eigen::Vector3f v3 = vertices[triangles[t].v3];

		Eigen::Vector2f t1 = texCoords[triangles[t].t1];
		Eigen::Vector2f t2 = texCoords[triangles[t].t2];
		Eigen::Vector2f t3 = texCoords[triangles[t].t3];

		glColor4f(v1[0], v1[1], v1[2], 1.0);
		glVertex3f(t1[0]*width, t1[1]*height, d1/maxDepth);

		glColor4f(v2[0], v2[1], v2[2], 1.0);
		glVertex3f(t2[0]*width, t2[1]*height, d2/maxDepth);

		glColor4f(v3[0], v3[1], v3[2], 1.0);
		glVertex3f(t3[0]*width, t3[1]*height, d3/maxDepth);
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		ROS_ERROR("Error brefore reading color buffer. %s\n", gluErrorString(errCode));

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,partialResult.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		ROS_ERROR("Error on reading color buffer. %s\n", gluErrorString(errCode));

	GLBuff.free();

	// copy channels into final output
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			cv::Vec4f pos3D = partialResult.at<cv::Vec4f>(r,c);

			if (pos3D[0] == 0.0 && pos3D[1] == 0.0 && pos3D[2] == 0.0)
			{
				pos3D[0] = pos3D[1] = pos3D[2] = NAN;
			}
			partialResult.at<cv::Vec4f>(r,c) = pos3D;
		}

	return partialResult;
}

cv::Mat gncTK::Mesh::projectToCamera(gncTK::FusionFunction camera, int width, int height, std::string channel)
{
	GLenum errCode;

	// calculate vertex depth values and find maximum depth
	float maxDepth = 0;
	Eigen::Vector3f cameraPos = camera.getCameraLocation();
	std::vector<float> vertexDepths;
	for (int v=0; v<vertices.size(); ++v)
	{
		float depth = (cameraPos-vertices[v]).norm();
		vertexDepths.push_back(depth);
		if (depth > maxDepth)
			maxDepth = depth;
	}

	// setup GL rendering context
	gncTK::Utils::GLBufferInfo GLBuff = gncTK::Utils::setupOffscreenGLBuffer(width, height, GL_RGBA32F);

	cv::Mat partialResult(height,width, CV_32FC4, cv::Scalar(0,1,0,0));
	cv::Mat result(height,width, CV_32F, std::numeric_limits<float>::quiet_NaN());

	// render mesh into buffer
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepthf(1.0f);
	//glDepthRange(1,0); // reverse depth buffer

	glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
	glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);

	// pre-calculate UV scaling factors
	float uScale = width;
	float vScale = height;

	// render depth, height and angle geometry to frame buffer
	glClearColor(-1000000.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBegin(GL_TRIANGLES);
	for (int t=0; t<triangles.size(); ++t)
	{
		float d1 = vertexDepths[triangles[t].v1];
		float d2 = vertexDepths[triangles[t].v2];
		float d3 = vertexDepths[triangles[t].v3];
		float p1,p2,p3;

		if (channel == "pointDensity")
		{
			p1 = vertexPointDensities[triangles[t].v1];
			p2 = vertexPointDensities[triangles[t].v2];
			p3 = vertexPointDensities[triangles[t].v3];
		}
		if (channel == "pointPixelDensity")
		{
			p1 = vertexPointPixelDensities[triangles[t].v1];
			p2 = vertexPointPixelDensities[triangles[t].v2];
			p3 = vertexPointPixelDensities[triangles[t].v3];
		}
		if (channel == "triangleProjectedGeometry")
		{
			p1 = p2 = p3 = triangleProjectedGeometryRatios[t];
		}
		if (channel == "triangleIncidentAngle")
		{
			p1 = p2 = p3 = triangleIncidentAngles[t];
		}

		// if any of the vertex projected values are NAN then don't render this triangle
		// openGL hardware doesn't handle float NANs!
		if (std::isnan(p1) || std::isnan(p2) || std::isnan(p2))
			continue;

		Eigen::Vector2f t1 = texCoords[triangles[t].t1];
		Eigen::Vector2f t2 = texCoords[triangles[t].t2];
		Eigen::Vector2f t3 = texCoords[triangles[t].t3];

		glColor4f(p1, 0.0, 0.0, 1.0);
		glVertex3f((t1[0]*uScale), (t1[1]*vScale), d1/maxDepth);

		glColor4f(p2, 0.0, 0.0, 1.0);
		glVertex3f((t2[0]*uScale), (t2[1]*vScale), d2/maxDepth);

		glColor4f(p3, 0.0, 0.0, 1.0);
		glVertex3f((t3[0]*uScale), (t3[1]*vScale), d3/maxDepth);
	}
	glEnd();

	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error before reading color buffer. %s\n", gluErrorString(errCode));

	// read partial buffer out
	glReadnPixels(0,0, width,height, GL_RGBA, GL_FLOAT, width*height*16,partialResult.ptr(0));
	if ((errCode=glGetError()) != GL_NO_ERROR)
		printf("Error on reading color buffer. %s\n", gluErrorString(errCode));

	GLBuff.free();

	// copy channels into final output
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			float value = partialResult.at<cv::Vec4f>(r,c)[0];
			if (value != -1000000)
				result.at<float>(r,c) = value;
		}

	partialResult.release();

	return result;
}

cv::Mat gncTK::Mesh::projectPointPixelDensityToCamera(int width, int height)
{
	// create 10x10 bins
	int binsX = ceil(width/10.0);
	int binsY = ceil(height/10.0);
	cv::Mat bins(binsY,binsX, CV_32F);
	bins = 0.0;

	// add all vertices into their respective bins
	for (int v=0; v<vertices.size(); ++v)
	{
		int bX = ( (texCoords[v][0])  * width ) / 10;
		int bY = ((1-texCoords[v][1]) * height) / 10;

		if (bX >= 0 && bX < binsX &&
			bY >= 0 && bY < binsY)
		{
			++bins.at<float>(bY, bX);
		}
	}

	// up-sample densities via point sampling to full frame size
	cv::Mat output(height, width, CV_32F);
	output = -1.0;
	for (int r=0; r<height; ++r)
		for (int c=0; c<width; ++c)
		{
			int bX = c / 10;
			int bY = r / 10;

			output.at<float>(r,c) = bins.at<float>(bY, bX) / 100.0;
		}

	return output;
}




