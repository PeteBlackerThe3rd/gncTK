/*-----------------------------------------------------------\\
||                                                           ||
||                 LIDAR fusion GNC project                  ||
||               ----------------------------                ||
||                                                           ||
||    Surrey Space Centre - STAR lab                         ||
||    (c) Surrey Iniversity 2017                             ||
||    Pete dot Blacker at Gmail dot com                      ||
||                                                           ||
\\-----------------------------------------------------------//

gncTK.h

Main include file for the GNC Toolkit library

This library includes a set of data storage objects and
processing objects for developing new GNC algorithms and
architectures for LIDAR fusion based ground robots.

Dependencies

ROS Kinetic
The Point Cloud Library (via ROS)
OpenCV (also via ROS)
The GridMap ROS Library
The Eigen matrix math library

-------------------------------------------------------------*/

#include <Eigen/Dense>

#include "utils.h"
#include "stats1d.h"
#include "lidar_sim.h"
#include "fusion_function.h"
#include "mesh.h"
#include "mesh_analysis.h"
#include "fusion.h"
//#include "feature_atlas.h"
#include "fusion_structured.h"
//#include "fusion_unstructured.h"
#include "fusion_quad_tree.h"
#include "calibration.h"
#include "dem.h"
#include "gradient_descent.h"
