#ifndef GNCTK_STATS1D_H_
#define GNCTK_STATS1D_H_

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

stats1d.h

one dimensional statistical summary storage object
---------------------------------------------------

This object stores the statistical summary of a set of
one dimensional continuous values. Including the Naive mean,
min, max, standard deviation, min_inlier, max_inlier and a
histogram of densities.

-------------------------------------------------------------*/

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace gncTK
{
	class Stats1D;
};

class gncTK::Stats1D
{
public:
	Stats1D(std::vector<double> values, int histogramBinCount = 20, float inlierThreshold = 3);

	// return a string list with basic stat header for CSV writing
	static std::vector<std::string> csvGetBasicHeaders();

	// return a string list with basic stats for CSV writing
	std::vector<std::string> csvGetBasic();

	// return a string list with all stats for CSV writing
	std::vector<std::string> csvGetAll();

	// return a cv::Mat with the statistics and histogram displayed
	cv::Mat getStatsImage();

	double mean;
	double min,max;
	double meanInlier, minInlier, maxInlier;
	double stdDev;
	unsigned int n;

	std::vector<double> histogramBins;
	std::vector<int> histogramTotals;
};

#endif /* GNCTK_STATS_1D_H_ */
