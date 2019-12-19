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

stats1d.cpp

one dimensional statistical summary storage object
---------------------------------------------------

This object stores the statistical summary of a set of
one dimensional continuous values. Including the Naive mean,
min, max, standard deviation, min_inlier, max_inlier and a
histogram of densities.

-------------------------------------------------------------*/
#include <vector>
#include <stdio.h>
#include <math.h>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stats1d.h>

// create stats object with summary of set of values given
gncTK::Stats1D::Stats1D(std::vector<double> values, int histogramBinCount, float inlierThreshold)
{
	if (values.size() == 0)
	{
		printf("Error : Trying to calculate statistical summary of an empty set of values!\n");
		return;
	}

	n = values.size();

	// calculate mean, min and max
	double sum = 0;
	min = max = values[0];
	for (int i=0; i<values.size(); ++i)
	{
		sum += values[i];
		if (values[i] < min) min = values[i];
		if (values[i] > max) max = values[i];
	}
	mean = sum / values.size();

	// calculate standard deviation
	double sumDist = 0;
	for (int i=0; i<values.size(); ++i)
	{
		sumDist += fabs(values[i] - mean);
	}
	stdDev = sumDist / values.size();

	// find the mean, min and max inliers
	sum = 0;
	int n = 0;
	minInlier = maxInlier = mean;
	for (int i=0; i<values.size(); ++i)
	{
		if (values[i] >= mean - inlierThreshold*stdDev &&
			values[i] <= mean + inlierThreshold*stdDev)
		{
			sum += values[i];
			n ++;
			if (values[i] < minInlier) minInlier = values[i];
			if (values[i] > maxInlier) maxInlier = values[i];
		}
	}
	meanInlier = sum / n;

	// create histogram bins covering the range of inliers
	histogramBins.push_back(minInlier);
	histogramTotals.push_back(0.0);
	for (int h=0; h<histogramBinCount; ++h)
	{
		histogramBins.push_back( (h/(double)histogramBinCount) * (maxInlier-minInlier) );
		histogramTotals.push_back(0);
	}

	// add values into histogram
	for (int i=0; i<values.size(); ++i)
	{
		int bin = ((values[i] - minInlier) / (maxInlier-minInlier)) * histogramBinCount;
		if (bin > 0 && bin <= histogramBinCount)
			++histogramTotals[bin];
	}
}

// return a string list with basic stats for CSV writing
std::vector<std::string> gncTK::Stats1D::csvGetBasicHeaders()
{
	std::vector<std::string> header;

	header.push_back("N");
	header.push_back("Mean");
	header.push_back("Min");
	header.push_back("Max");
	header.push_back("StdDev");
	header.push_back("Mean Inlier");
	header.push_back("Min Inlier");
	header.push_back("Max Inlier");

	return header;
}

// return a string list with basic stats for CSV writing
std::vector<std::string> gncTK::Stats1D::csvGetBasic()
{
	std::vector<std::string> output;

	char temp[20];

	sprintf(temp, "%d", n);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", mean);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", min);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", max);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", stdDev);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", meanInlier);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", minInlier);
	output.push_back(std::string(temp));
	sprintf(temp, "%f", maxInlier);
	output.push_back(std::string(temp));

	return output;
}

// return a string list with all stats for CSV writing
std::vector<std::string> gncTK::Stats1D::csvGetAll()
{
	std::vector<std::string> output = csvGetBasic();
	char temp[30];

	output.push_back(" ");

	for (int h=0; h<histogramTotals.size(); ++h)
	{
		sprintf(temp, "%d", histogramTotals[h]);
		output.push_back(std::string(temp));
	}

	output.push_back(" ");

	for (int h=0; h<histogramBins.size(); ++h)
	{
		sprintf(temp, "%f", histogramBins[h]);
		output.push_back(std::string(temp));
	}

	return output;
}

// return a cv::Mat with the statistics and histogram displayed
cv::Mat gncTK::Stats1D::getStatsImage()
{
	cv::Mat histogram(600, 800, CV_8UC3, cv::Scalar(0,0,0));

	cv::Scalar axesColor(255,255,255);
	cv::Scalar barsColor(255,0,0);

	// draw histogram bars
	int maxFreq = 0;
	for (int b=1; b<histogramTotals.size(); ++b)
	{
		if (histogramTotals[b] > maxFreq)
			maxFreq = histogramTotals[b];
	}
	for (int b=1; b<histogramBins.size(); ++b)
	{
		float width = 600.0 / (histogramBins.size()-1);
		float height = (histogramTotals[b] / (float)maxFreq) * 300;
		cv::rectangle(histogram,
					cv::Point(100 + width*(b-1), 500 - height),
					cv::Point(100 + width*b, 500),
					barsColor,
					CV_FILLED);
	}

	// draw histogram axes
	cv::line(histogram, cv::Point(100,200), cv::Point(100,500), axesColor);
	cv::line(histogram, cv::Point(100,500), cv::Point(700,500), axesColor);

	// determine spacing
	float horzSpacing = (600.0 / (histogramBins.size()-1)) * histogramBins[0];// = axesIdealSpacing(600.0 / (histogramBins.size()-1), 100.0);
	float horzFactor = 1;
	float idealSpacing = 100;

	bool done = false;
	while (!done)
	{
		done = true;

		// if increasing the spacing improves it
		if (fabs((horzSpacing * 2) - idealSpacing) < fabs(horzSpacing - idealSpacing))
		{
			horzSpacing *= 2;
			horzFactor *= 2;
			done = false;
		}

		// if decreasing the spacing improves it
		if (fabs((horzSpacing / 2) - idealSpacing) < fabs(horzSpacing - idealSpacing))
		{
			horzSpacing /= 2;
			horzFactor /= 2;
			done = false;
		}
	}

	printf("spacing = %f and factor = %f\n", horzSpacing, horzFactor);

	int b=0;
	for (float tick = 0; tick<=600; tick += horzSpacing)
	{
		cv::line(histogram, cv::Point(100+tick,500), cv::Point(100+tick,520), axesColor);
		char label[256];
		sprintf(label, "%.2fcm", (b++ / horzFactor) * 100);

		int baseLine=0;
		cv::Size tSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 2, &baseLine);
		cv::putText(histogram, label, cv::Point(100+tick-(tSize.width/2),520 + tSize.height), cv::FONT_HERSHEY_PLAIN, 1, axesColor, 2);
	}

	return histogram;
}


