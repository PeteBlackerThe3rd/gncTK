#ifndef GNCTK_GRADIENT_DESCENT_H_
#define GNCTK_GRADIENT_DESCENT_H_

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

gradient_descent.h

gradient descent optimiser object
----------------------------------------------


-------------------------------------------------------------*/

#include <stdio.h>
#include <Eigen/Dense>
#include <math.h>

#define gncTK_GDOP_DESCENT		-1
#define gncTK_GDOP_ASCENT		1

namespace gncTK
{
	class GradientDescent;
};

class gncTK::GradientDescent
{
public:

	/// Default constructor
	GradientDescent();

	void setInitialState(Eigen::VectorXd initialState);

	void setUniformWeights();

	// Method to execute this gradient descent optimiser, return the final quality metric
	float runOptimisation(bool showMessages);


	//void printFinalState();

	/// The initial state vector
	Eigen::VectorXd initialState;

	/// The final optimised state vector
	Eigen::VectorXd finalState;

	/// The number of dimensions in the State Space being evaluated
	int dimensionCount;

	/// The direction of the optimisation, 1 is gradient ascent -1 is gradient descent.
	int direction;

	float initialStepSize;
	float minimumStepSize;
	int maximumIterations;

	/// Expansion factor
	float alpha;

	/// Contraction factor
	float beta;

	/// Momentum factor
	float lambda;

	bool useWeights;
	Eigen::VectorXd weights;

	/// Pointer to quality function, accepts a state vector and returns a floating point quality metric
	double (*qualityFunction)(Eigen::VectorXd state);

	/// Pointer to function that will be executed after each iteration of the optimisation function has completed
	/**
	 * This callback function can be used to add special behaviour every iteration of the optimisation function,
	 * such as writing the current state vector to a log or adjusting the dimension weights as the solution converges.
	 */
	void (*iterationCallback)(gncTK::GradientDescent *optimiser,
							  int iteration,
							  Eigen::VectorXd currentState,
							  float stepSize,
							  float currentQuality);
};

#endif /* GNCTK_GRADIENT_DESCENT_H_*/
