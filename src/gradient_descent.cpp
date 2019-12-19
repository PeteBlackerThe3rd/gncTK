
#include <gradient_descent.h>
#include <stdio.h>

/// Default constructor resets all initial values
gncTK::GradientDescent::GradientDescent()
{
  iterationCallback = NULL;
  qualityFunction = NULL;

  alpha = 1.2;
  beta = 0.6;
  lambda = 0.9;

  dimensionCount = 0;

  maximumIterations = 1000;
  initialStepSize = 0.1;
  minimumStepSize = 0;

  useWeights = false;

  direction = gncTK_GDOP_DESCENT;
}

void gncTK::GradientDescent::setInitialState(Eigen::VectorXd initialState)
{
	this->initialState = initialState;
	dimensionCount = initialState.rows();
}

///
void gncTK::GradientDescent::setUniformWeights()
{
	weights = Eigen::VectorXd::Ones(dimensionCount);
}

/// Method to perform final target calibration on many fusion scans
float gncTK::GradientDescent::runOptimisation(bool showMessages)
{
  // clone the initial state into the current state vector
  Eigen::VectorXd currentState = initialState;

  // get initial quality metric
  float currentQuality = qualityFunction(currentState);

  if (std::isnan(currentQuality))
  {
	printf("Gradient Descent Error : quality function returned NAN\n");
	return -1;
  }

  if (!useWeights)
	setUniformWeights();

  if (showMessages)
	  printf("Starting gradient descent, initial quality metric [%f]\n", currentQuality);

  int iteration = 0;
  float stepSize = initialStepSize;

  Eigen::VectorXd previousStateDelta;
  Eigen::VectorXd stateDelta = Eigen::VectorXd::Zero(dimensionCount);

  do
  {
	float baseQuality = qualityFunction(currentState);

	//if (showMessages)
		//printf("Partial Differentials:\n");

	// calculate partial differentials at this point in the state space
	Eigen::VectorXd partialDiffs = Eigen::VectorXd::Zero(dimensionCount);
	float sumOfSquares = 0;
	for (int d=0; d<dimensionCount; ++d)
	{
	  currentState[d] += stepSize * weights[d];
	  float more = qualityFunction(currentState);
	  currentState[d] -= stepSize * weights[d];

	  if (showMessages)
	  {
		  printf("quality with dimension [%d] delta : %f\n", d, more);
	  }

	  partialDiffs[d] = ((more - baseQuality) / (stepSize));

	  //if (showMessages)
		  //printf("%d -> %f [a=%f b=%f s=%.10f]\n", d+1, partialDiffs[d], baseQuality, more, stepSize);

	  sumOfSquares += partialDiffs[d] * partialDiffs[d];
	}



	float gradient = sqrt(sumOfSquares);
	float cauchyCoeff;

	  //if (showMessages)
		  //printf("Grad = %f, couchy = %f\n", gradient, cauchyCoeff);

	// if the gradient is zero then this must be a local minimum (or possibly maximum)
	if (gradient == 0)
	{
	  printf("Gradient of zero reached.\n");
	  finalState = currentState;
	  return currentQuality;
	}

	// try different step sizes until an improved cost or limit is reached
	bool qualityImproved = false;
	double newQuality;
	do
	{
		// calculate update delta vector
		cauchyCoeff = stepSize / gradient;
		stateDelta.array() = partialDiffs.array() * weights.array() * direction * cauchyCoeff;

		// if there was a previous delta state then apply the lambda momentum factor
		if (previousStateDelta.rows() == currentState.rows())
			stateDelta.array() += previousStateDelta.array() * lambda;

		Eigen::VectorXd newState = Eigen::VectorXd::Zero(dimensionCount);
		newState.array() = currentState.array() + stateDelta.array();

		// find quality at new location
		newQuality = qualityFunction(newState);
		qualityImproved = ((newQuality * direction) > (currentQuality * direction));

		if (std::isnan(newQuality))
		{
			printf("Gradient Descent Error : quality function returned NAN\n");
			return -1;
		}

		if (qualityImproved)
		{
			stepSize *= alpha;
			printf("quality improved to [%f]\n", newQuality);
		}
		else
		{
			stepSize *= beta;
			printf("shortened Step Size to [%f] quality here [%f]\n", stepSize, newQuality);
		}

		// if an iteration callback has been set then call it
		if (iterationCallback != NULL)
		{
		  iterationCallback(this,
							iteration,
							currentState,
							stepSize,
							currentQuality);
		}

		++iteration;

	} while(!qualityImproved && iteration < maximumIterations && stepSize > minimumStepSize);

	printf("improved [%f]", newQuality);

	currentQuality = newQuality;
	currentState.array() += stateDelta.array();
	previousStateDelta = stateDelta;

	/*std::vector<float> deltaState;
	for (int d=0; d<dimensionCount; ++d)
	{
	  float delta = (cauchyCoeff * direction * partialDiffs[d] * weights[d]);

	  if (showMessages)
		  printf("delta [%d] -> %f\n", d+1, delta);

	  deltaState.push_back( delta );
	}*/

	// if there was a previous state delta (if this isn't the first iteration)
	/*if (previousStateDelta.size() == dimensionCount)
	{
	  for (int d=0; d<dimensionCount; ++d)
		deltaState[d] += lambda * previousStateDelta[d];
	}*/

	// advance state to next location using state delta vector
	/*std::vector<float> nextState;
	for (int d=0; d<dimensionCount; ++d)
	{
	  nextState.push_back(currentState[d] + deltaState[d]);
	}

	// get the new quality metric
	float newQuality = qualityFunction(&nextState);

	if (newQuality != newQuality)
	{
	  printf("Gradient Descent Error : quality function returned NAN\n");
	  finalState = currentState;
	  return -1;
	}

	// if the quality improved then we expand the gradient variable
	// or if the quality decreased we contract it
	if ((newQuality * direction) < (currentQuality * direction))
	{
	  // quality moved away from the target direction so contract step size and ignore this next state vector
	  stepSize *= beta;
	  printf("#### failed #####\n");
	}
	else
	{
	  // quality moves towards the target direction so expand the step size and adopt this next state vector
	  stepSize *= alpha;
	  currentState = nextState;
	  currentQuality = newQuality;

	  previousStateDelta = deltaState;

	  printf("---- improved -----\n");
	}

	++iteration;

    // if an iteration callback has been set then call it
	if (iterationCallback != NULL)
	{
	  iterationCallback(this,
			  	  	    iteration,
						&currentState,
						stepSize,
						currentQuality);
	}*/

  } while (stepSize > minimumStepSize && iteration < maximumIterations);

  // save final state vector
  finalState = currentState;

  return (currentQuality);
}


