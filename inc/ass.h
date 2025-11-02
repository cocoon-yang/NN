#pragma once
#include <random>
#include <cmath>

typedef float DataType;
typedef unsigned int uint;

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<double> dis(-0.5, 0.5);

static float getRandVal()
{
	float a = (float)dis(gen);
	return a;
}

/*
   https://github.com/libfann/fann

	The activation functions used for the neurons during training. The activation functions
	can either be defined for a group of neurons by <fann_set_activation_function_hidden> and
	<fann_set_activation_function_output> or it can be defined for a single neuron by <fann_set_activation_function>.

	The steepness of an activation function is defined in the same way by
	<fann_set_activation_steepness_hidden>, <fann_set_activation_steepness_output> and <fann_set_activation_steepness>.

   The functions are described with functions where:
   * x is the input to the activation function,
   * y is the output,
   * s is the steepness and
   * d is the derivation.

   LINEAR - Linear activation function.
	 * span: -inf < y < inf
	 * y = x*s, d = 1*s
	 * Can NOT be used in fixed point.

   THRESHOLD - Threshold activation function.
	 * x < 0 -> y = 0, x >= 0 -> y = 1
	 * Can NOT be used during training.

   THRESHOLD_SYMMETRIC - Threshold activation function.
	 * x < 0 -> y = 0, x >= 0 -> y = 1
	 * Can NOT be used during training.

   SIGMOID - Sigmoid activation function.
	 * One of the most used activation functions.
	 * span: 0 < y < 1
	 * y = 1/(1 + exp(-2*s*x))
	 * d = 2*s*y*(1 - y)

   SIGMOID_STEPWISE - Stepwise linear approximation to sigmoid.
	 * Faster than sigmoid but a bit less precise.

   SIGMOID_SYMMETRIC - Symmetric sigmoid activation function, aka. tanh.
	 * One of the most used activation functions.
	 * span: -1 < y < 1
	 * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
	 * d = s*(1-(y*y))

   SIGMOID_SYMMETRIC - Stepwise linear approximation to symmetric sigmoid.
	 * Faster than symmetric sigmoid but a bit less precise.

   GAUSSIAN - Gaussian activation function.
	 * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
	 * span: 0 < y < 1
	 * y = exp(-x*s*x*s)
	 * d = -2*x*s*y*s

   GAUSSIAN_SYMMETRIC - Symmetric gaussian activation function.
	 * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
	 * span: -1 < y < 1
	 * y = exp(-x*s*x*s)*2-1
	 * d = -2*x*s*(y+1)*s

   ELLIOT - Fast (sigmoid like) activation function defined by David Elliott
	 * span: 0 < y < 1
	 * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
	 * d = s*1/(2*(1+|x*s|)*(1+|x*s|))

   ELLIOT_SYMMETRIC - Fast (symmetric sigmoid like) activation function defined by David Elliott
	 * span: -1 < y < 1
	 * y = (x*s) / (1 + |x*s|)
	 * d = s*1/((1+|x*s|)*(1+|x*s|))

	LINEAR_PIECE - Bounded linear activation function.
	 * span: 0 <= y <= 1
	 * y = x*s, d = 1*s

	LINEAR_PIECE_SYMMETRIC - Bounded linear activation function.
	 * span: -1 <= y <= 1
	 * y = x*s, d = 1*s

	SIN_SYMMETRIC - Periodical sinus activation function.
	 * span: -1 <= y <= 1
	 * y = sin(x*s)
	 * d = s*cos(x*s)

	COS_SYMMETRIC - Periodical cosinus activation function.
	 * span: -1 <= y <= 1
	 * y = cos(x*s)
	 * d = s*-sin(x*s)

	SIN - Periodical sinus activation function.
	 * span: 0 <= y <= 1
	 * y = sin(x*s)/2+0.5
	 * d = s*cos(x*s)/2

	COS - Periodical cosinus activation function.
	 * span: 0 <= y <= 1
	 * y = cos(x*s)/2+0.5
	 * d = s*-sin(x*s)/2
*/
enum ACTIVATIONFUNC
{
	LINEAR = 0,
	POW,
	THRESHOLD,
	THRESHOLD_SYMMETRIC,
	SIGMOID,
	SIGMOID_STEPWISE,
	SIGMOID_SYMMETRIC,
	SIGMOID_SYMMETRIC_STEPWISE,
	GAUSSIAN,
	GAUSSIAN_SYMMETRIC,
	/* Stepwise linear approximation to gaussian.
	 * Faster than gaussian but a bit less precise.
	 * NOT implemented yet.
	 */
	 GAUSSIAN_STEPWISE,
	 ELLIOT,
	 ELLIOT_SYMMETRIC,
	 LINEAR_PIECE,
	 LINEAR_PIECE_SYMMETRIC,
	 SIN_SYMMETRIC,
	 COS_SYMMETRIC,
	 SIN,
	 COS
};


static DataType pow_real(DataType value, DataType order)
{
	DataType result = pow(value, order);
	return result;
}
static DataType pow_derive(DataType steepness, DataType value, DataType order)
{
	DataType result = pow(value, order - 1.0);
	result = steepness * result * order;
	if (0.0 == order)
	{
		result = 0.0;
	}
	return result;
}

static DataType sigmoid_real(DataType value)
{
	DataType result = 1.0f / (1.0f + exp(-2.0f * value));
	return result;
}
static DataType sigmoid_derive(DataType steepness, DataType value)
{
	DataType result = 2.0f * steepness * value * (1.0f - value);
	return result;
}

static DataType sigmoid_symmetric_real(DataType value)
{
	DataType result = 2.0f / (1.0f + exp(-2.0f * value)) - 1.0f;
	return result;
}
static DataType sigmoid_symmetric_derive(DataType steepness, DataType value)
{
	DataType result = steepness * (1.0f - (value * value));
	return result;
}


static DataType gaussian_real(DataType value)
{
	DataType result = exp(-value * value);
	return result;
}
static DataType gaussian_derive(DataType steepness, DataType value)
{
	DataType result = steepness * (1.0f - (value * value));
	return result;
}

static DataType sin_symmetric_real(DataType value)
{
	DataType result = sin(value);
	return result;
}
static DataType sin_symmetric_derive(DataType steepness, DataType value)
{
	DataType result = steepness * cos(steepness * value);
	return result;
}

static DataType sin_real(DataType value)
{
	DataType result = sin(value) / 2.0f + 0.5f;
	return result;
}
static DataType sin_derive(DataType steepness, DataType value)
{
	DataType result = steepness * cos(steepness * value) / 2.0f;
	return result;
}

static DataType cos_symmetric_real(DataType value)
{
	DataType result = cos(value);
	return result;
}
static DataType cos_symmetric_derive(DataType steepness, DataType value)
{
	DataType result = -steepness * sin(steepness * value);
	return result;
}

static DataType cos_real(DataType value)
{
	DataType result = cos(value) / 2.0f + 0.5f;
	return result;
}
static DataType cos_derive(DataType steepness, DataType value)
{
	DataType result = -steepness * sin(steepness * value) / 2.0f;
	return result;
}
