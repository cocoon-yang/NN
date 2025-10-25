#pragma once 
#include <iostream>
#include <vector> 
#include <memory>  
#include <random>
#include <cmath>

/* 
 ref  https://github.com/libfann/fann
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

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<double> dis(-1.0, 1.0);

static float getRandVal()
{
	float a = (float)dis(gen);
	return a;
}
 
typedef float DataType; 
typedef unsigned int uint;
 

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


class Connection;
class Layer;

class Neuro
{
public:
	Neuro(int num);
	virtual ~Neuro();

	Neuro(const Neuro& RHS);

	Neuro& operator = (const Neuro& RHS);

public:
	void init(Layer* pLayer = nullptr);

	void setWeight(std::vector<float>& values);

public:

	DataType run(std::shared_ptr<DataType[]> pVal);
	
	DataType getValue();

	void updataWeight(std::shared_ptr<DataType[]> pVal, DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate);

	void show(int index);

	std::string toStr();

	std::string toStrN();

	void setOrder(float val);

	int getOrder();

	std::shared_ptr<Connection> getConnection(size_t index);

private:
	DataType calcuOutput(DataType value);
	DataType calcuGrad(DataType diffVar, DataType value); 

public:
	int layerID;
	int index;

private:
	float order;
	bool ACTIVE;
	DataType _value;
	std::size_t _inputNum; 
	std::size_t _type;
	std::shared_ptr<DataType[]> _weights; 
	std::vector<std::shared_ptr<Connection>> _pConnections;
};


class Layer {
public:
	Layer(int inNum, int outNum, int id = 0);
	virtual ~Layer();

	Layer(const Layer& RHS);

	Layer& operator = (const Layer& RHS);

public:
	void init(Layer* pLayer = nullptr);

	void clear();

	int getID();

	int getInputNum();

	int getOutputNum();

	int getConnectedLayerID();

	void setConnectedLayerID(int index);

	std::shared_ptr<Neuro> getNeuron(size_t index);

	void show(int index);

	std::string toStr();

public:
	void forward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output);

	void backward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr);

private: 
	int _id; 
	int _connectLayerID;

	int _inputNum;
	int _outputNum;

	DataType* _pWeights; 
	DataType* _pBiases; 
	std::vector<std::shared_ptr<Neuro>> _pNeros; 
};



class Connection
{
public:
	Connection() {}
	~Connection() {}

public:
	DataType _weight;
	std::shared_ptr<Neuro> _pInputNero;
	std::shared_ptr<Neuro> _pOutputNero;
};

class NN {
public:
	NN();

	~NN();

public:
	void setModel(std::vector<uint> theTop);

	void init();

	void clear();

	void show();

	void load(const char* fileName); 

	void save(const char* fileName);

public:
	bool isFinish();
	void setFinish(bool state);

public:
	void train(float* input, float* y, float lr);
	  
	void softmax(float* input, int size);
	 
	int predict(float* input);
private:
	std::vector<Layer*> model;
	std::vector<uint> topology; 
	 
	std::vector<std::shared_ptr<DataType[]>> _hidenData; 

	bool _FINISH;
};

