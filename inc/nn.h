#pragma once 
#include <iostream>
#include <vector> 
#include <memory>  
#include <random>

/**
 * @brief https://zhuanlan.zhihu.com/p/31381234209 
*/

typedef float DataType; 
typedef unsigned int uint;

class Neuro
{
public:
	Neuro(int num);
	virtual ~Neuro();

	Neuro(const Neuro& RHS);

	Neuro& operator = (const Neuro& RHS);

public:
	void init();

public:
	//DataType run(DataType* pVal)
	//{
	//	DataType result = 0.0;
	//	DataType tmp = 1.0;
	//	if (_inputNum <= 0)
	//	{
	//		return result;
	//	} 
	//	result = 0.0f;
	//	
	//	for (size_t j = 0; j < _inputNum; j++)
	//	{
	//		tmp = pVal[j];
	//		result += tmp * (_weights[j]);
	//	}
	//	return result;
	//} 

	DataType run(std::shared_ptr<DataType[]> pVal);
	
	void updataWeight(std::shared_ptr<DataType[]> pVal, DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate);

	void show(int index);

private:
	std::size_t _inputNum; 
	std::shared_ptr<DataType[]> _weights;

	//std::vector<DataType> _weights;

	// DataType* _weights; 

	std::size_t order;
};


class Layer {
public:
	Layer(int inNum, int outNum);
	virtual ~Layer();

	Layer(const Layer& RHS);

	Layer& operator = (const Layer& RHS);

public:
	void init();

	void clear();

	int getInputNum();

	int getOutputNum();

	void show(int index);

public:
	void forward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output);

	void backward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr);
private:
	int _inputNum;
	int _outputNum;

	DataType* _pWeights; 
	DataType* _pBiases; 
	std::vector<std::shared_ptr<Neuro>> _pNeros; 
};



class NN {
public:
	NN();

	~NN();

public:
	void setModel(std::vector<uint> theTop);

	void init();

	void show();

public:
  

public:
	void train(float* input, float* y, float lr);
	 
	//void train( float* input, float* y, float lr) {
	// 
	//	Layer* inputLay = model[0]; 
	//	Layer* outputLay = model[1];
	//
	//	int HIDDEN_SIZE = inputLay->getOutputNum();
	//	int OUTPUT_SIZE = outputLay->getOutputNum();
	//
	//	float* hidden_output = new float[HIDDEN_SIZE]; 
	//	float* final_output = new float[OUTPUT_SIZE];
	//
	//	float* output_grad = new float[OUTPUT_SIZE] { 0 };
	//	float* hidden_grad = new float[HIDDEN_SIZE] { 0 };
	//	 
	//	 
	//	// 前向传递：从输入层到隐藏层
	//	inputLay->forward(input, hidden_output);
	// 
	//	// ReLU (Rectified Linear Unit)  
	//	for (int i = 0; i < HIDDEN_SIZE; i++)
	//	{
	//		hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  
	//	}
	// 
	//	// 前向传递：从隐藏层到输出层
	//	outputLay->forward( hidden_output, final_output);
	//	//softmax(final_output, OUTPUT_SIZE);
	//
	//	// 计算输出梯度
	//	for (int i = 0; i < OUTPUT_SIZE; i++)
	//	{
	//		float diff_val = final_output[i] - y[i];
	//		output_grad[i] = 2.0f * diff_val;
	//	}
	//
	//
	//	// 反向传播：从输出层到隐藏层
	//	outputLay->backward( hidden_output, output_grad, hidden_grad, lr);
	//
	//	// 通过 ReLU 激活函数反向传播
	//	for (int i = 0; i < HIDDEN_SIZE; i++)
	//	{
	//		hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;   
	//	}
	//
	//
	//	// 反向传播：从隐藏层到输入层
	//	inputLay->backward( input, hidden_grad, NULL, lr); 
	//
	//	delete[] hidden_output;
	//	delete[] final_output;
	//	delete[] output_grad;
	//	delete[] hidden_grad;
	//}
	 
	void softmax(float* input, int size);
	 
	int predict(float* input);
private:
	std::vector<Layer*> model;
	std::vector<uint> topology; 
	 
	std::vector<std::shared_ptr<DataType[]>> _hidenData;
};

