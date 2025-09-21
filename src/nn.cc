#include "nn.h"
 
Neuro::Neuro(int num) :_inputNum(num), order(1)
{
	_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
}

Neuro::~Neuro() 
{} 

Neuro::Neuro(const Neuro& RHS)
{
	_inputNum = RHS._inputNum;
	order = RHS.order;
	_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
	for (std::size_t i = 0; i < _inputNum; i++) {
		_weights[i] = RHS._weights[i];
	}
}

Neuro& Neuro::operator = (const Neuro& RHS)
{
	if (this != &RHS)
	{
		_inputNum = RHS._inputNum;
		order = RHS.order;
		_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
		for (std::size_t i = 0; i < _inputNum; i++) {
			_weights[i] = RHS._weights[i];
		}
	}
	return *this;
} 


void Neuro::init()
{
	if (!_weights)
	{
		_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<DataType> dis(-1.0, 1.0);

	float scale = sqrtf(2.0f / _inputNum);

	for (int i = 0; i < _inputNum; i++) {
		float tmp = (dis(gen) - 0.5f) * 2 * scale;
		_weights[i] = tmp;
	}

	return;
}  

DataType Neuro::run(std::shared_ptr<DataType[]> pVal)
{
	DataType result = 1.0;
	DataType tmp = 0.0;
	if (_inputNum <= 0)
	{
		return result;
	}

	for (size_t j = 0; j < _inputNum; j++)
	{
		tmp += pVal[j] * (_weights[j]);
	}

	for (size_t j = 0; j < order; j++)
	{
		result *= tmp;
	}
	return result;
}

void Neuro::updataWeight(std::shared_ptr<DataType[]> pVal, DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate)
{  
	for (int j = 0; j < _inputNum; j++)
	{
		DataType tmp = 1.0;
		for (int i = 0; i < order - 1; i++)
		{
			tmp *= pVal[j];
		}
		varGrad[j] += _weights[j] * diffVal * tmp * order;

		tmp *= pVal[j];
		_weights[j] -= learnRate * tmp * diffVal;
	} 
}

void Neuro::show(int index)
{
	std::cout << std::endl;
	std::cout << " Neuro " << index << std::endl;
	std::cout << "  Weights: " << std::endl;
	std::cout << "   ";
	for (int j = 0; j < _inputNum; j++)
	{
		std::cout << _weights[j];
		if (j < (_inputNum - 1))
		{
			std::cout << ", ";
		}
	}
	std::cout << std::endl;
}


Layer::Layer(int inNum, int outNum) :_inputNum(inNum)
, _outputNum(outNum), _pWeights(nullptr)
, _pBiases(nullptr)
{
}
Layer::~Layer() {
	clear();
}

Layer::Layer(const Layer& RHS)
{
	_inputNum = RHS._inputNum;
	_outputNum = RHS._outputNum;

	int n = _inputNum * _outputNum;
	_pWeights = new DataType[n];
	memcpy_s(_pWeights, n * sizeof(DataType), RHS._pWeights, n * sizeof(DataType));

	_pBiases = new DataType[_outputNum];
	memcpy_s(_pBiases, n * sizeof(DataType), RHS._pBiases, _outputNum * sizeof(DataType));
}

Layer& Layer::operator = (const Layer& RHS)
{
	if (this != &RHS)
	{
		_inputNum = RHS._inputNum;
		_outputNum = RHS._outputNum;

		int n = _inputNum * _outputNum;
		_pWeights = new DataType[n];
		memcpy_s(_pWeights, n * sizeof(DataType), RHS._pWeights, n * sizeof(DataType));

		_pBiases = new DataType[_outputNum];
		memcpy_s(_pBiases, n * sizeof(DataType), RHS._pBiases, _outputNum * sizeof(DataType));
	}
	return *this;
} 

void Layer::init()
{
	clear();
	int n = (int)_pNeros.size();
	if (0 == n)
	{
		for (int i = 0; i < _outputNum; i++)
		{
			std::shared_ptr<Neuro> p = std::make_shared<Neuro>(_inputNum);
			p->init();
			_pNeros.push_back(p);
		}
	}
	else {
		for (int i = 0; i < _outputNum; i++)
		{
			std::shared_ptr<Neuro> p = _pNeros[i];
			if (!p)
			{
				continue;
			}
			p->init();
		}
	} 
}

void Layer::clear()
{
	if (nullptr != _pWeights)
	{
		delete[] _pWeights;
		_pWeights = nullptr;
	}
	if (nullptr != _pBiases)
	{
		delete[] _pBiases;
		_pBiases = nullptr;
	} 
}

int Layer::getInputNum() {
	return _inputNum;
}

int Layer::getOutputNum() {
	return _outputNum;
}

void Layer::show(int index)
{
	std::cout << std::endl;
	std::cout << "Layer " << index << std::endl;
	std::cout << "----------------- " << std::endl;
	std::cout << "   Input: " << _inputNum << std::endl;
	std::cout << "  Output: " << _outputNum << std::endl;
	for (int i = 0; i < _outputNum; i++)
	{
		std::shared_ptr<Neuro> p = _pNeros[i];
		if (!p)
		{
			continue;
		}
		p->show(i);
	}
	std::cout << std::endl;
}
 
void Layer::forward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output) {
	for (int i = 0; i < _outputNum; i++) {
		std::shared_ptr<Neuro> p = _pNeros[i];
		if (!p)
		{
			continue;
		}
		output[i] = p->run(input); 
	}
}

void Layer::backward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr)
{  
	for (int i = 0; i < _outputNum; i++) {

		std::shared_ptr<Neuro> pNeuron = _pNeros[i];
		if (!pNeuron)
		{
			continue;
		}
		pNeuron->updataWeight(input, output_grad[i], input_grad, lr);

		//_pBiases[i] -= lr * output_grad[i];
	}
}



NN::NN() { 
}

NN::~NN() {
	size_t n = model.size();
	for (size_t i = 0; i < n - 1; i++)
	{
		Layer* pLay = model[i];
		delete pLay;
		pLay = nullptr;
	}
} 

void NN::setModel(std::vector<uint> theTop)
{
	topology = theTop;
	size_t n = topology.size();
	for (size_t i = 0; i < n - 1; i++)
	{
		int in = topology[i];
		int out = topology[i + 1];
		Layer* pLay = new Layer(in, out);
		pLay->init();
		model.push_back(pLay);
	}

	for (size_t i = 0; i < n; i++)
	{
		int in = topology[i];
		std::shared_ptr<DataType[]> pTmp = std::shared_ptr<DataType[]>(new DataType[in]);
		_hidenData.push_back(pTmp);
	}
}

void NN::init()
{
	size_t n = model.size();
	for (size_t i = 0; i < n; i++)
	{
		Layer* pLay = model[i];
		if (pLay)
		{
			pLay->init();
		}
	}
}

void NN::show()
{
	std::cout << std::endl;

	size_t n = model.size();
	std::cout << "Layer Number: " << n << std::endl;
	for (size_t i = 0; i < n; i++)
	{
		Layer* pLay = model[i];
		pLay->show((int)i);
	}
	std::cout << std::endl;
}

void NN::train(float* input, float* y, float lr)
{
	predict(input);

	size_t n = topology.size();
	int OUTPUT_SIZE = topology[n - 1];

	std::shared_ptr<DataType[]> pInput = _hidenData[n - 2];
	std::shared_ptr<DataType[]> pY_bar = _hidenData[n - 1];

	if ((!pInput) || (!pY_bar))
	{
		//
		std::cout << "NN::train:  Invalid input or y_bar data pointer. " << std::endl;
		// 
		return;
	}
	 
	bool OK = true;
	std::shared_ptr<DataType[]> pDiff = std::shared_ptr<DataType[]>(new DataType[OUTPUT_SIZE]);
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		pDiff[i] = 2.0f * (pY_bar[i] - y[i]);
		if (fabs(pDiff[i]) > 100.0f)
		{
			OK = false;
			break;
		}
	}

	if (!OK)
	{
		//
		std::cout << "Train:  ======  RESET Neurons ======= " << std::endl;
		// 

		init();
		 
		predict(input);

		for (size_t i = 0; i < OUTPUT_SIZE; i++)
		{
			pDiff[i] = 2.0f * (pY_bar[i] - y[i]);
			if (fabs(pDiff[i]) > 1.0e2)
			{
				OK = false;
			}
		}
	}


	//// DEBUG -- BEGIN --  
	//std::cout << std::endl;
	std::cout << "Train:" << std::endl;
	//std::cout << " Y:" << std::endl;
	//for (size_t i = 0; i < OUTPUT_SIZE; i++)
	//{
	//	std::cout << "  " << y[i] << std::endl;
	//} 
	//std::cout << "pY_bar:" << std::endl;
	//for (size_t i = 0; i < OUTPUT_SIZE; i++)
	//{
	//	std::cout << "  " << pY_bar[i] << " " << std::endl;
	//} 
	std::cout << " Diff:" << std::endl;
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << "  " << pDiff[i] << " " << std::endl;
	}
	//// DEBUG -- END --  

	size_t m = model.size();
	for (int i = (int)m - 1; i >= 0; i--)
	{
		pInput = _hidenData[i];
		Layer* pLayer = model[i];
		if (!pLayer)
		{
			return;
		}
		int inputNum = pLayer->getInputNum();
		std::shared_ptr<DataType[]> input_Grad = std::shared_ptr<DataType[]>(new DataType[inputNum]);
		for (int i = 0; i < inputNum; i++)
		{
			input_Grad[i] = 0.0f;
		}

		pLayer->backward(pInput, pDiff, input_Grad, lr);

		//  ReLU Method 
		for (int i = 0; i < inputNum; i++)
		{
			input_Grad[i] *= fabs(input_Grad[i]) > 2.0f ? 0.0f : input_Grad[i];
		}

		pDiff.reset();
		pDiff = input_Grad;
	}
}

void NN::softmax(float* input, int size) {
	float max = input[0], sum = 0;
	for (int i = 1; i < size; i++)
		if (input[i] > max) max = input[i];
	for (int i = 0; i < size; i++) {
		input[i] = expf(input[i] - max);
		sum += input[i];
	}
	for (int i = 0; i < size; i++)
		input[i] /= sum;
}

int NN::predict(float* input)
{
	int result = -1;

	size_t n = model.size();
	Layer* pLayer = model[0];

	int INPUT_SIZE = pLayer->getInputNum();
	int OUTPUT_SIZE = pLayer->getOutputNum();

	std::shared_ptr<DataType[]> final_input = _hidenData[0]; //    new float[INPUT_SIZE];
	std::shared_ptr<DataType[]> final_output = _hidenData[1]; //   new float[OUTPUT_SIZE];

	if ((!final_input) || (!final_output))
	{
		return result;
	}

	for (size_t i = 0; i < INPUT_SIZE; i++)
	{
		final_input[i] = input[i];
	}
	 
	pLayer->forward(final_input, final_output);

	for (size_t i = 1; i < n; i++)
	{
		pLayer = model[i];
		if (!pLayer)
		{
			return result;
		}

		final_input = _hidenData[i];
		final_output = _hidenData[i + 1];

		if ((!final_input) || (!final_output))
		{
			return result;
		}

		pLayer->forward(final_input, final_output);
	} 
	return result;
}