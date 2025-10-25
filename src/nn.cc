#include "nn.h"
#include "string"
#include <fstream> 
 
#define _DEBUG_ 1    

Neuro::Neuro(int num) :_inputNum(num), order(1), ACTIVE(true), _type(1)
{ 
	order = 0.5;
	_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
}

Neuro::~Neuro() 
{} 

Neuro::Neuro(const Neuro& RHS)
{
	_inputNum = RHS._inputNum;
	order = RHS.order;
	_type = RHS._type;
	ACTIVE = RHS.ACTIVE;

	_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
	for (std::size_t i = 0; i < _inputNum; i++) {
		_weights[i] = RHS._weights[i];
	}

	_pConnections.clear();
	for (std::size_t i = 0; i < _inputNum; i++) {
		_pConnections.push_back(RHS._pConnections[i]);
	} 
}

Neuro& Neuro::operator = (const Neuro& RHS)
{
	if (this != &RHS)
	{
		_inputNum = RHS._inputNum;
		order = RHS.order;
		_type = RHS._type;
		ACTIVE = RHS.ACTIVE;
		_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
		for (std::size_t i = 0; i < _inputNum; i++) {
			_weights[i] = RHS._weights[i];
		}

		_pConnections.clear();
		for (std::size_t i = 0; i < _inputNum; i++) {
			_pConnections.push_back(RHS._pConnections[i]);
		} 
	}
	return *this;
} 


void Neuro::init(Layer* pInputLayer)
{
	float scale = sqrtf(2.0f / _inputNum);

	if (nullptr == pInputLayer)
	{
		_pConnections.clear();
		for (int i = 0; i < _inputNum; i++) {
			std::shared_ptr<Connection> p = std::make_shared<Connection>();

			float tmp = (getRandVal() - 0.5f) * 2 * scale;
			p->_weight = tmp;
			p->_pInputNero = nullptr; 
			_pConnections.push_back(p); 
		}
	}
	else { 
		_pConnections.clear();
		for (int i = 0; i < _inputNum; i++) {
			std::shared_ptr<Connection> p = std::make_shared<Connection>();

			float tmp = (getRandVal() - 0.5f) * 2 * scale; 
			p->_weight = tmp;
			p->_pInputNero = pInputLayer->getNeuron(i); 
			_pConnections.push_back(p); 
		}
	}
	 
	if (!_weights)
	{
		_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
	} 

	for (int i = 0; i < _inputNum; i++) {
		float tmp = (getRandVal() - 0.5f) * 2 * scale;
		_weights[i] = tmp;
	}

	return;
}  


void Neuro::setWeight(std::vector<float>& values)
{
	// Quick Return 
	if (_inputNum <= 0)
	{
		return;
	}

	std::size_t n = values.size();
	if (_inputNum > n)
	{
		return;
	}
	if (!_weights)
	{
		_weights = std::shared_ptr<DataType[]>(new DataType[_inputNum]);
	}
	for (int i = 0; i < _inputNum; i++) {
		float tmp = values[i];
		_weights[i] = tmp;
	}
	 
	for (int i = 0; i < _inputNum; i++) {
		float tmp = values[i];
		_pConnections[i]->_weight = tmp;
	}

	return;
}


DataType Neuro::run(std::shared_ptr<DataType[]> pVal)
{
	_value = 1.0;
	DataType tmp = 0.0;
	if (_inputNum <= 0)
	{
		return _value;
	}
	if(!ACTIVE)
	{
		return tmp;
	}

#ifdef _DEBUG_
	std::cout << std::endl;
	std::cout << "Neuro::run():" << std::endl; 
#endif

	for (size_t j = 0; j < _inputNum; j++)
	{
		// tmp += pVal[j] * (_weights[j]); 

		tmp += pVal[j] * _pConnections[j]->_weight;

#ifdef _DEBUG_ 
		std::cout << "        tmp += " << pVal[j] << " * " << _weights[j] << " = " << tmp << std::endl;
#endif
	}

	if (0.0 == order)
	{
		_value = tmp;
	}
	else {
		_value = calcuOutput(tmp); //  pow(tmp, order);
	}
	if (!std::isfinite(_value))
	{
		_value = tmp;
	} 
#ifdef _DEBUG_ 
	std::cout << "      result:" << _value << std::endl;
	std::cout << std::endl;
#endif

	return _value;
}

DataType Neuro::getValue()
{
	return _value;
}


std::shared_ptr<Connection> Neuro::getConnection(size_t index)
{
	std::shared_ptr<Connection> p;
	if (index < _inputNum)
	{
		p = _pConnections[index];
	}
	return p;
}

DataType Neuro::calcuGrad(DataType diffVar, DataType value)
{
	DataType result = 0.0;
	switch (_type)
	{
	case LINEAR:
		result = (DataType)value;
		break;
	case SIN:
		result = (DataType)sin_derive(diffVar, value);
		break;
	case COS:
		result = (DataType)cos_derive(diffVar, value);
		break;
	case GAUSSIAN_STEPWISE:
		result = 0;
		break;
	case ACTIVATIONFUNC::POW:
		result = pow_derive(diffVar, value, order);
	}
	return result;
}

DataType Neuro::calcuOutput(DataType value)
{
	DataType result = 0.0;
	switch (_type)
	{
	case LINEAR:
		result = (DataType)value;
		break;
	case LINEAR_PIECE:
		result = (DataType)((value < 0) ? 0 : (value > 1) ? 1 : value);
		break;
	case LINEAR_PIECE_SYMMETRIC:
		result = (DataType)((value < -1) ? -1 : (value > 1) ? 1 : value);
		break;
	case SIGMOID:
	{
		result = sigmoid_real(value);
	} 
		break;
	case SIGMOID_SYMMETRIC:
		result = (DataType)sigmoid_symmetric_real(value);
		break; 
	case THRESHOLD:
		result = (DataType)((value < 0) ? 0 : 1);
		break;
	case THRESHOLD_SYMMETRIC:
		result = (DataType)((value < 0) ? -1 : 1);
		break;
	case GAUSSIAN:
		result = (DataType)gaussian_real(value);
		break; 
	case SIN:
		result = (DataType)sin_real(value);
		break;
	case COS:
		result = (DataType)cos_real(value);
		break;
	case GAUSSIAN_STEPWISE:
		result = 0;
		break;
	case ACTIVATIONFUNC::POW:
		result = pow_real(value, order);
	}
	return result;
}
 
void Neuro::updataWeight(std::shared_ptr<DataType[]> pVal, DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate)
{
	//  DEBUG -- BEGIN -- 
#ifdef _DEBUG_
	std::cout << std::endl;
	std::cout << "Neuro::updataWeight():" << std::endl;
	std::cout << "       order:" << order << std::endl;
	std::cout << "input number:" << _inputNum << std::endl;
	std::cout << "input:" << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << pVal[i] << "  " << std::endl;
	}
	std::cout << "diffVal:" << diffVal << std::endl;

	std::cout << "varGrad:" << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << varGrad[i] << "  " << std::endl;
	}
	std::cout << std::endl;
#endif
	//  DEBUG -- END --  

	for (int j = 0; j < _inputNum; j++)
	{
		DataType tmp = 1.0; 
		tmp = calcuGrad(diffVal, pVal[j]);
		varGrad[j] += _weights[j] * tmp; 
 

		tmp = calcuOutput(pVal[j]); 
		_weights[j] -= learnRate * tmp * diffVal;


		_pConnections[j]->_weight -= learnRate * tmp * diffVal;

		if (fabs(tmp) > 1000.0)
		{
			ACTIVE = false;
		} 
	}

	// DEBUG -- BEGIN -- 
#ifdef _DEBUG_
	std::cout << "new weights:" << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << _pConnections[i]->_weight << "  " << std::endl;
	}
	std::cout << "input_grad:" << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << varGrad[i] << "  " << std::endl;
	}
	std::cout << std::endl; 
#endif
	//  DEBUG -- END --  
	return;
}



void Neuro::show(int index)
{
	std::cout << std::endl;
	std::cout << " Neuro " << index << std::endl; 
	std::cout << "    ACTIVE: " << ACTIVE << std::endl;
	std::cout << "     order: " << order << std::endl;
	std::cout << "   Weights: " << std::endl;
	std::cout << "   ";
	for (int j = 0; j < _inputNum; j++)
	{
		std::cout << _pConnections[j]->_weight; // _weights[j];
		if (j < (_inputNum - 1))
		{
			std::cout << ", ";
		}
	}
	std::cout << std::endl;
}

std::string Neuro::toStr()
{
	std::string result = std::to_string(_inputNum) + " " + std::to_string(order) + " ";
	for (int j = 0; j < _inputNum; j++)
	{ 
		if (!ACTIVE)
		{
			result = result + std::to_string(0.0) + " ";
		}
		else {
			result = result + std::to_string(_weights[j]) + " ";
		} 
	}
	result = result + "\n";

	return result;
}

std::string Neuro::toStrN()
{
	std::string result = std::to_string(_inputNum) + " " + std::to_string(order) + " ";
	for (int j = 0; j < _inputNum; j++)
	{
		if (!ACTIVE)
		{
			result = result + std::to_string(0.0) + " ";
		}
		else {
			 result = result + std::to_string(_pConnections[j]->_weight) + " ";
			 std::shared_ptr<Neuro> pIn = _pConnections[j]->_pInputNero;
			 if (pIn)
			 {
				 result = result + std::to_string(pIn->layerID) + " ";
				 result = result + std::to_string(pIn->index) + " ";
			 }
			 else {
				 result = result + std::to_string(-1) + " ";
				 result = result + std::to_string(-1) + " ";
			 }
		}
	}
	result = result + "\n";

	return result;
}

void Neuro::setOrder(float val)
{
	if (val < 0)
	{
		return;
	}

	order = val * 1.0;
}

int Neuro::getOrder()
{
	return  order;
}


Layer::Layer(int inNum, int outNum, int id) :_inputNum(inNum)
, _outputNum(outNum), _pWeights(nullptr), _id(id), _connectLayerID(0)
, _pBiases(nullptr)
{
}
Layer::~Layer() {
	clear();
}

Layer::Layer(const Layer& RHS)
{
	_id = RHS._id;
	_connectLayerID = RHS._connectLayerID;

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
		_id = RHS._id;
		_connectLayerID = RHS._connectLayerID;

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

void Layer::init(Layer* pLayer)
{
	clear();
	if (nullptr == pLayer)
	{
		_connectLayerID = -1;
	}
	else {
		_connectLayerID = pLayer->getID();
	}

	int n = (int)_pNeros.size();
	if (0 == n)
	{
		for (int i = 0; i < _outputNum; i++)
		{
			std::shared_ptr<Neuro> p = std::make_shared<Neuro>(_inputNum);
			p->init();
			p->setOrder(i);
			p->layerID = _id;
			p->index = i;
			for (int j = 0; j < _inputNum; j++)
			{
				std::shared_ptr<Connection> pConnect = p->getConnection(j);
				if (pConnect)
				{
					pConnect->_pOutputNero = p;
				}

				if (nullptr != pLayer)
				{
					std::shared_ptr<Neuro> pInNeuro = pLayer->getNeuron(j);
					pConnect->_pInputNero = pInNeuro; 
				}
			} 
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
			p->setOrder(i);  
			for (int j = 0; j < _inputNum; j++)
			{
				std::shared_ptr<Connection> pConnect = p->getConnection(j);
				if (pConnect)
				{
					pConnect->_pOutputNero = p;
				}
			}
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

int Layer::getID()
{
	return _id;
}

int Layer::getInputNum() {
	return _inputNum;
}

int Layer::getOutputNum() {
	return _outputNum;
}

int Layer::getConnectedLayerID()
{
	return _connectLayerID;
}

void  Layer::setConnectedLayerID(int index)
{
	_connectLayerID = index;
}

std::shared_ptr<Neuro> Layer::getNeuron(size_t index)
{
	std::shared_ptr<Neuro> p;
	if (index < _outputNum)
	{
		p = _pNeros[index];
	} 
	return p; 
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
 

std::string Layer::toStr()
{
	std::string result = std::to_string(_inputNum) + " " + std::to_string(_outputNum)  
		+ " " + std::to_string(_connectLayerID)
		+ "\n"; 
	for (int i = 0; i < _outputNum; i++)
	{
		std::shared_ptr<Neuro> p = _pNeros[i];
		if (!p)
		{
			result += "\n";
		}
		else {
			result += p->toStrN();
		} 
	}
	return result;
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


	//  DEBUG -- BEGIN --   
#ifdef _DEBUG_ 
	std::cout << "Layer::forward()" << std::endl; 
	std::cout << "    output:" << std::endl;
	for (int j = 0; j < _outputNum; j++)
	{
		std::cout << output[j] << "  ";
	}
	std::cout << std::endl;
#endif _DEBUG_ 
	//   DEBUG -- END --  
}

void Layer::backward(std::shared_ptr<DataType[]> input, std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr)
{
	//   DEBUG -- BEGIN --  
#ifdef _DEBUG_ 
	std::cout << std::endl;
	std::cout << "Layer::backward():" << std::endl;
	std::cout << "output number:" << _outputNum  << std::endl;
	std::cout << "output_grad:" << std::endl;
	for (size_t i = 0; i < _outputNum; i++)
	{
		std::cout << output_grad[i] << "  " << std::endl;
	}
	std::cout << "input_grad:" << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << input_grad[i] << "  " << std::endl;
	}
#endif _DEBUG_ 
	//   DEBUG -- END --  
	 
	for (int i = 0; i < _outputNum; i++) {

		std::shared_ptr<Neuro> pNeuron = _pNeros[i];
		if (!pNeuron)
		{
			continue;
		}
		pNeuron->updataWeight(input, output_grad[i], input_grad, lr); 
	}
}
 

NN::NN() { 
	_FINISH = false;
}

NN::~NN() {
	clear();
} 

void NN::setModel(std::vector<uint> theTop)
{
	topology = theTop;
	size_t n = topology.size();
	for (size_t i = 0; i < n - 1; i++)
	{
		int in = topology[i];
		int out = topology[i + 1];
		Layer* pLay = new Layer(in, out, i); 
		if (i > 0)
		{
			Layer* pInLay = model[i-1];
			pLay->init(pInLay);
		}
		else {
			pLay->init();
		}

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
	_FINISH = false;
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

void NN::clear()
{
	int n = (int)model.size();
	for (int i = 0; i < n - 1; i++)
	{
		Layer* pLay = model[i];
		delete pLay;
		pLay = nullptr;
	}
	model.clear();

	n = (int)_hidenData.size();
	for (int i = 0; i < n - 1; i++)
	{
		std::shared_ptr<DataType[]> pData = _hidenData[i]; 
		if (!pData)
		{
			pData.reset();
		} 
	}
	_hidenData.clear(); 

	topology.clear();
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

void NN::load(const char* fileName)
{
	if (nullptr == fileName)
	{
		return;
	}
	clear();

	char data[100];
	 
	std::ifstream thefile;
	thefile.open(fileName); 
	thefile.get(data, 200); 

	// Layer Number 
	int layNum = 0; 
	thefile >> data; 
	layNum = atoi(data);
	//std::cout << layNum << std::endl;

	// Topology  
	uint N = 0; 
	for (int i = 0; i < layNum + 1; i++)
	{
		thefile >> data; 
		N = atoi(data);
		topology.push_back(N); 
	}

	for (size_t i = 0; i < layNum; i++)
	{
		int in = topology[i];
		int out = topology[i + 1];
		Layer* pLay = new Layer(in, out);
		pLay->init();
		model.push_back(pLay);
	}

	for (size_t i = 0; i < layNum + 1;  i++)
	{
		int in = topology[i];
		std::shared_ptr<DataType[]> pTmp = std::shared_ptr<DataType[]>(new DataType[in]);
		_hidenData.push_back(pTmp);
	}
	  
	for (int i = 0; i < layNum; i++)
	{
		thefile >> data;
		int inputNum = atoi(data); 
		thefile >> data; 
		int outputNum = atoi(data);   
		thefile >> data;
		int connectedLayerID = atoi(data);

		Layer* pLayer = model[i]; 
		pLayer->setConnectedLayerID(connectedLayerID);

		for (int j = 0; j < outputNum; j++)
		{
			thefile >> data;
			inputNum = atoi(data); 
			thefile >> data;
			double val = std::stod(data);  
			std::shared_ptr<Neuro> pNeuron = pLayer->getNeuron(j);

			pNeuron->setOrder(val);

			std::vector<float>  weights;
			for (int k = 0; k < inputNum; k++)
			{
				thefile >> data;
				val = std::stod(data); 
				//std::cout << val << "  " ; 

				weights.push_back(val);

				thefile >> data;
				int inLayerNum = atoi(data);
				thefile >> data;
				int inNeuronIndex = atoi(data);

				std::shared_ptr<Connection> p = std::make_shared<Connection>();
				p->_weight = val; 
				if (i > 0)
				{
					Layer* pInputLayer = model[connectedLayerID];
					if (pInputLayer)
					{
						std::shared_ptr<Neuro> pInputNeuron = pInputLayer->getNeuron(inNeuronIndex);
						p->_pInputNero = pInputNeuron;
					} 
				} 
			}
			//std::cout << std::endl; 

			pNeuron->setWeight(weights);
		}
	}


	thefile.close();
	return;
}

void NN::save(const char* fileName)
{ 
	std::ofstream myfile;
	myfile.open(fileName);
	myfile << "Writing this to a file.\n";

	size_t n = model.size();
	myfile << n << std::endl;
	for (size_t i = 0; i < n + 1; i++)
	{
		myfile << topology[i] << " ";
	}
	myfile << std::endl;
	for (size_t i = 0; i < n; i++)
	{
		Layer* pLay = model[i];
		myfile << pLay->toStr();
	}
	myfile << std::endl; 
	myfile.close();
}

bool NN::isFinish()
{
	return _FINISH;
}

void NN::setFinish(bool state)
{
	_FINISH = state;
	return;
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

		//show();

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
	 
	// DEBUG -- BEGIN --  
#ifdef _DEBUG_ 
	std::cout << std::endl;
	std::cout << "Train:" << std::endl;
	std::cout << " Y:" << std::endl;
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << "  " << y[i] << std::endl;
	} 
	std::cout << "pY_bar:" << std::endl;
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << "  " << pY_bar[i] << " " << std::endl;
	} 

	float diffSum = 0.0f;
	std::cout << " Diff:" << std::endl;
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << "  " << pDiff[i] << " " << std::endl;
		diffSum += pDiff[i];
	}

	if (fabs(diffSum) < 0.01)
	{
		std::cout << "  Sum of the errors: " << diffSum << " " << std::endl;
		std::cout << "    " << std::endl;
		_FINISH = true;
		return;
	}
#endif
	// DEBUG -- END --  

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

	 
	// DEBUG -- BEGIN --  
#ifdef _DEBUG_
	std::cout << "predict:" << std::endl;
	pLayer->show(0);
	std::cout << "final_input:" << std::endl;
	for (size_t i = 0; i < INPUT_SIZE; i++)
	{
		std::cout << final_input[i] << "  " << std::endl;
	}
#endif  
	 // DEBUG -- BEGIN --  
	

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

	// 
	// DEBUG -- BEGIN --   
// #ifdef _DEBUG_
	OUTPUT_SIZE = topology[n];
	std::cout << std::endl;
	std::cout << "NN::Predict Result:" << std::endl;
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		std::cout << "   " << final_output[i] << std::endl;
	}
	std::cout << std::endl; 
// #endif
	// DEBUG -- BEGIN --  

	return result;
}