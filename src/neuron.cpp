#include "neuron.h"
#include "connection.h" 
#include "layer.h"
#include "string"

Neuron::Neuron(int num):_value(0.0f), _bias(0.0f), _inputNum(num), _ALIVE(true), order(0.0f), _type(1)
{
}

Neuron::~Neuron()
{
	_pConnections.clear();
}

Neuron::Neuron(const Neuron& RHS)
{
	_inputNum = RHS._inputNum;
	order = RHS.order;
	_type = RHS._type;
	_ALIVE = RHS._ALIVE;
	_value = RHS._value;
	_bias = RHS._bias;

	_pConnections.clear();
	for (std::size_t i = 0; i < _inputNum; i++) {
		_pConnections.push_back(RHS._pConnections[i]);
	} 
}

Neuron& Neuron::operator = (const Neuron& RHS)
{
	if (this != &RHS)
	{
		_inputNum = RHS._inputNum;
		order = RHS.order;
		_type = RHS._type;
		_ALIVE = RHS._ALIVE;
		_value = RHS._value;
		_bias = RHS._bias;

		_pConnections.clear();
		for (std::size_t i = 0; i < _inputNum; i++) {
			_pConnections.push_back(RHS._pConnections[i]);
		} 
	}
	return *this;
}

std::shared_ptr<Neuron> Neuron::getPtr()
{
	return shared_from_this();
}

void Neuron::init(std::shared_ptr<Layer> pInputLayer)
{ 
	float scale = sqrtf(2.0f);
	if (_inputNum > 0)
	{ 
		scale = sqrtf(2.0f / _inputNum);
	} 
	_bias = 0.0f; // (getRandVal() - 0.5f) * 2 * scale;;

	if (0 == _pConnections.size())
	{
		// This is an EMPTY Neuron 
		for (int i = 0; i < _inputNum; i++) {
			std::shared_ptr<Connection> p = std::make_shared<Connection>();

			float tmp = (getRandVal() - 0.5f) * 2 * scale;
			p->_weight = tmp;
			if (nullptr == pInputLayer)
			{
				p->_pInputNero = nullptr;
			}
			else { 
				p->_pInputNero = pInputLayer->getNeuron(i); 
			}

			_pConnections.push_back(p);
		}
	}
	else {
		// This Neuron has been intialized, 
		// we do NOT modify its connection, 
		// only reset the weights.
		for (int i = 0; i < _inputNum; i++) { 
			std::shared_ptr<Connection> p = _pConnections[i];
			if (!p)
			{
				float tmp = (getRandVal() - 0.5f) * 2 * scale; 
				p->_weight = tmp;
			} 
		}
	}
	 
	//if (nullptr == pInputLayer)
	//{
	//	_pConnections.clear();
	//	for (int i = 0; i < _inputNum; i++) {
	//		std::shared_ptr<Connection> p = std::make_shared<Connection>();
	//
	//		float tmp = (getRandVal() - 0.5f) * 2 * scale;
	//		p->_weight = tmp; 
	//		p->_pInputNero = nullptr;
	//		_pConnections.push_back(p); 
	//	}
	//}
	//else { 
	//	_pConnections.clear();
	//	for (int i = 0; i < _inputNum; i++) {
	//		std::shared_ptr<Connection> p = std::make_shared<Connection>();
	//
	//		float tmp = (getRandVal() - 0.5f) * 2 * scale;
	//		p->_weight = tmp;
	//		p->_pInputNero = pInputLayer->getNeuron(i);
	//		_pConnections.push_back(p); 
	//	}
	//}
	return;
}

void Neuron::setWeight(std::vector<float>& values)
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

	for (int i = 0; i < _inputNum; i++) {
		float tmp = values[i];
		if (nullptr != _pConnections[i])
		{
			_pConnections[i]->_weight = tmp;
		} 
	} 
} 

void Neuron::setWeight(uint connectionIndex, DataType val)
{
	uint n = _pConnections.size();

	if (connectionIndex >= n)
	{
		std::cout << "Neuron::setWeight(): Connection Index: " << connectionIndex << " overflow." << std::endl;
		return; 
	}
	std::shared_ptr<Connection> pCon = _pConnections[connectionIndex];
	if (nullptr == pCon)
	{
		std::cout << "Neuron::setWeight(): Invalid Connection." << std::endl;
		return;
	}
	pCon->_weight = val;
	return;
}

/**
\brief Get the output of the neuron
*/
DataType Neuron::getValue()
{
	return _value;
}

void Neuron::setValue(DataType val)
{
	_value = val;
}

DataType Neuron::getBias()
{
	return _bias;
}

void Neuron::setBias(DataType val)
{
	_bias = val;
}

void Neuron::setOrder(float val)
{
	if (val < 0)
	{
		return;
	}

	order = val * 1.0;
}

int Neuron::getOrder()
{
	return  order;
}

void Neuron::setType(uint val)
{
	_type = val;
}

/***
Output Formate
 inputNum | neuron type | order | bias | 
 weight_i | input neuron layer id | input neuron id | 
 ... 
*/
std::string Neuron::toStr()
{
	std::string result = std::to_string(_inputNum) + " " + std::to_string(_type) + " "
		+ std::to_string(order) + " " + std::to_string(_bias) + " ";
	for (int j = 0; j < _inputNum; j++)
	{
		if (!_ALIVE)
		{
			result = result + std::to_string(0.0) + " ";
		}
		else {
			result = result + std::to_string(_pConnections[j]->_weight) + " ";
			std::shared_ptr<Neuron> pIn = _pConnections[j]->_pInputNero;
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


/**
\brief Calculate the output of the neuron
@param[in] pVal: input variable vector
*/
DataType Neuron::run()
{ 
	DataType tmp = 0.0; 

	if (!_ALIVE)
	{
		return tmp;
	}

#ifdef _DEBUG_
	std::cout << std::endl;
	std::cout << "Neuro::run():" << std::endl;
#endif

	tmp = _bias;

#ifdef _DEBUG_ 
	std::cout << "   var  = " << _bias  << std::endl;
#endif

	for (size_t j = 0; j < _inputNum; j++)
	{
		if (!_pConnections[j])
		{
			continue;
		}

		// Overlap inactive Connections 
		if (!(_pConnections[j]->ALIVE)) {
			continue;
		}

		if(!(_pConnections[j]->_pInputNero))
		{
			continue;
		}
		 
		DataType val = _pConnections[j]->_pInputNero->_value;
		DataType w = _pConnections[j]->_weight;
		 
		tmp += val * w;

#ifdef _DEBUG_ 
		std::cout << "   var += " << val << " * " << w << " => " << tmp << std::endl;
#endif
	}
	  
	// 
	// Calculating the output value 
	if (0.0 == order)
	{
		_value = tmp;
	}
	else {
		_value = calcuOutput(tmp); //  pow(tmp, order);
	}

#ifdef _DEBUG_ 
	std::cout << "   result: " << _value << std::endl;
#endif

	return _value;
}

/**
\brief  Backpropagration -- updating the weights of input neurons.
@param[in] diffVal: output variable gradient  
@param[out] diffVal: input variable gradient vector 
@param[in] learnRate: learning rate.
*/
void Neuron::updataWeight(DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate)
{
#ifdef _DEBUG_ 
	std::cout << std::endl;
	std::cout << "  Neuron::updataWeight() "   << std::endl;
#endif
  
	std::shared_ptr<DataType[]> pVal = std::shared_ptr<DataType[]>(new DataType[_inputNum], [](DataType* p) {delete[] p; p = nullptr; });
	if (!pVal)
	{
		return;
	}
	for (size_t j = 0; j < _inputNum; j++)
	{
		if (!_pConnections[j])
		{
			continue;
		}
		if (!_pConnections[j]->ALIVE)
		{
			continue;
		}

		if (!(_pConnections[j]->_pInputNero))
		{
			continue;
		}
		DataType w = _pConnections[j]->_weight;
		DataType val = _pConnections[j]->_pInputNero->_value;

		pVal[j] = val;
	} 
	 
	// Gradient of the ERROR 
	DataType errorGrad = learnRate * diffVal;

#ifdef _DEBUG_ 
	std::cout << "   Error Gradient: " << errorGrad << std::endl;
#endif

	_bias -= errorGrad;

	//std::cout << " bias: " << _bias << std::endl;
	//std::cout << "   AZ: " << AZ << std::endl;
	//std::cout << "   XZ: " << XZ << std::endl;

	//if (2 == _iteration)
	//{  
	//	_bias = XZ; 
	//}
	 
	for (int j = 0; j < _inputNum; j++)
	{
		if (!_pConnections[j])
		{
			continue;
		}
		if (!_pConnections[j]->ALIVE)
		{
			continue;
		}

		DataType tmp = 1.0;
		tmp = calcuGrad(diffVal, pVal[j]);

		//
		//varGrad[j] += _weights[j] * tmp; 
		if (_pConnections[j])
		{
			varGrad[j] += _pConnections[j]->_weight * tmp;
		}
		 
		if (!_pConnections[j]->_pInputNero)
		{
			continue;
		}
		 
		//tmp = calcuOutput(pVal[j]);
		//_weights[j] -= learnRate * tmp * diffVal;
		tmp = _pConnections[j]->_pInputNero->_value;
		  
		_pConnections[j]->_weight -= learnRate  * diffVal * tmp;

		tmp = _pConnections[j]->_weight;
		if (fabs(tmp) > 1000.0)
		{
			_pConnections[j]->ALIVE = false;
			_pConnections[j]->_weight = 0.0f; 
		}

		//if (fabs(tmp) < 0.001)
		//{
		//	_pConnections[j]->ALIVE = false;
		//	_pConnections[j]->_weight = 0.0f;
		//}
	}



	// DEBUG -- BEGIN -- 
#ifdef _DEBUG_  
	std::cout << "   bias: " << _bias << std::endl;
	std::cout << "   weights: " << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << "    " << _pConnections[i]->_weight << "  " << std::endl;
	}
	std::cout << "   input_grad: " << std::endl;
	for (size_t i = 0; i < _inputNum; i++)
	{
		std::cout << "    " << varGrad[i] << "  " << std::endl;
	}
	std::cout << std::endl;
#endif 
	//  DEBUG -- END --  
}

/**
 * @brief Kill one connection of this Neuron 
 * @param connectionIndex: uint, index of the connection  
*/
void Neuron::killConnection(uint connectionIndex)
{
	if (connectionIndex >= _inputNum)
	{
		return;
	}
	if (!_pConnections[connectionIndex])
	{
		return;
	}
	_pConnections[connectionIndex]->ALIVE = false;
	return;
}

/**
 * @brief Active one connection of this Neuron
 * @param connectionIndex: uint, index of the connection
*/
void Neuron::activeConnection(uint connectionIndex)
{
	uint n = _pConnections.size();

	if (connectionIndex >= n)
	{
		return;
	}
	if (!_pConnections[connectionIndex])
	{
		return;
	}
	_pConnections[connectionIndex]->ALIVE = true;
	return;
}

/**
\brief Get the number of connections attached to the Neuron.
*/
uint Neuron::getConnectionNum()
{
	return _pConnections.size();
}

/**
\brief Connect a neuron as input neuron, 
       and create a new connection for it.
*/
void Neuron::connectNeuron(std::shared_ptr<Neuron> pNeuron)
{
	if (!pNeuron)
	{
		return;
	} 
	size_t index = 0;
	for (index = 0; index < _inputNum; index++)
	{
		if (!_pConnections[index])
		{
			continue;
		}
		if (pNeuron == _pConnections[index]->_pInputNero)
		{
			return;
		}
	}
	std::shared_ptr<Connection> p = std::make_shared<Connection>();

	float scale = sqrtf(2.0f);
	if (_inputNum > 0)
	{
		scale = sqrtf(2.0f / _inputNum);
	}

	float tmp = (getRandVal() - 0.5f) * 2 * scale;
	p->_weight = tmp;
	p->_pInputNero = pNeuron;
	p->_pOutputNero = getPtr();

	_pConnections.push_back(p);
	_inputNum = _pConnections.size();
	return;
}

/**
\brief Remove the connection to a neuron. 
*/
void Neuron::disconnectNeuron(std::shared_ptr<Neuron> pNeuron)
{
	if (!pNeuron)
	{
		return;
	}
	size_t index = 0;
	for (index = 0; index < _inputNum; index++)
	{
		if (!_pConnections[index])
		{
			continue;
		}
		if (pNeuron == _pConnections[index]->_pInputNero)
		{
			//break;
			swap(_pConnections[index], _pConnections[_inputNum - 1]);
			_pConnections.pop_back();
			break;
		}
	}
	_inputNum = _pConnections.size();
	return;
}


/**
\brief Get one Connection pointer of the neuron 
 * @param index: uint, index of the connection
*/
std::shared_ptr<Connection> Neuron::getConnection(size_t index)
{
	std::shared_ptr<Connection> p;  
	if (index < _inputNum)
	{
		p = _pConnections[index];
	}
	return p;
}

DataType Neuron::calcuGrad(DataType steepness, DataType var)
{
	DataType result = 0.0;
	switch (_type)
	{
	case LINEAR:
		result = (DataType)var;
		break;
	case SIN:
		result = (DataType)sin_derive(steepness, var);
		break;
	case COS:
		result = (DataType)cos_derive(steepness, var);
		break;
	case GAUSSIAN_STEPWISE:
		result = 0;
		break;
	case ACTIVATIONFUNC::POW:
		result = pow_derive(steepness, var, order);
	}
	return result;
}

DataType Neuron::calcuOutput(DataType var)
{
	DataType result = 0.0;
	switch (_type)
	{
	case LINEAR:
		result = (DataType)var;
		break;
	case LINEAR_PIECE:
		result = (DataType)((var < 0) ? 0 : (var > 1) ? 1 : var);
		break;
	case LINEAR_PIECE_SYMMETRIC:
		result = (DataType)((var < -1) ? -1 : (var > 1) ? 1 : var);
		break;
	case SIGMOID:
	{
		result = sigmoid_real(var);
	}
	break;
	case SIGMOID_SYMMETRIC:
		result = (DataType)sigmoid_symmetric_real(var);
		break;
		//case SIGMOID_SYMMETRIC_STEPWISE:
		//	result = (DataType)stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, var);
		//	break;
		//case SIGMOID_STEPWISE:
		//	result = (DataType)stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, var);
		//	break;
	case THRESHOLD:
		result = (DataType)((var < 0) ? 0 : 1);
		break;
	case THRESHOLD_SYMMETRIC:
		result = (DataType)((var < 0) ? -1 : 1);
		break;
	case GAUSSIAN:
		result = (DataType)gaussian_real(var);
		break;
		//case GAUSSIAN_SYMMETRIC:
		//	result = (DataType)gaussian_symmetric_real(var);
		//	break;
		//case ELLIOT:
		//	result = (DataType)elliot_real(var);
		//	break;
		//case ELLIOT_SYMMETRIC:
		//	result = (DataType)elliot_symmetric_real(var);
		//	break;
		//case SIN_SYMMETRIC:
		//	result = (DataType)sin_symmetric_real(var);
		//	break;
		//case COS_SYMMETRIC:
		//	result = (DataType)cos_symmetric_real(var);
		//	break;
	case SIN:
		result = (DataType)sin_real(var);
		break;
	case COS:
		result = (DataType)cos_real(var);
		break;
	case GAUSSIAN_STEPWISE:
		result = 0;
		break;
	case ACTIVATIONFUNC::POW:
		result = pow_real(var, order);
	}
	return result;
}
