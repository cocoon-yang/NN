#include "neuron.h"
#include "connection.h" 
#include "layer.h"
#include "string"

Neuron::Neuron(int num):_value(0.0f), _inputNum(num), ACTIVE(true), order(0.0f), _type(1)
{

}

Neuron::~Neuron()
{

}

Neuron::Neuron(const Neuron& RHS)
{
	_inputNum = RHS._inputNum;
	order = RHS.order;
	_type = RHS._type;
	ACTIVE = RHS.ACTIVE;
	_value = RHS._value;

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
		ACTIVE = RHS.ACTIVE;
		_value = RHS._value;

		_pConnections.clear();
		for (std::size_t i = 0; i < _inputNum; i++) {
			_pConnections.push_back(RHS._pConnections[i]);
		} 
	}
	return *this;
}

void Neuron::init(Layer* pInputLayer)
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
		_pConnections[i]->_weight = tmp;
	} 
} 


/**
\brief Calculate the output of the neuron
@param[in] pVal: input variable vector
*/
DataType Neuron::run()
{
	// _value = 1.0;
	DataType tmp = 0.0; 

	if (!ACTIVE)
	{
		return tmp;
	}

	for (size_t j = 0; j < _inputNum; j++)
	{
		if (!_pConnections[j])
		{
			continue;
		}
		if(!(_pConnections[j]->_pInputNero))
		{
			continue;
		}
		DataType val = _pConnections[j]->_pInputNero->_value;
		DataType w = _pConnections[j]->_weight;

		// tmp += pVal[j] * (_weights[j]);  
		tmp += val * w;

#ifdef _DEBUG_ 
		std::cout << "        tmp += " << val << " * " << w << " = " << tmp << std::endl;
#endif
	}
	// Calculating the output value 
	if (0.0 == order)
	{
		_value = tmp;
	}
	else {
		_value = calcuOutput(tmp); //  pow(tmp, order);
	}
	return _value;
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

std::string Neuron::toStr()
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

void Neuron::updataWeight(DataType diffVal, std::shared_ptr<DataType[]> varGrad, DataType learnRate)
{
	std::shared_ptr<DataType[]> pVal = std::shared_ptr<DataType[]>(new DataType[_inputNum]);

	for (size_t j = 0; j < _inputNum; j++)
	{
		if (!_pConnections[j])
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



	for (int j = 0; j < _inputNum; j++)
	{
		DataType tmp = 1.0;
		tmp = calcuGrad(diffVal, pVal[j]);

		//varGrad[j] += _weights[j] * tmp; 
		if (_pConnections[j])
		{
			varGrad[j] += _pConnections[j]->_weight * tmp;
		}


		//tmp = calcuOutput(pVal[j]);
		//_weights[j] -= learnRate * tmp * diffVal;
		tmp = _pConnections[j]->_pInputNero->_value;




		_pConnections[j]->_weight -= learnRate  * diffVal/ tmp;

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
}


/**
\brief Get one Connection pointer of the neuron
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

DataType Neuron::calcuGrad(DataType diffVar, DataType value)
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

DataType Neuron::calcuOutput(DataType value)
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
		//case SIGMOID_SYMMETRIC_STEPWISE:
		//	result = (DataType)stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, value);
		//	break;
		//case SIGMOID_STEPWISE:
		//	result = (DataType)stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, value);
		//	break;
	case THRESHOLD:
		result = (DataType)((value < 0) ? 0 : 1);
		break;
	case THRESHOLD_SYMMETRIC:
		result = (DataType)((value < 0) ? -1 : 1);
		break;
	case GAUSSIAN:
		result = (DataType)gaussian_real(value);
		break;
		//case GAUSSIAN_SYMMETRIC:
		//	result = (DataType)gaussian_symmetric_real(value);
		//	break;
		//case ELLIOT:
		//	result = (DataType)elliot_real(value);
		//	break;
		//case ELLIOT_SYMMETRIC:
		//	result = (DataType)elliot_symmetric_real(value);
		//	break;
		//case SIN_SYMMETRIC:
		//	result = (DataType)sin_symmetric_real(value);
		//	break;
		//case COS_SYMMETRIC:
		//	result = (DataType)cos_symmetric_real(value);
		//	break;
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
