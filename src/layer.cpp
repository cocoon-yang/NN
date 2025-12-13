#include "layer.h"
#include "connection.h" 
#include "string"

Layer::Layer(int inNum, int outNum, int id) :_inputNum(inNum)
, _outputNum(outNum), _id(id) 
{
}
Layer::~Layer()
{

}

Layer::Layer(const Layer& RHS)
{
	_id = RHS._id;  
	_inputNum = RHS._inputNum;
	_outputNum = RHS._outputNum;
	 
	//_pBiases = new DataType[_outputNum];
	//memcpy_s(_pBiases, n * sizeof(DataType), RHS._pBiases, _outputNum * sizeof(DataType));
}

Layer& Layer::operator = (const Layer& RHS)
{
	if (this != &RHS)
	{
		_id = RHS._id;  
		_inputNum = RHS._inputNum;
		_outputNum = RHS._outputNum;
		 
		//_pBiases = new DataType[_outputNum];
		//memcpy_s(_pBiases, n * sizeof(DataType), RHS._pBiases, _outputNum * sizeof(DataType));
	}
	return *this;
}

void Layer::clear()
{
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

void Layer::init(std::shared_ptr<Layer> pLayer)
{
	clear();
	//if (nullptr == pLayer)
	//{
	//	_connectLayerID = -1;
	//}
	//else {
	//	_connectLayerID = pLayer->getID();
	//}

	int n = (int)_pNerons.size();
	if (0 == n)
	{
		for (int i = 0; i < _outputNum; i++)
		{
			std::shared_ptr<Neuron> p = std::make_shared<Neuron>(_inputNum);
			p->init(pLayer);
			p->setOrder(i);
			p->layerID = _id;
			p->index = i;
			for (int j = 0; j < _inputNum; j++)
			{
				std::shared_ptr<Connection> pConnect = p->getConnection(j);
				if (pConnect)
				{
					pConnect->_pOutputNero = p;

					if (nullptr != pLayer)
					{
						std::shared_ptr<Neuron> pInNeuro = pLayer->getNeuron(j);
						pConnect->_pInputNero = pInNeuro;
					} 
				} 
			}
			_pNerons.push_back(p);
		}
	}
	else {
		for (int i = 0; i < _outputNum; i++)
		{
			std::shared_ptr<Neuron> p = _pNerons[i];
			if (!p)
			{
				continue;
			}
			p->init();
			//p->setOrder(i);
			//uint n = p->getConnectionNum();
			//for (int j = 0; j < n; j++)
			//{
			//	std::shared_ptr<Connection> pConnect = p->getConnection(j);
			//	if (pConnect)
			//	{
			//		pConnect->_pOutputNero = p;
			//	}
			//}
		}
	} 
}

std::shared_ptr<Neuron> Layer::getNeuron(size_t index)
{
	std::shared_ptr<Neuron> p;
	if (index < _outputNum)
	{
		p = _pNerons[index];
	}
	return p;
} 

 
std::string Layer::toStr()
{
	std::string result = std::to_string(_inputNum) + " " + std::to_string(_outputNum)
		// + " " + std::to_string(_connectLayerID)
		+ "\n";
	for (int i = 0; i < _outputNum; i++)
	{
		std::shared_ptr<Neuron> p = _pNerons[i];
		if (!p)
		{
			result += "\n";
		}
		else {
			result += p->toStr();
		}
	}
	return result;
}

void Layer::show()
{
	std::cout << toStr() << std::endl;
}

void Layer::setOrder(float val)
{
	for (int i = 0; i < _outputNum; i++) {
		std::shared_ptr<Neuron> pNeuron = _pNerons[i];
		if (!pNeuron)
		{
			continue;
		}
		pNeuron->setOrder(val);
	}
}

void Layer::killConnection(uint neuronIndex, uint connectionIndex)
{
	if (neuronIndex >= _outputNum)
	{
		return;
	}
	std::shared_ptr<Neuron> pNeuron = _pNerons[neuronIndex];
	if (!pNeuron)
	{
		return;
	}
	pNeuron->killConnection(connectionIndex);
	return;
}

void Layer::activeConnection(uint neuronIndex, uint connectionIndex)
{
	if (neuronIndex >= _outputNum)
	{
		return;
	}
	std::shared_ptr<Neuron> pNeuron = _pNerons[neuronIndex];
	if (!pNeuron)
	{
		return;
	}
	pNeuron->activeConnection(connectionIndex);
	return;
}


void Layer::activeRandConnection(uint num)
{
	if (num >= _outputNum)
	{
		return;
	}

	for (int i = 0; i < _outputNum; i++) {
		std::shared_ptr<Neuron> pNeuron = _pNerons[i];
		if (!pNeuron)
		{
			continue;
		}
		uint n = pNeuron->getConnectionNum();
		for (int j = 0; j < n; j++)
		{
			pNeuron->killConnection(j);
		}
		float tmp = getRandVal();
		n = tmp * (n + 1);
		pNeuron->activeConnection(n);
	}
	 
	return;
}

void Layer::forward()
{
#ifdef _DEBUG_
	std::cout << std::endl;
	std::cout << "Layer " << _id << " forward():" << std::endl;
#endif

	for (int i = 0; i < _outputNum; i++) {
		std::shared_ptr<Neuron> pNeuron = _pNerons[i];
		if (!pNeuron)
		{
			continue;
		}
		pNeuron->run( );
	} 
	return;
} 

void Layer::backward(std::shared_ptr<DataType[]> output_grad, std::shared_ptr<DataType[]> input_grad, float lr)
{
	for (int i = 0; i < _outputNum; i++) {
		std::shared_ptr<Neuron> pNeuron = _pNerons[i];
		if (!pNeuron)
		{
			continue;
		}
		pNeuron->updataWeight( output_grad[i], input_grad, lr); 
	}
}