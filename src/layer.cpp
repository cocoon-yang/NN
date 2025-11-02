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

void Layer::init(Layer* pLayer)
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

void Layer::forward()
{
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

		//_pBiases[i] -= lr * output_grad[i];
	}
}