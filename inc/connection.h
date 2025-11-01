#pragma once
#include "neuron.h"
#include "ass.h"

class Connection
{
public:
	Connection();
	virtual ~Connection();

public:
	DataType _weight;
	std::shared_ptr<Neuron> _pInputNero;
	std::shared_ptr<Neuron> _pOutputNero;
};

