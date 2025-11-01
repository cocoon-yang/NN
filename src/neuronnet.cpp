#include "neuronnet.h" 
#include <fstream> 

NeuronNet::NeuronNet():_FINISH(false)
{

} 

NeuronNet::~NeuronNet()
{

} 

void NeuronNet::setModel(std::vector<uint> theTop)
{
	topology = theTop;
	size_t n = topology.size(); 
	int in = 1;
	int out = topology[0];

	// Input Layer 
	std::vector<float> weightVec;
	for (size_t i = 0; i < out; i++)
	{
		weightVec.push_back(1.0f);
	}
	Layer* pLay = new Layer(1, out, 0);
	pLay->init(); 
	for (size_t i = 0; i < out; i++)
	{
		std::shared_ptr<Neuron> pN = pLay->getNeuron(i);
		pN->setWeight(weightVec);
	} 
	model.push_back(pLay);

	for (size_t i = 1; i < n; i++)
	{
		in = topology[i-1];
		out = topology[i];
		pLay = new Layer(in, out, i);
		if (i > 0)
		{
			Layer* pInLay = model[i - 1];
			pLay->init(pInLay);
		}
		else {
			pLay->init();
		} 
		model.push_back(pLay);
	}

	if (n > 2)
	{
		pLay = model[1];
		if (!pLay)
		{
			return;
		}
		out = pLay->getOutputNum();
		for (size_t i = 0; i < out; i++)
		{
			weightVec.push_back(1.0f);
		}
		for (size_t i = 0; i < out; i++)
		{
			std::shared_ptr<Neuron> pN = pLay->getNeuron(i);
			pN->setWeight(weightVec);
		}
	}
	 
	// Output Layer  

}

void NeuronNet::init()
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

void NeuronNet::predict(float* input)
{
	size_t n = model.size();
	Layer* pLayer = model[0]; 

	if (nullptr == pLayer)
	{
		return;
	}

	int m = pLayer->getOutputNum();
	for (int i = 0; i < m; i++)
	{
		std::shared_ptr<Neuron> p = pLayer->getNeuron(i); 
		if (!p)
		{
			continue;
		}
		p->setValue(input[i]);
	}

	for (size_t i = 1; i < n; i++)
	{
		pLayer = model[i];
		if (!pLayer)
		{
			return;
		} 
		pLayer->forward( );
	}
} 

void NeuronNet::train(float* input, float* y, float lr)
{
	predict(input);

	size_t n = topology.size();
	int OUTPUT_SIZE = topology[n - 1];

	size_t m = model.size();
	Layer* pLayer = model[m - 1]; 
	if (!pLayer)
	{
		return;
	}
	bool OK = true;
	std::shared_ptr<DataType[]> pDiff = std::shared_ptr<DataType[]>(new DataType[OUTPUT_SIZE]);
	for (size_t i = 0; i < OUTPUT_SIZE; i++)
	{
		pDiff[i] = 0.0;
		std::shared_ptr<Neuron> pNeuron = pLayer->getNeuron(i);
		if (!pNeuron)
		{
			continue;
		}
		float tmp = pNeuron->getValue();
		pDiff[i] = 2.0f * (tmp - y[i]);
		if (fabs(pDiff[i]) > 100.0f)
		{
			OK = false;
			break;
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
	//// DEBUG -- END --  


	for (int i = (int)m - 1; i > 0; i--)
	{
		pLayer = model[i];
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

		pLayer->backward( pDiff, input_Grad, lr);

		//  ReLU Method 
		for (int i = 0; i < inputNum; i++)
		{
			input_Grad[i] *= fabs(input_Grad[i]) > 2.0f ? 0.0f : input_Grad[i];
		}

		pDiff.reset();
		pDiff = input_Grad;
	}
}

void NeuronNet::save(const char* fileName)
{ 
	std::ofstream myfile;
	myfile.open(fileName);
	myfile << "Writing this to a file.\n";

	size_t n = model.size();
	myfile << n << std::endl;
	for (size_t i = 0; i < n; i++)
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