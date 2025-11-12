#pragma once
#include <iostream>

class GoldDiv
{
public:
	GoldDiv() {
		Index = 0;
		II = 0;
	}
	~GoldDiv() {}

public:
	void update(double val)
	{
		OBJ = val;
		switch (Index)
		{
		case 0:
		{
			FL = val;
			init();
			break;
		}
		case 1:
		{
			setF1(val);
			break;
		}
		case 2:
		{
			setF2(val);
			break;
		}
		case 3:
		{
			refreshFU(val);
			break;
		}
		default:
			break;
		}

		if (OBJ < 0.01)
		{
			retriveALPHA();
		}
		return;
	}

	void setXL(double val)
	{
		XL = val;
	}

	void setXU(double val)
	{
		XU = val;
	}

	double getX()
	{
		switch (Index)
		{
		case 0:
			Alpha = XL;
			break;
		case 1:
			Alpha = X1;
			break;
		case 2:
			Alpha = X2;
			break;
		case 3:
			Alpha = XU;
			break;
		}
		return Alpha;
	}

	double getAlpha()
	{
		return Alpha;
	}

	void show()
	{
		std::cout << std::endl;
		std::cout << " GOLDEN SECTION METHOD RESULTS " << std::endl;
		std::cout << " SEARCH RANGE: [" << XL << ", " << XU << "]"
			<< std::endl;
		std::cout << " FL = " << FL << std::endl;
		std::cout << " F1 = " << F1 << std::endl;
		std::cout << " F2 = " << F2 << std::endl;
		std::cout << " FU = " << FU << std::endl;
		std::cout << " ALPHA = " << Alpha;
		std::cout << " OBJ = " << OBJ << std::endl;
		return;
	}

	int getIndex()
	{
		return Index;
	}
private:
	void init() {
		X1 = XL + 0.38196601 * (XU - XL);
		X2 = XU - 0.38196601 * (XU - XL);
		Index = 1;
	}

	void setF1(const double val)
	{
		F1 = val;
		Index = 2;
	}

	void setF2(const double val)
	{
		F2 = val;
		Index = 3;
	}

	void refreshFU(const double val)
	{
		FU = val;
		Index = 4;

		updateX();
	}

	void retriveALPHA()
	{
		Alpha = XL;
		OBJ = FL;
		if (F1 < OBJ) {
			Alpha = X1;
			OBJ = F1;
		}
		if (F2 < OBJ) {
			Alpha = X2;
			OBJ = F2;
		}
		if (FU < OBJ) {
			Alpha = XU;
			OBJ = FU;
		}
		Index = -1;
		return;
	}

	void updateX()
	{
		II++;
		if (II > 10)
		{
			retriveALPHA();
			return;
		}
		double Del = fabs(XU - XL);
		if (Del < 0.01)
		{
			retriveALPHA();
			return;
		}

		if (F2 <= F1)
		{
			XL = X1;
			FL = F1;

			X1 = X2;
			F1 = F2;

			X2 = XU - 0.38196601 * (XU - XL);
			Index = 2;
		}
		else {
			XU = X2;
			FU = F2;

			X2 = X1;
			F2 = F1;

			X1 = XL + 0.38196601 * (XU - XL);
			Index = 1;
		}


		return;
	}

private:
	int Index;
	double XL, X1, X2, XU;
	double FL, F1, F2, FU;
	double Alpha;
	double OBJ;

	int II;
};

