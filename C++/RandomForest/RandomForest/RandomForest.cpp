// RandomForest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "functions.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

int trainingNum = 1;
int imageSize = 120;
int _tmain(int argc, _TCHAR* argv[])
{
	vector<Mat*> trainingData = produceTrainingData(trainingNum,imageSize);
	Mat rectFeatures = createRectFeatures(trainingData,trainingNum,imageSize);

	return 0;
}

