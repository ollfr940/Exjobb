// RandomForest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "functions.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool training = true;
int trainingNum = 50;
int characters = 10;
int imageSize = 120;

int _tmain(int argc, _TCHAR* argv[])
{
	if(training)
	{
		vector<Mat*> trainingData = produceData(0,characters,trainingNum,imageSize);
		Mat responses = createResponses(trainingNum,characters);
		printf("Create rect features....\n");
		Mat trainingFeatures = createRectFeatures(trainingData,trainingNum,imageSize);
		//Mat trainingFeatures = creatSumFeatures(trainingData,trainingNum,imageSize, false);
		writeMatToFile(trainingFeatures, responses, trainingNum,"test.txt");

		CvRTrees tree;
		printf("Training....\n");
		tree.train(trainingFeatures,CV_ROW_SAMPLE,responses);
		tree.save("tree.xml");
		/*CvMLData cvml;
		cvml.read_csv("test.txt");
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(characters*trainingNum, true);
		cvml.set_train_test_split(&cvtts);
		tree.train(&cvml);*/
	}
	else
	{
		CvRTrees tree;
		tree.load("tree.xml");
		vector<Mat*> testData = produceData(7,1,1,imageSize);
		Mat testFeatures = createRectFeatures(testData,1, imageSize);
		
		cout << tree.predict(testFeatures) << endl;
	}

	return 0;
}

