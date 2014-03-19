// RandomForest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "functions.h"
#include "Classes.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

bool training = false;
int numOfChars = 500;
string type = "uppercase";
int charSize = 120;
int imageSize = 1200;
int overlap = 4;
int tileNum = imageSize/charSize*overlap - (overlap-1);

int _tmain(int argc, _TCHAR* argv[])
{
	if(training)
	{
		RandomCharacters trainingData = produceData(numOfChars,charSize,type);
		//Mat responses = createResponses(trainingNum,characters);
		printf("Create rect features....\n");
		Mat trainingFeatures = createRectFeatures(trainingData,numOfChars,charSize);
		//Mat trainingFeatures = creatSumFeatures(trainingData,trainingNum,imageSize, false);
		//writeMatToFile(trainingFeatures, responses, trainingNum,"test.txt");

		CvRTrees tree;
		printf("Training....\n");
		tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses);
		tree.save("lowercase.xml");
		/*CvMLData cvml;
		cvml.read_csv("test.txt");
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(characters*trainingNum, true);
		cvml.set_train_test_split(&cvtts);
		tree.train(&cvml);*/
	}
	else
	{
		RandomCharactersImages testIm = createTestImages(1,50,charSize,imageSize,"uppercase");
		imshow("test",*testIm.randChars[0]);
		waitKey();
		/*imshow("res",*testIm.responses[0]);
		waitKey();*/
		CvRTrees tree;
		tree.load("uppercase.xml");
		vector<Mat*> predictions = predictImages(testIm,tree,1,imageSize,charSize,overlap,tileNum,type);
		evaluateResult(predictions,testIm,imageSize,charSize,tileNum,overlap);
		//int testNum = 20;
		//evaluateRect(tree,testNum,charSize,type);
	}

	return 0;
}


