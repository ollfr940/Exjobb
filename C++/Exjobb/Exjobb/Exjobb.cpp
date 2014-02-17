// Exjobb.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include"functions.h"
#include<vector>
#include <cstdlib>
#include "cvplot.h"
using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	bool training =  false;

	int images = 100;
	int trainingImages = 100;
	int tileSize = 64;
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;

	if(training)
	{
		vector<char> trainingResponses1D = getResponses1D(images,0,tileSize,imageSize,tileNum);
		//vector<char> trainingResponses2D = getResponses2D(images,0,tileSize,imageSize,tileNum);

		//Mat features = createSimpleFeatures(images,tileSize,imageSize,tileNum);
		Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);

		writeMatToFile(trainingData,trainingResponses1D,"LBP100_1D.txt");
		//writeFileToMatlab(trainingData,"Matlab_LBP265x64.txt");

		CvMLData cvml;
		cvml.read_csv("LBP100_1D.txt");
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(trainingImages*tileNum*tileNum, true);
		cvml.set_train_test_split(&cvtts);

		CvBoost boost;
		printf("Training....\n");
		boost.train(&cvml, CvBoostParams(CvBoost::GENTLE, 30, 0.5, 1, false, 0), false);


		vector<float> train_responses, test_responses;
		//Calculate the training error
		float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR);
		//Calculate the test error
		float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR);

		cout << "Train error: " <<fl1 << endl 
			<< "Test error: " << fl2 << endl << endl;

		boost.save("./LBP100_boost30_1D.xml", "boost");
		CvSeq* weights = boost.get_weak_predictors();
		cout << weights->total << endl;


		int scaleDown = 4;
		int firstImage = 100;
		const int imNum = 165;
		vector<char> testResponses1D = getResponses1D(imNum,firstImage,tileSize,imageSize,tileNum);

		printf("Calculate features for testing:\n\n");
		Mat testData = createLBPFeatures(imNum,firstImage,tileSize,imageSize,tileNum);
		writeMatToXML(testData,"LBP165_1D.xml");
		//testImages(firstImage,scaleDown,tileSize,imageSize,tileNum,boost,testData,testResponses);

		float b[imNum], c[imNum];
		testForPlot(firstImage,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses1D,b,c);
	}
	else
	{
		//Test with saved data
		int scaleDown = 8;
		int firstImage = 100;
		const int imNum = 165;
		Mat d = createFastCornerFeatures(imNum,firstImage,tileSize,imageSize,tileNum,100);
		vector<char> testResponses = getResponses1D(imNum,firstImage,tileSize,imageSize,tileNum);
		CvBoost boost;
		boost.load("LBP100_boost30_1D.xml");
		Mat testData = readMatFromXML("LBP165_1D.xml");
		testImages(firstImage,10,scaleDown,tileSize,imageSize,tileNum,boost,testData,testResponses);

		//float b[imNum], c[imNum]; testForPlot(firstImage,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c);
	}
	return 0;

}