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
	bool cascadeTesting = false;
	int images = 100;
	int trainingImages = 100;
	int tileSize = 32;
	float strongClassThres = -5;
	string codeType = "all";
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;

	if(training)
	{		
		vector<char> trainingResponses = getResponses(images,0,tileSize,imageSize,tileNum,true,codeType);

		Mat trainingData = createStdFeatures(images,0,tileSize,imageSize,tileNum);					//
		//Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createDistanceFeatures(images,0,tileSize,imageSize,tileNum,100,300);
		//Mat trainingData = createFastCornerFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createBRISKFeatures(images,0,tileSize,imageSize,tileNum);
		//visualizeFeature(0,imageSize,tileSize,tileNum,0,trainingData);
		writeMatToFile(trainingData,trainingResponses,"std100_2D.txt");							//

		CvMLData cvml;
		cvml.read_csv("std100_2D.txt");															//
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(trainingImages*tileNum*tileNum, true);
		cvml.set_train_test_split(&cvtts);

		CvBoost boost;
		printf("Training....\n");
		boost.train(&cvml, CvBoostParams(CvBoost::GENTLE, 100, 0.5, 1, false, 0), false);				//


		//Calculate the training error
		float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR);
		//Calculate the test error
		float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR);

		cout << "Train error: " <<fl1 << endl 
			<< "Test error: " << fl2 << endl << endl;

		boost.save("./std100_boost100_2D.xml", "boost");											//


		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 165;																		//

		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		printf("Calculate features for testing:\n\n");
		//Mat testData = createDistanceFeatures(imNum,firstImage,tileSize,imageSize,tileNum,100,300);	//
		Mat testData = createStdFeatures(imNum,firstImage,tileSize,imageSize,tileNum);

		writeMatToXML(testData,"std165_2D.xml");													//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,50,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);
	}
	else if(cascadeTesting)
	{

	}
	else
	{
		//Test with saved data
		int scaleDown = 8;
		int firstImage = 100;
		const int imNum = 165;																		//
		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		CvBoost boost;											
		boost.load("std100_boost100_2D.xml");																//
		Mat testData = readMatFromXML("std165_2D.xml");											//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,100,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);

	}
	return 0;

}