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
	bool cascadeTesting = true;
	int images = 100;
	int trainingImages = 100;
	int tileSize = 32;
	float strongClassThres = -1;
	string codeType = "2D";
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;

	//For testing
	int scaleDown = 8;
	int firstImage = 100;
	const int imNum = 165;

	if(training)
	{		
		vector<char> trainingResponses = getResponses(images,0,tileSize,imageSize,tileNum,true,codeType);

		//Mat trainingData = createStdFeatures(images,0,tileSize,imageSize,tileNum);					//
		//Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createDistanceFeatures(images,0,tileSize,imageSize,tileNum,100,300);
		Mat trainingData = createFastCornerFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createBRISKFeatures(images,0,tileSize,imageSize,tileNum);
		//visualizeFeature(0,imageSize,tileSize,tileNum,0,trainingData);
		writeMatToFile(trainingData,trainingResponses,"fast100_all.txt");							//

		CvMLData cvml;
		cvml.read_csv("fast100_all.txt");															//
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

		boost.save("./fast100_boost100_all.xml", "boost");											//


		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 165;																		//

		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		printf("Calculate features for testing:\n\n");
		//Mat testData = createDistanceFeatures(imNum,firstImage,tileSize,imageSize,tileNum,100,300);	//
		Mat testData = createFastCornerFeatures(imNum,firstImage,tileSize,imageSize,tileNum);

		writeMatToXML(testData,"fast165_all.xml");													//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,50,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);
	}
	else if(cascadeTesting)
	{
		//string cascadeStep1 = "std";
		//string cascadeStep2 = "FAST";
		vector<char> testResponsesAll = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,"all");
		vector<char> testResponses2D = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,"2D");
		vector<CvBoost*> boost;
		vector<double> strongClassVector;
		CvBoost boost1;
		CvBoost boost2;
		boost1.load("std100_boost100_all.xml");
		boost2.load("fast100_boost100_2D.xml");
		boost.push_back(&boost1);
		boost.push_back(&boost2);
		strongClassVector.push_back(-1);
		strongClassVector.push_back(-1);
		Mat predictions = cascade(firstImage,imNum,tileSize,imageSize,tileNum,boost,testResponsesAll,testResponses2D,strongClassVector);
		float b[imNum], c[imNum];
		evaluateResult2(firstImage,20,scaleDown,imNum,tileSize,imageSize,tileNum,testResponses2D,predictions,b,c);
	}
	else
	{
		//Test with saved data																		//
		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		CvBoost boost;											
		boost.load("fast100_boost100_2D.xml");																//
		Mat testData = readMatFromXML("fast165_2D.xml");											//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,100,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);

	}
	return 0;

}