// Exjobb.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "functions.h"
#include "Classes.h"
#include <vector>
#include <cstdlib>
#include "cvplot.h"
using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{
	bool training =  false;
	bool cascadeTesting = true;
	int images = 50;
	int trainingImages = 50;
	int tileSize = 32;
	float strongClassThres = -1;
	string codeType = "1D";
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;

	//For testing
	int scaleDown = 4;
	int firstImage = 50;
	const int imNum = 90;

	if(training)
	{		
		vector<char> trainingResponses = getResponses(images,0,tileSize,imageSize,tileNum,true,codeType);

		//Mat trainingData = createStdFeatures(images,0,tileSize,imageSize,tileNum);					//
		//Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createDistanceFeatures(images,0,tileSize,imageSize,tileNum,100,300);
		//Mat trainingData = createFastCornerFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createBRISKFeatures(images,0,tileSize,imageSize,tileNum);
		Mat trainingData = createI1DFeatures(images,0,tileSize,imageSize,tileNum);
		//visualizeFeature(0,imageSize,tileSize,tileNum,0,trainingData);
		writeMatToFile(trainingData,trainingResponses,"i1d50_1D.txt");							//

		CvMLData cvml;
		cvml.read_csv("i1d50_1D.txt");															//
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

		boost.save("./i1d50_boost50_1D.xml", "boost");											//


		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 90;																		//

		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		printf("Calculate features for testing:\n\n");
		//Mat testData = createDistanceFeatures(imNum,firstImage,tileSize,imageSize,tileNum,100,300);	//
		Mat testData = createI1DFeatures(imNum,firstImage,tileSize,imageSize,tileNum);

		writeMatToXML(testData,"i1d90_1D.xml");													//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,20,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);
	}
	else if(cascadeTesting)
	{
		vector<char> testResponses1D = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,"1D");
		vector<char> testResponses2D = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,"2D");
		vector<CvBoost*> boost;
		vector<double> strongClassVector;
		vector<CalcSample*> featureFuncs;
		featureFuncs.push_back(new CalcSTDSample(1,cv::Mat(1,2,CV_32FC1)));
		featureFuncs.push_back(new CalcFASTSample(3,cv::Mat(1,4,CV_32FC1)));
		featureFuncs.push_back(new CalcLBPSample(256,cv::Mat(1,257,CV_32FC1),cv::Mat::zeros(tileSize-2,tileSize-2,CV_32FC1),tileSize));
		featureFuncs.push_back(new CalcI1DSample(1,cv::Mat(1,2,CV_32FC1)));

		CvBoost boost1;
		CvBoost boost2;
		CvBoost boost3;
		CvBoost boost4;
		boost1.load("std50_boost50_all.xml");
		boost2.load("fast50_boost50_2D.xml");
		boost3.load("LBP50_boost50_all.xml");
		boost4.load("i1d50_boost50_1D.xml");
		boost.push_back(&boost1);
		boost.push_back(&boost2);
		boost.push_back(&boost3);
		boost.push_back(&boost4);
		strongClassVector.push_back(-1);
		strongClassVector.push_back(-1);
		strongClassVector.push_back(-1);
		strongClassVector.push_back(-3);

		Mat predictions = cascade(firstImage,imNum,tileSize,imageSize,tileNum,boost,strongClassVector,featureFuncs);
		float b[imNum], c[imNum], d[imNum], e[imNum];
		evaluateCascade(firstImage,20,scaleDown,imNum,tileSize,imageSize,tileNum,testResponses1D,testResponses2D,predictions,b,c,d,e);
	}
	else
	{
		//Test with saved data																		//
		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,false,codeType);
		CvBoost boost;											
		boost.load("fast50_boost50_2D.xml");																//
		Mat testData = readMatFromXML("fast90_2D.xml");											//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,0,scaleDown,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c,strongClassThres);

	}
	return 0;

}