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
	bool training =  true;
	bool cascadeTesting = false;
	int images = 100;
	int trainingImages = 100;
	int tileSize = 48;
	string codeType = "1D";
	int downSample = 2;
	int imageSize = 2048/downSample;
	int overlap = 2;
	int tileNum = imageSize/tileSize*overlap - (overlap-1);
	
	float thresholdStd = -6;//-5; //-4.5;
	float thresholdFAST = -7;//-7; //-1.5;
	float thresholdLBP = -10; //-4; //-3;
	float thresholdI1D = -2.5; //-2.5;//-1.5;
	float thresholdDist = -2; //-1.5;
	float strongClassThres = thresholdI1D;

	//For testing
	int scaleDown = 8;
	int firstImage = 100;
	const int imNum = 165;

	if(training)
	{		
		vector<char> trainingResponses = getResponses(images,0,tileSize,imageSize,tileNum,overlap,downSample,true,codeType);

		//CalcSample* featureFunc = new CalcSTDSample(1,cv::Mat(1,1,CV_32FC1),false);
		CalcSample* featureFunc = new CalcI1DSample(1,cv::Mat(1,1,CV_32FC1),false);
		//CalcSample* featureFunc = new CalcDistSample(2,cv::Mat(1,2,CV_32FC1),false);
		//CalcSample* featureFunc = new CalcFASTSample(3,cv::Mat(1,3,CV_32FC1),false);
		//CalcSample* featureFunc = new CalcLBPSample(256,cv::Mat(1,256,CV_32FC1),false,cv::Mat::zeros(tileSize-2,tileSize-2,CV_32FC1),tileSize);
		Mat trainingData = createFeatures(images,0,tileSize,imageSize,tileNum,overlap,downSample,true,featureFunc);
		writeMatToFile(trainingData,trainingResponses,"i1d100x48_1D_downsample2_overlap2_laplace5_vec.txt");							//

		CvMLData cvml;
		cvml.read_csv("i1d100x48_1D_downsample2_overlap2_laplace5_vec.txt");															//
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(trainingImages*tileNum*tileNum, true);
		cvml.set_train_test_split(&cvtts);

		CvBoost boost;
		printf("Training....\n");
		boost.train(&cvml, CvBoostParams(CvBoost::GENTLE, 100, 0.95, 1, false, 0), false);


		//Calculate the training error
		float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR);
		//Calculate the test error
		float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR);

		cout << "Train error: " <<fl1 << endl 
			<< "Test error: " << fl2 << endl << endl;

		boost.save("./i1d100x48_boost100_1D_downsample2_overlap2_laplace5_vec.xml", "boost");											//


		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 165;

		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,overlap,downSample,false,codeType);
		printf("Calculate features for testing:\n\n");
		Mat testData = createFeatures(imNum,firstImage,tileSize,imageSize,tileNum,overlap,downSample,true,featureFunc);

		writeMatToXML(testData,"i1d165x48_1D_downsample2_overlap2_laplace5_vec.xml");													//
		float b[imNum], c[imNum];
		evaluateResult(firstImage,20,scaleDown,imNum,tileSize,imageSize,tileNum,overlap,downSample,boost,testData,testResponses,b,c,strongClassThres);
	}
	else if(cascadeTesting)
	{
		vector<char> testResponses1D = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,overlap,downSample,false,"1D");
		vector<char> testResponses2D = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,overlap,downSample,false,"2D");
		vector<CvBoost*> boost;
		vector<float> strongClassVector;
		vector<CalcSample*> featureFuncs;
		featureFuncs.push_back(new CalcSTDSample(1,cv::Mat(1,2,CV_32FC1),true));
		featureFuncs.push_back(new CalcI1DSample(1,cv::Mat(1,2,CV_32FC1),true));
		featureFuncs.push_back(new CalcDistSample(2,cv::Mat(1,3,CV_32FC1),true));
		featureFuncs.push_back(new CalcFASTSample(3,cv::Mat(1,4,CV_32FC1),true));
		featureFuncs.push_back(new CalcLBPSample(256,cv::Mat(1,257,CV_32FC1),true,cv::Mat::zeros(tileSize-2,tileSize-2,CV_32FC1),tileSize));

		CvBoost boost1;
		CvBoost boost2;
		CvBoost boost3;
		CvBoost boost4;
		CvBoost boost5;
		boost1.load("std100x48_boost100_all_downsample2_overlap2_laplace5.xml");
		boost2.load("i1d100x48_boost100_1D_downsample2_overlap2_laplace5.xml");
		boost3.load("dist100x64_boost100_1D_downsample2_overlap2.xml");
		boost4.load("fast100x48_boost100_2D_downsample2_overlap2.xml");
		boost5.load("LBP100x64_boost100_2D_downsample2_overlap2.xml");
		boost.push_back(&boost1);
		boost.push_back(&boost2);
		boost.push_back(&boost3);
		boost.push_back(&boost4);
		boost.push_back(&boost5);
		strongClassVector.push_back(thresholdStd);
		strongClassVector.push_back(thresholdI1D);
		strongClassVector.push_back(thresholdDist);
		strongClassVector.push_back(thresholdFAST);
		strongClassVector.push_back(thresholdLBP);

		vector<Mat*> predictions = cascade(firstImage,imNum,tileSize,imageSize,tileNum,overlap,downSample,false,boost,strongClassVector,featureFuncs);
		float b[imNum], c[imNum], d[imNum], e[imNum];
		evaluateCascade(firstImage,20,scaleDown,imNum,tileSize,imageSize,tileNum,overlap,downSample,testResponses1D,testResponses2D,predictions,b,c,d,e);
	}
	else
	{
		//Test with saved data																		
		vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum,overlap,downSample,false,codeType);
		CvBoost boost;											
		boost.load("dist100x48_boost100_1D_downsample2_overlap2.xml");
		Mat testData = readMatFromXML("dist165x48_1D_downsample2_overlap2.xml");
		float b[imNum], c[imNum];
		evaluateResult(firstImage,100,scaleDown,imNum,tileSize,imageSize,tileNum,overlap,downSample,boost,testData,testResponses,b,c,strongClassThres);

	}
	return 0;

}