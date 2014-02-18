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
		//vector<char> trainingResponses = getResponses1D(images,0,tileSize,imageSize,tileNum);			
		vector<char> trainingResponses = getResponses2D(images,0,tileSize,imageSize,tileNum);

		//Mat features = createSimpleFeatures(images,tileSize,imageSize,tileNum);
		Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);
		//Mat trainingData = createDistanceFeatures(images,2,tileSize,imageSize,tileNum,100,300);
		//visualizeFeature(0,imageSize,tileNum,0,trainingData);
		writeMatToFile(trainingData,trainingResponses,"LBP100_2D.txt");

		CvMLData cvml;
		cvml.read_csv("LBP100_2D.txt");
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(trainingImages*tileNum*tileNum, true);
		cvml.set_train_test_split(&cvtts);

		CvBoost boost;
		printf("Training....\n");
		boost.train(&cvml, CvBoostParams(CvBoost::GENTLE, 100, 0.95, 1, false, 0), false);


		vector<float> train_responses, test_responses;
		//Calculate the training error
		float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR);
		//Calculate the test error
		float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR);

		cout << "Train error: " <<fl1 << endl 
			<< "Test error: " << fl2 << endl << endl;

		boost.save("./LBP100_boost100_2D.xml", "boost");
		CvSeq* weights = boost.get_weak_predictors();
		cout << weights->total << endl;


		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 165;
		//vector<char> testResponses = getResponses1D(imNum,firstImage,tileSize,imageSize,tileNum);
		vector<char> testResponses = getResponses2D(imNum,firstImage,tileSize,imageSize,tileNum);

		printf("Calculate features for testing:\n\n");
		//Mat testData = createDistanceFeatures(imNum,firstImage,tileSize,imageSize,tileNum,100,300);
		Mat testData = createLBPFeatures(imNum,firstImage,tileSize,imageSize,tileNum);
		writeMatToXML(testData,"LBP165_2D.xml");
		//testImages(firstImage,scaleDown,tileSize,imageSize,tileNum,boost,testData,testResponses);

		float b[imNum], c[imNum]; testForPlot(firstImage,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c);
	}
	else
	{
		//Test with saved data
		int scaleDown = 8;
		int firstImage = trainingImages;
		const int imNum = 115;
		vector<char> testResponses = getResponses2D(imNum,firstImage,tileSize,imageSize,tileNum);
		CvBoost boost;
		boost.load("LBP100_boost100_2D.xml");
		Mat testData = readMatFromXML("LBP165_1D.xml");
		//testImages(firstImage,50,scaleDown,tileSize,imageSize,tileNum,boost,testData,testResponses);

		float b[imNum], c[imNum]; testForPlot(firstImage,imNum,tileSize,imageSize,tileNum,boost,testData,testResponses,b,c);
	}
	return 0;

}