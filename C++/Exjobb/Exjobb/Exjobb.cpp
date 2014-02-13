// Exjobb.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include"functions.h"
#include<vector>
#include <cstdlib>
//#include "cvplot.h"


int _tmain(int argc, _TCHAR* argv[])
{
	
	int images = 10;
	int trainingImages = 10;
	int tileSize = 64;
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;
	
	vector<char> trainingResponses = getResponses(images,0,tileSize,imageSize,tileNum);

	//Mat features = createSimpleFeatures(images,tileSize,imageSize,tileNum);
	Mat trainingData = createLBPFeatures(images,0,tileSize,imageSize,tileNum);
	
	cout << trainingData.size() << endl;
	writeMatToFile(trainingData,trainingResponses,"LBP265x64.txt");
	writeFileToMatlab(trainingData,"Matlab_LBP265x64.txt");
	
	
	CvMLData cvml;
	cvml.read_csv("LBP140x64.txt");
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
	
	boost.save("./LBP140x50_boost30.xml", "boost");
	CvSeq* weights = boost.get_weak_predictors();
	cout << weights->total << endl;
	
	
	int scaleDown = 4;
	int firstImage = 10;
	int imNum = 18;
	printf("Calculte features for testing....\n\n");
	vector<char> testResponses = getResponses(imNum,firstImage,tileSize,imageSize,tileNum);
	Mat testData = createLBPFeatures(imNum,firstImage,tileSize,imageSize,tileNum);
	Mat Im = testImages(firstImage,scaleDown,tileSize,imageSize,tileNum,boost,testData,testResponses);
	namedWindow("Test image",CV_WINDOW_AUTOSIZE);
	imshow("Test image", Im);
	waitKey();
	
	//Mat imForPlot = testForPlot(firstImage,imNum,tileSize,imageSize,tileNum,boost,LBP,responses);
	//CvPlot::plot("RGB", &imForPlot  , imNum, 3);
	//CvPlot::label("B");
	
	return 0;

}