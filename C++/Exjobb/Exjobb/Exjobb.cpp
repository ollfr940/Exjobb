// Exjobb.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include"functions.h"
#include<vector>
#include <cstdlib>



int _tmain(int argc, _TCHAR* argv[])
{

	int images = 140;
	int training_images = 50;
	int tileSize = 64;
	int imageSize = 2048;
	int tileNum = imageSize/tileSize;
	
	//vector<char> responses = getResponses(images,tileSize,imageSize,tileNum);

	//Mat features = createSimpleFeatures(images,tileSize,imageSize,tileNum);
	//Mat LBP = createLBPFeatures(images,tileSize,imageSize,tileNum);
	
	//writeMatToFile(LBP,responses,"LBP140x64.txt");
	
	
	CvMLData cvml;
	cvml.read_csv("LBP140x64.txt");
	cvml.set_response_idx(0);
	CvTrainTestSplit cvtts(training_images*tileNum*tileNum, true);
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
		<< "Test error: " << fl2 << endl;
	
	//boost.save("./LBP140x50x30_boost30.xml", "boost");
	CvSeq* weights = boost.get_weak_predictors();
	cout << weights->total << endl;
    //boost.save("weights_LBP140*50*30.txt");
	
	return 0;

}