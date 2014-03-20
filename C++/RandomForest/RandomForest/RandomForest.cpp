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
int numOfChars = 1000;
int numOfImages = 3;
double desicionThres = 0;
string type = "uppercase";
int charSize = 120;
int imageSize = 1200;
double angle = 0;
int overlap = 8;
int tileNum = imageSize/charSize*overlap - (overlap-1);

//Random forest parameters
int maxDepth = 100;
int minSampleCount =10;
float regressionAccuracy = 0;
bool useSurrugate = false;
int maxCategories = 10;
const float *priors;
bool calcVarImportance = false;
int nactiveVars = 0;
int maxNumOfTreesInForest = 1000;
float forestAccuracy = 0;
int termCritType = CV_TERMCRIT_ITER;

int _tmain(int argc, _TCHAR* argv[])
{

	CvRTrees tree;
	if(training)
	{
		RandomCharacters trainingData = produceData(numOfChars,charSize,type,angle);
		//Mat responses = createResponses(trainingNum,characters);
		printf("Create rect features....\n");
		Mat trainingFeatures = createRectFeatures(trainingData,numOfChars,charSize);
		//Mat trainingFeatures = creatSumFeatures(trainingData,trainingNum,imageSize, false);
		//writeMatToFile(trainingFeatures, responses, trainingNum,"test.txt");

		printf("Training....\n");
		tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
		tree.save("uppercase.xml");
		/*CvMLData cvml;
		cvml.read_csv("test.txt");
		cvml.set_response_idx(0);
		CvTrainTestSplit cvtts(characters*trainingNum, true);
		cvml.set_train_test_split(&cvtts);
		tree.train(&cvml);*/
	}
	else
	{
		RandomCharactersImages testIm = createTestImages(numOfImages,50,charSize,imageSize,type,angle);
		imshow("test",*testIm.randChars[0]);
		waitKey();
		/*imshow("res",*testIm.responses[0]);
		waitKey();*/
		tree.load("uppercase.xml");
		double numOfTrees = tree.get_tree_count();
		vector<Mat*> predictions = predictImages(testIm,tree,numOfImages,imageSize,charSize,overlap,tileNum,numOfTrees,desicionThres,type);
		evaluateResult(predictions,testIm,imageSize,charSize,tileNum,overlap);
		//int testNum = 20;
		//evaluateRect(tree,testNum,charSize,type);
	}

	return 0;
}


