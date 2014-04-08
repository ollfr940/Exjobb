// RandomForest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "functions.h"
#include "features.h"
#include "Classes.h"
#include "helperFunctions.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

vector<Point*> points;

bool training = true;
bool trainFromImage = true;
bool dataFromRealImage = false;
bool falseClass = true;
int numOfChars = 120;
int numOfImages = 1;
double desicionThres = 0.5;
string charType = "uppercase";
string featureType = "points";
int upSample = 2;
int numOfClasses = 26;
int charSize = 128;
double fontSize = charSize/30;
int imageWidth = 1024;
int imageHeight = 1024;
double angle = 10;
int charDiv = 15;
int charOrg = 20;
int overlap = 8;
int numOfPointPairs = 100000;

//Random forest parameters
int numOfForests = 12;
int maxDepth = 10;
int minSampleCount =numOfChars/100;
float regressionAccuracy = 0.2;
bool useSurrugate = false;
int maxCategories = 10;
const float *priors;
bool calcVarImportance = false;
int nactiveVars = 0; //0 = square root of total number of features in every split
int maxNumOfTreesInForest = 100;
float forestAccuracy = 0;
int termCritType = CV_TERMCRIT_ITER;

//Global variables for drawing boxes
bool destroy=false;
cv::Rect box;
vector<Rect*> boxVector;
vector<char> boxResponses;
bool drawing_box = false;
bool firstBox = true;
int firstBoxWidth, firstBoxHeight;

int _tmain(int argc, _TCHAR* argv[])
{
	CvRTrees tree;

	if(training)
	{
		bool n = true;
		if(argc == 1)
		{
			for(int i=0; i<numOfForests; i++)
			{
				RandomCharacters trainingData = produceData(numOfChars,charSize,charType,angle, charDiv, charOrg,fontSize,numOfClasses, falseClass);
				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStr(i,numOfChars,numOfClasses,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,n).c_str());
				n = false;
			}
		}
		else if(argc == 3)
		{
			for(int i=*argv[2]-48; i<numOfForests; i += *argv[1]-48)
			{
				RandomCharacters trainingData = produceData(numOfChars,charSize,charType,angle, charDiv,charOrg,fontSize,numOfClasses, falseClass);
				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStr(i,numOfChars,numOfClasses,maxDepth,maxNumOfTreesInForest,angle,charType,featureType, n).c_str());
				n = false;
			}
		}
		else
			cout << "Wrong number of arguments" << endl;
	}
	else if(trainFromImage)
	{
		Mat im, image;

		if(dataFromRealImage)
		{
			im = imread("im.jpg",CV_LOAD_IMAGE_GRAYSCALE);
			imageWidth = im.size().width/2;
			imageHeight = im.size().height/2;
			resize(im,image,Size(imageWidth,imageHeight));
		}
		else
		{
			RandomCharactersImages testIm = createTestImages(1,50,charSize,imageWidth,imageHeight,charType,angle,fontSize,numOfClasses);
			image = *testIm.randChars[0];
		}

		String name = "Draw bounding boxes";
		namedWindow(name);
		box = Rect(0,0,1,1);

		Mat imageCopy = image.clone();
		if(!image.data)
		{
			printf("!!! Failed \n");
			return 2;
		}

		Mat temp = image.clone();
		setMouseCallback(name, mouseCallback, &image);

		while(1)
		{
			temp = image.clone();
			if (drawing_box)
				draw_box(&temp, box);

			imshow(name, temp);

			if(waitKey(15) == 27)
				break;
		}

		RandomCharacters trainingData = produceDataFromImage(boxVector,boxResponses,numOfChars,angle,imageCopy,dataFromRealImage);
		Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType);

		printf("Training....\n");
		tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
		tree.save("test_im.xml");
		writeSizeToFile(trainingData.randChars[0]->size().width,trainingData.randChars[0]->size().height, "charSize.txt");
	}
	else
	{
		vector<CvRTrees*> forestVector;
		for(int i=0; i<numOfForests; i++)
		{
			forestVector.push_back(new CvRTrees);
			cout << "loading: " << intToStr(i,numOfChars,numOfClasses,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,false) << endl << endl;
			forestVector[i]->load(intToStr(i,numOfChars,numOfClasses,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,false).c_str());
		}

		CSize charSizeXY = loadSizeFromFile("charSize.txt");
		charSizeXY.height = charSize;
		charSizeXY.width = charSize;
		evaluateIm(forestVector,100,charSize,charType,featureType, charDiv,charOrg,fontSize,numOfClasses,numOfPointPairs,angle,maxNumOfTreesInForest,desicionThres, falseClass);
		//RandomCharactersImages testIm = createTestImages(numOfImages,50,charSize,imageWidth,imageHeight,charType,angle,fontSize,numOfClasses);
		//vector<Mat*> predictions = predictImages(testIm,forestVector,numOfImages,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,overlap,maxNumOfTreesInForest,desicionThres,numOfPointPairs,charType,featureType);
		//evaluateResult(predictions,testIm,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,numOfImages,overlap,upSample);

	}
	return 0;
}


