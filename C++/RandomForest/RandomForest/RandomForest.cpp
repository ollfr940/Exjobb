// RandomForest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "functions.h"
#include "features.h"
#include "Classes.h"
#include "helperFunctions.h"
#include "functionsForRealImages.h"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

vector<Point*> points;

bool training = true;
bool trainFromImage = false;
bool dataFromRealImage = false;
bool evaluateRealImageUsingAfont = true;
bool falseClass = true;
bool useAfont = true;
bool useNoise = true;

int numOfChars = 30;
int numOfFalseData = 60;
int numOfImages = 1;
double desicionThres = 0.4;
string charType = "digitsAndLetters";
string featureType = "points";
int downSample = 2;
int charSize = 128;
double fontSize = charSize/30;
int imageWidth = 1024;
int imageHeight = 1024;
double angle = 10;
int charDivX = 25;
int charDivY = 10;
int charOrg = 25;
int overlap = 16;
int numOfPointPairs = 100000;

//Parameters for clustering
int minCluster = 5; 
int pixelConnectionThres = 5;

//Random forest parameters 
int numOfForests = 10;
int maxDepth = 10;
int minSampleCount =numOfChars/100;
float regressionAccuracy = 0.4;
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
				RandomCharacters trainingData;
				if(useAfont)
					trainingData = produceDataFromAfont(numOfChars,charType,numOfFalseData,charDivX,charDivY,angle,falseClass,downSample,useNoise);
				//else
					//trainingData = produceData(numOfChars,charSize,charType,angle, charDivX,charDivY, charOrg,fontSize, falseClass,useNoise);

				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType, downSample,useNoise);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStr(i,numOfChars,charSize,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,n,useNoise, useAfont).c_str());
				n = false;
			}
		}
		else if(argc == 3)
		{
			for(int i=*argv[2]-48; i<numOfForests; i += *argv[1]-48)
			{
				RandomCharacters trainingData;
				if(useAfont)
					trainingData = produceDataFromAfont(numOfChars,charType,numOfFalseData,charDivX,charDivY,angle,falseClass,downSample,useNoise);
				//else
					//trainingData = produceData(numOfChars,charSize,charType,angle, charDivX,charDivY, charOrg,fontSize, falseClass,useNoise);

				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType,downSample,useNoise);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStr(i,numOfChars,charSize,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass, n,useNoise, useAfont).c_str());
				n = false;
			}
		}
		else
			cout << "Wrong number of arguments" << endl;
	}
	else if(trainFromImage)
	{
		/*Mat im, image;

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
			printf("Failed \n");
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
		Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs,featureType, downSample,useNoise);

		printf("Training....\n");
		tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
		tree.save("test_im.xml");
		writeSizeToFile(trainingData.randChars[0]->size().width,trainingData.randChars[0]->size().height, "charSize.txt");*/
	}
	else if(evaluateRealImageUsingAfont)
	{
		Mat im, image;
		im = imread("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\Images\\im8.jpg",CV_LOAD_IMAGE_GRAYSCALE);
		imageWidth = im.size().width/2;
		imageHeight = im.size().height/2;
		resize(im,image,Size(imageWidth,imageHeight));
		String name = "Draw bounding boxes for adjusting image";
		namedWindow(name);
		box = Rect(0,0,1,1);

		Mat imageCopy = image.clone();

		Mat temp = image.clone();
		setMouseCallback(name, drawSquareToAdjustImage, &image);

		while(1)
		{
			temp = image.clone();
			if (drawing_box)
				draw_box(&temp, box);

			imshow(name, temp);

			if(waitKey(15) == 27)
				break;

			if(waitKey(15) == 32)
 				image = imageCopy.clone();
		}
		cv::destroyWindow(name);
		//Resize image to adjust tile to size 64x64
		cout << boxVector[0]->size().height << endl;
		cout << boxVector[0]->height << endl;
		int imageSizeX = imageCopy.cols/(boxVector[0]->height/((float)charSize/downSample));
		int imageSizeY = imageCopy.rows/(boxVector[0]->height/((float)charSize/downSample));
		int charSizeX = charSize/downSample;
		int charSizeY = charSize/downSample;

		Mat reSizedImage;
		resize(imageCopy,reSizedImage,Size(imageSizeX,imageSizeY));
		vector<Mat*> testIm;
		testIm.push_back(&reSizedImage);
		vector<CvRTrees*> forestVector = loadForests(numOfForests,maxDepth,maxNumOfTreesInForest,numOfChars,charSize,angle,charType,featureType,falseClass,useNoise, useAfont);
		//cv::threshold(reSizedImage,reSizedImage,128,255,CV_8UC1);
		//imshow("LSDFJ",reSizedImage);
		//waitKey();

		vector<Mat*> predictions = predictRealImages(testIm,forestVector,numOfImages,imageSizeX,imageSizeY,charSizeX,charSizeY,overlap,
			maxNumOfTreesInForest,desicionThres,numOfPointPairs,charType,featureType, downSample, useNoise);

		evaluateResultRealImage(predictions,testIm,imageSizeX*downSample,imageSizeY*downSample,charSizeX*downSample,charSizeY*downSample,numOfImages,overlap,downSample,minCluster,pixelConnectionThres);

	}
	else
	{
		vector<CvRTrees*> forestVector = loadForests(numOfForests,maxDepth,maxNumOfTreesInForest,numOfChars,charSize,angle,charType,featureType,falseClass,useNoise, useAfont);
		
		CSize charSizeXY = loadSizeFromFile("charSize.txt");
		charSizeXY.height = charSize;
		charSizeXY.width = charSize;

		//evaluateIm(forestVector,100,charSize,charType,featureType, charDivX,charDivY,charOrg,fontSize,charType,numOfPointPairs,angle,maxNumOfTreesInForest,desicionThres, falseClass,useAfont, downSample,useNoise);
		
		
		RandomCharactersImages testIm;
		if(useAfont)
			testIm = createTestImagesAfont(numOfImages,50,charSize,charDivX,charDivY,imageWidth,imageHeight,charType,angle,fontSize,charType, downSample,useNoise);
		//else
			//testIm = createTestImages(numOfImages,50,charSize,imageWidth,imageHeight,charType,angle,fontSize,numOfClasses);
		
		vector<Mat*> predictions = predictImages(testIm,forestVector,numOfImages,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,overlap,maxNumOfTreesInForest,
			desicionThres,numOfPointPairs,charType,featureType, downSample,useNoise);

		evaluateResult(predictions,testIm,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,numOfImages,overlap,downSample,minCluster,pixelConnectionThres);
		
	}
	return 0;
}


