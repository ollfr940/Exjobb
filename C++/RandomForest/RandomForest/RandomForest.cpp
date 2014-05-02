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
bool backgroundDistinction = false;
bool trainFromImage = false;
bool dataFromRealImage = false;
bool evaluateRealImageUsingAfont = true;
bool falseClass = true;
bool useNoise = false;

int numOfTrueChars = 90;
int numOfFalseChars = 150;
int numOfImages = 1;
double desicionThres1 = 0.2;
double desicionThres2 = 0.9;
string charType = "digitsAndLetters";
string featureType = "rects";
int tileSizeX = 54;
int tileSizeY = 64;
int imageWidth = 1024;
int imageHeight = 1024;
double angle = 5;
int charDivX = 20; //25;
int charDivY = 20;
int overlap = 16;
int numOfPointPairs1 = 100000;
int numOfPointPairs2 = 1000;
int numOfPointPairs3 = 10000;

//Parameters for clustering
int minCluster = 5; 
int pixelConnectionThres = 2;

//Random forest parameters 
int numOfForests = 1;
int maxDepth = 10;
int minSampleCount = (numOfTrueChars*36 + numOfFalseChars)/100; //snumOfChars/100;
float regressionAccuracy = 0.95;
bool useSurrugate = false;
int maxCategories = 10;
const float *priors;
bool calcVarImportance = false;
int nactiveVars = 0; //0 = square root of total number of features in every split
int maxNumOfTreesInForest = 1000;
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
		bool useDiff = false;
		bool n = true;
		if(argc == 1)
		{
			for(int i=0; i<numOfForests; i++)
			{
				RandomCharacters trainingData = produceDataFromAfont(numOfTrueChars,charType,numOfFalseChars, tileSizeX, tileSizeY, charDivX,charDivY,angle,falseClass,useNoise);

				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs1,featureType,tileSizeX, tileSizeY,useNoise);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStrOCR(i,numOfTrueChars, numOfFalseChars,tileSizeX,tileSizeY,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,n,useNoise).c_str());
				n = false;
			}
		}
		else if(argc == 3)
		{
			for(int i=*argv[2]-48; i<numOfForests; i += *argv[1]-48)
			{
				RandomCharacters trainingData = produceDataFromAfont(numOfTrueChars,charType,numOfFalseChars, tileSizeX, tileSizeY,charDivX,charDivY,angle,falseClass,useNoise);

				Mat trainingFeatures = calcFeaturesTraining(trainingData,numOfPointPairs1,featureType,tileSizeX, tileSizeY,useNoise);
				cout << "Training forest nr: " << i << endl;
				tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
				CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
				tree.save(intToStrOCR(i,numOfTrueChars,numOfFalseChars,tileSizeX, tileSizeY,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass, n,useNoise).c_str());
				n = false;
			}
		}
		else
			cout << "Wrong number of arguments" << endl;
	}
	else if(backgroundDistinction)
	{
		bool n = true;
		useNoise = false;
		int numOfTrueCharacters = 1500; //1000; //500;
		int numOfFalseCharacters = 20000; //20000; // 20000; //43000;
		int numOfFalseImages = 300; //100;
		maxNumOfTreesInForest = 100;
		int reSizeTo = 8;
		minSampleCount = (numOfTrueCharacters*36+numOfFalseImages*(imageWidth/tileSizeX)*(imageHeight/tileSizeY) + numOfFalseCharacters)/100;

		vector<Mat*> randomImages = drawRandomImages(numOfFalseImages,imageWidth,50, 10, useNoise);
		RandomCharacters trainingData = produceDataFromAfont(numOfTrueCharacters,charType,numOfFalseCharacters, tileSizeX, tileSizeY, charDivX,charDivY,angle,true,useNoise);
		RandomImagesAndCharacters FeaturesAndResponses = calcPointPairFeaturesScales(randomImages, trainingData,tileSizeX,tileSizeY,reSizeTo, imageWidth,numOfPointPairs2,numOfTrueCharacters,numOfFalseCharacters,charType, useNoise);
		//RandomImagesAndCharacters FeaturesAndResponses = calcStandardDeviationFeatures(randomImages, trainingData, tileSizeX,tileSizeY, imageWidth,numOfTrueCharacters,numOfFalseCharacters,charType, reSizeTo);

		cout << "Training forest" << endl;
		tree.train(FeaturesAndResponses.features,CV_ROW_SAMPLE,FeaturesAndResponses.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));

		tree.save(intToStrBackground(0,numOfTrueCharacters,numOfFalseCharacters,numOfFalseImages,reSizeTo,featureType,n).c_str());
		n = false;


		reSizeTo = 16;
		minSampleCount = (numOfTrueCharacters*36+numOfFalseImages*(imageWidth/tileSizeX)*(imageHeight/tileSizeY) + numOfFalseCharacters)/100;

		randomImages = drawRandomImages(numOfFalseImages,imageWidth,50, 10, useNoise);
		trainingData = produceDataFromAfont(numOfTrueCharacters,charType,numOfFalseCharacters, tileSizeX, tileSizeY, charDivX,charDivY,angle,true,useNoise);
		FeaturesAndResponses = calcPointPairFeaturesScales(randomImages, trainingData,tileSizeX,tileSizeY,reSizeTo, imageWidth,numOfPointPairs3,numOfTrueCharacters,numOfFalseCharacters,charType, useNoise);
		//RandomImagesAndCharacters FeaturesAndResponses = calcStandardDeviationFeatures(randomImages, trainingData, tileSizeX,tileSizeY, imageWidth,numOfTrueCharacters,numOfFalseCharacters,charType, reSizeTo);

		cout << "Training forest" << endl;
		tree.train(FeaturesAndResponses.features,CV_ROW_SAMPLE,FeaturesAndResponses.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));

		tree.save(intToStrBackground(0,numOfTrueCharacters,numOfFalseCharacters,numOfFalseImages,reSizeTo,featureType,n).c_str());
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
		int imageNum = 2;
		Mat image;
		image = imread(getImageAndGroundTruthName(imageNum,'p',false).c_str(),CV_LOAD_IMAGE_GRAYSCALE);

		//Adjust image to fit the screen
		if(image.size().width > image.size().height)
		{
			imageWidth = 1600;
			imageHeight = imageWidth*(image.size().height/image.size().width);
		}
		else
		{
			imageHeight = 1200;
			imageWidth = imageHeight*((float)image.size().width/(float)image.size().height);
		}

		resize(image,image,Size(imageWidth,imageHeight));
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
		cout << boxVector[0]->size().width << endl;
		int imageSizeX = imageCopy.cols/((float)boxVector[0]->width/((float)tileSizeX));
		int imageSizeY = imageCopy.rows/((float)boxVector[0]->height/((float)tileSizeY));

		Mat reSizedImage;
		resize(imageCopy,reSizedImage,Size(imageSizeX,imageSizeY));
		vector<Mat*> testIm;
		testIm.push_back(&reSizedImage);
		vector<CvRTrees*> forestVector1 = loadForestsOCR(numOfForests,maxDepth,maxNumOfTreesInForest,numOfTrueChars,numOfFalseChars,tileSizeX, tileSizeY,angle,charType,featureType,falseClass,useNoise);
		vector<CvRTrees*> forestVector2 = loadForestsBackground();

		//evaluateBackground(*testIm[0],forestVector2,tileSizeX,tileSizeY,numOfPointPairs2,desicionThres2,overlap);

		vector<Mat*> predictions = predictRealImages(testIm,forestVector1,forestVector2,numOfImages,imageSizeX,imageSizeY,tileSizeX,tileSizeY,overlap,
			maxNumOfTreesInForest,desicionThres1, desicionThres2,numOfPointPairs1,numOfPointPairs2,numOfPointPairs3 ,charType,featureType, useNoise);

		evaluateResultRealImage(predictions,testIm,imageSizeX*2,imageSizeY*2,tileSizeX*2,tileSizeY*2,numOfImages,overlap,minCluster,pixelConnectionThres,true, imageNum);

	}
	else
	{
		vector<CvRTrees*> forestVector1 = loadForestsOCR(numOfForests,maxDepth,maxNumOfTreesInForest,numOfTrueChars,numOfFalseChars,tileSizeX, tileSizeY,angle,charType,featureType,falseClass,useNoise);
		vector<CvRTrees*> forestVector2 = loadForestsBackground();
		calcTreeForPlot(forestVector1[0],numOfPointPairs1,tileSizeX,tileSizeY,charDivX,charDivY,useNoise, maxDepth);

		/*
		CvForestTree* tree = forestVector1[0]->get_tree(0);
		const CvDTreeNode* treeNode = tree->get_root();
		CvDTreeSplit* split = treeNode->split;
		cout << treeNode->left->class_idx << endl;
		*/
		CSize charSizeXY = loadSizeFromFile("charSize.txt");
		charSizeXY.height = 64;
		charSizeXY.width = 64;

		//evaluateIm(forestVector,100,charSize,charType,featureType, charDivX,charDivY,charOrg,fontSize,charType,numOfPointPairs1,angle,maxNumOfTreesInForest,desicionThres1, falseClass,useAfont, downSample,useNoise);
		
		
		
		RandomCharactersImages testIm;
		testIm = createTestImagesAfont(numOfImages,50,tileSizeX, tileSizeY,charDivX,charDivY,imageWidth,imageHeight,charType,angle,charType,useNoise);

		evaluateBackground(*testIm.randChars[0],forestVector2,charSizeXY.width,charSizeXY.height,numOfPointPairs2,desicionThres2,overlap);
		/*
		vector<Mat*> predictions = predictImages(testIm,forestVector1,forestVector2,numOfImages,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,overlap,maxNumOfTreesInForest,
			desicionThres1, desicionThres2 ,numOfPointPairs1,numOfPointPairs2, charType,featureType, downSample,useNoise);

		evaluateResult(predictions,testIm,imageWidth,imageHeight,charSizeXY.width,charSizeXY.height,numOfImages,overlap,minCluster,pixelConnectionThres);
		*/
	}
	return 0;
}


