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

vector<Point*> points;

bool training = false;
bool trainFromImage = false;
int numOfChars = 1000;
int numOfImages = 1;
double desicionThres = 0.8;
string type = "lowercase";
int numOfClasses = 5;
int charSize = 120;
double fontSize = charSize/30;
int imageSize = 1200;
double angle = 25;
int overlap = 8;
int tileNum = imageSize/charSize*overlap - (overlap-1);

//Random forest parameters
int numOfForests = 2;
int maxDepth = 15;
int minSampleCount =numOfChars/100;
float regressionAccuracy = 0;
bool useSurrugate = false;
int maxCategories = 10;
const float *priors;
bool calcVarImportance = false;
int nactiveVars = 100;
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
		for(int i=0; i<numOfForests; i++)
		{
			RandomCharacters trainingData = produceData(numOfChars,charSize,type,angle,fontSize,numOfClasses);
			Mat trainingFeatures = createRectFeatures(trainingData);

			cout << "Training forest nr: " << i << endl;
			tree.train(trainingFeatures,CV_ROW_SAMPLE,trainingData.responses,Mat(),Mat(),Mat(),Mat(),
			CvRTParams(maxDepth,minSampleCount,regressionAccuracy,useSurrugate,maxCategories,priors,calcVarImportance,nactiveVars,maxNumOfTreesInForest,forestAccuracy,termCritType));
			tree.save(intToStr(i,numOfChars,numOfClasses,type).c_str());
		}
	}
	else if(trainFromImage)
	{
		RandomCharactersImages testIm = createTestImages(1,50,charSize,imageSize,type,angle,fontSize,numOfClasses);
		String name = "Draw bounding boxes";
		namedWindow(name);
		box = Rect(0,0,1,1);

		Mat image = *testIm.randChars[0];
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
		//for(int i=0; i<boxVector.size(); i++)
			//cout << boxVector[i]->x << endl << boxVector[i]->height << endl;
		RandomCharacters trainingData = produceDataFromImage(boxVector,boxResponses,numOfChars,angle,imageCopy);
		Mat trainingFeatures = createRectFeatures(trainingData);
		
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
			forestVector[i]->load(intToStr(i,numOfChars,numOfClasses,type).c_str());
		}

		CSize charSizeXY = loadSizeFromFile("charSize.txt");
		charSizeXY.height = charSize;
		charSizeXY.width = charSize;
		cout << charSizeXY.width << endl << charSizeXY.height << endl;
		RandomCharactersImages testIm = createTestImages(numOfImages,50,charSize,imageSize,type,angle,fontSize,numOfClasses);
		RandomCharacters proxData = produceProxData(type,numOfClasses,charSize,fontSize);
		Mat proxDataFeatures = createRectFeatures(proxData);
		//tree.load("test_im.xml");
		//double numOfTrees = tree.get_tree_count();
		vector<Mat*> predictions = predictImages(testIm,forestVector,numOfImages,imageSize,charSizeXY.width,charSizeXY.height,overlap,maxNumOfTreesInForest,desicionThres,type, proxDataFeatures);
		evaluateResult(predictions,testIm,imageSize,charSizeXY.width,charSizeXY.height,numOfImages,overlap);

		cout << "tree count: " << tree.get_tree_count() << endl;

	}

	return 0;
}


