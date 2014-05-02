#include "stdafx.h"
#include "functions.h"
#include "helperFunctions.h"
using namespace std;
using namespace cv;

/*
RandomCharacters produceData(int numOfChars, int charSize, string type,double angle, int charDivX, int charDivY, int charOrg, double fontSize, bool falseClass, bool useNoise)
{
	printf("Produce data....\n\n");

	Mat* image;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);
	double randomAngle, scale = 1.0;
	char d, dd, randd;
	string dstr;
	int numOfDataForEachClass;

	if(type == "digits")
	{
		d = '0';
		numOfDataForEachClass = 10;
	}
	else if(type == "uppercase")
	{
		d = 'A';
		numOfDataForEachClass = 26;
	}
	else if(type == "lowercase")
	{
		d = 'a';
		numOfDataForEachClass = 26;
	}
	else if(type == "digitsAndLetters")
	{
		d = '0';
		numOfDataForEachClass = 36;
	}
	else
	{
		printf("error");
		abort();
	}

	RandomCharacters chars;
	if(falseClass)
		chars.responses = Mat::zeros(numOfChars*(numOfDataForEachClass+1),1,CV_32SC1);
	else
		chars.responses = Mat::zeros(numOfChars*numOfDataForEachClass,1,CV_32SC1);
	dd = d;

	for(int i=0; i<numOfDataForEachClass; i++)
	{
		for(int c=0; c<numOfChars; c++)
		{

			chars.responses.at<int>(i*numOfChars+c,0) = (int)dd;

			image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
			cv::add(*image,255,*image);
			if(useNoise)
				randn(*image,50,30);
			cv::Point org;
			
			if(dd == 'I')
			{
				org.x = rng.uniform(10,charSize-30);
				org.y = rng.uniform(charSize-charOrg-charDivX, charSize-charOrg+charDivY);
			}
			else
			{
				org.x = rng.uniform(charOrg-charDivX,charOrg+charDivX);
				org.y = rng.uniform(charSize-charOrg-charDivY, charSize-charOrg+charDivY);
			}
			dstr = dd;
			randomAngle = rng.uniform(-angle,angle);
			cv::putText(*image,dstr , org, 0, fontSize ,0, 10, 8,false);
			rotate(*image,randomAngle,charSize,charSize,scale);
			chars.randChars.push_back(image);
			//cv::imshow("im", *image);
			//cv::waitKey();
		}
		if(d =='9')
			d = 'A';
		else if(d == 'Z')
			d = 'a';
		else
			dd++;
	}
	
	if(falseClass)
	{
		char d1, d2;
		Point org1, org2;
		string dstr1, dstr2;

		for(int i=0; i<numOfChars; i++)
		{
			chars.responses.at<int>(numOfDataForEachClass*numOfChars+i,0) = 0;
			image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
			cv::add(*image,255,*image);	
			if(useNoise)
				randn(*image,50,30);

			d1 = rng.uniform(d,d+numOfDataForEachClass);
			d2 = rng.uniform(d,d+numOfDataForEachClass);
			dstr1 = d1;
			dstr2 = d2;

			org1.x = -charSize/2;
			org1.y = charSize-27;
			org2.x = charSize-30;
			org2.y = charSize-27;

			randomAngle = rng.uniform(-angle,angle);
			cv::putText(*image,dstr1 , org1, 0, fontSize ,0, 10, 8,false);
			cv::putText(*image,dstr2 , org2, 0, fontSize ,0, 10, 8,false);
			rotate(*image,randomAngle,charSize,charSize,scale);
			chars.randChars.push_back(image);
			//cv::imshow("im", *image);
			//cv::waitKey();
		}
	}

	return chars;
}
*/
RandomCharacters produceDataFromImage(vector<Rect*> boxVec, vector<char> boxRes, int numOfCharacters, double angle, Mat& image, bool useRealIm)
{
	printf("Produce data from image....\n\n");
	uint64 initValue = time(0);
	RNG rng(initValue);
	RandomCharacters chars;
	Mat* imRect,imCopy, imCopyTrans;
	int x, y;
	int	width = boxVector[0]->width;
	int	height = boxVector[0]->height;
	double randomAngle, scale = 1.0;
	chars.responses = Mat::zeros((int)boxVector.size()*numOfCharacters,1,CV_32SC1);

	for(int i=0; i<boxVector.size(); i++)
	{
		imCopy = image(*boxVector[i]).clone();

		if(useRealIm)
			preProcessRect(imCopy,128);
		cout << imCopy.channels();

		for(int j=0; j<numOfCharacters; j++)
		{
			imRect = new Mat(Mat::zeros(height,width,CV_8UC1));
			//imCopyTrans = Mat::zeros(width,height,CV_8UC1);
			add(*imRect,255,*imRect);
			x = rng.uniform(-width/5, width/5);
			y = rng.uniform(-height/5, height/5);

			if(x >= 0 && y >= 0)
				imCopy(Rect(x,y, width-x, height-y)).copyTo((*imRect)(Rect(0,0,width-x, height-y))); 
			else if(x >= 0 && y <= 0)
				imCopy(Rect(x,0,width-x, height+y)).copyTo((*imRect)(Rect(0,-y, width-x, height+y)));
			else if(x <= 0 && y <= 0)
				imCopy(Rect(0,0,width+x, height+y)).copyTo((*imRect)(Rect(-x,-y, width+x, height+y)));
			else if(x <= 0 && y >= 0)
				imCopy(Rect(-x,0, width+x, height-y)).copyTo((*imRect)(Rect(0,y,width+x, height-y))); 

			//imCopy = image(*boxVector[i]).clone();
			randomAngle = rng.uniform(-angle,angle);
			//imRect = new Mat(imCopyTrans.clone()); 
			//imCopy.deallocate();
			rotate(*imRect,randomAngle,width, height, scale);
			chars.randChars.push_back(imRect);
			chars.responses.at<int>(i*numOfCharacters+j,0) = boxRes[i];
			//cv::imshow("im", *imRect);
			//cv::waitKey();
		}
	}
	return chars;
}

RandomCharacters produceDataFromAfont(int numOfChars, string type, int numOfFalseData, int tileSizeX, int tileSizeY, int charDivX, int charDivY, double angle, bool falseClass, bool useNoise)
{
	printf("Produce data from OCR A-font....\n\n");
	uint64 initValue = time(0);
	RNG rng(initValue);
	int numOfDataForEachClass;
	RandomCharacters chars;
	char d;

	if(type == "digits")
	{
		d = '0';
		numOfDataForEachClass = 10;
	}
	else if(type == "uppercase")
	{
		d = 'A';
		numOfDataForEachClass = 26;
	}
	else if(type == "lowercase")
	{
		d = 'a';
		numOfDataForEachClass = 26;
	}
	else if(type == "digitsAndLetters")
	{
		d = '0';
		numOfDataForEachClass = 36;
	}
	else
	{
		printf("error");
		abort();
	}

	if(falseClass)
		chars.responses = Mat::zeros(numOfFalseData+numOfDataForEachClass*numOfChars,1,CV_32SC1);
	else
		chars.responses = Mat::zeros(numOfDataForEachClass*numOfChars,1,CV_32SC1);

	Mat* imRect, *imRectDownSampled,imCopy, imCopy1, imCopy2, imCopyTrans;
	int x, y, x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b, typeOfFalseData, randNoisePar1, randNoisePar2;
	int	width = tileSizeX*2;
	int	height = tileSizeY*2;
	double randomAngle, scale = 1.0;
	char d1, d2;
	string dstr, dstr1, dstr2, *name, *name1, *name2;

	for(int i=0; i<numOfDataForEachClass; i++)
	{
		dstr = d;
		name = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr+".png");
		imCopy = imread(name->c_str(),CV_LOAD_IMAGE_GRAYSCALE);

		for(int j=0; j<numOfChars; j++)
		{
			imRect = new Mat(Mat::zeros(height,width,CV_8UC1));
			imRectDownSampled = new Mat(Mat::zeros(tileSizeY,tileSizeX,CV_8UC1));
			add(*imRectDownSampled,255,*imRectDownSampled);
			if(useNoise)
			{
				randNoisePar1 = rng.uniform(30,150);
				randNoisePar2 = rng.uniform(0,8);
				randn(*imRectDownSampled,randNoisePar1,randNoisePar2);
			}
			add(*imRect,255,*imRect);

			if(width == height)
			{
				x = rng.uniform(-8-charDivX,-8+charDivX);
				y = rng.uniform(8-charDivY, 8+charDivY);
			}
			else
			{
				x = rng.uniform(-charDivX,charDivX);
				y = rng.uniform(8-charDivY,8+charDivY);
			}

			if(x >= 0 && y >= 0)
				imCopy(Rect(x,y, width-x, height-y)).copyTo((*imRect)(Rect(0,0,width-x, height-y))); 
			else if(x >= 0 && y <= 0)
				imCopy(Rect(x,0,width-x, height+y)).copyTo((*imRect)(Rect(0,-y, width-x, height+y)));
			else if(x <= 0 && y <= 0)
				imCopy(Rect(0,0,width+x, height+y)).copyTo((*imRect)(Rect(-x,-y, width+x, height+y)));
			else if(x <= 0 && y >= 0)
				imCopy(Rect(0,y, width+x, height-y)).copyTo((*imRect)(Rect(-x,0,width+x, height-y))); 

			randomAngle = rng.uniform(-angle,angle);
			rotate(*imRect,randomAngle,width, height, scale);
			resize(*imRect,*imRect,Size(tileSizeX,tileSizeY));
			threshold(*imRect,*imRect,128,1,CV_8UC1);
			multiply(*imRect,*imRectDownSampled,*imRectDownSampled);
			chars.randChars.push_back(imRectDownSampled);
			chars.responses.at<int>(i*numOfChars+j,0) = (int)d;
			//cv::imshow("im", *imRectDownSampled);
			//cv::waitKey();
		}
		if(d =='9')
			d = 'A';
		else
			d++;
	}

	if(falseClass)
	{
		for(int i=0; i<numOfFalseData; i++)
		{
			d1 = d + rng.uniform(-26,-1);
			d2 = d + rng.uniform(-26,-1);
			dstr1 = d1;
			dstr2 = d2;
			name1 = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr1+".png");
			name2 = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr2+".png");
			imCopy1 = imread(name1->c_str(),CV_LOAD_IMAGE_GRAYSCALE);
			imCopy2 = imread(name2->c_str(),CV_LOAD_IMAGE_GRAYSCALE);
			imRect = new Mat(Mat::zeros(height,width,CV_8UC1));
			imRectDownSampled = new Mat(Mat::zeros(tileSizeY,tileSizeX,CV_8UC1));
			add(*imRectDownSampled,255,*imRectDownSampled);
			if(useNoise)
			{
				randNoisePar1 = rng.uniform(30,150);
				randNoisePar2 = rng.uniform(0,8);
				randn(*imRectDownSampled,randNoisePar1,randNoisePar2);
			}
			add(*imRect,255,*imRect);

			if(width == height)
			{
				x1a = 80;
				y1a = 10;
				x2a = -80;
				y2a = 10;
			}
			else
			{
				x1a = 70;
				y1a = 10;
				x2a = -70;
				y2a = 10;
			}
			x1b = rng.uniform(0,20);
			x2b = rng.uniform(0,20);
			y1b = rng.uniform(60,100);
			y2b = rng.uniform(50,90);

			typeOfFalseData = rng.uniform(0,5);

			if(typeOfFalseData == 0)
				imCopy1(Rect(x1a,y1a, width-x1a, height-y1a)).copyTo((*imRect)(Rect(0,0,width-x1a, height-y1a)));
			else if(typeOfFalseData == 1)
				imCopy2(Rect(0,y2a, width+x2a, height-y2a)).copyTo((*imRect)(Rect(-x2a,0,width+x2a, height-y2a))); 
			else if(typeOfFalseData == 2)
			{
				imCopy1(Rect(x1a,y1a, width-x1a, height-y1a)).copyTo((*imRect)(Rect(0,0,width-x1a, height-y1a)));
				imCopy2(Rect(0,y2a, width+x2a, height-y2a)).copyTo((*imRect)(Rect(-x2a,0,width+x2a, height-y2a)));
			}
			else if(typeOfFalseData == 3)
			{
				imCopy1(Rect(x1b,y1b, width-x1b, height-y1b)).copyTo((*imRect)(Rect(x1b,0,width-x1b, height-y1b)));
			}
			else if(typeOfFalseData == 4)
			{
				imCopy2(Rect(x2b,0, width-x2b, y2b)).copyTo((*imRect)(Rect(x2b,height-y2b,width-x2b, y2b))); 
			}
			else if(typeOfFalseData == 5)
			{
				imCopy1(Rect(x1b,y1b, width-x1b, height-y1b)).copyTo((*imRect)(Rect(x1b,0,width-x1b, height-y1b)));
				imCopy2(Rect(x2b,0, width-x2b, y2b)).copyTo((*imRect)(Rect(x2b,height-y2b,width-x2b, y2b))); 
			}

			randomAngle = rng.uniform(-angle,angle);
			rotate(*imRect,randomAngle,width, height, scale);
			resize(*imRect,*imRect,Size(tileSizeX,tileSizeY));
			threshold(*imRect,*imRect,128,1,CV_8UC1);
			multiply(*imRect,*imRectDownSampled,*imRectDownSampled);
			chars.randChars.push_back(imRectDownSampled);
			chars.responses.at<int>(numOfDataForEachClass*numOfChars+i,0) = 0;
			//cv::imshow("im", *imRectDownSampled);
			//cv::waitKey();
		}
	}
	return chars;
}
/*
RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageWidth, int imageHeight, string type,double angle, double fontSize, int numOfClasses)
{
	printf("Create test images....\n\n");
	RandomCharactersImages charImages;
	Mat* image, *responses;
	Mat characterRect, responseRect;
	int xPos, yPos;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);
	double randomAngle, scale = 1.0;
	char d, randd;
	string dstr;

	if(type == "digits")
		d = '0';
	else if(type == "uppercase")
		d = 'A';
	else if(type == "lowercase")
		d = 'a';
	else
	{
		printf("error");
		abort();
	}

	for(int im=0; im<numOfImages; im++)
	{
		image = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		responses = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		cv::add(*image,255,*image);

		for(int c=0; c<numOfChars; c++)
		{
			randd = d + rng.uniform(0,numOfClasses);

			xPos = rng.uniform(0,imageWidth-charSize);
			yPos = rng.uniform(0,imageHeight-charSize);

			if(!sum((*responses)(Rect(xPos,yPos,charSize,charSize)))(0))
			{
				dstr = randd;
				characterRect = Mat::zeros(charSize,charSize,CV_8UC1);
				responseRect = Mat::zeros(charSize,charSize,CV_8UC1);
				add(characterRect,255,characterRect);
				add(responseRect,(int)randd,responseRect);
				cv::Point org(10,charSize-10);
				randomAngle = rng.uniform(-angle,angle);
				cv::putText(characterRect,dstr , org, 0, fontSize ,0, 10, 8,false);
				rotate(characterRect,randomAngle,charSize,charSize,scale);
				characterRect.copyTo((*image)(Rect(xPos,yPos,charSize,charSize)));
				responseRect.copyTo((*responses)(Rect(xPos,yPos,charSize,charSize)));

			}
		}
		charImages.randChars.push_back(image);
		charImages.responses.push_back(responses);
	}
	return charImages;
}*/

RandomCharactersImages createTestImagesAfont(int numOfImages, int numOfChars, int tileSizeX, int tileSizeY, int charDivX, int charDivY, int imageWidth, int imageHeight, string type,double angle, 
	string charType, bool useNoise)
{
	printf("Create test images OCR A-font....\n\n");
	RandomCharactersImages charImages;
	Mat* image, *responses;
	Mat characterRect, responseRect;
	int xPos, yPos, randd;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);
	double scale = 1.0;
	char d;
	string dstr;
	d = 'A';

	for(int im=0; im<numOfImages; im++)
	{
		RandomCharacters characters = produceDataFromAfont(numOfChars,charType,0,tileSizeX, tileSizeY,charDivX,charDivY,angle,false,useNoise);
		image = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		responses = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		add(*image,255,*image);
		if(useNoise)
				randn(*image,200,30);

		for(int c=0; c<numOfChars; c++)
		{
			xPos = rng.uniform(0,imageWidth-tileSizeX);
			yPos = rng.uniform(0,imageHeight-tileSizeY);

			if(!sum((*responses)(Rect(xPos,yPos,tileSizeX,tileSizeY)))(0))
			{
				randd = rng.uniform(0,(int)characters.randChars.size()-1);
				characterRect = *characters.randChars[randd];
				responseRect = Mat::zeros(tileSizeY,tileSizeX,CV_8UC1);
				add(responseRect,(int)characters.responses.at<int>(randd,0),responseRect);
				characterRect.copyTo((*image)(Rect(xPos,yPos,tileSizeX,tileSizeY)));
				responseRect.copyTo((*responses)(Rect(xPos,yPos,tileSizeX,tileSizeY)));
			}
		}
		charImages.randChars.push_back(image);
		charImages.responses.push_back(responses);
		//imshow("sldf",*image);
		//waitKey();
	}
	return charImages;
}

Mat createResponses(int trainingNum, int characters)
{
	int res = 0;
	Mat responses = Mat::zeros(trainingNum*characters,1,CV_32SC1);
	for(int i = 0; i< characters; i++)
	{
		for(int j=0; j<trainingNum; j++)
			responses.at<int>(i*trainingNum+j,0) = res;
		res++;
	}
	return responses;
}



void evaluateIm(vector<CvRTrees*> forestVector, int testNum, int tileSizeX, int tileSizeY,string type, string featureType,int charDivX, int charDivY,int fontSize,
	string charType, int numOfPoints, double angle, int numOfTrees, double threshold, bool falseClass, bool useNoise)
{
	RandomCharacters testData = produceDataFromAfont(testNum, charType,0,tileSizeX, tileSizeY,charDivX,charDivY,angle,falseClass,useNoise);

	int numOfDataForEachClass;
	if(type == "digits")
		numOfDataForEachClass = 10;
	else if(type == "uppercase")
		numOfDataForEachClass = 26;
	else if(type == "lowercase")
		numOfDataForEachClass = 26;
	else if(type == "digitsAndLetters")
		numOfDataForEachClass = 36;
	else
	{
		printf("error");
		abort();
	}

	Mat testFeatures = calcFeaturesTraining(testData,numOfPoints,featureType,tileSizeX, tileSizeY,useNoise);
	int truePred = 0;
	double trueConf = 0;
	int numOfForests = (int)forestVector.size();
	CvForestTree* tree;
	Mat treePred;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	double minVal, maxVal;
	
	printf("Calculate predictions....\n");
	for(int im=0; im<testData.randChars.size(); im++)
	{
		imshow("im",*testData.randChars[im]);
		waitKey();
		treePred = Mat::zeros(256,1, CV_32SC1);
		for(int f=0; f<numOfForests; f++)
		{
			for(int t=0; t<(int)numOfTrees; t++)
			{
				tree = forestVector[f]->get_tree(t);
				treePred.at<int>(static_cast<int>(tree->predict(testFeatures.row(im))->value),0)++;
			}
		}
		cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);

		cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
		if(*maxIndx == testData.responses.at<int>(im,0))
		{
			truePred++;
			trueConf += maxVal/(numOfTrees*numOfForests);
		}
	}

	cout << "Number of test images: " << testNum*numOfDataForEachClass << endl;
	cout << "True detections :" << truePred << endl;
	cout << "Average amount of correct classifications: " << trueConf/(testNum*numOfDataForEachClass) << endl;
}


vector<Mat*> predictImages(RandomCharactersImages& randIms, vector<CvRTrees*> forestVector1, vector<CvRTrees*> forestVector2, int imNum, int imageWidth, int imageHeight, int charSizeX, int charSizeY, int overlap, 
	int numOfTrees,double desicionThres1, double desicionThres2, int numOfPointPairs1, int numOfPointPairs2, string charType, string featureType, int downSample,bool useNoise)
{
	printf("Detecting characters in images....\n\n");
	//CalcRectSample calcRect;
	DWORD start, stop;
	int xPos, yPos, predPosx, predPosy;
	int charSizeXUpSampled = charSizeX*downSample;
	int charSizeYUpSampled = charSizeY*downSample;
	int overlapTileX = charSizeX/overlap;
	int overlapTileY = charSizeY/overlap;

	Mat imRect,imRect8x8, integralRect; 
	Mat featureMat1;
	Mat featureMat2 = Mat::zeros(1,numOfPointPairs2,CV_32FC1);
	Scalar mean, std;
	int imRectUp,imRectDown, imRectMiddleHor, imRectMiddleVert, rectFiltNum, imRectSum, imRectLeft, imRectRight, imRectUpUp, imRectDownDown;
	//int imRectUp,imRectDown, imRectMiddleHor, imRectMiddleVert, rectFiltNum;
	vector<Mat*> predictions;
	Mat* pred;
	int tileNumX = imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeY*overlap - (overlap-1);

	if(featureType == "rects")
	{
		rectFiltNum = calcRectFiltNum(charSizeX,charSizeY)+1;
		featureMat1 = Mat::zeros(1,rectFiltNum,CV_32FC1);
	}
	else if(featureType == "points")
	{
		featureMat1 = Mat::zeros(1,numOfPointPairs1,CV_32FC1);
	}

	Mat treePred;
	CvForestTree* tree;
	double minVal, maxVal;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	int numOfForests = (int)forestVector1.size();

	Mat pointPairVector1 = Mat::zeros(numOfPointPairs1,4,CV_32SC1);
	Mat pointPairVector2 = Mat::zeros(numOfPointPairs2,4,CV_32SC1);

	if(featureType == "points")
	{
		cv::RNG rng(0);
		int distThreshold = 20;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPointPairs1; i++)
		{
			x1 = 0;
			y1 = 0;
			x2 = 0;
			y2 = 0;
			
			while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
			{
				x1 = rng.uniform(charSizeXUpSampled/4,charSizeXUpSampled*3/4);
				y1 = rng.uniform(charSizeYUpSampled/4,charSizeYUpSampled*3/4);
				x2 = rng.uniform(0,charSizeXUpSampled);
				y2 = rng.uniform(0,charSizeYUpSampled);
			}
			pointPairVector1.at<int>(i,0) = x1/downSample;
			pointPairVector1.at<int>(i,1) = y1/downSample;
			pointPairVector1.at<int>(i,2) = x2/downSample;
			pointPairVector1.at<int>(i,3) = y2/downSample;
		}
	}

	RNG rng(0);
	int distThreshold = 20;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPointPairs2; i++)
	{
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;

		while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
		{
			x1 = rng.uniform(charSizeX/4,charSizeX*3/4);
			y1 = rng.uniform(charSizeY/4,charSizeY*3/4);
			x2 = rng.uniform(0,charSizeX);
			y2 = rng.uniform(0,charSizeY);
		}

		pointPairVector2.at<int>(i,0) = x1/(charSizeX/8);
		pointPairVector2.at<int>(i,1) = y1/(charSizeY/8);
		pointPairVector2.at<int>(i,2) = x2/(charSizeX/8);
		pointPairVector2.at<int>(i,3) = y2/(charSizeY/8);
	}


	start = GetTickCount();
	for(int im=0; im<imNum; im++)
	{
		pred = new Mat(Mat::zeros(tileNumY,tileNumX, CV_8UC1));
		predPosx = 0;
		predPosy = 0;
		for(int y=0; y<tileNumY; y++)
		{
			yPos = y*overlapTileY;
			for(int x=0; x<tileNumX; x++)
			{
				xPos = x*overlapTileY;
				//int rectArea = charSizeXDownSampled*charSizeYDownSampled*255;
				imRect = (*randIms.randChars[im])(Rect(xPos,yPos,charSizeX,charSizeY));
				/*imRectUp = static_cast<int>(sum(imRect(Rect(0,0,charSizeXDownSampled,charSizeYDownSampled/4)))(0));
				imRectDown = static_cast<int>(sum(imRect(Rect(0,charSizeYDownSampled*3/4,charSizeXDownSampled,charSizeYDownSampled/4)))(0));
				imRectMiddleHor = static_cast<int>(sum(imRect(Rect(0,charSizeYDownSampled*7/16,charSizeXDownSampled,charSizeYDownSampled/8)))(0));
				imRectMiddleVert = static_cast<int>(sum(imRect(Rect(charSizeXDownSampled*7/16,0,charSizeXDownSampled/8, charSizeYDownSampled)))(0));
				*/
				resize(imRect,imRect8x8,Size(8,8));
				featureMat2 = Mat::zeros(1,numOfPointPairs2,CV_32FC1);
				calcPointPairsFeaturesTile(imRect8x8,featureMat2,pointPairVector2,numOfPointPairs2,0, false);

				treePred = Mat::zeros(2,1, CV_32SC1);

				for(int f=0; f<forestVector2.size(); f++)
				{
					for(int t=0; t<(int)numOfTrees; t++)
					{
						tree = forestVector2[f]->get_tree(t);
						treePred.at<int>(static_cast<int>(tree->predict(featureMat2)->value),0)++;
					}
				}
				
				cv::meanStdDev(imRect,mean,std);
				imRectSum = std.val[0];
				cv::meanStdDev(imRect(Rect(0,0,charSizeX,charSizeY/4)),mean,std);
				imRectUp = std.val[0];
				cv::meanStdDev(imRect(Rect(0,charSizeY*3/4,charSizeX,charSizeY/4)),mean,std);
				imRectDown = std.val[0];
				cv::meanStdDev(imRect(Rect(0,charSizeY*7/16,charSizeX,charSizeY/8)),mean,std);
				imRectMiddleHor = std.val[0];
				cv::meanStdDev(imRect(Rect(charSizeX*7/16,0,charSizeX/8, charSizeY)),mean,std);
				imRectMiddleVert = std.val[0];
				cv::meanStdDev(imRect(Rect(0,0,charSizeX/64,charSizeY)),mean,std);
				imRectLeft = std.val[0];
				cv::meanStdDev(imRect(Rect(charSizeY*59/64,0,charSizeX/64,charSizeY)),mean,std);
				imRectRight = std.val[0];
				cv::meanStdDev(imRect(Rect(0,0,charSizeX,charSizeY/64)),mean,std);
				imRectUpUp = std.val[0];
				cv::meanStdDev(imRect(Rect(0,charSizeY*59/64,charSizeX,charSizeY/64)),mean,std);
				imRectDownDown = std.val[0];
				//if(imRectUp < rectArea/4 && imRectDown < rectArea/4 && imRectMiddleVert < rectArea/8 && imRectMiddleHor < rectArea/8)
				int stdThres = 40;
				//if(imRectSum > stdThres && imRectUp > stdThres && imRectDown > stdThres && imRectMiddleVert > stdThres && imRectMiddleHor > stdThres)
				if(treePred.at<int>(1,0) > desicionThres2*numOfTrees)
				{
					//imshow("sfsdf",imRect);
					//waitKey();
					if(featureType == "rects")
						calcRectFeatureTile(imRect,featureMat1,charSizeX,charSizeY,0);
					else if(featureType == "points")
					{
						featureMat1 = Mat::zeros(1,numOfPointPairs1,CV_32FC1);
						calcPointPairsFeaturesTile(imRect,featureMat1,pointPairVector1,numOfPointPairs1,0,useNoise);
					}
					else
						abort();

					//Loop over all trees

					treePred = Mat::zeros(256,1, CV_32SC1);

					for(int f=0; f<numOfForests; f++)
					{
						for(int t=0; t<(int)numOfTrees; t++)
						{
							tree = forestVector1[f]->get_tree(t);
							treePred.at<int>(static_cast<int>(tree->predict(featureMat1)->value),0)++;
						}
					}
					cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);
					//cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
					if(maxVal > numOfTrees*numOfForests*desicionThres1)
						pred->at<uchar>(predPosy,predPosx) = *maxIndx;
						
				}
				predPosx++;

			}
			//cout << y << endl;
			predPosx = 0;
			predPosy++;
		}
		predictions.push_back(pred);
	}
	stop = GetTickCount();
	std::cout << "Average time per image: " << (float)(stop - start)/((float)imNum)/1000 << std::endl << std::endl;
	return predictions;
}


void evaluateResult(vector<Mat*> predictions, RandomCharactersImages& randIms,  int imageWidth, int imageHeight, int charSizeX, int charSizeY, int numOfImages, int overlap, 
	 int minCluster, int pixelConnectionThres)
{
	printf("Visulize result....\n\n");
	int overlapTileX = charSizeX/overlap;
	int overlapTileY = charSizeY/overlap;
	int tileNumX = imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeY*overlap - (overlap-1);
	char p;
	int numOfTrue, numOfFalse;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	Mat characterRect, responseRect;
	Mat visulizeClusters = Mat::zeros(imageHeight, imageWidth,CV_8UC3);
	add(visulizeClusters, 255, visulizeClusters);
	Mat visulizePred = Mat::zeros(imageHeight,imageWidth,CV_8UC3);
	add(visulizePred,255,visulizePred);
	//histogram parameters
	/*MatND hist;
	int channels[2];
	int	histSize[1];
	float range[2];
	channels[0] = 0; channels[1] = 1; 
	histSize[0] = 256;
	range[0] = 0; range[1] = 256; //charSize*charSize/(overlap*overlap);
	const float* ranges[] = {range};
	double minVal, maxVal;
	int minIndx, maxIndx;
	vector<Mat> mergeIm;*/
	int maxIndx;

	for(int i=0; i<numOfImages; i++)
	{
		numOfTrue = 0;
		numOfFalse = 0;

		for(int x=0; x<tileNumX; x++)
		{
			for(int y=0; y<tileNumY; y++)
			{
				//cout << x << endl;
				responseRect = (*randIms.responses[i])(Rect(x*overlapTileX,y*overlapTileY,charSizeX,charSizeY));

				maxIndx = calcMaxIndex(responseRect,256);
				//calcHist(&responseRect,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);
				//cv::minMaxIdx(hist,&minVal,&maxVal,&minIndx,&maxIndx);

				if(predictions[i]->at<uchar>(y,x))
				{
					p = predictions[i]->at<uchar>(y,x);
					charStr = p;
					characterRect = Mat::zeros(overlapTileY,overlapTileX,CV_8UC3);
					//add(characterRect,255,characterRect);
					cv::Point org(overlapTileX/12,overlapTileY-overlapTileY/12);

					if(p == (char)maxIndx)
					{
						cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(0,255,0), 1, 8,false);
						numOfTrue++;
					}
					else if(p == 0)
					{
						characterRect.setTo(Scalar(0,0,255));
					}
					else
					{
						add(characterRect,255,characterRect);
						cv::putText(characterRect,charStr , org, 0, charSizeR ,CV_RGB(255,0,0), 1, 8,false);
						numOfFalse++;
					}

					characterRect.copyTo(visulizePred(Rect(x*overlapTileX,y*overlapTileY,overlapTileX,overlapTileY)));
				}
				else if(maxIndx != 0)
				{
					characterRect = Mat::zeros(overlapTileY,overlapTileX,CV_8UC3);
					characterRect.copyTo(visulizePred(Rect(x*overlapTileX,y*overlapTileY,overlapTileX,overlapTileY)));
				}
			}
		}
		calcClusters(*predictions[i],visulizeClusters, *randIms.responses[i],pixelConnectionThres,imageWidth,imageHeight,charSizeX,charSizeX/40,minCluster,overlapTileX,overlapTileY);

		cout << "Number of true detections: " << numOfTrue << endl;
		cout << "Number of false detections: " << numOfFalse << endl;
		imshow("image", *randIms.randChars[i]);
		imshow("visulize predictions",visulizePred);
		imshow("visulize clusters",visulizeClusters);
		waitKey();
	}
}

void evaluateBackground(Mat& image, vector<CvRTrees*> forestVector1, int tileSizeX, int tileSizeY, int numOfPointPairs1, double desicionThres2, int overlap )
{
	printf("Visulize result....\n\n");
	Mat imRect, imRect8x8, imageThresholded;
	Mat imageCopy = image.clone();
	int imageHeight = image.rows;
	int imageWidth = image.cols;
	int xPos = 0;
	int	yPos = 0;
	int	predPosx = 0;
	int	predPosy = 0;
	Mat pointPairVector1 = Mat::zeros(numOfPointPairs1,4,CV_32SC1);
	Mat featureMat1 = Mat::zeros(1,numOfPointPairs1,CV_32FC1);
	Mat treePred;
	CvForestTree* tree;
	int tileNumX = imageWidth/(tileSizeX/overlap) - 2;
	int tileNumY = imageHeight/(tileSizeY/overlap) - 2;
	int overlapTileX = tileSizeX/overlap;
	int overlapTileY = tileSizeY/overlap;
	cv::adaptiveThreshold(image,imageThresholded,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,tileSizeX+1,20);
	
	imshow("ljglj",imageThresholded);
	waitKey();
	RNG rng(0);
	int distThreshold = 10;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPointPairs1; i++)
	{
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;

		while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
		{
			x1 = rng.uniform(tileSizeX/4,tileSizeX*3/4);
			y1 = rng.uniform(tileSizeY/4,tileSizeY*3/4);
			x2 = rng.uniform(0,tileSizeX);
			y2 = rng.uniform(0,tileSizeY);
		}

		pointPairVector1.at<int>(i,0) = x1/(tileSizeX/8);
		pointPairVector1.at<int>(i,1) = y1/(tileSizeY/8);
		pointPairVector1.at<int>(i,2) = x2/(tileSizeX/8);
		pointPairVector1.at<int>(i,3) = y2/(tileSizeY/8);
	}


		while(yPos < imageHeight - tileSizeY)
		{
			while(xPos < imageWidth - tileSizeX)
			{
				/*imRect = image(Rect(xPos,yPos,tileSizeX,tileSizeY));
				resize(imRect,imRect8x8,Size(8,8));
				featureMat1 = Mat::zeros(1,numOfPointPairs1,CV_32FC1);
				calcPointPairsFeaturesTile(imRect8x8,featureMat1,pointPairVector1,numOfPointPairs1,0, false);
				*/
				int reSizeTo = 8;
				imRect = imageThresholded(Rect(xPos,yPos,tileSizeX,tileSizeY));
				featureMat1 = Mat::zeros(1,reSizeTo*reSizeTo,CV_32FC1);
				calcStdTile(imRect,featureMat1,0,reSizeTo);
				treePred = Mat::zeros(2,1, CV_32SC1);

				for(int f=0; f<forestVector1.size(); f++)
				{
					for(int t=0; t< forestVector1[0]->get_tree_count(); t++)
					{
						tree = forestVector1[f]->get_tree(t);
						treePred.at<int>(static_cast<int>(tree->predict(featureMat1)->value),0)++;
					}
				}
				if(treePred.at<int>(1,0) > desicionThres2*forestVector1[0]->get_tree_count())
				{
					//imshow("Evaluate background",imRect);
					//waitKey();
					rectangle(imageCopy,Point(xPos,yPos),Point(xPos+tileSizeX,yPos+tileSizeY),Scalar(0,0,0));
				}
				predPosx++;
				xPos += overlapTileX;
			}
			xPos = 0;
			yPos += overlapTileY;
			predPosx = 0;
			predPosy++;
		}
		imshow("Evaluate background",imageCopy);
		waitKey();
}


void calcTreeForPlot(CvRTrees* forest, int numOfPoints,int tileSizeX, int tileSizeY, int charDivX, int charDivY, bool useNoise, int maxDepth)
{

	RandomCharacters testData = produceDataFromAfont(1,"digitsAndLetters",0, tileSizeX, tileSizeY, charDivX,charDivY,0,false,useNoise);
	Mat pointPairVector = Mat::zeros(numOfPoints,4,CV_32SC1);
	Mat* character;
	vector<vector<int>*> pred;
	cv::RNG rng(0);
	int distThreshold = 10;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPoints; i++)
	{
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;

		while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
		{
			x1 = rng.uniform(tileSizeX/4,tileSizeX*3/4);
			y1 = rng.uniform(tileSizeY/4,tileSizeY*3/4);
			x2 = rng.uniform(0,tileSizeX);
			y2 = rng.uniform(0,tileSizeY);
		}

		pointPairVector.at<int>(i,0) = x1;
		pointPairVector.at<int>(i,1) = y1;
		pointPairVector.at<int>(i,2) = x2;
		pointPairVector.at<int>(i,3) = y2;
	}

	CvForestTree* tree = forest->get_tree(0);
	CvDTreeSplit* split; 

	for(int im = 0; im < testData.randChars.size(); im++)
	{
		pred.push_back(new vector<int>);
		character = testData.randChars[im];
		const CvDTreeNode* treeNode = tree->get_root();


		for(int j=0; j<maxDepth+1; j++)
		{

			if(treeNode->left)
			{
				split = treeNode->split;
				x1 = pointPairVector.at<int>(split->var_idx,0);
				y1 = pointPairVector.at<int>(split->var_idx,1);
				x2 = pointPairVector.at<int>(split->var_idx,2);
				y2 = pointPairVector.at<int>(split->var_idx,3);

				if(character->at<uchar>(y1,x1) >= character->at<uchar>(y2,x2))
				{
					//cout << 1;
					pred[im]->push_back(1);
					treeNode = treeNode->right;
				}
				else if(character->at<uchar>(y1,x1) < character->at<uchar>(y2,x2))
				{
					//cout << -1;
					pred[im]->push_back(-1);
					treeNode = treeNode->left;
				}
			}
			else
				pred[im]->push_back(0);
		}

	}
	writeMatToFile(pred,"testTree.txt");
}
