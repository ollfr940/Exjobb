#include "stdafx.h"
#include "functions.h"
#include "helperFunctions.h"
using namespace std;
using namespace cv;


RandomCharacters produceData(int numOfChars, int charSize, string type,double angle, int charDivX, int charDivY, int charOrg, double fontSize, int numOfClasses, bool falseClass, bool useNoise)
{
	printf("Produce data....\n\n");
	RandomCharacters chars;
	if(falseClass)
		chars.responses = Mat::zeros(numOfChars*(numOfClasses+1),1,CV_32SC1);
	else
		chars.responses = Mat::zeros(numOfChars*numOfClasses,1,CV_32SC1);

	Mat* image;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);
	double randomAngle, scale = 1.0;
	char d, dd, randd;
	string dstr;

	if(type == "numbers")
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

	dd = d;

	for(int i=0; i<numOfClasses; i++)
	{
		for(int c=0; c<numOfChars; c++)
		{

			chars.responses.at<int>(i*numOfChars+c,0) = (int)dd;

			image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
			cv::add(*image,255,*image);
			if(useNoise)
				randn(*image,50,30);
			cv::Point org;
			
			/*while(sum(image->col(1))(0) != charSize*255 ||  sum(image->col(charSize-2))(0) != charSize*255 || sum(image->row(1))(0) != charSize*255 || sum(image->row(charSize-2))(0) != charSize*255)
			{
				*image = Mat::zeros(charSize,charSize,CV_8UC1);
				cv::add(*image,255,*image);
				org.x = rng.uniform(0,charSize-10);
				org.y = rng.uniform(charSize/2,charSize);
				dstr = dd;
				randomAngle = rng.uniform(-angle,angle);
				cv::putText(*image,dstr , org, 0, fontSize ,0, 10, 8,false);
				rotate(*image,randomAngle,charSize,charSize,scale);
				//cv::imshow("im", *image);
				//cv::waitKey();
			}*/
			
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
			cv::imshow("im", *image);
			cv::waitKey();
		}
		dd++;
	}
	
	if(falseClass)
	{
		char d1, d2;
		Point org1, org2;
		string dstr1, dstr2;

		for(int i=0; i<numOfChars; i++)
		{
			chars.responses.at<int>(numOfClasses*numOfChars+i,0) = 0;
			image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
			cv::add(*image,255,*image);	
			if(useNoise)
				randn(*image,50,30);

			d1 = rng.uniform(d,d+numOfClasses);
			d2 = rng.uniform(d,d+numOfClasses);
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
	chars.responses = Mat::zeros(boxVector.size()*numOfCharacters,1,CV_32SC1);

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

RandomCharacters produceDataFromAfont(int numOfChars, int numOfClasses, int numOfFalseData, int charDivX, int charDivY, double angle, bool falseClass, float downSample, bool useNoise)
{
	printf("Produce data from OCR A-font....\n\n");
	uint64 initValue = time(0);
	RNG rng(initValue);
	RandomCharacters chars;

	if(falseClass)
		chars.responses = Mat::zeros(numOfFalseData+numOfClasses*numOfChars,1,CV_32SC1);
	else
		chars.responses = Mat::zeros(numOfClasses*numOfChars,1,CV_32SC1);

	Mat* imRect, *imRectDownSampled,imCopy, imCopy1, imCopy2, imCopyTrans;
	int x, y, x1, y1, x2, y2;
	int	width = 128;
	int	height = 128;
	double randomAngle, scale = 1.0;
	char d = 'A';
	char d1, d2;
	string dstr, dstr1, dstr2, *name, *name1, *name2;

	for(int i=0; i<numOfClasses; i++)
	{
		dstr = d;
		name = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr+".png");
		imCopy = imread(name->c_str(),CV_LOAD_IMAGE_GRAYSCALE);

		for(int j=0; j<numOfChars; j++)
		{
			imRect = new Mat(Mat::zeros(height,width,CV_8UC1));
			imRectDownSampled = new Mat(Mat::zeros(height/downSample,width/downSample,CV_8UC1));
			add(*imRectDownSampled,255,*imRectDownSampled);
			add(*imRect,255,*imRect);

			x = rng.uniform(-10-charDivX,-10+charDivX);
			y = rng.uniform(10-charDivY, 10+charDivY);

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
			if(useNoise)
				randn(*imRect,200,30);
			resize(*imRect,*imRectDownSampled,Size(width/downSample,height/downSample));
			chars.randChars.push_back(imRectDownSampled);
			chars.responses.at<int>(i*numOfChars+j,0) = (int)d;
			cv::imshow("im", *imRectDownSampled);
			cv::waitKey();
		}
		d++;
	}

	if(falseClass)
	{
		for(int i=0; i<numOfFalseData; i++)
		{
			d1 = d + rng.uniform(-numOfClasses,0);
			d2 = d + rng.uniform(-numOfClasses,0);
			dstr1 = d1;
			dstr2 = d2;
			name1 = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr1+".png");
			name2 = new string("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\" + dstr2+".png");
			imCopy1 = imread(name1->c_str(),CV_LOAD_IMAGE_GRAYSCALE);
			imCopy2 = imread(name2->c_str(),CV_LOAD_IMAGE_GRAYSCALE);
			imRect = new Mat(Mat::zeros(height,width,CV_8UC1));
			imRectDownSampled = new Mat(Mat::zeros(height/downSample,width/downSample,CV_8UC1));
			add(*imRectDownSampled,255,*imRectDownSampled);
			if(useNoise)
				randn(*imRectDownSampled,50,30);
			add(*imRect,255,*imRect);
			x1 = 80;
			y1 = 10;
			x2 = -80;
			y2 = 10;

			imCopy1(Rect(x1,y1, width-x1, height-y1)).copyTo((*imRect)(Rect(0,0,width-x1, height-y1))); 
			imCopy2(Rect(0,y2, width+x2, height-y2)).copyTo((*imRect)(Rect(-x2,0,width+x2, height-y2))); 

			randomAngle = rng.uniform(-angle,angle);
			rotate(*imRect,randomAngle,width, height, scale);
			resize(*imRect,*imRectDownSampled,Size(width/downSample,height/downSample));
			chars.randChars.push_back(imRectDownSampled);
			chars.responses.at<int>(numOfClasses*numOfChars+i,0) = 0;
			//cv::imshow("im", *imRectDownSampled);
			//cv::waitKey();
		}
	}
	return chars;
}

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

	if(type == "numbers")
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
}

RandomCharactersImages createTestImagesAfont(int numOfImages, int numOfChars, int charSize, int charDivX, int charDivY, int imageWidth, int imageHeight, string type,double angle, double fontSize, 
	int numOfClasses, float downSample, bool useNoise)
{
	printf("Create test images OCR A-font....\n\n");
	RandomCharactersImages charImages;
	Mat* image, *responses;
	Mat characterRect, responseRect;
	int xPos, yPos, randd;
	int charSizeDownSampled = charSize/downSample;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);
	double randomAngle, scale = 1.0;
	char d;
	string dstr;
	d = 'A';

	for(int im=0; im<numOfImages; im++)
	{
		RandomCharacters characters = produceDataFromAfont(numOfChars,numOfClasses,0,charDivX,charDivY,angle,false, downSample,useNoise);
		image = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		responses = new Mat(Mat::zeros(imageWidth,imageHeight,CV_8UC1));
		add(*image,255,*image);

		for(int c=0; c<numOfChars; c++)
		{
			xPos = rng.uniform(0,imageWidth-charSizeDownSampled);
			yPos = rng.uniform(0,imageHeight-charSizeDownSampled);

			if(!sum((*responses)(Rect(xPos,yPos,charSizeDownSampled,charSizeDownSampled)))(0))
			{
				randd = rng.uniform(0,characters.randChars.size()-1);
				characterRect = *characters.randChars[randd];
				responseRect = Mat::zeros(charSizeDownSampled,charSizeDownSampled,CV_8UC1);
				add(responseRect,(int)characters.responses.at<int>(randd,0),responseRect);
				characterRect.copyTo((*image)(Rect(xPos,yPos,charSizeDownSampled,charSizeDownSampled)));
				responseRect.copyTo((*responses)(Rect(xPos,yPos,charSizeDownSampled,charSizeDownSampled)));
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

/*
Mat createRectFeatures(RandomCharacters trainingData)
{
	printf("Create rect features....\n\n");
	CalcRectSample calcRect;
	Mat integralIm, integralRect;
	float p;
	int width = trainingData.randChars[0]->size().width;
	int height = trainingData.randChars[0]->size().height;
	int rectFiltNum = calcRectFiltNum(width,height)+1;
	int numOfCharacters = (int)trainingData.randChars.size();
	Mat featureMat = Mat::zeros(numOfCharacters,rectFiltNum,CV_32FC1);
	for(int im=0; im<numOfCharacters; im++)
	{
		width = trainingData.randChars[im]->size().width;
		height = trainingData.randChars[im]->size().height;

		int indx = 0; 
		for(int rectx=0; rectx <8; rectx++)
		{
			for(int recty=0; recty <8; recty++)
			{
				int rectSizex = 12 + rectx*4;
				int rectSizey = 12 + recty*4;
				int rectNumx = width/rectSizex;
				int rectNumy = height/rectSizey;

				for(int i=0; i<rectNumx; i++)
				{
					for(int j=0; j<rectNumy; j++)
					{
						integral((*trainingData.randChars[im])(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect,CV_32FC1);
						calcRect.operator()(integralRect,featureMat,indx,rectSizex,rectSizey,im);
					}
				}
			}
		}

	}
	return featureMat;
}*/


void evaluateIm(vector<CvRTrees*> forestVector, int testNum, int imageSize,string type, string featureType,int charDivX, int charDivY, int charOrg,int fontSize,
	int numOfClasses, int numOfPoints, double angle, int numOfTrees, double threshold, bool falseClass, bool useAfont, float downSample, bool useNoise)
{
	RandomCharacters testData;
	if(useAfont)
		testData = produceDataFromAfont(testNum,numOfClasses,0,charDivX,charDivY,angle,falseClass, downSample,useNoise);
	else
		testData = produceData(testNum,imageSize,type,angle,charDivX,charDivY,charOrg,fontSize,numOfClasses, falseClass,useNoise);

	Mat testFeatures = calcFeaturesTraining(testData,numOfPoints,featureType,downSample,useNoise);
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
		//imshow("im",*testData.randChars[im]);
		//waitKey();
		treePred = Mat::zeros(256,1, CV_32SC1);
		for(int f=0; f<numOfForests; f++)
		{
			for(int t=0; t<(int)numOfTrees; t++)
			{
				tree = forestVector[f]->get_tree(t);
				treePred.at<int>(tree->predict(testFeatures.row(im))->value,0)++;
			}
		}
		cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);

		//cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
		if(*maxIndx == testData.responses.at<int>(im,0))
		{
			truePred++;
			trueConf += maxVal/(numOfTrees*numOfForests);
		}
	}

	cout << "Number of test images: " << testNum*numOfClasses << endl;
	cout << "True detections :" << truePred << endl;
	cout << "Average amount of correct classifications: " << trueConf/(testNum*numOfClasses) << endl;
}
/*
vector<Mat*> predictImages(RandomCharactersImages& randIms, vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, int charSizeY, int overlap, int numOfTrees,double desicionThres, int numOfPointPairs, string charType, string featureType, float downSample)
{
	printf("Detecting characters in images....\n\n");
	//CalcRectSample calcRect;
	DWORD start, stop;
	int xPos, yPos, predPosx, predPosy;
	Mat imRect, integralRect, featureMat;
	int imRectUp,imRectDown, imRectMiddleHor, imRectMiddleVert, imRectRight, imRectLeft, imRectSum, rectFiltNum;
	vector<Mat*> predictions;
	Mat* pred;
	char proxIndx;
	int tileNumX = imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeY*overlap - (overlap-1);

	if(featureType == "rects")
	{
		rectFiltNum = calcRectFiltNum(charSizeX,charSizeY)+1;
		featureMat = Mat::zeros(1,rectFiltNum,CV_32FC1);
	}
	else if(featureType == "points")
	{
		featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
	}

	Mat treePred;
	CvForestTree* tree;
	double minVal, maxVal;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	CvMat sample1, sample2;
	float proxOld, proxNew;
	char prox;
	int numOfForests = (int)forestVector.size();
	Mat pointPairVector = Mat::zeros(numOfPointPairs,4,CV_32SC1);

	if(featureType == "points")
	{
		cv::RNG rng(0);
		int distThreshold = 20;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPointPairs; i++)
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
			pointPairVector.at<int>(i,0) = x1/downSample;
			pointPairVector.at<int>(i,1) = y1/downSample;
			pointPairVector.at<int>(i,2) = x2/downSample;
			pointPairVector.at<int>(i,3) = y2/downSample;
		}
	}

	start = GetTickCount();
	for(int im=0; im<imNum; im++)
	{
		pred = new Mat(Mat::zeros(tileNumX,tileNumY, CV_8UC1));
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;
		while(yPos < imageHeight-charSizeY && predPosy < tileNumY)
		{
			while(xPos < imageWidth-charSizeX && predPosx < tileNumX)
			{
				int rectArea = charSizeX*charSizeY*255;
				imRect = (*randIms.randChars[im])(Rect(xPos,yPos,charSizeX,charSizeY));
				//imRectSum = sum(imRect)(0);
				imRectUp = sum(imRect(Rect(0,0,charSizeX,charSizeY/4)))(0);
				imRectDown = sum(imRect(Rect(0,charSizeY*3/4,charSizeX,charSizeY/4)))(0);
				imRectMiddleHor = sum(imRect(Rect(0,charSizeY*7/16,charSizeX,charSizeY/8)))(0);
				imRectMiddleVert = sum(imRect(Rect(charSizeX*7/16,0,charSizeX/8, charSizeY)))(0);
				//imRectLeft = sum(imRect(Rect(0,0,charSizeX*4/16, charSizeY)))(0);
				//imRectRight = sum(imRect(Rect(charSizeX*12/16,0,charSizeX*4/16, charSizeY)))(0);
				if(imRectUp < rectArea/4 && imRectDown < rectArea/4 && imRectMiddleVert < rectArea/8 && imRectMiddleHor < rectArea/8)
				{
					//imshow("sfsdf",imRect);
					//waitKey();
					if(featureType == "rects")
						calcRectFeatureTile(imRect,featureMat,charSizeX,charSizeY,0);
					else if(featureType == "points")
					{
						featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
						calcPointPairsFeaturesTile(imRect,featureMat,pointPairVector,numOfPointPairs,0);
					}
					else
						abort();

					//Loop over all trees

					treePred = Mat::zeros(256,1, CV_32SC1);

					for(int f=0; f<numOfForests; f++)
					{
						for(int t=0; t<(int)numOfTrees; t++)
						{
							tree = forestVector[f]->get_tree(t);
							treePred.at<int>(tree->predict(featureMat)->value,0)++;
						}
					}
					cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);
					//cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
					if(maxVal > numOfTrees*numOfForests*desicionThres)
						pred->at<uchar>(predPosx,predPosy) = *maxIndx;
						
					xPos += charSizeX/overlap;
					predPosx++;
					//yPos += charSize/overlap;
				}
				else
				{
					xPos += charSizeX/overlap; // charSize;
					predPosx++; // += overlap;
					//yPos += charSize;
				}
			}
			predPosx = 0;
			predPosy++;
			xPos = 0;
			yPos += charSizeY/overlap;
		}
		predictions.push_back(pred);
	}
	stop = GetTickCount();
	std::cout << "Average time per image: " << (float)(stop - start)/((float)imNum)/1000 << std::endl << std::endl; 
	return predictions;
}*/

vector<Mat*> predictImages(RandomCharactersImages& randIms, vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, int charSizeY, int overlap, int numOfTrees,double desicionThres, int numOfPointPairs, string charType, string featureType, float downSample,bool useNoise)
{
	printf("Detecting characters in images....\n\n");
	//CalcRectSample calcRect;
	DWORD start, stop;
	int xPos, yPos, predPosx, predPosy;
	int charSizeXDownSampled = charSizeX/downSample;
	int charSizeYDownSampled = charSizeY/downSample;
	int overlapTileX = charSizeXDownSampled/overlap;
	int overlapTileY = charSizeYDownSampled/overlap;

	Mat imRect, integralRect, featureMat;
	int imRectUp,imRectDown, imRectMiddleHor, imRectMiddleVert, rectFiltNum;
	vector<Mat*> predictions;
	Mat* pred;
	char proxIndx;
	int tileNumX = imageWidth/charSizeXDownSampled*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeYDownSampled*overlap - (overlap-1);

	if(featureType == "rects")
	{
		rectFiltNum = calcRectFiltNum(charSizeX,charSizeY)+1;
		featureMat = Mat::zeros(1,rectFiltNum,CV_32FC1);
	}
	else if(featureType == "points")
	{
		featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
	}

	Mat treePred;
	CvForestTree* tree;
	double minVal, maxVal;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	int numOfForests = (int)forestVector.size();
	Mat pointPairVector = Mat::zeros(numOfPointPairs,4,CV_32SC1);

	if(featureType == "points")
	{
		cv::RNG rng(0);
		int distThreshold = 20;
		int x1, x2, y1, y2;
		for(int i=0; i<numOfPointPairs; i++)
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
			pointPairVector.at<int>(i,0) = x1/downSample;
			pointPairVector.at<int>(i,1) = y1/downSample;
			pointPairVector.at<int>(i,2) = x2/downSample;
			pointPairVector.at<int>(i,3) = y2/downSample;
		}
	}

	start = GetTickCount();
	for(int im=0; im<imNum; im++)
	{
		pred = new Mat(Mat::zeros(tileNumX,tileNumY, CV_8UC1));
		predPosx = 0;
		predPosy = 0;
		for(int y=0; y<tileNumY; y++)
		{
			yPos = y*overlapTileY;
			for(int x=0; x<tileNumX; x++)
			{
				xPos = x*overlapTileY;
				int rectArea = charSizeXDownSampled*charSizeYDownSampled*255;
				imRect = (*randIms.randChars[im])(Rect(xPos,yPos,charSizeXDownSampled,charSizeYDownSampled));
				imRectUp = sum(imRect(Rect(0,0,charSizeXDownSampled,charSizeYDownSampled/4)))(0);
				imRectDown = sum(imRect(Rect(0,charSizeYDownSampled*3/4,charSizeXDownSampled,charSizeYDownSampled/4)))(0);
				imRectMiddleHor = sum(imRect(Rect(0,charSizeYDownSampled*7/16,charSizeXDownSampled,charSizeYDownSampled/8)))(0);
				imRectMiddleVert = sum(imRect(Rect(charSizeXDownSampled*7/16,0,charSizeXDownSampled/8, charSizeYDownSampled)))(0);

				if(imRectUp < rectArea/4 && imRectDown < rectArea/4 && imRectMiddleVert < rectArea/8 && imRectMiddleHor < rectArea/8)
				{
					//imshow("sfsdf",imRect);
					//waitKey();
					if(featureType == "rects")
						calcRectFeatureTile(imRect,featureMat,charSizeX,charSizeY,0);
					else if(featureType == "points")
					{
						featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
						calcPointPairsFeaturesTile(imRect,featureMat,pointPairVector,numOfPointPairs,0,useNoise);
					}
					else
						abort();

					//Loop over all trees

					treePred = Mat::zeros(256,1, CV_32SC1);

					for(int f=0; f<numOfForests; f++)
					{
						for(int t=0; t<(int)numOfTrees; t++)
						{
							tree = forestVector[f]->get_tree(t);
							treePred.at<int>(tree->predict(featureMat)->value,0)++;
						}
					}
					cv::minMaxIdx(treePred,&minVal,&maxVal,minIndx,maxIndx);
					//cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
					if(maxVal > numOfTrees*numOfForests*desicionThres)
						pred->at<uchar>(predPosx,predPosy) = *maxIndx;
						
				}
				predPosx++;

			}
			predPosx = 0;
			predPosy++;
		}
		predictions.push_back(pred);
	}
	stop = GetTickCount();
	std::cout << "Average time per image: " << (float)(stop - start)/((float)imNum)/1000 << std::endl << std::endl; 
	return predictions;
}



/*void evaluateResult(vector<Mat*> predictions, RandomCharactersImages& randIms,  int imageWidth, int imageHeight, int charSizeX, int charSizeY, int numOfImages, int overlap, float downSample)
{
	printf("Visulize result....\n\n");
	int overlapTileX = charSizeX/overlap;
	int overlapTileY = charSizeY/overlap;
	int tileNumX = imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeY*overlap - (overlap-1);
	char p;
	int numOfTrue, numOfFalse;
	char maxRes;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	Mat characterRect, responseRect;
	Mat visulizePred = Mat::zeros(imageHeight,imageWidth,CV_8UC3);
	add(visulizePred,255,visulizePred);
	//histogram parameters
	MatND hist;
	int channels[2];
	int	histSize[1];
	float range[2];
	channels[0] = 0; channels[1] = 1; 
	histSize[0] = 256;
	range[0] = 0; range[1] = 256; //charSize*charSize/(overlap*overlap);
	const float* ranges[] = {range};
	double minVal, maxVal;
	int minIndx, maxIndx;
	vector<Mat> mergeIm;

	for(int i=0; i<numOfImages; i++)
	{
		numOfTrue = 0;
		numOfFalse = 0;

		for(int x=0; x<tileNumX; x++)
		{
			for(int y=0; y<tileNumY; y++)
			{
				cout << x << endl;
				responseRect = (*randIms.responses[i])(Rect(x*overlapTileX,y*overlapTileY,charSizeX,charSizeY));
				calcHist(&responseRect,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);
				cv::minMaxIdx(hist,&minVal,&maxVal,&minIndx,&maxIndx);

				if(predictions[i]->at<uchar>(x,y))
				{
					p = predictions[i]->at<uchar>(x,y);
					charStr = p;
					characterRect = Mat::zeros(overlapTileY,overlapTileX,CV_8UC3);
					//add(characterRect,255,characterRect);
					cv::Point org(overlapTileX/12,overlapTileY-overlapTileY/12);

					if(p == (char)maxIndx)
					{
						cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(0,255,0), 1, 8,false);
						numOfTrue++;
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
		cout << "Number of true detections: " << numOfTrue << endl;
		cout << "Number of false detections: " << numOfFalse << endl;
		//Mat upSampledVis, upSampledIm;
		//resize(visulizePred,upSampledVis,Size(imageHeight*upSample,imageWidth*upSample));
		//resize(*randIms.randChars[i],upSampledIm,Size(imageHeight*upSample,imageWidth*upSample));
		imshow("image", *randIms.randChars[i]);
		imshow("visulize predictions",visulizePred);
		waitKey();
	}
}*/

void evaluateResult(vector<Mat*> predictions, RandomCharactersImages& randIms,  int imageWidth, int imageHeight, int charSizeX, int charSizeY, int numOfImages, int overlap, float downSample)
{
	printf("Visulize result....\n\n");
	int charSizeXDownSampled = charSizeX/downSample;
	int charSizeYDownSampled = charSizeY/downSample;
	int overlapTileX = charSizeXDownSampled/overlap;
	int overlapTileY = charSizeYDownSampled/overlap;
	int tileNumX = imageWidth/charSizeXDownSampled*overlap - (overlap-1);
	int tileNumY = imageHeight/charSizeYDownSampled*overlap - (overlap-1);
	char p;
	int numOfTrue, numOfFalse;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	Mat characterRect, responseRect;
	Mat visulizePred = Mat::zeros(imageHeight,imageWidth,CV_8UC3);
	add(visulizePred,255,visulizePred);
	//histogram parameters
	MatND hist;
	int channels[2];
	int	histSize[1];
	float range[2];
	channels[0] = 0; channels[1] = 1; 
	histSize[0] = 256;
	range[0] = 0; range[1] = 256; //charSize*charSize/(overlap*overlap);
	const float* ranges[] = {range};
	double minVal, maxVal;
	int minIndx, maxIndx;
	vector<Mat> mergeIm;

	for(int i=0; i<numOfImages; i++)
	{
		numOfTrue = 0;
		numOfFalse = 0;

		for(int x=0; x<tileNumX; x++)
		{
			for(int y=0; y<tileNumY; y++)
			{
				//cout << x << endl;
				responseRect = (*randIms.responses[i])(Rect(x*overlapTileX,y*overlapTileY,charSizeXDownSampled,charSizeYDownSampled));
				calcHist(&responseRect,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);
				cv::minMaxIdx(hist,&minVal,&maxVal,&minIndx,&maxIndx);

				if(predictions[i]->at<uchar>(x,y))
				{
					p = predictions[i]->at<uchar>(x,y);
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
		cout << "Number of true detections: " << numOfTrue << endl;
		cout << "Number of false detections: " << numOfFalse << endl;
		imshow("image", *randIms.randChars[i]);
		imshow("visulize predictions",visulizePred);
		waitKey();
	}
}