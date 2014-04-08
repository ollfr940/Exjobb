#include "stdafx.h"
#include "functions.h"
#include "helperFunctions.h"
using namespace std;
using namespace cv;


RandomCharacters produceData(int numOfChars, int charSize, string type,double angle, int charDiv, int charOrg, double fontSize, int numOfClasses, bool falseClass)
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
			cv::imshow("im", *image);
			cv::waitKey();
		}
	}


	for(int i=0; i<numOfClasses; i++)
	{
		for(int c=0; c<numOfChars; c++)
		{

			chars.responses.at<int>(i*numOfChars+c,0) = (int)d;

			image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
			cv::add(*image,255,*image);
			cv::Point org;
			org.x = rng.uniform(charOrg-charDiv,charOrg+charDiv);
			org.y = rng.uniform(charSize-charOrg-charDiv, charSize-charOrg+charDiv);
			dstr = d;
			randomAngle = rng.uniform(-angle,angle);
			cv::putText(*image,dstr , org, 0, fontSize ,0, 10, 8,false);
			rotate(*image,randomAngle,charSize,charSize,scale);
			chars.randChars.push_back(image);
			//cv::imshow("im", *image);
			//cv::waitKey();
		}
		d++;
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


void evaluateIm(vector<CvRTrees*> forestVector, int testNum, int imageSize,string type, string featureType,int charDiv, int charOrg,int fontSize, int numOfClasses, int numOfPoints, double angle, int numOfTrees, double threshold, bool falseClass)
{
	RandomCharacters testData = produceData(testNum,imageSize,type,angle,charDiv,charOrg,fontSize,numOfClasses, falseClass);
	Mat testFeatures = calcFeaturesTraining(testData,numOfPoints,featureType);
	int truePred = 0;
	double trueConf = 0;
	int numOfForests = (int)forestVector.size();
	CvForestTree* tree;
	Mat* pred, treePred;
	int minIndx[2] = {0,0};
	int maxIndx[2] = {0,0};
	double minVal, maxVal;

	/*Mat pointPairVector = Mat::zeros(numOfPoints,4,CV_32SC1);
	cv::RNG rng(0);
	int distThreshold = 20;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPoints; i++)
	{ 
		x1 = 0;
		y1 = 0;
		x2 = 0;
		y2 = 0;

		while(abs(x1-x2) < distThreshold && abs(y1-y2) < distThreshold)
		{
			x1 = rng.uniform(imageSize/4,imageSize*3/4);
			y1 = rng.uniform(imageSize/4,imageSize*3/4);
			x2 = rng.uniform(0,imageSize);
			y2 = rng.uniform(0,imageSize);
		}
		pointPairVector.at<int>(i,0) = x1;
		pointPairVector.at<int>(i,1) = y1;
		pointPairVector.at<int>(i,2) = x2;
		pointPairVector.at<int>(i,3) = y2;
	}
	*/
	

	printf("Calculate predictions....\n");
	for(int im=0; im<testData.randChars.size(); im++)
	{
		//Mat testFeatures = Mat::zeros(1,numOfPoints,CV_32FC1);
		//calcPointPairsFeaturesTile(*testData.randChars[im],testFeatures,pointPairVector,numOfPoints,0);
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
		//if(maxVal > numOfTrees*numOfForests*threshold)
			//truePred++;
	}

	cout << "Number of test images: " << testNum*numOfClasses << endl;
	cout << "True detections :" << truePred << endl;
	cout << "Average amount of correct classifications: " << trueConf/(testNum*numOfClasses) << endl;
}

vector<Mat*> predictImages(RandomCharactersImages& randIms, vector<CvRTrees*> forestVector,int imNum, int imageWidth, int imageHeight, int charSizeX, int charSizeY, int overlap, int numOfTrees,double desicionThres, int numOfPointPairs, string charType, string featureType)
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
			pointPairVector.at<int>(i,0) = x1;
			pointPairVector.at<int>(i,1) = y1;
			pointPairVector.at<int>(i,2) = x2;
			pointPairVector.at<int>(i,3) = y2;
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
					imshow("sfsdf",imRect);
					waitKey();
					/*int indx = 0; 
					for(int rectx=0; rectx <8; rectx++)
					{
						for(int recty=0; recty <8; recty++)
						{
							int rectSizex = 12 + rectx*4;
							int rectSizey = 12 + recty*4;
							int rectNumx = charSizeX/rectSizex;
							int rectNumy = charSizeY/rectSizey;

							for(int i=0; i<rectNumx; i++)
							{
								for(int j=0; j<rectNumy; j++)
								{
									integral(imRect(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect,CV_32FC1);
									calcRect.operator()(integralRect,featureMat,indx,rectSizex,rectSizey,0);
								}
							}
						}
					}*/
					if(featureType == "rects")
						calcRectFeatureTile(imRect,featureMat,charSizeX,charSizeY,0);
					else if(featureType == "points")
					{
						featureMat = Mat::zeros(1,numOfPointPairs,CV_32FC1);
						calcPointPairsFeaturesTile(imRect,featureMat,pointPairVector,numOfPointPairs,0);
					}
					else
						abort();

					//pred->at<uchar>(predPosx,predPosy) = forest.predict(featureMat);


					// Use proxData
					/*
					prox = forest.predict(featureMat);
					if(type == "uppercase")
					sample1 = proxDataFeatures.row(prox-65);
					else if(type == "lowercase")
					sample1 = proxDataFeatures.row(prox-97);
					else if(type == "numbers")
					sample1 = proxDataFeatures.row(prox-48);
					sample2 = featureMat;

					if(forest.get_proximity(&sample1,&sample2) > desicionThres)
					pred->at<uchar>(predPosx,predPosy) = prox;

					*/

					/*
					proxOld = 0;
					for(int p=0; p<10; p++)
					{
					proxIndx = '0';
					sample1 = proxDataFeatures.row(p);
					sample2 = featureMat;
					proxNew = forest.get_proximity(&sample1,&sample2);
					if(proxNew > proxOld)
					{
					proxOld = proxNew;
					proxIndx += p;
					}

					}

					if(proxOld > desicionThres)
					pred->at<uchar>(predPosx,predPosy) = proxIndx;

					*/
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
					cout << (char)(*maxIndx) << "\t" << maxVal/(numOfForests*numOfTrees) << endl;
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
}


void evaluateResult(vector<Mat*> predictions, RandomCharactersImages& randIms,  int imageWidth, int imageHeight, int charSizeX, int charSizeY, int numOfImages, int overlap, int upSample)
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
}
