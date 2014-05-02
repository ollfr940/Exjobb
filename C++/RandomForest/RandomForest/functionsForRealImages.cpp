#include "stdafx.h"
#include "functionsForRealImages.h"
using namespace std;
using namespace cv;

vector<Mat*> predictRealImages(vector<Mat*> imageVector, vector<CvRTrees*> forestVector1, vector<CvRTrees*> forestVector2, int imNum, int imageWidth, int imageHeight, int tileSizeX, int tileSizeY, int overlap, 
	int numOfTrees,double desicionThres1,double desicionThres2, int numOfPointPairs1, int numOfPointPairs2, int numOfPointPairs3, string charType, string featureType, bool useNoise)
{
	printf("Detecting characters in images....\n\n");
	//CalcRectSample calcRect;
	DWORD start, stop;
	Scalar mean, std;
	int xPos, yPos, predPosx, predPosy, rectFiltNum;
	int overlapTileX = tileSizeX/overlap;
	int overlapTileY = tileSizeY/overlap;

	Mat imRect, imRectReSize, imRectThres, integralRect,thresholdedImage, imRectThresholded;
	Mat featureMat1;
	Mat featureMat2 = Mat::zeros(1,numOfPointPairs2,CV_32FC1);
	Mat featureMat3 = Mat::zeros(1,numOfPointPairs3,CV_32FC1);
	vector<Mat*> predictions;
	Mat* pred;
	int tileNumX = imageWidth/(tileSizeX/overlap) - 2;//imageWidth/charSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/(tileSizeY/overlap) - 2; //imageHeight/charSizeY*overlap -(overlap-1);

	if(featureType == "rects")
	{
		rectFiltNum = calcRectFiltNum(tileSizeX,tileSizeY)+1;
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
	Mat pointPairVector3 = Mat::zeros(numOfPointPairs3,4,CV_32SC1);

	if(featureType == "points")
	{
		cv::RNG rng(0);
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
			pointPairVector1.at<int>(i,0) = x1;
			pointPairVector1.at<int>(i,1) = y1;
			pointPairVector1.at<int>(i,2) = x2;
			pointPairVector1.at<int>(i,3) = y2;
		}
	}

	RNG rng(0);
	int distThreshold = 10;
	int x1, x2, y1, y2;
	for(int i=0; i<numOfPointPairs2; i++)
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

		pointPairVector2.at<int>(i,0) = x1/(tileSizeX/8);
		pointPairVector2.at<int>(i,1) = y1/(tileSizeY/8);
		pointPairVector2.at<int>(i,2) = x2/(tileSizeX/8);
		pointPairVector2.at<int>(i,3) = y2/(tileSizeY/8);
	}

	for(int i=0; i<numOfPointPairs3; i++)
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

		pointPairVector3.at<int>(i,0) = x1/(tileSizeX/16);
		pointPairVector3.at<int>(i,1) = y1/(tileSizeY/16);
		pointPairVector3.at<int>(i,2) = x2/(tileSizeX/16);
		pointPairVector3.at<int>(i,3) = y2/(tileSizeY/16);
	}

	start = GetTickCount();
	for(int im=0; im<imNum; im++)
	{
		//cv::Laplacian(*imageVector[im],laplacianImage,-1,5);
		pred = new Mat(Mat::zeros(tileNumY,tileNumX, CV_8UC1));
		cv::adaptiveThreshold(*imageVector[im],thresholdedImage,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,tileSizeX+1,20);
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;

		while(yPos < imageHeight - tileSizeY)
		{
			while(xPos < imageWidth - tileSizeX)
			{

				imRect = (*imageVector[im])(Rect(xPos,yPos,tileSizeX,tileSizeY));
				resize(imRect,imRectReSize,Size(8,8));
				featureMat2 = Mat::zeros(1,numOfPointPairs2,CV_32FC1);
				calcPointPairsFeaturesTile(imRectReSize,featureMat2,pointPairVector2,numOfPointPairs2,0, false);

				imRectThresholded = thresholdedImage(Rect(xPos,yPos,tileSizeX,tileSizeY));
				meanStdDev(imRectThresholded,mean,std);
				/*
				int reSizeTo = 8;
				featureMat2 = Mat::zeros(1,reSizeTo*reSizeTo,CV_32FC1);
				calcStdTile(imRectThresholded,featureMat2,0, reSizeTo);*/

				treePred = Mat::zeros(2,1, CV_32SC1);

			
				for(int t=0; t<forestVector2[0]->get_tree_count(); t++)
				{
					tree = forestVector2[0]->get_tree(t);
					treePred.at<int>(static_cast<int>(tree->predict(featureMat2)->value),0)++;
				}

				if(treePred.at<int>(1,0) > desicionThres2*forestVector2[0]->get_tree_count())
				{
					//imshow("sfsdf",imRectThresholded);
					//waitKey();

						if(featureType == "rects")
							calcRectFeatureTile(imRect,featureMat1,tileSizeX,tileSizeY,0);
						else if(featureType == "points")
						{
							featureMat1 = Mat::zeros(1,numOfPointPairs1,CV_32FC1);
							calcPointPairsFeaturesTile(imRectThresholded,featureMat1,pointPairVector1,numOfPointPairs1,0, useNoise);
						}
						else
							abort();

						//Loop over all trees

						treePred = Mat::zeros(256,1, CV_32SC1);

						for(int f=0; f<numOfForests; f++)
						{
							for(int t=0; t<forestVector1[f]->get_tree_count(); t++)
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
				xPos += overlapTileX;
			}
			xPos = 0;
			yPos += overlapTileY;
			predPosx = 0;
			predPosy++;
		}
		removeFalsePredictions(*pred);
		predictions.push_back(pred);
	}
	stop = GetTickCount();
	std::cout << "Average time per image: " << (float)(stop - start)/((float)imNum)/1000 << std::endl << std::endl; 
	return predictions;
}

void evaluateResultRealImage(vector<Mat*> predictions, vector<Mat*> imageVector,  int imageWidth, int imageHeight, int tileSizeX, int tileSizeY, int numOfImages, 
	int overlap, int minCluster, int pixelConnectionThres, bool useGroundTruth, int imageNum)
{
	printf("Visulize result....\n\n");
	int overlapTileX = tileSizeX/overlap;
	int overlapTileY = tileSizeY/overlap;
	int tileNumX = imageWidth/tileSizeX*overlap - (overlap-1);
	int tileNumY = imageHeight/tileSizeY*overlap - (overlap-1);
	char p;
	int xPos, yPos, predPosx, predPosy;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	Mat characterRect, *groundTruth;
	Mat visulizeClusters = Mat::zeros(imageHeight, imageWidth,CV_8UC3);
	add(visulizeClusters,255,visulizeClusters);
	Mat visulizePred = Mat::zeros(imageHeight,imageWidth,CV_8UC3);
	add(visulizePred,255,visulizePred);

	for(int i=0; i<numOfImages; i++)
	{
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;

		while(yPos < imageHeight - tileSizeY)
		{
			while(xPos < imageWidth - tileSizeX)
			{
				//cout << x << endl;

				if(predictions[i]->at<uchar>(predPosy,predPosx))
				{
					p = predictions[i]->at<uchar>(predPosy,predPosx);
					charStr = p;
					characterRect = Mat::zeros(overlapTileY,overlapTileX,CV_8UC3);
					add(characterRect,255,characterRect);
					cv::Point org(overlapTileX/12,overlapTileY-overlapTileY/12);

					if(useGroundTruth)
					{
						groundTruth =  new Mat(imread(getImageAndGroundTruthName(imageNum,p,true).c_str(),CV_LOAD_IMAGE_GRAYSCALE));

						if(groundTruth->empty())
							cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(255,0,0), 1, 8,false);
						else
						{
							resize(*groundTruth,*groundTruth,Size(imageWidth,imageHeight));
							if(sum((*groundTruth)(Rect(xPos,yPos,tileSizeX,tileSizeY)))(0))
								cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(0,255,0), 1, 8,false);
							else
								cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(255,0,0), 1, 8,false);
						}
						delete groundTruth;

					}
					else
						cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(255,0,0), 1, 8,false);

					characterRect.copyTo(visulizePred(Rect(xPos,yPos,overlapTileX,overlapTileY)));
				}

				predPosx++;
				xPos += overlapTileX;
			}
			xPos = 0;
			yPos += overlapTileY;
			predPosx = 0;
			predPosy++;
		}
		calcClustersRealImage(*predictions[i],visulizeClusters,pixelConnectionThres,imageWidth,imageHeight,tileSizeX,tileSizeY,tileSizeX/40,minCluster,overlap,useGroundTruth,imageNum);
		imshow("image", *imageVector[i]);
		resize(visulizePred,visulizePred,Size(1024,1024));
		imshow("visulize predictions",visulizePred);
		resize(visulizeClusters,visulizeClusters,Size(1024,1024));
		imshow("visulize clusters",visulizeClusters);
		waitKey();
	}
}


void calcClustersRealImage(Mat& predictions, Mat& visulizeClusters, int connectionThres, int imageWidth, int imageHeight, int tileSizeX, int tileSizeY, int fontSize,
	int minCluster,int overlap, bool useGroundTruth, int imageNum)
{
	printf("Detecting Clusters....\n\n");
	int clusterNum = 0;
	int clusterFound = 0;
	vector<int> clusterSize;
	clusterSize.clear();
	string charStr;
	Point org;
	org.x = 10;
	org.y = tileSizeY*3/4-25;
	int overlapTileX = tileSizeX/overlap;
	int overlapTileY = tileSizeY/overlap;
	Mat imRect, responseRect, *groundTruth;
	Mat clusters = Mat::zeros(predictions.size(), CV_8UC1);
	int xIndex, yIndex, xMax, yMax, maxPred, maxPredIndex;
	Mat predInClusters;
	int	truePos = 0;
	int	trueNeg = 0;
	int	falsePos= 0;
	int	falseNeg = 0;

	for(int y=0; y<predictions.rows; y++)
	{
		for(int x=0; x<predictions.cols; x++)
		{
			if(predictions.at<uchar>(y,x))
			{
				if(!clusters.at<uchar>(y,x))
				{
					clusterNum++;
					clusterSize.push_back(1);
					clusters.at<uchar>(y,x) = clusterNum;
					findClusters(clusters, predictions,x,y,clusterSize,clusterNum,connectionThres);
				}

			}
		}
	}


	for(int i=1; i<=clusterNum; i++)
	{
		predInClusters = Mat::zeros(256,1,CV_32SC1);
		if(clusterSize[i-1] >= minCluster)
		{
			xIndex = 1000000;
			yIndex = 1000000;
			xMax = 0;
			yMax = 0;
			for(int y=0; y<predictions.rows; y++)
			{
				for(int x=0; x<predictions.cols; x++)
				{
					if(clusters.at<uchar>(y,x) == (int)i)
					{
						predInClusters.at<int>((int)predictions.at<uchar>(y,x),0)++;
						if(x < xIndex)
							xIndex = x;
						if(y < yIndex)
							yIndex = y;
						if(x > xMax)
							xMax = x;
						if(y > yMax)
							yMax = y;

					}
				}
			}

			maxPred = 0;
			maxPredIndex = 0;

			if(xMax - xIndex < overlap && yMax - yIndex < overlap)
			{
				for(int j=0; j<256; j++)
				{
					if(predInClusters.at<int>(j,0) > maxPred)
					{
						maxPred = predInClusters.at<int>(j,0);
						maxPredIndex = j;
					}
				}

				charStr = (char)maxPredIndex;

				imRect = Mat::zeros(tileSizeY*3/4,tileSizeX*3/4,CV_8UC3);
				//add(imRect,128,imRect);
				xIndex = xIndex*overlapTileX; 
				yIndex = yIndex*overlapTileY; 


				groundTruth =  new Mat(imread(getImageAndGroundTruthName(imageNum,(char)maxPredIndex,true).c_str(),CV_LOAD_IMAGE_GRAYSCALE));

				if(groundTruth->empty())
				{
					falsePos++;
					cv::putText(imRect,charStr , org, 0, fontSize,CV_RGB(255,0,0), 4, 8,false);
				}
				else
				{
					resize(*groundTruth,*groundTruth,Size(imageWidth,imageHeight));

					if(sum((*groundTruth)(Rect(xIndex,yIndex,tileSizeX,tileSizeY)))(0))
					{
						truePos++;
						cv::putText(imRect,charStr , org, 0, fontSize,CV_RGB(0,255,0), 4, 8,false);
					}
					else
					{
						falsePos++;
						cv::putText(imRect,charStr , org, 0, fontSize,CV_RGB(255,0,0), 4, 8,false);
					}
				}
				delete groundTruth;

				imRect.copyTo(visulizeClusters(Rect(xIndex,yIndex,tileSizeX*3/4,tileSizeY*3/4)));
			}
		}
	}
	cout << "Number of true detections:\t" << truePos << endl;
	cout << "Number of false detections:\t" << falsePos << endl;
}


void findCluster(Mat& clusters, Mat& pred, int x, int y, vector<int>& clusterSize, int clusterNum, int connectionThres)
{
	bool k = false;
	for(int i=y-connectionThres; i<=y+connectionThres; i++)
	{
		for(int j=x-connectionThres; j<=x+connectionThres; j++)
		{
			if(i >=0 && j>=0 && j<clusters.cols && i<clusters.rows && pred.at<uchar>(i,j) && !clusters.at<uchar>(i,j))
			{
				clusters.at<uchar>(i,j) = clusterNum;
				clusterSize[clusterNum-1]++;
				k = true;
				findClusters(clusters,pred,j,i,clusterSize,clusterNum,connectionThres);
			}
		}
	}

	if(!k)
		return;
}

