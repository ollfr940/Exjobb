#include "stdafx.h"
#include "functions.h"
using namespace std;
using namespace cv;

void writeMatToFile(Mat& m, Mat& r,int imageNum, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}

	for(int i=0; i<m.rows; i++)
	{
		fout << r.at<int>(i,0) << ',';
		for(int j=0; j<m.cols-1; j++)
		{
			fout<<m.at<float>(i,j)<<',';
		}
		fout << m.at<float>(i,m.cols-1);
		fout<<std::endl;
	}

	fout.close();
}

void rotate(Mat& src, double angle,int imageSize, double scale)
{
	/*Mat transformation = Mat::zeros(2,3,CV_32FC1);
	transformation.at<float>(0,0) = scale;//*cos(angle*3.1415/180);
	transformation.at<float>(0,1) = scale;//*sin(angle*3.1415/180);
	transformation.at<float>(1,0) = -scale*sin(angle*3.1415/180);
	transformation.at<float>(1,1) = scale*cos(angle*3.1415/180);*/
	//transformation.at<float>(0,2) = scale*imageSize/2;
	//transformation.at<float>(1,2) = scale*imageSize/2;
	Point2f pt((float)imageSize/2, (float)imageSize/2);
	Mat transformation = getRotationMatrix2D(pt, angle, scale);

	warpAffine(src, src, transformation, Size(imageSize*(int)scale, imageSize*(int)scale),0,cv::BORDER_CONSTANT,Scalar(255,255,255));
}


RandomCharacters produceData(int numOfChars, int charSize, string type,double angle)
{
	RandomCharacters chars;
	chars.responses = Mat::zeros(numOfChars,1,CV_32SC1);
	//vector<Mat*> imageVec;
	Mat* image;
	double fontSize = charSize/30 + 1;
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

	for(int c=0; c<numOfChars; c++)
	{
		if(type == "numbers")
			randd = d + rng.uniform(0,9);
		else if(type == "uppercase" || type == "lowercase")
			randd = d + rng.uniform(0,25);

		chars.responses.at<int>(c,0) = (int)randd;

		image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
		cv::add(*image,255,*image);
		cv::Point org;
		org.x = rng.uniform(-10,30);
		org.y = rng.uniform(charSize-20, charSize+20);
		dstr = randd;
		randomAngle = rng.uniform(-angle,angle);
		cv::putText(*image,dstr , org, 0, fontSize ,0, 4, 8,false);
		rotate(*image,randomAngle,charSize,scale);
		chars.randChars.push_back(image);
		//cv::imshow("im", *image);
		//cv::waitKey();
	}
	return chars;
}

RandomCharactersImages createTestImages(int numOfImages, int numOfChars, int charSize, int imageSize, string type,double angle)
{
	RandomCharactersImages charImages;
	Mat* image, *responses;
	Mat characterRect, responseRect;
	int xPos, yPos;
	int charSizeR = charSize/30 + 1;
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
		image = new Mat(Mat::zeros(imageSize,imageSize,CV_8UC1));
		responses = new Mat(Mat::zeros(imageSize,imageSize,CV_8UC1));
		cv::add(*image,255,*image);

		for(int c=0; c<numOfChars; c++)
		{
			if(type == "numbers")
				randd = d + rng.uniform(0,9);
			else if(type == "uppercase" || type == "lowercase")
				randd = d + rng.uniform(0,25);

			xPos = rng.uniform(0,imageSize-charSize);
			yPos = rng.uniform(0,imageSize-charSize);

			if(!sum((*responses)(Rect(xPos,yPos,charSize,charSize)))(0))
			{
				dstr = randd;
				characterRect = Mat::zeros(charSize,charSize,CV_8UC1);
				responseRect = Mat::zeros(charSize,charSize,CV_8UC1);
				add(characterRect,255,characterRect);
				add(responseRect,(int)randd,responseRect);
				cv::Point org(10,charSize-10);
				randomAngle = rng.uniform(-angle,angle);
				cv::putText(characterRect,dstr , org, 0, charSizeR ,0, 4, 8,false);
				rotate(characterRect,randomAngle,charSize,scale);
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

Mat createRectFeatures(RandomCharacters trainingData,int numOfChars, int charSize)
{
	CalcRectSample calcRect;
	Mat integralIm, integralRect;
	float p;
	int filtNum = 28578;
	Mat featureMat = Mat::zeros(numOfChars,filtNum,CV_32FC1);
	for(int im=0; im<numOfChars; im++)
	{
		//integral(*trainingData[im], integralIm);
		int indx = 0; 
		for(int rectx=0; rectx <8; rectx++)
		{
			for(int recty=0; recty <8; recty++)
			{
				int rectSizex = 12 + rectx*4;
				int rectSizey = 12 + recty*4;
				int rectNumx = charSize/rectSizex;
				int rectNumy = charSize/rectSizey;

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
}

void evaluateRect(CvRTrees& tree, int testNum, int imageSize,string type, double angle)
{
	RandomCharacters testData = produceData(testNum,imageSize,type,angle);
	Mat testFeatures = createRectFeatures(testData,testNum, imageSize);

	for(int im=0; im<testNum; im++)
	{
		cout << "Real value: " << (char)testData.responses.at<int>(im,0) << endl;
		cout << "Predicted value: " << (char)tree.predict(testFeatures.row(im)) << endl << endl;
	}
}

vector<Mat*> predictImages(RandomCharactersImages& randIms, CvRTrees& forest,int imNum, int imageSize, int charSize, int overlap, int tileNum, int numOfTrees,double desicionThres, string type)
{
	CalcRectSample calcRect;
	int xPos, yPos, predPosx, predPosy;
	Mat imRect, integralRect, imRectHalf1,imRectHalf2,imRectHalf3,imRectHalf4;
	vector<Mat*> predictions;
	Mat* pred;
	int filtNum = 28578;
	Mat featureMat = Mat::zeros(1,filtNum,CV_32FC1);
	Mat treePred;
	double minVal, maxVal;
	int minIndx, maxIndx;

	for(int im=0; im<imNum; im++)
	{
		pred = new Mat(Mat::zeros(tileNum,tileNum,CV_8UC1));
		xPos = 0;
		yPos = 0;
		predPosx = 0;
		predPosy = 0;
		while(yPos < imageSize-charSize)
		{
			while(xPos < imageSize-charSize)
			{
				imRect = (*randIms.randChars[im])(Rect(xPos,yPos,charSize,charSize));
				imRectHalf1 = imRect(Rect(0,0,charSize/2,charSize));
				imRectHalf2 = imRect(Rect(charSize/2,0,charSize/2,charSize));
				imRectHalf3 = imRect(Rect(0,0,charSize,charSize/2));
				imRectHalf4 = imRect(Rect(0,charSize/2,charSize,charSize/2));
				if(sum(imRect)(0) < charSize*charSize*255 && sum(imRectHalf1)(0) < charSize*charSize*255/2 && sum(imRectHalf2)(0) < charSize*charSize*255/2 && sum(imRectHalf3)(0) < charSize*charSize*255/2 && sum(imRectHalf4)(0) < charSize*charSize*255/2)
				{
					int indx = 0; 
					for(int rectx=0; rectx <8; rectx++)
					{
						for(int recty=0; recty <8; recty++)
						{
							int rectSizex = 12 + rectx*4;
							int rectSizey = 12 + recty*4;
							int rectNumx = charSize/rectSizex;
							int rectNumy = charSize/rectSizey;

							for(int i=0; i<rectNumx; i++)
							{
								for(int j=0; j<rectNumy; j++)
								{
									integral(imRect(Rect(i*rectSizex,j*rectSizey,rectSizex,rectSizey)),integralRect,CV_32FC1);
									calcRect.operator()(integralRect,featureMat,indx,rectSizex,rectSizey,0);
								}
							}
						}
					}

					//pred->at<uchar>(predPosx,predPosy) = forest.predict(featureMat);

					treePred = Mat::zeros(256,1,CV_8UC1);

					for(int t=0; t<numOfTrees; t++)
					{
						CvForestTree* tree = forest.get_tree(t);
						treePred.at<uchar>(tree->predict(featureMat)->value,0)++;
					}
					cv::minMaxIdx(treePred,&minVal,&maxVal,&minIndx,&maxIndx);
					//cout << treePred << endl;
					//cout << numOfTrees << endl;
					if(maxVal > numOfTrees*desicionThres)
						pred->at<uchar>(predPosx,predPosy) = maxIndx;

					xPos += charSize/overlap;
					predPosx++;
					//yPos += charSize/overlap;
				}
				else
				{
					xPos += charSize/overlap; // charSize;
					predPosx++; // += overlap;
					//yPos += charSize;
				}
			}
			predPosx = 0;
			predPosy++;
			xPos = 0;
			yPos += charSize/overlap;
		}
		predictions.push_back(pred);
	}
	return predictions;
}

void evaluateResult(vector<Mat*> predictions, RandomCharactersImages& randIms,  int imageSize, int charSize, int tileNum, int overlap)
{
	int overlapTile = charSize/overlap;
	char p;
	char maxRes;
	string charStr;
	double charSizeR = 0.25;//charSize/(100*overlap) + 1;
	int charPos = overlapTile/12;
	Mat characterRect, responseRect;
	Mat visulizePred = Mat(Mat::zeros(imageSize,imageSize,CV_8UC3));
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

	for(int i=0; i<predictions.size(); i++)
	{
		//mergeIm.push_back(*randIms.randChars[i]);
		//mergeIm.push_back(*randIms.randChars[i]);
		//mergeIm.push_back(*randIms.randChars[i]);
		//merge(mergeIm,visulizePred);

		for(int x=0; x<tileNum; x++)
		{
			for(int y=0; y<tileNum; y++)
			{
				responseRect = (*randIms.responses[i])(Rect(x*overlapTile,y*overlapTile,charSize,charSize));
				calcHist(&responseRect,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);
				cv::minMaxIdx(hist,&minVal,&maxVal,&minIndx,&maxIndx);

				if(predictions[i]->at<uchar>(x,y))
				{
					p = predictions[i]->at<uchar>(x,y);
					charStr = p;
					characterRect = Mat::zeros(overlapTile,overlapTile,CV_8UC3);
					//add(characterRect,255,characterRect);
					cv::Point org(charPos,overlapTile-charPos);

					if(p == (char)maxIndx)
					{
						cv::putText(characterRect,charStr , org, 0, charSizeR,CV_RGB(0,255,0), 1, 8,false);
					}
					else
					{
						add(characterRect,255,characterRect);
						cv::putText(characterRect,charStr , org, 0, charSizeR ,CV_RGB(255,0,0), 1, 8,false);
					}
					
					characterRect.copyTo(visulizePred(Rect(x*overlapTile,y*overlapTile,overlapTile,overlapTile)));
				}
				else if(maxIndx != 0)
				{
					characterRect = Mat::zeros(overlapTile,overlapTile,CV_8UC3);
					characterRect.copyTo(visulizePred(Rect(x*overlapTile,y*overlapTile,overlapTile,overlapTile)));
				}
			}
		}
		imshow("visulize predictions",visulizePred);
		waitKey();
	}
}
