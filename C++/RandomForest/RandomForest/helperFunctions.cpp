#include "stdafx.h"
#include "helperFunctions.h"
using namespace std;
using namespace cv;

int calcMaxIndex(Mat& matrix, int numOfValues)
{
	int maxPred = 0;
	int maxPredIndex = 0;
	Mat histMat = Mat::zeros(numOfValues,1,CV_32SC1);

	for(int x=0; x<matrix.rows; x++)
	{
		for(int y=0; y<matrix.rows; y++)
		{
			histMat.at<int>(matrix.at<uchar>(y,x),0)++;
		}
	}

	for(int i=0; i<numOfValues; i++)
	{
		if(histMat.at<int>(i,0) > maxPred)
		{
			maxPred = histMat.at<int>(i,0);
			maxPredIndex = i;
		}
	}
	return maxPredIndex;
}

void removeFalsePredictions(Mat& pred)
{
	int zeroSum;
	for(int y=0; y<pred.rows; y++)
	{
		for(int x=0; x<pred.cols; x++)
		{
			if(pred.at<uchar>(y,x))
			{
				zeroSum = 0;
				for(int yy=y-1; yy <= y+1; yy++)
				{
					for(int xx=x-1; xx <= x+1; xx++)
					{
						if(yy < 0 || yy >= pred.rows || xx < 0 || xx >= pred.cols)
							zeroSum++;
						else if(!pred.at<uchar>(yy,xx))
							zeroSum++;
					}
				}
				if(zeroSum > 6)
					pred.at<uchar>(y,x) = 0;
			}
		}
	}
}
void calcClusters(Mat& predictions, Mat& visulizeClusters, Mat& responses, int connectionThres, int imageWidth, int imageHeight, int charSize, int fontSize, int minCluster,int overlapTileX, int overlapTileY)
{
	printf("Detecting Clusters....\n\n");
	int clusterNum = 0;
	int clusterFound = 0;
	vector<int> clusterSize;
	clusterSize.clear();
	string charStr;
	Point org;
	org.x = 10;
	org.y = charSize*3/4-25;
	Mat imRect, responseRect;
	Mat clusters = Mat::zeros(predictions.size(), CV_8UC1);
	int xIndex, yIndex, maxPred, maxPredIndex, maxResponseIndx;
	Mat predInClusters;

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
					findCluster(clusters, predictions,x,y,clusterSize,clusterNum,connectionThres);
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
					}
				}
			}

			maxPred = 0;
			maxPredIndex = 0;
			for(int j=0; j<256; j++)
			{
				if(predInClusters.at<int>(j,0) > maxPred)
				{
					maxPred = predInClusters.at<int>(j,0);
					maxPredIndex = j;
				}
			}

			charStr = (char)maxPredIndex;

			xIndex = xIndex*overlapTileX; 
			yIndex = yIndex*overlapTileY; 
			
			responseRect = responses(Rect(xIndex,yIndex,charSize,charSize));
			maxResponseIndx = calcMaxIndex(responseRect,256);
			charStr = (char)maxPredIndex;

			imRect = Mat::zeros(charSize,charSize,CV_8UC3);
			add(imRect,255,imRect);

			if(maxPredIndex == maxResponseIndx)
				putText(imRect,charStr,org, 0, fontSize ,CV_RGB(0,255,0),4, 8,false);
			else
				putText(imRect,charStr,org, 0, fontSize ,CV_RGB(255,0,0),4, 8,false);

			imRect.copyTo(visulizeClusters(Rect(xIndex,yIndex,charSize,charSize)));
		}
	}
}


void findClusters(Mat& clusters, Mat& pred, int x, int y, vector<int>& clusterSize, int clusterNum, int connectionThres)
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
				findCluster(clusters,pred,j,i,clusterSize,clusterNum,connectionThres);
			}
		}
	}

	if(!k)
		return;
}

vector<CvRTrees*> loadForestsBackground()
{
	string name;
	vector<CvRTrees*> forestVector;
	cout << "loading classifier for character and background separation...." << endl;

	//name = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\DetectFalseTilespoints_NumOfTrueData1000_NumOfFalseData10000_NumOfFalseImages200\\Forest_8x8_" + ret + ".xml";
		
	name = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\DetectFalseTilespoints_NumOfTrueData1500_NumOfFalseData20000_NumOfFalseImages300\\Forest_8x8_0.xml";
	forestVector.push_back(new CvRTrees);
	forestVector[0]->load(name.c_str());

	/*name = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\DetectFalseTilespoints_NumOfTrueData200_NumOfFalseData1000_NumOfFalseImages10\\Forest_16x16_0.xml";
	forestVector.push_back(new CvRTrees);
	forestVector[1]->load(name.c_str());*/

	return forestVector;
}		

vector<CvRTrees*> loadForestsOCR(int numOfForests, int maxDepth, int maxNumOfTreesInForest, int numOfChars, int numOfFalseImages, int tileSizeX, int tileSizeY, double angle, 
	string charType, string featureType, bool falseClass, bool useNoise)
{
	vector<CvRTrees*> forestVector;


	for(int i=0; i<numOfForests; i++)
	{
		forestVector.push_back(new CvRTrees);
		cout << "loading: " << intToStrOCR(i,numOfChars,numOfFalseImages,tileSizeX, tileSizeY,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,false,useNoise) << endl << endl;
		forestVector[i]->load(intToStrOCR(i,numOfChars,numOfFalseImages,tileSizeX, tileSizeY,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,false,useNoise).c_str());
	}
	return forestVector;
}		

int calcRectFiltNum(int charSizeX, int charSizeY)
{
	int filtNum = 0;
	for(int i=0; i<8; i++)
	{
		for(int j=0; j<8; j++)
		{
			int rectSizex = 12+4*i;
			int rectSizey = 12+4*j;
			filtNum += (charSizeX/rectSizex)*(charSizeY/rectSizey);
		}
	}
	return filtNum*17;
}

string getImageAndGroundTruthName(int imageNum, char p, bool getGroundTrue)
{
	string name;
	string mapName;
	stringstream s1;
	string ret1 = "";
	s1 << imageNum; s1 >> ret1;
	mapName = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\testImages\\im" + ret1;

	if(getGroundTrue)
		name = mapName + "\\im" + ret1 + "_mask" + p + ".png";
	else
		name = mapName + ".jpg";
	
	return name;
}

string intToStrBackground(int i, int numOfTrueChars, int numOfFalseChars, int numOfFalseImages, int reSizeTo, string featureType, bool n)
{
	string name;
	string mapName;

	stringstream s1, s2, s3, s4, s5;
	string ret1 = "";
	string ret2 = "";
	string ret3 = "";
	string ret4 = "";
	string ret5 = "";
	s1 << numOfTrueChars; s1 >> ret1;
	s2 << numOfFalseChars; s2 >> ret2;
	s3 << numOfFalseImages; s3 >> ret3;
	s4 << reSizeTo; s4 >> ret4;
	s5 << i; s5 >> ret5;
	mapName = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\DetectFalseTiles" + featureType +"_NumOfTrueData" + ret1 + "_NumOfFalseData" + ret2 + "_NumOfFalseImages" + ret3;

	if(n)
		CreateDirectoryA(mapName.c_str(),NULL);

	name = mapName + "\\" +"Forest" + "_" + ret4 + "x" + ret4 + "_" + ret5 + ".xml";
	return name;
}

string intToStrOCR(int i, int numOfChars, int numOfFalseImages, int tileSizeX, int tileSizeY, int depth, int treeNum, double angle, string charType, string featureType, bool falseClass, 
	bool n, bool useNoise)
{
	string name;
	string mapName;

	stringstream s2, s3, s4, s5, s6, s7, s8;
	string ret2 = "";
	string ret3 = "";
	string ret4 = "";
	string ret5 = "";
	string ret6 = "";
	string ret7 = "";
	string ret8 = "";

	s2 << numOfChars; s2 >> ret2;
	s3 << i; s3 >> ret3;
	s4 << depth; s4 >> ret4;
	s5 << treeNum; s5 >> ret5;
	s6 << angle; s6 >> ret6;
	s7 << tileSizeX; s7 >> ret7;
	s8 << tileSizeY; s8 >> ret8;

	mapName = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\" + charType + "_" + featureType + "_DataForEachClass" + 
		ret2 + "_imageSize" + ret7 + "x" + ret8 + "_depth" + ret4 + "_treeNum" + ret5 + "_angle" + ret6;

	if(falseClass)
		mapName += "_usingFalseClass";

	if(useNoise)
		mapName += "_withNoise";

	if(n)
		CreateDirectoryA(mapName.c_str(),NULL);
	name = mapName + "\\" +"Forest" + "_" + ret3 + ".xml";

	return name;
}

void mouseCallback( int event, int x, int y, int flags, void* param )
{
	Mat* frame = (Mat*) param;
	Rect *saveBox;
	char character;

	switch( event )
	{
	case CV_EVENT_MOUSEMOVE:
		{
			if( drawing_box )
			{
				box.width = x-box.x;
				box.height = y-box.y;
			}
			if(firstBox)
			{
				firstBoxWidth = box.width;
				firstBoxHeight = box.height;
			}
		}
		break;

	case CV_EVENT_LBUTTONDOWN:
		{   
			drawing_box = true;
			box = Rect( x, y, 0, 0 );
		}
		break;

	case CV_EVENT_LBUTTONUP:
		{  
			drawing_box = false;

			if( box.width < 0 )
			{   
				box.x += box.width;
				box.width *= -1;      
			}

			if( box.height < 0 )
			{   
				box.y += box.height;
				box.height *= -1; 
			}
			if(firstBox)
				firstBox = false;
			else
			{
				box.x += (box.width - firstBoxWidth)/2;
				box.y += (box.height - firstBoxHeight)/2;
				box.width = firstBoxWidth;
				box.height = firstBoxHeight;
			}
			saveBox = new Rect(box);
			boxVector.push_back(saveBox);
			draw_box(frame, box);
			cout << "Type the character and press Enter: ";
			cin >> character;
			cout << endl << endl;
			boxResponses.push_back(character);
		}
		break;

	default:
		break;
	}
}

void drawSquareToAdjustImage( int event, int x, int y, int flags, void* param )
{
	Mat* frame = (Mat*) param;
	Rect *saveBox;
	Mat sample = imread("C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\OCRAFont\\A.png",CV_LOAD_IMAGE_GRAYSCALE);
	resize(sample,sample,Size(64,64));
	int sampleHeight = calcCharHeight(sample);

	switch( event )
	{
	case CV_EVENT_MOUSEMOVE:
		{
			if( drawing_box )
			{
				box.width = x-box.x;
				box.height = y-box.y;
			}
		}
		break;

	case CV_EVENT_LBUTTONDOWN:
		{   
			drawing_box = true;
			box = Rect( x, y, 0, 0 );
		}
		break;

	case CV_EVENT_LBUTTONUP:
		{  
			drawing_box = false;

			if( box.width < 0 )
			{   
				box.x += box.width;
				box.width *= -1;      
			}

			if( box.height < 0 )
			{   
				box.y += box.height;
				box.height *= -1; 
			}
			box.height = (64/(float)sampleHeight)*box.height;
			box.width = ((float)54/64)*box.height;
			/*
			if(box.height > box.width)
				box.width = box.height;
			else
				box.height = box.width;
				*/
			saveBox = new Rect(box);
			if(!boxVector.empty())
				boxVector.pop_back();
			boxVector.push_back(saveBox);
			draw_box(frame, box);
		}
		break;

	default:
		break;
	}
}


void draw_box(Mat * img, Rect rect)
{
	rectangle(*img, Point(box.x, box.y), Point(box.x+box.width,box.y+box.height),Scalar(0,0,255) ,2);

	Rect rect2=Rect(box.x,box.y,box.width,box.height);
}

void writeSizeToFile(int imSizeX, int imSizeY, const char* filename)
{
	std::ofstream fout(filename);
	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}
	fout << imSizeX << endl << imSizeY;
	fout.close();
}

CSize loadSizeFromFile(const char* filename)
{
	CSize s;
	std::ifstream fin(filename);
	fin >> s.width >> s.height;
	return s;
}

void createAndSavePointPairs(int numOfPoints, int width, int height, string filename)
{
	/*std::ofstream fout(filename);
	if(!fout)
	{
	std::cout<<"File Not Opened"<<std::endl;  return;
	}*/
	printf("Create random point pairs and save to file....\n\n");
	Mat pointPairVector = Mat::zeros(numOfPoints,4,CV_32SC1);
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
			x1 = rng.uniform(width/4,width*3/4);
			y1 = rng.uniform(height/4,height*3/4);
			x2 = rng.uniform(0,width);
			y2 = rng.uniform(0,height);
		}

		pointPairVector.at<int>(i,0) = x1;
		pointPairVector.at<int>(i,1) = y1;
		pointPairVector.at<int>(i,2) = x2;
		pointPairVector.at<int>(i,3) = y2;
	}

	cv::imwrite(filename, pointPairVector); 
	//fout.close();
}


void rotate(Mat& src, double angle,int imageSizex, int imageSizey, double scale)
{
	Point2f pt((float)imageSizex/2, (float)imageSizey/2);
	Mat transformation = getRotationMatrix2D(pt, angle, scale);

	warpAffine(src, src, transformation, Size(imageSizex*(int)scale, imageSizey*(int)scale),0,cv::BORDER_CONSTANT,Scalar(255,255,255));
}

void preProcessRect(Mat& image, double threshold)
{
	cv::threshold(image, image, threshold, 255,0);
}

vector<Mat*> drawRandomImages(int numOfImages,int imageSize, int numOfLines, int numOfRectangles, bool useNoise)
{
	printf("Create random images for false data....\n\n");
	Mat* image;
	vector<Mat*> imageVector;
	uint64 initValue = time(0);
	cv::RNG rng(initValue);

	int thickness, randNoisePar1, randNoisePar2;
	int lineType = 8;
	int x1 = -imageSize/2;
	int x2 = imageSize*3/2;
	int y1 = -imageSize/2;
	int y2 = imageSize*3/2;

	Point pt1, pt2;
	for(int im=0; im<numOfImages; im++)
	{
		image = new Mat(Mat::zeros(imageSize,imageSize, CV_8UC1));
		if(useNoise)
		{
			randNoisePar1 = rng.uniform(50,200);
			randNoisePar2 = rng.uniform(5,15);
			randn(*image,randNoisePar1,randNoisePar2);
		}

		for( int i = 0; i < numOfLines; i++ )
		{
			pt1.x = rng.uniform( x1, x2 );
			pt1.y = rng.uniform( y1, y2 );
			pt2.x = rng.uniform( x1, x2 );
			pt2.y = rng.uniform( y1, y2 );

			line(*image, pt1, pt2, rng.uniform(0,255), rng.uniform(1, 50), 8 );
		}
		/*
		thickness = rng.uniform( -3, 10 );

		for( int i = 0; i < numOfRectangles; i++ )
		{
		pt1.x = rng.uniform( x1, x2 );
		pt1.y = rng.uniform( y1, y2 );
		pt2.x = rng.uniform( x1, x2 );
		pt2.y = rng.uniform( y1, y2 );

		rectangle(*image, pt1, pt2, rng.uniform(0,255), MAX( thickness, -1 ), lineType );
		}
		*/
		cv::adaptiveThreshold(*image,*image,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,33,20);
		imageVector.push_back(image);
		//imshow("sdf", *image);
		//waitKey();

	}
	return imageVector;
}

void writeMatToFile(vector<vector<int>*> v, const char* filename)
{
	std::ofstream fout(filename);

	if(!fout)
	{
		std::cout<<"File Not Opened"<<std::endl;  return;
	}

	for(int i=0; i<v.size(); i++)
	{
		vector<int> p = *v[i];
		for(int j=0; j<p.size(); j++)
		{
			fout << p[j] <<',';
		}
		fout<<std::endl;
	}

	fout.close();
}


int calcCharHeight(Mat& im)
{
	for(int y=0; y<im.size().height; y++)
	{
		for(int x=0; x<im.size().width; x++)
		{
			if(im.at<uchar>(y,x) == 0)
				return im.size().height - y;
		}
	}
}
/*
RandomCharacters produceProxData(string type, int numOfClasses, int charSize, double fontSize)
{
RandomCharacters v;
Mat* image;
char d;
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

for(int c=0; c<numOfClasses; c++)
{
image = new Mat(Mat::zeros(charSize,charSize,CV_8UC1));
cv::add(*image,255,*image);
cv::Point org(10, charSize-10);
dstr = d;
cv::putText(*image,dstr , org, 0, fontSize ,0, 6, 8,false);
v.randChars.push_back(image);
d++;
//cv::imshow("im", *image);
//cv::waitKey();
}
return v;
}*/