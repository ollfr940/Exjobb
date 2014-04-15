#include "stdafx.h"
#include "helperFunctions.h"
using namespace std;
using namespace cv;

vector<CvRTrees*> loadForests(int numOfForests, int maxDepth, int maxNumOfTreesInForest, int numOfChars,int numOfClasses, int charSize, int angle, string charType, string featureType, bool falseClass)
{
	vector<CvRTrees*> forestVector;
	for(int i=0; i<numOfForests; i++)
		{
			forestVector.push_back(new CvRTrees);
			cout << "loading: " << intToStr(i,numOfChars,numOfClasses,charSize,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,false) << endl << endl;
			forestVector[i]->load(intToStr(i,numOfChars,numOfClasses,charSize,maxDepth,maxNumOfTreesInForest,angle,charType,featureType,falseClass,false).c_str());
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

string intToStr(int i, int numOfChars,int numOfClasses, int charSize, int depth, int treeNum, int angle, string charType, string featureType, bool falseClass, bool n)
{
	stringstream s1, s2, s3, s4, s5, s6, s7;
	string ret1 = "";
	string ret2 = "";
	string ret3 = "";
	string ret4 = "";
	string ret5 = "";
	string ret6 = "";
	string ret7 = "";
	string name;
	string mapName;

	s1 << numOfClasses; s1 >> ret1;
	s2 << numOfChars; s2 >> ret2;
	s3 << i; s3 >> ret3;
	s4 << depth; s4 >> ret4;
	s5 << treeNum; s5 >> ret5;
	s6 << angle; s6 >> ret6;
	s7 << charSize; s7 >> ret7;

	if(falseClass)
	{
		mapName = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\" + charType + "_" + featureType + "_NumOfData" + 
			ret1 + "x" + ret2 + "_imageSize" + ret7 + "_depth" + ret4 + "_treeNum" + ret5 + "_angle" + ret6 + "_usingFalseClass";
	}
	else
	{
		mapName = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\" + charType + "_" + featureType + "_NumOfData" + 
			ret1 + "x" + ret2 + "_imageSize" + ret7 + "_depth" + ret4 + "_treeNum" + ret5 + "_angle" + ret6;
	}

	if(n)
	CreateDirectoryA(mapName.c_str(),NULL);
	name = mapName + "\\" +"Forest" + "_" + ret3 + ".xml";
	//name = charType + "_" + featureType + "_NumOfData " + ret1 + "x" + ret2 + "_" + ret3 + ".xml";
	//name = "C:\\Users\\tfridol\\git\\Exjobb\\C++\\RandomForest\\RandomForest\\uppercase_rects_NumOfData 5200\\uppercase_NumOfData 5200_" + ret3 + ".xml";
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

		if(box.height > box.width)
			box.width = box.height;
		else
			box.height = box.width;

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

RandomCharacters produceProxData(string type, int numOfClasses, int charSize, double fontSize)
{
	RandomCharacters v;
	Mat* image;
	char d;
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
}