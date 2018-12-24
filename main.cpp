#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <fstream>

#include <time.h>

using namespace std;
using namespace cv;

void process(Mat srcImg)
{
	cv::resize(srcImg, srcImg, Size(800, 600));
	cv::flip(srcImg, srcImg, 1);
	//1.分离通道
	vector<Mat> imgSplit;
	split(srcImg, imgSplit);

	//2.滤波
	Mat imgMedianBlur;
	medianBlur(imgSplit[0], imgMedianBlur, 5);

	//3.二值化
	Mat imgBinary;
	threshold(imgMedianBlur, imgBinary, 100, 255, CV_THRESH_BINARY);

	//4.边缘提取
	Mat imgEdge;
	Canny(imgBinary, imgEdge, 5, 120);

	//5.霍夫线变换
	vector<Vec2f> lines;
	vector<vector<Point>> points;
	HoughLines(imgEdge, lines, 0.78, CV_PI / 180, 95, 0, 0);
	//选前四条直线
	int linesNum = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0]; //半径
		float theta = lines[i][1]; //角度
		Point pt1, pt2;
		vector<Point> temp;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));

		temp.push_back(pt1);
		temp.push_back(pt2);

		if (points.empty())
		{
			points.push_back(temp);
		}
		else
		{
			if (linesNum <= 3 && hasPoints(temp, points))
			{
				points.push_back(temp);
			}
		}
		linesNum++;
		if (linesNum > 4)
			break;
	}
	for (int i = 0; i < points.size(); i++)
	{
		cout << "coord: (" << points[i][0].x << "," << points[i][0].y << "), (" << points[i][1].x << "," << points[i][1].y << ")" << endl;
		//cv::line(srcImg, points[i][0], points[i][1], Scalar(255, 0, 0));
	}
	cv::imshow("line", srcImg);

	//5.求直线交点
	vector<cv::Point2f> corners = findCorners(points, srcImg.cols, srcImg.rows);
	cout << "corners.size(): " << corners.size() << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		cout << corners[i].x << " " << corners[i].y << endl;
		//cv::circle(srcImg, cv::Point(corners[i].x, corners[i].y), 10, cv::Scalar(0, 0, 255), 4, 4);
	}
	cv::imshow("corners", srcImg);

	//6.交点排序
	vector<cv::Point2f> cornersSorted = sortCorner(corners);
	std::cout << "After sorted " << endl;
	for (int i = 0; i < cornersSorted.size(); i++)
	{
		cout << "x:" << cornersSorted[i].x << " y:" << cornersSorted[i].y << endl;
	}

	//6.找到卡片位置
	Mat imgTrans = Mat::zeros(540, 856, CV_8UC1);
	vector<cv::Point2f> transPts = { cv::Point2f(0,0), cv::Point2f(imgTrans.cols,0), cv::Point2f(0,imgTrans.rows), cv::Point2f(imgTrans.cols,imgTrans.rows) };
	Mat transMat;
	int cornerNum = cornersSorted.size();
	switch (cornerNum)
	{
	case 4:
		transMat = getPerspectiveTransform(cornersSorted, transPts);
		warpPerspective(srcImg, imgTrans, transMat, imgTrans.size());
		break;
	default:
		break;
	}
	cv::imshow("imgTrans", imgTrans);
	cv::waitKey(0);

	//imgProc(imgTrans, cv::Size(17, 5), cv::Size(3, 3), 200);

	unevenLightCompensate(imgTrans, 32);
	cv::imshow("unevenLightCompensate", imgTrans);

	//4.Sobel算子，x方向求梯度
	Mat sobel; //,sobel_x,sobel_y;
	Sobel(imgTrans, sobel, CV_8U, 1, 0, 3);
	convertScaleAbs(sobel, sobel);

	//5.二值化
	Mat binary;
	threshold(sobel, binary, 0, 255, THRESH_OTSU + THRESH_BINARY);
	//Mat dest;
	//blur(binary, dest, Size(3, 3));

	//6.膨胀和腐蚀操作核设定
	Mat element1 = getStructuringElement(MORPH_RECT, Size(5, 5));
	//控制高度设置可以控制上下行的膨胀程度，例如3比4的区分能力更强,但也会造成漏检
	Mat element2 = getStructuringElement(MORPH_RECT, Size(5,5));
	//Mat element3 = getStructuringElement(MORPH_RECT, Size(5, 3));

	//7.膨胀一次，让轮廓突出
	Mat dilate1;
	dilate(binary, dilate1, element1);

	//8.腐蚀一次，去掉细节，表格线等。这里去掉的是竖直的线
	Mat erode1;
	erode(dilate1, erode1, element2);

	cv::imshow("erode1", erode1);
	cv::waitKey(0);

	Mat res = erode1;
	vector<cv::RotatedRect> rects = findTextRegion(res);
	cout << "rect.size(): " << rects.size() << endl;
	for (auto rect : rects) {
		Point2f P[4];
		rect.points(P);

		line(srcImg, Point((int)P[0].x, (int)P[0].y), Point((int)P[1].x, (int)P[1].y), Scalar(0, 255, 0));
		line(srcImg, Point((int)P[1].x, (int)P[1].y), Point((int)P[2].x, (int)P[2].y), Scalar(0, 255, 0));
		line(srcImg, Point((int)P[2].x, (int)P[2].y), Point((int)P[3].x, (int)P[3].y), Scalar(0, 255, 0));
		line(srcImg, Point((int)P[3].x, (int)P[3].y), Point((int)P[0].x, (int)P[0].y), Scalar(0, 255, 0));

		Mat mask(res.size(), CV_8UC1, Scalar::all(255));
		Rect dwRect = rect.boundingRect();
		if (dwRect.x<0)
			dwRect.x = 0;
		if (dwRect.x + dwRect.width > res.cols)
			dwRect.width = res.cols - dwRect.x;
		if (dwRect.y<0)
			dwRect.y = 0;
		if (dwRect.y + dwRect.height > res.rows)
			dwRect.height = res.rows - dwRect.y;
		//rectangle(srcImg, dwRect, Scalar(0, 0, 255));
		mask(dwRect).setTo(255);
		Mat imgROI;
		srcImg.copyTo(imgROI, mask);
		Mat reImg = normalizedMatByRoi(imgROI, rect);
		cvtColor(reImg, reImg, COLOR_BGR2GRAY);
		equalizeHist(reImg, reImg);
		reImg = remove_noise_and_smooth(reImg);
		cv::imshow("reImg", reImg);
		cv::waitKey(0);
		mProcImgs.push_back(reImg);
		//imwrite("I:\\xx.tif", reImg);
	}
	cv::imshow("srcImg", srcImg);
	cv::waitKey(0);

}

bool setSortRule(const Point2f& p1, const Point2f& p2)
{
	return p1.y < p2.y;
}

vector<cv::Point2f> sortCorner(vector<cv::Point2f> corners)
{
	vector<cv::Point2f> cornersSorted;
	vector<cv::Point2f> cornersRes;
	if (corners.size() == 4)
	{
		sort(corners.begin(), corners.end(), setSortRule);
		Point2f tl, tr, bl, br;
		if (corners[0].x < corners[1].x)
		{
			tl = corners[0];
			tr = corners[1];
		}
		else
		{
			tl = corners[1];
			tr = corners[0];
		}
		if (corners[2].x < corners[3].x)
		{
			bl = corners[2];
			br = corners[3];
		}
		else
		{
			bl = corners[3];
			br = corners[2];
		}
		float d1 = (tl.x - tr.x)*(tl.x - tr.x) + (tl.y - tr.y)*(tl.y - tr.y);
		float d2 = (tl.x - bl.x)*(tl.x - bl.x) + (tl.y - bl.y)*(tl.y - bl.y);
		if (d1 > d2)
		{
			cornersSorted.push_back(tl);
			cornersSorted.push_back(tr);
			cornersSorted.push_back(bl);
			cornersSorted.push_back(br);
		}
		else
		{
			cornersSorted.push_back(tr);
			cornersSorted.push_back(br);
			cornersSorted.push_back(tl);
			cornersSorted.push_back(bl);
		}
	}
	else
	{
		for (int i = 0; i < corners.size(); i++)
			cornersSorted.push_back(corners[i]);
	}
	return cornersSorted;
}

std::vector<cv::Point2f> findCorners(std::vector<std::vector<cv::Point>> points, int w, int h)
{
	vector<cv::Point2f> corners;
	for (size_t i = 0; i < points.size(); i++)
	{
		for (size_t j = i + 1; j < points.size(); j++)
		{
			int x1 = points[i][0].x, y1 = points[i][0].y, x2 = points[i][1].x, y2 = points[i][1].y;
			int x3 = points[j][0].x, y3 = points[j][0].y, x4 = points[j][1].x, y4 = points[j][1].y;

			//x1  y1 line1.end
			//x2  y2 line1.start
			//x3  y3 line2.end
			//x4  y4 line2.start
			float x11 = x1 - x2;
			float y11 = y1 - y2;
			float x22 = x3 - x4;
			float y22 = y3 - y4;
			float x21 = x4 - x2;
			float y21 = y4 - y2;
			float dd = y11*x22 - y22*x11;

			if (abs(dd) > 1000.0)
			{
				Point2f pt;
				pt.x = (x11*x22*y21 + y11*x22*x2 - y22*x11*x4) / dd;
				pt.y = -(y11*y22*x21 + x11*y22*y2 - x22*y11*y4) / dd;
				cout << " x: " << pt.x << "  y: " << pt.y << endl;
				if (abs(pt.x) <= w && abs(pt.y) <= h)
					corners.push_back(pt);
			}
		}
	}
	return corners;
}

bool hasPoints(vector<Point> temp, vector<vector<Point>> points)
{
	bool flag = true;
	for (auto iter = points.begin(); iter != points.end(); iter++)
	{
		if ((abs((*iter)[0].x - temp[0].x) + abs((*iter)[0].y - temp[0].y)) < 50 || (abs((*iter)[1].x - temp[1].x) + abs((*iter)[1].y - temp[1].y)) < 50)
		{
			flag = false;
			break;
		}
	}
	return flag;
}

void unevenLightCompensate(Mat &image, int blockSize)
{
	if (image.channels() == 3) cvtColor(image, image, 7);
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}

vector<RotatedRect> findTextRegion(Mat &img)
{
	vector<RotatedRect> rects;
	//1.查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//drawContours(mSrcImg, contours, -1, Scalar(0, 0, 255));

	//2.筛选护照
	for (int i = 0; i < contours.size(); i++)
	{
		//找到最小矩形，该矩形可能有方向
		RotatedRect rect = minAreaRect(contours[i]);
		//rectangle(mSrcImg, rect.boundingRect(),Scalar(0,0,255));
		//计算高和宽
		int m_width = rect.boundingRect().width;
		int m_height = rect.boundingRect().height;
		//筛选那些太细的矩形，留下扁的
		if (m_height > m_width || m_width<200)
			continue;
		if (!verifySize(rect))
			continue;

		//计算当前轮廓的面积
		/*double area = contourArea(contours[i]);

		//面积小于 200000（1080的图,其他的要测试调整）的全部筛选掉
		if (area < 15000)
		continue;*/

		drawContours(mSrcImg, contours, i, Scalar(0, 0, 255));

		//轮廓近似，作用较小，approxPolyDP函数有待研究
		double epsilon = 0.001*arcLength(contours[i], true);
		Mat approx;
		approxPolyDP(contours[i], approx, epsilon, true);

		//符合条件的rect添加到rects集合中
		rects.push_back(rect);
	}
	return rects;
}

int main()
{
	string imageList = "pics.txt";
	std::ifstream finImage(imageList, std::ios::in);
	string line;
	Mat image;

	while (getline(finImage, line))
	{
		image = imread(line);

		process(image);
		
		char c = waitKey(0) & 0xFF;
		if (c == 'q')
			break;
	}
}