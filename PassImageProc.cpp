#include "PassImageProc.h"

using namespace cv;
using namespace std;


bool setSortRule_X(const cv::Point2f& p1, const cv::Point2f& p2)
{
	return p1.x < p2.x;
}

bool setSortRule_Y(const cv::Point2f& p1, const cv::Point2f& p2)
{
	return p1.y < p2.y;
}

float getDistance(cv::Point2f &pa, cv::Point2f &pb)
{
	return sqrt((pa.x - pb.x)*(pa.x - pb.x) + (pa.y - pb.y)*(pa.y - pb.y));
}

std::vector<cv::Point> findSharpCorners(const std::vector<cv::Point> bigestContour)
{
	vector<Point> sharpContour;
	int icount = bigestContour.size();
	float fmax = -1;//用于保存局部最大值
	int   imax = -1;
	bool  bstart = false;
	for (int i = 0; i<bigestContour.size(); i++) {
		Point2f pa = (Point2f)bigestContour[(i + icount - 7) % icount];
		Point2f pb = (Point2f)bigestContour[(i + icount + 7) % icount];
		Point2f pc = (Point2f)bigestContour[i];
		//两支撑点距离
		float fa = getDistance(pa, pb);
		float fb = getDistance(pa, pc) + getDistance(pb, pc);
		float fang = fa / fb;
		float fsharp = 1 - fang;
		if (fsharp>0.05) {
			bstart = true;
			if (fsharp>fmax) {
				fmax = fsharp;
				imax = i;
			}
		}
		else {
			if (bstart) {
				sharpContour.push_back(bigestContour[imax]);
				//circle(srcImgScale, bigestContour[imax], 10, Scalar(255, 255, 255), 4);
				imax = -1;
				fmax = -1;
				bstart = false;
			}
		}
	}
	return sharpContour;
}

PassImageProc::PassImageProc()
{

}

PassImageProc::PassImageProc(cv::Mat srcImg)
	:mSrcImg(srcImg)
{
}


PassImageProc::~PassImageProc()
{
}

Mat PassImageProc::getROIImage()
{
	Mat roi = mSrcImg(Rect(4, mSrcImg.rows - 120, mSrcImg.cols - 8, 120));
	return roi;
}

cv::Mat PassImageProc::findCardRegion(Mat srcImg, float scale)
{
	Mat imgTrans0 = Mat::zeros(srcImg.rows, srcImg.cols, srcImg.type());;

	Mat srcImgCopy;
	srcImg.copyTo(srcImgCopy);

	int src_H = srcImg.rows;
	int src_W = srcImg.cols;

	int scale_H = int(src_H*scale);
	int scale_W = int(src_W*scale);

	//cv::flip(srcImg, srcImg, 1);

	Mat srcImgScale;
	//cv::resize(srcImg, srcImgScale, Size(scale_W, scale_H));
	cv::resize(srcImg, srcImgScale, Size(800, 600));

	//1.分离通道
	vector<Mat> imgSplit;
	split(srcImgScale, imgSplit);

	//2.滤波
	Mat imgMedianBlur;
	medianBlur(imgSplit[0], imgMedianBlur, 5);

	//3.二值化
	Mat imgBinary;
	threshold(imgMedianBlur, imgBinary, 120, 255, THRESH_OTSU);
	cv::imshow("imgBinary", imgBinary);

	//4.边缘提取
	Mat imgEdge;
	Canny(imgBinary, imgEdge, 5, 120);
	cv::imshow("imgDege", imgEdge);

	//轮廓提取
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgEdge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	std::cout << "contours.size():" << contours.size() << endl;

	if (!contours.size())
		return imgTrans0;

	//保留最大的轮廓，即为证件位置
	int maxSize = 0;
	int idx = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		RotatedRect rect = minAreaRect(contours[i]);
		float area = rect.boundingRect().area();		
		if (maxSize < area)
		{
			maxSize = area;
			idx = i;
		}
	}
	if (maxSize < 200)
		return imgTrans0;

	std::cout << "contours[idx].size(): " << contours[idx].size() << std::endl;
	RotatedRect rect = minAreaRect(contours[idx]);

	//遍历轮廓，求出所有支撑角度
	vector<Point> sharpContours = findSharpCorners(contours[idx]);
	for (int i = 0; i < sharpContours.size(); i++)
	{
		//circle(srcImgScale, sharpContours[i], 10, Scalar(255, (40*i)%255, 255), 4);
	}
	
	////角点检测
	int rect_x = rect.boundingRect().x;
	int rect_y = rect.boundingRect().y;
	int rect_w = rect.boundingRect().width;
	int rect_h = rect.boundingRect().height;
	cv::Rect rectTmp;

	if (rect_x < 0)
		rectTmp.x = 0;
	else
		rectTmp.x = rect_x;
	if (rect_y < 0)
		rectTmp.y = 0;
	else
		rectTmp.y = rect_y;

	if (rectTmp.x + rect_w >= srcImgScale.cols)
		rectTmp.width = srcImgScale.cols - rect_x - 1;
	else
		rectTmp.width = rect_w;

	if (rectTmp.y + rect_h >= srcImgScale.rows)
		rectTmp.height = srcImgScale.rows - rect_y - 1;
	else
		rectTmp.height = rect_h;

	/////角点检测
	cv::Mat imgCorner;
	imgBinary(rectTmp).copyTo(imgCorner);
	cv::imshow("imgCorner", imgCorner);

	std::vector<cv::Point2f> cornersGoodFeatures;
	cv::goodFeaturesToTrack(imgCorner, cornersGoodFeatures, 200, 0.05, 10, Mat(), 5);

	std::cout << "Number of corners detected: " << cornersGoodFeatures.size() << std::endl;
	int cornersGoodFeaturesSize = cornersGoodFeatures.size();
	for (int i = 0; i < cornersGoodFeatures.size(); i++)
	{
		//circle(srcImgScale, cv::Point(cornersGoodFeatures[i].x + rect_x, cornersGoodFeatures[i].y + rect_y), 4, Scalar(0, 0, 255), -1, 8, 0);
	}
	sort(cornersGoodFeatures.begin(), cornersGoodFeatures.end(), setSortRule_X);
	cv::Point pt_right = cornersGoodFeatures[0];
	cv::Point pt_left = cornersGoodFeatures[cornersGoodFeaturesSize - 1];
	sort(cornersGoodFeatures.begin(), cornersGoodFeatures.end(), setSortRule_Y);
	cv::Point pt_top = cornersGoodFeatures[0];
	cv::Point pt_bot = cornersGoodFeatures[cornersGoodFeaturesSize - 1];

	//circle(srcImgScale, cv::Point(pt_right.x + rect_x, pt_right.y + rect_y), 8, Scalar(0, 0, 255), -1, 8, 0);
	//circle(srcImgScale, cv::Point(pt_left.x + rect_x, pt_left.y + rect_y), 8, Scalar(0, 0, 255), -1, 8, 0);
	//circle(srcImgScale, cv::Point(pt_top.x + rect_x, pt_top.y + rect_y), 8, Scalar(0, 0, 255), -1, 8, 0);
	//circle(srcImgScale, cv::Point(pt_bot.x + rect_x, pt_bot.y + rect_y), 8, Scalar(0, 0, 255), -1, 8, 0);
	//////////////// 角点检测 end


	//cv::drawContours(srcImgScale, contours, idx, (255, 0, 255), 1);
	//cv::rectangle(srcImgScale, rect.boundingRect(), (255, 0, 255), 1);

	cv::imshow("contours", srcImgScale);
	//cv::waitKey(0);
	
	std::cout << "rect.angle:  " << rect.angle
			 << " rect.center: " << rect.center
			 << " rect.size(): " << rect.size << std::endl;		

	float angle = rect.angle;

	//if (rect.angle < 0)
	//	angle = rect.angle + 90;

	cv::Point2f center;
	center.x = rect.center.x / 0.5;
	center.y = rect.center.y / 0.5;

	cv::Mat rot_mat = cv::getRotationMatrix2D(rect.center, rect.angle, 1.0);
	cv::warpAffine(srcImgScale, srcImgScale, rot_mat, srcImgScale.size());

	//cv::rectangle(srcImgScale, rectTmp, cv::Scalar(255, 0, 255),2);

	cv::imshow("srcImg affter warpAffine ", srcImgScale);
	cv::waitKey(0);
	
	int cropLength = max(rectTmp.width, rectTmp.height) + 1;
	int cropCenterX = rectTmp.x + rectTmp.width / 2 - 1;
	int cropCenterY = rectTmp.y + rectTmp.height / 2 - 1;
	int lt_x = max(cropCenterX - cropLength / 2 + 1, 0);
	int lt_y = max(cropCenterY - cropLength / 2 + 1, 0);

	std::cout << "cropCenterX: " << lt_x << " " << lt_y << " " << cropLength << std::endl;
	std::cout << "srcImgScale: " << srcImgScale.cols << " " << srcImgScale.rows << std::endl;

	//cv::rectangle(srcImgScale, cv::Rect(lt_x, lt_y, cropLength, cropLength), cv::Scalar(255, 255, 0));

	srcImgScale(cv::Rect(lt_x, lt_y, cropLength, cropLength)).copyTo(imgTrans0);
	
	return imgTrans0;

	/*
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
	if (hasPoints(temp, points) && linesNum <= 3)
	{
	points.push_back(temp);
	}
	}
	//linesNum++;
	//if (linesNum > 4)
	//	break;
	}
	for (int i = 0; i < points.size(); i++)
	{
	cout << "coord: (" << points[i][0].x << "," << points[i][0].y << "), (" << points[i][1].x << "," << points[i][1].y << ")" << endl;
	cv::line(srcImg, points[i][0], points[i][1], Scalar(255, 0, (30*i)%255));
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
	//cv::imshow("corners", srcImg);

	//6.交点排序
	vector<cv::Point2f> cornersSorted = sortCorner(corners);
	std::cout << "After sorted " << endl;
	for (int i = 0; i < cornersSorted.size(); i++)
	{
	cout << "x:" << cornersSorted[i].x << " y:" << cornersSorted[i].y << endl;
	}

	//7.找到卡片位置
	Mat imgTrans = Mat::zeros(540, 856, CV_8UC3);
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

	return imgTrans;
	*/
}

void PassImageProc::imgProc(Mat &srcImg, const Size &closeSize, const Size &erSize, const double binaryMinVal)
{
	//找到证件位置并旋转，此时证件有两种情况，正、倒，都是非镜像的
	float scale = 0.5;
	cv::flip(srcImg, srcImg, 1);

	Mat imgTrans;
	imgTrans = findCardRegion(srcImg, scale);

	if (imgTrans.cols == 0 || imgTrans.rows == 0)
		return;

	cv::imshow("imgTrans0", imgTrans);
	cv::waitKey(0);

	imgTrans = findCardRegion(imgTrans, scale);

	if (imgTrans.cols == 0 || imgTrans.rows == 0)
		return;

	cv::imshow("imgTrans1", imgTrans);
	cv::waitKey(0);

	//cv::imwrite("D:\\code\\OCR\\photos\\GA2\\90_1_"+string("crop2.jpg"), imgTrans);

	//cv::imshow("imgTrans", imgTrans);
	//cv::waitKey(0);

	Mat imgGary;
	cvtColor(imgTrans, imgGary, CV_RGB2GRAY);

	//光照补偿
	unevenLightCompensate(imgGary, 16);
	//cv::imshow("unevenLightCompensate", imgGary);

	//Sobel算子，x方向求梯度
	Mat sobel; //,sobel_x,sobel_y;
	Sobel(imgGary, sobel, CV_8U, 1, 0, 3);
	//cv::imshow("sobel", sobel);

	convertScaleAbs(sobel, sobel);
	//cv::imshow("convertScaleAbs", sobel);

	//二值化
	Mat binary;
	threshold(sobel, binary, binaryMinVal, 255, THRESH_OTSU + THRESH_BINARY);
	//cv::imshow("binary", binary);
	//Mat dest;
	//blur(binary, dest, Size(3, 3));

	//膨胀和腐蚀核
	Mat element1 = getStructuringElement(MORPH_RECT, closeSize);
	Mat element2 = getStructuringElement(MORPH_RECT, erSize);
	//Mat element3 = getStructuringElement(MORPH_RECT, Size(5, 3));

	//膨胀
	Mat dilate1;
	dilate(binary, dilate1, element1);

	//腐蚀
	Mat erode1;
	erode(dilate1, erode1, element2);

	//cv::imshow("erode1", erode1);
	//cv::waitKey(0);

	Mat res = erode1;
	vector<cv::RotatedRect> rects = findTextRegion(res, imgTrans);
	cout << "rect.size(): " << rects.size() << endl;
	for (auto rect : rects) {
		Point2f P[4];
		rect.points(P);

		//line(imgTrans, Point((int)P[0].x, (int)P[0].y), Point((int)P[1].x, (int)P[1].y), Scalar(0, 255, 0));
		//line(imgTrans, Point((int)P[1].x, (int)P[1].y), Point((int)P[2].x, (int)P[2].y), Scalar(0, 255, 0));
		//line(imgTrans, Point((int)P[2].x, (int)P[2].y), Point((int)P[3].x, (int)P[3].y), Scalar(0, 255, 0));
		//line(imgTrans, Point((int)P[3].x, (int)P[3].y), Point((int)P[0].x, (int)P[0].y), Scalar(0, 255, 0));

		Rect dwRect = rect.boundingRect();
		if (dwRect.x<0)
			dwRect.x = 0;
		if (dwRect.x + dwRect.width > res.cols)
			dwRect.width = res.cols - dwRect.x;
		if (dwRect.y<0)
			dwRect.y = 0;
		if (dwRect.y + dwRect.height > res.rows)
			dwRect.height = res.rows - dwRect.y;
		//rectangle(imgTrans, dwRect, Scalar(0, 0, 255));
		//mask(dwRect).setTo(255);

		//cv::imshow("mask", mask);
		//cv::waitKey(0);

		Mat imgROI;
		imgGary(dwRect).copyTo(imgROI);

		//cv::imshow("imgROI", imgROI);
		//cv::waitKey(0);

		//Mat reImg = normalizedMatByRoi(imgROI, rect);
		Mat reImg = imgROI;
		threshold(reImg, reImg, 100, 255, THRESH_OTSU);
		//cvtColor(reImg, reImg, CV_BGR2GRAY);
		//equalizeHist(reImg, reImg);
		//reImg = remove_noise_and_smooth(reImg);
		//cv::imshow("reImg", reImg);
		//cv::waitKey(0);
		mProcImgs.push_back(reImg);

		cv::transpose(reImg, reImg);
		cv::flip(reImg, reImg, 0);
		cv::transpose(reImg, reImg);
		cv::flip(reImg, reImg, 0);

		mProcImgs.push_back(reImg);
		//imwrite("I:\\xx.tif", reImg);
	}
	//cv::imshow("srcImg", srcImg);
	//cv::waitKey(0);
}

bool setSortRule(const Point2f& p1, const Point2f& p2)
{
	return p1.y < p2.y;
}

vector<cv::Point2f> PassImageProc::sortCorner(vector<cv::Point2f> corners)
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

std::vector<cv::Point2f> PassImageProc::findCorners(std::vector<std::vector<cv::Point>> points, int w, int h)
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

bool PassImageProc::hasPoints(vector<Point> temp, vector<vector<Point>> points)
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

void PassImageProc::imgProc1(Mat &srcImg, const Size &closeSize, const Size &erSize, const double binaryMinVal)
{
	//1、减弱对比度
	Mat weekContractImg;
	srcImg.convertTo(weekContractImg, srcImg.type(), 0.8, 0);
	//Mat weekContractImg = srcImg;
	//2、灰度化并中值滤波去噪
	Mat gray;
	cvtColor(weekContractImg, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 3);

	/*//1、高斯模糊
	Mat out;
	GaussianBlur(srcImg, out, Size(3, 3), 0, 0, BORDER_DEFAULT);

	//2、灰度化
	Mat gray;
	cvtColor(out, gray, COLOR_BGR2GRAY);*/


	//3、灰度化后光线补偿
	unevenLightCompensate(gray, 32);

	//4.Sobel算子，x方向求梯度
	Mat sobel; //,sobel_x,sobel_y;
	Sobel(gray, sobel, CV_8U, 1, 0, 3);
	convertScaleAbs(sobel, sobel);

	//5.二值化
	Mat binary;
	threshold(sobel, binary, binaryMinVal, 255, THRESH_OTSU + THRESH_BINARY);
	//Mat dest;
	//blur(binary, dest, Size(3, 3));

	//6.膨胀和腐蚀操作核设定
	Mat element1 = getStructuringElement(MORPH_RECT, closeSize);
	//控制高度设置可以控制上下行的膨胀程度，例如3比4的区分能力更强,但也会造成漏检
	Mat element2 = getStructuringElement(MORPH_RECT, erSize);
	//Mat element3 = getStructuringElement(MORPH_RECT, Size(5, 3));

	//7.膨胀一次，让轮廓突出
	Mat dilate1;
	dilate(binary, dilate1, element1);

	//8.腐蚀一次，去掉细节，表格线等。这里去掉的是竖直的线
	Mat erode1;
	erode(dilate1, erode1, element2);

	//imshow("erode1", erode1);
	//waitKey(0);

	Mat res = erode1;
	vector<cv::RotatedRect> rects = findTextRegion(res, res);
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
		mProcImgs.push_back(reImg);
		//imwrite("I:\\xx.tif", reImg);
	}
}

//补偿图像中的不均匀光线
void PassImageProc::unevenLightCompensate(Mat &image, int blockSize)
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

void PassImageProc::GetStringSize(HDC hDC, const char* str, int* w, int* h)
{
	SIZE size;
	GetTextExtentPoint32A(hDC, str, strlen(str), &size);
	if (w != 0) *w = size.cx;
	if (h != 0) *h = size.cy;
}

void PassImageProc::putTextZH(Mat &dst, const char* str, Point org, Scalar color, int fontSize, const char* fn, bool italic, bool underline)
{
	CV_Assert(dst.data != 0 && (dst.channels() == 1 || dst.channels() == 3));
	int x, y, r, b;
	if (org.x > dst.cols || org.y > dst.rows)
		return;
	x = org.x < 0 ? -org.x : 0;
	y = org.y < 0 ? -org.y : 0;
	LOGFONTA lf;
	lf.lfHeight = -fontSize;
	lf.lfWidth = 0;
	lf.lfEscapement = 0;
	lf.lfOrientation = 0;
	lf.lfWeight = 5;
	lf.lfItalic = italic;
	lf.lfUnderline = underline;
	lf.lfStrikeOut = 0;
	lf.lfCharSet = DEFAULT_CHARSET;
	lf.lfOutPrecision = 0;
	lf.lfClipPrecision = 0;
	lf.lfQuality = PROOF_QUALITY;
	lf.lfPitchAndFamily = 0;
	strcpy_s(lf.lfFaceName, fn);
	HFONT hf = CreateFontIndirectA(&lf);
	HDC hDC = CreateCompatibleDC(0);
	HFONT hOldFont = (HFONT)SelectObject(hDC, hf);
	int strBaseW = 0, strBaseH = 0;
	int singleRow = 0;
	char buf[1 << 12];
	strcpy_s(buf, str);
	char *bufT[1 << 12];
	int nnh = 0;
	int cw, ch;
	const char* ln = strtok_s(buf, "\n", bufT);
	while (ln != 0) {
		PassImageProc::GetStringSize(hDC, ln, &cw, &ch);
		strBaseW = max(strBaseW, cw);
		strBaseH = max(strBaseH, ch);
		ln = strtok_s(0, "\n", bufT);
		nnh++;
	}
	singleRow = strBaseH;
	strBaseH *= nnh;
	if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
	{
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hDC);
		return;
	}
	r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
	b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
	org.x = org.x < 0 ? 0 : org.x;    org.y = org.y < 0 ? 0 : org.y;
	BITMAPINFO bmp = { 0 };
	BITMAPINFOHEADER& bih = bmp.bmiHeader;
	int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));
	bih.biSize = sizeof(BITMAPINFOHEADER);
	bih.biWidth = strBaseW;
	bih.biHeight = strBaseH;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = BI_RGB;
	bih.biSizeImage = strBaseH * strDrawLineStep;
	bih.biClrUsed = 0;
	bih.biClrImportant = 0;
	void* pDibData = 0;
	HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);
	CV_Assert(pDibData != 0);
	HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);
	SetTextColor(hDC, RGB(255, 255, 255));
	SetBkColor(hDC, 0);
	strcpy_s(buf, str);
	ln = strtok_s(buf, "\n", bufT);
	int outTextY = 0;
	while (ln != 0) {
		TextOutA(hDC, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow;
		ln = strtok_s(0, "\n", bufT);
	}
	uchar* dstData = (uchar*)dst.data;
	int dstStep = dst.step / sizeof(dstData[0]);
	unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
	unsigned char* pStr = (unsigned char*)pDibData + x * 3;
	for (int tty = y; tty <= b; ++tty)
	{
		unsigned char* subImg = pImg + (tty - y) * dstStep;
		unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
		for (int ttx = x; ttx <= r; ++ttx)
		{
			for (int n = 0; n < dst.channels(); ++n) {
				double vtxt = subStr[n] / 255.0;
				int cvv = vtxt * color.val[n] + (1 - vtxt) * subImg[n];
				subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
			}
			subStr += 3;
			subImg += dst.channels();
		}
	}
	SelectObject(hDC, hOldBmp);
	SelectObject(hDC, hOldFont);
	DeleteObject(hf);
	DeleteObject(hBmp);
	DeleteDC(hDC);
}

vector<RotatedRect> PassImageProc::findTextRegion(Mat img, Mat srcImg)
{
	vector<RotatedRect> rects;
	//1.查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(mSrcImg, contours, -1, Scalar(0, 0, 255));

	std::cout << "contours.size(): " << contours.size() << std::endl;
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
		if (m_height > m_width*0.7 || m_width<500)
			continue;
		//if (!verifySize(rect))
		//	continue;

		//计算当前轮廓的面积
		/*double area = contourArea(contours[i]);

		//面积小于 200000（1080的图,其他的要测试调整）的全部筛选掉
		if (area < 15000)
		continue;*/

		//drawContours(srcImg, contours, i, Scalar(0, 255, 0));
		rectangle(srcImg, rect.boundingRect(), Scalar(0, 0, 255), 2);

		//轮廓近似，作用较小，approxPolyDP函数有待研究
		//double epsilon = 0.001*arcLength(contours[i], true);
		//Mat approx;
		//approxPolyDP(contours[i], approx, epsilon, true);

		//符合条件的rect添加到rects集合中
		rects.push_back(rect);
	}

	//cv::imshow("contours", srcImg);
	//cv::waitKey(0);

	return rects;
}

bool PassImageProc::verifySize(RotatedRect mr)
{
	float error = 0.3;
	float aspect = 37.33333;
	//Set a min and max area. All other patchs are discarded
	float min = 3 * aspect * 3; // minimum area
	float max = 2000 * aspect * 2000; // maximum area
									  //Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;
	float area = mr.size.height * mr.size.width*1.0f;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
		r = (float)mr.size.height / (float)mr.size.width;
	if ((area < min || area > max) || (r < rmin || r > rmax))
		return false;
	return true;
}

Mat PassImageProc::ImgRotate(const Mat& ucmatImg, double dDegree)
{
	Mat ucImgRotate;
	double a = sin(dDegree  * CV_PI / 180);
	double b = cos(dDegree  * CV_PI / 180);
	int width = ucmatImg.cols;
	int height = ucmatImg.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));
	Point center = Point(ucmatImg.cols / 2, ucmatImg.rows / 2);
	Mat map_matrix = getRotationMatrix2D(center, dDegree, 1.0);
	map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // 修改坐标偏移
	map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // 修改坐标偏移
	warpAffine(ucmatImg, ucImgRotate, map_matrix, { width_rotate, height_rotate }, CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT, cvScalarAll(255));
	return ucImgRotate;
}

Mat PassImageProc::normalizedMatByRoi(const Mat &cpsrcMat, const RotatedRect &rotatedRect)
{
	Mat rotated;
	//角度
	float angle = 0;
	if (rotatedRect.size.width < rotatedRect.size.height) {
		angle = rotatedRect.angle + 180;
	}
	else {
		angle = rotatedRect.angle + 90;
	}
	angle = angle - 90;
	//缩放比例
	float rad = 1;
	float maxrectbian = rotatedRect.size.height>rotatedRect.size.width ? rotatedRect.size.height : rotatedRect.size.width;
	float minrectbian = rotatedRect.size.height<rotatedRect.size.width ? rotatedRect.size.height : rotatedRect.size.width;
	float radw = cpsrcMat.cols / maxrectbian;
	float radh = cpsrcMat.rows / minrectbian;
	rad = radw<radh ? radw : radh;
	rad *= 0.95;
	//执行旋转
	rotated = ImgRotate(cpsrcMat, angle);

	//裁剪
	int pading = 5;
	double angleHUD = angle  * CV_PI / 180.; // 弧度
	double _sin = abs(sin(angleHUD)), _cos = abs(cos(angleHUD)), _tan = abs(tan(angleHUD));
	double oldx = rotatedRect.center.x, oldy = rotatedRect.center.y;
	double newpX = 0; double newpY = 0;
	if (angle<0) {
		newpX = cpsrcMat.rows * _sin + oldx * _cos - oldy * _sin;//新坐标系下rect中心坐标
		newpY = oldy / _cos + (oldx - oldy * _tan) * _sin;//新坐标系下rect中心坐标
	}
	else if (angle >= 0) {
		newpX = oldx*_cos + oldy*_sin;
		newpY = oldy / _cos + (cpsrcMat.cols - (oldx + oldy*_tan))*_sin;
	}
	int startrow = (int)(newpY - minrectbian / 2) - pading;
	if (startrow<0)startrow = 0;

	int endrow = (int)(newpY + minrectbian / 2) + pading;
	if (endrow >= rotated.rows)endrow = rotated.rows;

	int startcls = (int)(newpX - maxrectbian / 2) - pading;
	if (startcls<0)startcls = 0;

	int endcls = (int)(newpX + maxrectbian / 2) + pading;
	if (endcls >= rotated.cols)endcls = rotated.cols;
	rotated = rotated.rowRange(startrow, endrow).clone();
	rotated = rotated.colRange(startcls, endcls).clone();
	return  rotated;
}

Mat  PassImageProc::image_smoothening(Mat &roiImage, double thresholdValue)
{
	Mat retImage;
	threshold(roiImage, retImage, thresholdValue, 255, THRESH_BINARY);
	threshold(retImage, retImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
	GaussianBlur(retImage, retImage, cv::Size(1, 1), 0);
	threshold(retImage, retImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
	return retImage;
}

Mat PassImageProc::remove_noise_and_smooth(Mat &img)
{
	Mat filtered;
	adaptiveThreshold(img, filtered, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 7); //47
	Mat opening;
	Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(filtered, opening, MORPH_OPEN, element);
	Mat closing;
	morphologyEx(opening, closing, MORPH_CLOSE, element);
	img = image_smoothening(img, 91); //45
	Mat or_image;
	bitwise_or(img, closing, or_image);
	return or_image;
}

std::vector<cv::Mat>  PassImageProc::getResultImageArray()
{
	return mProcImgs;
}
