#include "PassImageProc.h"
#include <string> 
#include <io.h>
#include <process.h>
#include <ctype.h>
#include <time.h>
#include <fstream>

using namespace cv;
using namespace std;

#define CAM
//#define PIC

bool bInit = true;
bool bStopRecognize = false;
bool bContinueProc = true;
std::vector<cv::Mat> g_ResultImgs;
std::vector<std::string> g_PassNumberArray;
std::vector<std::string> g_PassNameArray;
std::vector<std::string> g_TextArray;

std::vector<cv::Vec4i> g_lines;

clock_t start, finish;
double  duration;

///调整图片分辨率
bool SetResolution(const char* path, int iResolution) {
	FILE * file = fopen(path, "rb+");// - 打开图片文件 	
	if (!file)return false;
	int len = _filelength(_fileno(file));// - 获取文件大小
	char* buf = new char[len];
	fread(buf, sizeof(char), len, file);// - 将图片数据读入缓存 	
	char * pResolution = (char*)&iResolution;// - iResolution为要设置的分辨率的数值，如72dpi 	
											 // - 设置JPG图片的分辨率 	
	buf[0x0D] = 1;// - 设置使用图片密度单位 	
				  // - 水平密度，水平分辨率 	
	buf[0x0E] = pResolution[1];
	buf[0x0F] = pResolution[0]; 	// - 垂直密度，垂直分辨率 	
	buf[0x10] = pResolution[1];
	buf[0x11] = pResolution[0];  	// - 将文件指针移动回到文件开头 	
	fseek(file, 0, SEEK_SET); 	// - 将数据写入文件，覆盖原始的数据，让修改生效 	
	fwrite(buf, sizeof(char), len, file);
	fclose(file);
	return true;
}

///验证护照编码
byte Passport_checkSumSubString(byte *src, int len)
{
	int i, chk = 0;
	byte wmap[] = { 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 };
	byte xmap[] = { 0,1,2,3,4,5,6,7,8,9 };
	byte cmap[] = { 7,3,1 };
	int cmapindex = 0;
	for (i = 0; i<len; i++)
	{
		if ((src[i] >= 'A') && (src[i] <= 'Z'))
		{
			chk += wmap[src[i] - 'A'] * cmap[cmapindex++];
		}
		else if ((src[i] >= '0') && (src[i] <= '9'))
		{
			chk += xmap[src[i] - '0'] * cmap[cmapindex++];
		}
		else
		{
			continue;
		}
		cmapindex = cmapindex % 3;
	}

	return (byte)(chk % 10 + '0');
}

///护照冗余校验
bool PassPortCheckCRC(const char *s)
{
	int len = strlen(s);
	if (len<28)return false;
	if (*s != 'E')return false;
	/*char subs[29] = { '\0' };
	strncpy(subs, s, 28);

	//---第一串校验  护照号
	//E006621742CHN0811220H1706192MNPEHPKGHILLA076

	//E006621742
	char Bu[20] = {'\0'};
	strncpy(Bu, subs, 12);
	byte bCrc = Passport_checkSumSubString((byte *)Bu, 9);
	if (bCrc != Bu[9])return  false;
	//0811220
	memset(Bu, '\0', 20);
	strncpy(Bu, subs+13, 7);
	bCrc = Passport_checkSumSubString((byte *)Bu, 6);
	if (bCrc != Bu[6])return  false;
	//1706192
	memset(Bu, '\0', 20);
	strncpy(Bu, subs + 21, 7);
	bCrc = Passport_checkSumSubString((byte *)Bu, 6);
	if (bCrc != Bu[6])return  false;*/

	return true;
}

///名字校验
bool NameCheck(const char *s)
{
	int len = strlen(s);
	if (len < 5)
		return false;
	if (*s != 'P')
		return false;
	/*const char *p = strchr(s, '<');
	if (p == 0)
	return false;
	bool isAlpa = true;
	int prenum = p - s;
	for (int i = 0; i < prenum; i++) {
	if (!isalpha(*(s + i))) {
	isAlpa = false;
	break;
	}
	}
	if (!isAlpa)
	return false;
	if (*(p + 1) != '<' || !isalpha(*(p + 2)))
	return false;
	if (*(s + len - 3) != '<')
	return false;
	*/
	return true;
}

///港澳通行证
bool GangaoPass(const char *s)
{
	//CSC624902070220802807120406
	int len = strlen(s);
	if (len<27)return false;
	//if(s.contains("CS")) return true;
	char ss[4] = { '\0' };
	strncpy(ss, s, 2);
	if (!(isalpha(ss[0]) && isalnum(ss[1])))
		return false;

	char Bu[20] = { '\0' };
	strncpy(Bu, s, 10);
	byte bCrc = Passport_checkSumSubString((byte *)Bu, 9);
	if (bCrc != Bu[9])
		return  false;
	// C62490207
	int n = 2;
	n += 10;

	memset(Bu, '\0', 20);
	strncpy(Bu, s + 12, 7);
	bCrc = Passport_checkSumSubString((byte *)Bu, 6);
	if (bCrc != Bu[6])
		return  false;
	memset(Bu, '\0', 20);
	strncpy(Bu, s + 19, 7);
	bCrc = Passport_checkSumSubString((byte *)Bu, 6);
	if (bCrc != Bu[6])
		return  false;
	return true;
}

void trim(std::string &s)
{
	int index = 0;
	if (!s.empty()) {
		while ((index = s.find(' ', index)) != std::string::npos) {
			s.erase(index, 1);
		}
	}
}

int main()
{

	PassImageProc preProc;
	cv::Mat image;
	std::vector<cv::Mat> procImgs;

	clock_t total_start, total_end;
	clock_t detect_start, detect_end;

#ifdef CAM
	VideoCapture cap(0);

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);

	if (!cap.isOpened()) {
		std::cout << "read camera error!" << std::endl;
		system("pause");
		return -1;
	}
	while (true) {
		total_start = clock();

		clock_t readImg_start, readImg_end;
		readImg_start = clock();

		cap.read(image);

		readImg_end = clock();
		std::cout << "Read image time: " << readImg_end - readImg_start << "ms" << std::endl;

		if (image.empty())
		{
			std::cout << "Frame get failed!" << std::endl;
			continue;
		}

		//Mat *pDstImg = new Mat(frame.rows, frame.cols, frame.type());
		//frame.copyTo(*pDstImg);
		//Mat srcFrame = *((Mat *)pDstImg);

		detect_start = clock();

		preProc.imgProc(image, Size(28, 8));
		procImgs = preProc.getResultImageArray();

		detect_end = clock();
		std::cout << "   Detect time: " << detect_end - detect_start << "ms" << std::endl;
				
		imshow("frame", image);
		char c = waitKey(1);
		if (c == 27)
			break;

		total_end = clock();
		std::cout << "    Total time: " << total_end - total_start << "ms" << std::endl;
	}
	cap.release();	
	destroyAllWindows();
#endif // CAM
	
#ifdef PIC
	string imageList = "pics_new.txt";
	//string imageList = "pics_pp.txt";
	//string imageList = "pics.txt";
	std::ifstream finImage(imageList, std::ios::in);
	string line;

	while (getline(finImage, line))
	{
		std::cout << line << std::endl;
		image = imread(line);
		preProc.imgProc(image, Size(28, 8));
		procImgs = preProc.getResultImageArray();
		char c = waitKey(0) & 0xFF;
		if (c == 'q')
			break;
	}
#endif // PIC

	//system("pause");
	return 0;
}
