# OCR
## 用于辅助多证件读卡器读取证件或者火车票信息
+ 证件：机读码信息，首先找到机读码位置，利用tesseract进行机读码识别，再利用人脸检测对证件上的头像进行检测
+ 火车票：二维码信息，寻找二维码位置，利用ZBAR库进行二维码识别返回识别结果