#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace dnn;


string labelfile = "label.txt";
string model = "yolov5s.onnx";
string videofile = "v1.mp4";
vector<string> labels;



int main(int argc, char** argv) {
	ifstream f(labelfile.c_str());
	string ln;
	while (getline(f, ln)) {
		labels.push_back(ln);
	}

	// Load model
	ifstream mdl(model.c_str());
	if (mdl.good()) {
		cout << "Model " << model << " found" << endl;
	}
	else {
		cout << "ERROR: Model " << model << " NOT found!" << endl;
		return -1;
	}
	Net net = readNetFromONNX(model.c_str());

	// Load vedio
	VideoCapture capture;
	capture.open("v1.mp4");
	if (!capture.isOpened()) {
		cout << "ERROR: Fail to open video!" << endl;
		return -1;
	}

	namedWindow("YOLOv5", 1);
	for (;;) {
		Mat frame;
		Mat input;
		capture >> frame;
		if (!frame.empty()) {
			imshow("YOLOv5", frame);
			blobFromImage(frame, input, 1/255.0, Size(640, 640));
			//Scalar((0, 0, 0), );

			net.setInput(input);
		}
		else {
			if (waitKey(25) >= 0) {
				cout << "ERROR: No image!" << endl;
				return 0;
			}
		}
	}
	video.release();





	return 0;
}