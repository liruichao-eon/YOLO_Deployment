#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;
using namespace dnn;

float confThreshold = 0.25;
float nmsThreshold = 0.45;
string labelfile = "label.txt";
string model = "yolov5s.onnx";
string videofile = "v1.mp4";
vector<string> labels;


//int nms() {
//	vector<Rect> nmsBoxes;
//	vector<float> nmsConfidences;
//	vector<int> nmsClassIds;
//
//	vector<Rect> localBoxes;
//	vector<float> localConfidences;
//	vector<size_t> classIndices = it->second;
//	NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
//}

vector<String> getOutputsNames(const Net& net) {
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i) {
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}

//float getFPS(){
//	tm.stop();
//	double fps = counter / tm.getTimeSec();
//	tm.start();
//	return static_cast<float>(fps);
//}

//void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {

//	return;
//}


int main(int argc, char** argv) {
	Net net;
	ifstream f(labelfile.c_str());
	string ln;
	while (getline(f, ln)) {
		labels.push_back(ln);
	}

	// Load model
	ifstream mdl(model.c_str());
	if (mdl.good()) {
		cout << "Model " << model << " loading..." << endl;
		net = readNetFromONNX(model.c_str());
	}
	else {
		cout << "\nERROR: Model " << model << " NOT found!" << endl;
		return -1;
	}

	// Load video
	VideoCapture capture(videofile);
	bool ret = capture.open(videofile, CAP_FFMPEG);//.c_str(), CAP_FFMPEG);		//CAP_FFMPEG
	if (!ret) {
	//if (!capture.isOpened()) {
		cout << "\nERROR: Fail to open video!" << endl;
		return -1;
	}

	double fps = capture.get(CAP_PROP_FPS);


	namedWindow("YOLOv5", WINDOW_AUTOSIZE);

	while (capture.isOpened()) {
		Mat frame;
		float scalefactor;
		vector<Mat> output;
		vector<String> outNames = net.getUnconnectedOutLayersNames();
		//vector<String> outNames = getOutputsNames(net);

		capture >> frame;
		if (!frame.empty()) {
			imshow("YOLOv5", frame);
			// https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html
			Mat input = blobFromImage(frame, scalefactor=1/255.0, Size(640, 640), Scalar(0, 0, 0), false, false);
			net.setInput(input);
			output = net.forward();	// output, outNames);	// output, outputName	name for layer which output is needed to get
			
			cout << output.size() << endl;

			//for (size_t i = 0; i <= output.size(); ++i) {
			//	const uchar* result = output.ptr<uchar>(i);
			//	printf("* result: %d\n", * result);
			//}
			
		}
		else {
			if (waitKey(int(30)) >= 0) {
				cout << "ERROR: No image!" << endl;
				return 0;
			}
		}
	}
	capture.release();





	return 0;
}