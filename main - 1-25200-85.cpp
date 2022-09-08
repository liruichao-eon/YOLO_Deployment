#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

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
float scoreThreshold = 0.2;
string labelfile = "label.txt";
//string model = "./models/customed_ori/epoch10.onnx";	// fps=1.55, mem=222mb
string model = "./models/customed_ori/epoch50_tuned.onnx";	// fps=1.6, mem=216mb
string videofile = "./videos/v1.mp4";
vector<string> labels;
bool crowded = false;
int crowdedThreshold = 20;


int nms(const Mat& frame, const Mat& output, vector<int>& classIds, vector<float>& confidences, 
				vector<Rect>& bboxes, vector<int>& indices, bool& crowded) {

	Mat_<float> data(output);

	float x_factor = frame.cols / 640.0;
	float y_factor = frame.rows / 640.0;
	//float y_factor = 1.125;

	for (int i = 0; i < output.size[1]; i++) {
		float confidence = data(0, i, 4);
		if (confidence >= confThreshold) {
			vector<float> classes_scores;
			for (int j = 5; j < 85; j++) {		// printf("%f ", data(0, i, j));
				classes_scores.push_back(data(0, i, j));
			}
			auto max_class_score = max_element(begin(classes_scores), end(classes_scores));
			if (*max_class_score > scoreThreshold) {
				classIds.push_back(distance(begin(classes_scores), max_class_score));
				confidences.push_back(confidence);

				float x = data(0, i, 0);
				float y = data(0, i, 1);
				float w = data(0, i, 2);
				float h = data(0, i, 3);
				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				bboxes.push_back(cv::Rect(left, top, width, height));
			}
		} 
	}
	NMSBoxes(bboxes, confidences, scoreThreshold, nmsThreshold, indices);	// output indices
	if (indices.size() >= crowdedThreshold) {
		*(&crowded) = true;
	}

	return 0;
}


int visualizer(const Mat& frame, vector<int>& classIds, vector<float>& confidences,
				vector<Rect>& bboxes, vector<int>& indices, float fps, bool& crowded) {
	Rect box;
	int cls;
	for (size_t i = 0; i < indices.size(); ++i){
		int idx = indices[i];
		int cls = classIds[idx];
		box = bboxes[idx];
		String confidence = to_string(confidences[idx]);
		String count = to_string(indices.size()) + " Vehicles";
		rectangle(frame, box, Scalar(0, 255, 0), 0.5, LINE_4);
		rectangle(frame, Point(box.x, box.y - 20), Point(box.x + box.width, box.y), Scalar(0, 255, 0), FILLED);
		putText(frame, labels[cls].c_str(), Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		putText(frame, confidence.c_str(), Point(box.x + 45, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0));
		putText(frame, count.c_str(), Point(0, frame.cols), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255));
		if (crowded) {
			putText(frame, "Crowding!", Point(0, frame.cols-20), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255));
		}
	}
	imshow("YOLOv5", frame);
	return 0;
}


//float getFPS(){
//	tm.stop();
//	double fps = counter / tm.getTimeSec();
//	tm.start();
//	return static_cast<float>(fps);
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

	float n = 0;
	float fps = 0;
	clock_t start, end;
	start = clock();


	//namedWindow("YOLOv5", WINDOW_AUTOSIZE);

	while (capture.isOpened()) {
		Mat frame;
		float scalefactor;
		Mat output;
		vector<String> outNames = net.getUnconnectedOutLayersNames();
		//vector<String> outNames = getOutputsNames(net);

		capture >> frame;
		if (!frame.empty()) {
			
			// https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html
			Mat input = blobFromImage(frame, scalefactor=1/255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
			net.setInput(input);
			output = net.forward();	// output, outNames);	// output, outputName	name for layer which output is needed to get

			vector<int> classIds;
			vector<float> confidences;
			vector<Rect> bboxes;
			vector<int> indices;
			
			nms(frame, output, classIds, confidences, bboxes, indices, crowded);
			cout << indices.size() << endl;

			//FPS
			n += 1;
			end = clock();
			fps = 1000 * n / (end - start);
			cout << "FPS: " << fps << endl;

			visualizer(frame, classIds, confidences, bboxes, indices, fps, crowded);
			waitKey(30);
		}
		else {
			if (waitKey(int(1000/30)) >= 0) {
				cout << "ERROR: No image!" << endl;
				return 0;
			}
		}
	}
	//capture.release();


	return 0;
}