// Leave-one-out test; when training data is not enough
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include	<iostream>
#include	<opencv2/opencv.hpp>
//#include	<opencv2/highgui/highgui.hpp>
#include	<fstream>
#include	"Supp.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int, char**) {
	// Section 1: Get the Hu moments from the images
	char		filename[10][128];
	const int	noOfImagePerCol = 1, noOfImagePerRow = 3; // create window partition 
	int			winI = 0, rows, cols;
	char		name[128];
	Mat			srcI, largeWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol], tt;

	int labels[6] = { 1, 1, 1, 2, 2, 2 };
	float trainingData[6][7];
	float dataForValidate[4][7];

	std::fstream fs;
	fs.open("Week9activity_fruits/activitywk9.txt", std::fstream::in);
	for (int j = 0; j < 10; j++) {
		fs >> filename[j];

		srcI = imread(filename[j], IMREAD_GRAYSCALE);
		if (srcI.empty()) {
			cout << "cannot open " << filename << endl;
			return -1;
		}
		rows = srcI.rows, cols = srcI.cols;
		winI = 0;
		createWindowPartition(srcI, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);

		srcI.copyTo(win[winI]);
		putText(legend[winI++], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		threshold(srcI, win[winI], 108, 255, THRESH_OTSU);
		putText(legend[winI++], "threshold", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		win[1].copyTo(tt); // prepare to generate moments from tt (region or edge)

		// generate Hu moments here.
		Moments 	mom;
		double		hu[7];

		mom = moments(tt, true); // get moments of the thresholded image in win[1]
		HuMoments(mom, hu); // generate the Hu moments

		if (j < 6) {
			for (int k = 0; k < 7; k++) {
				trainingData[j][k] = hu[k];
			}
		}
		else {
			for (int k = 0; k < 7; k++) {
				dataForValidate[j - 6][k] = hu[k];
			}
		}

		waitKey(5);
	}

	Mat labelsMat(6, 1, CV_32SC1, labels);
	Mat trainingDataMat(6, 7, CV_32FC1, trainingData);


	// Section 2: Do normalization of the data
	// Xnew = (X - mean) / sigma

	vector<double> mean, sigma;
	for (int i = 0; i < trainingDataMat.cols; i++) {  //take each of the features in vector
		Scalar meanOut, sigmaOut;
		meanStdDev(trainingDataMat.col(i), meanOut, sigmaOut);  //get mean and std deviation
		mean.push_back(meanOut[0]);
		sigma.push_back(sigmaOut[0]);
	}
	for (size_t i = 0; i < trainingDataMat.cols; i++)
		trainingDataMat.col(i) = (trainingDataMat.col(i) - mean[i]) / sigma[i];


	// Section 3: Train the SVM
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);


	// Section 4: check the training and validating data
	Mat			input;

	cout << "Test on training data";
	for (int j = 0; j < 6; j++) { // Test on training data
		if (j % 3 == 0) cout << "\n";
		cout << svm->predict(trainingDataMat.row(j)) << ' ';
	}
	cout << "\n\n";

	cout << "Test on validating data" << endl;
	for (int j = 0; j < 4; j++) {
		for (int k = 0; k < 7; k++)
			// normalize all features using the same mean and sigma from section 2
			dataForValidate[j][k] = (dataForValidate[j][k] - mean[k]) / sigma[k];
		input = (Mat_<float>(1, 7) << dataForValidate[j][0], dataForValidate[j][1], dataForValidate[j][2], dataForValidate[j][3],
			dataForValidate[j][4], dataForValidate[j][5], dataForValidate[j][6]);
		cout << svm->predict(input) << ' ';
	}
	cout << "\n\n";

	system("pause");
	waitKey(0);
}
