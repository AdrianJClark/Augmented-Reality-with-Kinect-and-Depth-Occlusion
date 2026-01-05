#ifdef KINECTCALIB
#include <XnOS.h>
#include <XnCppWrapper.h>

#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <iostream>

using namespace std; 
using namespace xn;

// ****** Defines ******
#define SAMPLE_XML_PATH "Data/SamplesConfig.xml" // OpenNI config file

// OpenNI Global
Context niContext;
ImageGenerator niImage;

using namespace std;

void captureFrame(IplImage* frame, vector<CvPoint2D32f> Corners, int CornerCount);
void computeError(const CvMat* object_points, const CvMat* rot_vects, const CvMat* trans_vects, const CvMat* camera_matrix, const CvMat* dist_coeffs, const CvMat* image_points, const CvMat* point_counts);
void calibrateCamera(CvSize imgSize);

const int chessboard_width = 8;
const int chessboard_height = 6;
const float squareSize = 2.9f;

int numberFrames = 0;
vector<CvPoint2D32f> allCorners;

int main() {
	EnumerationErrors errors;
	switch (XnStatus rc = niContext.InitFromXmlFile(SAMPLE_XML_PATH, &errors)) {
		case XN_STATUS_OK:
			break;
		case XN_STATUS_NO_NODE_PRESENT:
			XnChar strError[1024];	errors.ToString(strError, 1024);
			printf("%s\n", strError);
			return rc; break;
		default:
			printf("Open failed: %s\n", xnGetStatusString(rc));
			return rc;
	}

	niContext.FindExistingNode(XN_NODE_TYPE_IMAGE, niImage);

	vector<CvPoint2D32f> Corners; Corners.resize(chessboard_width* chessboard_height);
	int CornerCount;

	while (true) {
		if (XnStatus rc = niContext.WaitAnyUpdateAll() != XN_STATUS_OK) {
			printf("Read failed: %s\n", xnGetStatusString(rc));
			return rc;
		}

		// Update MetaData containers
		ImageMetaData niImageMD;
		niImage.GetMetaData(niImageMD);

		// Extract Colour Image
		IplImage *new_frame = cvCreateImage(cvSize(niImageMD.XRes(), niImageMD.YRes()), IPL_DEPTH_8U, 3);
		memcpy(new_frame->imageData, niImageMD.Data(), new_frame->imageSize); cvCvtColor(new_frame, new_frame, CV_RGB2BGR);

		int Result = cvFindChessboardCorners(new_frame, cvSize(chessboard_width, chessboard_height), &Corners[0], &CornerCount);

		IplImage *imgCorners = cvCloneImage(new_frame);
		cvDrawChessboardCorners(imgCorners, cvSize(chessboard_width, chessboard_height), &Corners[0], CornerCount, Result);

		cvNamedWindow("Input Images"); cvShowImage("Input Images", imgCorners);

		cvReleaseImage(&imgCorners);

		switch (cvWaitKey(1)) {
			case 27:
				exit(0); break;
			case ' ':
				if (Result) {
					cout << "Capturing Image" << endl;
					captureFrame(new_frame, Corners, CornerCount);
				}
				break;
			case 13:
				calibrateCamera(cvGetSize(new_frame));
		}
		cvReleaseImage(&new_frame); 
	}

}


void captureFrame(IplImage* frame, vector<CvPoint2D32f> Corners, int CornerCount) {

	IplImage *fBW = cvCreateImage(cvGetSize(frame), frame->depth, 1); cvConvertImage(frame, fBW);
	cvFindCornerSubPix(fBW, &Corners[0], CornerCount, cvSize(11,11), cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01) );
	cvReleaseImage(&fBW);


	allCorners.insert(allCorners.end(), Corners.begin(), Corners.end());
	numberFrames++;
}

void calibrateCamera(CvSize imgSize) {
	//Initialise the chessboard point array
	vector<CvPoint3D32f> chessboardPoints; chessboardPoints.resize(chessboard_width * chessboard_height);
	for (int i=0; i<chessboard_height; i++) 
		for (int j=0; j<chessboard_width; j++) 
			chessboardPoints[i*chessboard_width+j] = cvPoint3D32f(i*squareSize, j*squareSize, 0);

	//Fill up all the points
	vector<CvPoint3D32f> allChessboardPoints;
	for (int i=0; i<numberFrames; i++) 
		allChessboardPoints.insert(allChessboardPoints.end(), chessboardPoints.begin(), chessboardPoints.end());

	int totalPointCount = chessboard_width * chessboard_height * numberFrames;

	CvMat mChessboardPoints = cvMat(1, totalPointCount, CV_32FC3, &allChessboardPoints[0] );
	CvMat mImagePoints = cvMat(1, totalPointCount, CV_32FC2, &allCorners[0] );
	
	vector<int> pointCount; pointCount.resize(numberFrames, chessboard_width * chessboard_height);
	CvMat mpointCount = cvMat(1, pointCount.size(), CV_32S, &pointCount[0] );
	
    double _camera[9], _dist_coeffs[4];
    CvMat mCamMatrix = cvMat( 3, 3, CV_64F, _camera );
    CvMat mDistCoeffs = cvMat( 1, 4, CV_64F, _dist_coeffs );

	CvMat *extr_params = cvCreateMat( numberFrames, 6, CV_32FC1 );
	CvMat rot_vects, trans_vects;
    cvGetCols( extr_params, &rot_vects, 0, 3 );
    cvGetCols( extr_params, &trans_vects, 3, 6 );

    cout << "Running stereo calibration ..." << endl;
	cvCalibrateCamera2(&mChessboardPoints, &mImagePoints, &mpointCount, imgSize, &mCamMatrix, &mDistCoeffs, &rot_vects, &trans_vects);
    cout << "Done." << endl;

	computeError(&mChessboardPoints, &rot_vects, &trans_vects, &mCamMatrix, &mDistCoeffs, &mImagePoints, &mpointCount);

    // save intrinsic parameters
    CvFileStorage* fstorage = cvOpenFileStorage("kinect.yml", NULL, CV_STORAGE_WRITE);
    cvWriteInt( fstorage, "image_width", imgSize.width );
    cvWriteInt( fstorage, "image_height", imgSize.height );
    cvWrite( fstorage, "camera_matrix", &mCamMatrix );
    cvWrite( fstorage, "distortion_coefficients", &mDistCoeffs );
    cvReleaseFileStorage(&fstorage);


}

void computeError(const CvMat* object_points, const CvMat* rot_vects, const CvMat* trans_vects,
        const CvMat* camera_matrix, const CvMat* dist_coeffs, const CvMat* image_points, const CvMat* point_counts)
{
    CvMat* image_points2 = cvCreateMat( image_points->rows, image_points->cols, image_points->type );
    double total_err = 0; int points_so_far = 0;
    
    for(int i = 0; i < numberFrames; i++ )
    {
        CvMat object_points_i, image_points_i, image_points2_i;
        int point_count = point_counts->data.i[i];
        CvMat rot_vect, trans_vect;

        cvGetCols( object_points, &object_points_i, points_so_far, points_so_far + point_count );
        cvGetCols( image_points,  &image_points_i,  points_so_far, points_so_far + point_count );
        cvGetCols( image_points2, &image_points2_i, points_so_far, points_so_far + point_count );
        points_so_far += point_count;

        cvGetRow( rot_vects, &rot_vect, i );
        cvGetRow( trans_vects, &trans_vect, i );

        cvProjectPoints2( &object_points_i, &rot_vect, &trans_vect, camera_matrix, dist_coeffs, &image_points2_i);

		total_err += cvNorm( &image_points_i, &image_points2_i, CV_L1 );
    }
    
    cvReleaseMat( &image_points2 );
    printf( "avg err = %g\n", total_err/points_so_far );
}
#endif