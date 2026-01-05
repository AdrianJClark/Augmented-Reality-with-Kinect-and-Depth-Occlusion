#ifndef KINECTCALIB

#include <XnOS.h>
#include <XnCppWrapper.h>

//OpenCV
#include "cv.h"
#include "highgui.h"

//OPIRA
#include "CaptureLibrary.h"
#include "OPIRALibrary.h"
#include "OPIRALibraryMT.h"
#include "RegistrationAlgorithms/OCVSurf.h"

//OpenGL
#include "GL/glut.h"
GLuint GLTextureID;

#include "leastsquaresquat.h"

using namespace std; 
using namespace xn;

// ****** Defines ******
#define SAMPLE_XML_PATH "Data/SamplesConfig.xml" // OpenNI config file

// OpenNI Global
Context niContext;
DepthGenerator niDepth;
ImageGenerator niImage;

bool running = true;

int WINDOW_WIDTH = 640, WINDOW_HEIGHT = 480;

void depthMouseFunc(int _event, int x, int y, int flags, void* param);
bool getTransform(IplImage *arImage, CvMat *capParams, CvMat *capDistortion, double **transform);
void draw(IplImage* frame_input, double *projectionMat, double* translationMat, IplImage* depthImage, CvMat* kinectTransform);
bool calcKinectOpenGLTransform(IplImage *colourIm, IplImage* depthIm, CvMat** transform);

bool loadKinectParams(char *filename, CvMat **params, CvMat **distortion);

void inpaintDepth(DepthMetaData *niDepthMD, bool halfSize);

CvPoint* rotateRect(CvRect r, CvPoint2D32f center, float rotAngle);
CvPoint* rotatePoints(CvPoint* p, int cornCount, CvPoint2D32f center, float rotAngle);

CvMat *kinectParams, *kinectDistort;
CvMat *kinectTransform =0;

Registration *arReg;
Registration *kinectReg;

CvSize markerSize;

CvPoint3D32f markCorn[4];

IplImage *debugDepth;

bool useKinect = true;

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutCreateWindow("SimpleTest");

	//Set up Materials 
	GLfloat mat_specular[] = { 0.4, 0.4, 0.4, 1.0 };
	GLfloat mat_diffuse[] = { .8,.8,.8, 1.0 };
	GLfloat mat_ambient[] = { .4,.4,.4, 1.0 };

	glShadeModel(GL_SMOOTH);//smooth shading
	glMatrixMode(GL_MODELVIEW);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialf(GL_FRONT, GL_SHININESS, 100.0);//define the material
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);//enable the material
	glEnable(GL_NORMALIZE);

	//Set up Lights
	GLfloat light0_ambient[] = {0.1, 0.1, 0.1, 0.0};
	float light0_diffuse[] = { 0.8f, 0.8f, 0.8, 1.0f };
	glLightfv (GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glEnable(GL_LIGHT0);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_NORMALIZE);
	glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable (GL_LINE_SMOOTH);	

	debugDepth = cvCreateImage(cvSize(640, 480), IPL_DEPTH_16U, 1);

	markerSize.width = -1; markerSize.height = -1;
	//Initialise the OpenCV Image for GLRendering
	glGenTextures(1, &GLTextureID); 	// Create a Texture object
    glBindTexture(GL_TEXTURE_2D, GLTextureID);  //Use this texture
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	
	glBindTexture(GL_TEXTURE_2D, 0);

	Capture *kinectColour, *kinectDepth;

	if (useKinect) {
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

		loadKinectParams("kinect.yml", &kinectParams, &kinectDistort);
		kinectDistort =0;
		kinectParams->data.db[2]=320.0; kinectParams->data.db[5]=240.0;

		CvFileStorage* fs = cvOpenFileStorage( "KinectTransform.yml", 0, CV_STORAGE_READ );
		if (fs!=0) {
			CvFileNode* fileparams = cvGetFileNodeByName( fs, NULL, "KinectTransform" );
			markerSize.width = cvReadIntByName(fs, 0, "MarkerWidth", -1);
			markerSize.height = cvReadIntByName(fs, 0, "MarkerHeight", -1);
			kinectTransform = (CvMat*)cvRead( fs, fileparams );
			cvReleaseFileStorage( &fs );
		}

		niContext.FindExistingNode(XN_NODE_TYPE_DEPTH, niDepth);
		niContext.FindExistingNode(XN_NODE_TYPE_IMAGE, niImage);

		niDepth.GetMirrorCap().SetMirror(false);

		//Align the depth image and colourImage
		niDepth.GetAlternativeViewPointCap().SetViewPoint(niImage);
	} else {
		kinectColour = new Video("kinect-colour.avi");
		kinectDepth = new Video("kinect-depth.avi");
	}

	Capture *capture = new Camera("camera.yml");
	CvMat *cParams = cvCloneMat(capture->getParameters());
	float scaleFactor = (float)capture->getWidth()/320.0;
	cParams->data.db[0]/= scaleFactor; cParams->data.db[4]/= scaleFactor;
	cParams->data.db[2]=160.0; cParams->data.db[5]=120.0;

	arReg = new RegistrationOPIRA(new OCVSurf());
	if (markerSize.width != -1)
		arReg->addResizedScaledMarker("Celica.bmp", 400,markerSize.width);

	kinectReg = new RegistrationOPIRA(new OCVSurf());
	kinectReg->addResizedMarker("Celica.bmp", 400);

	cvNamedWindow("Depth Image");

	while (running) {
		IplImage *colourIm, *depthIm;
		if (useKinect) {
			if (XnStatus rc = niContext.WaitAnyUpdateAll() != XN_STATUS_OK) {
				printf("Read failed: %s\n", xnGetStatusString(rc));
				return rc;
			}

			// Update MetaData containers
			DepthMetaData niDepthMD; ImageMetaData niImageMD;
			niDepth.GetMetaData(niDepthMD); niImage.GetMetaData(niImageMD);
			//inpaintDepth(&niDepthMD, true);

			// Extract Colour Image
			colourIm = cvCreateImage(cvSize(niImageMD.XRes(), niImageMD.YRes()), IPL_DEPTH_8U, 3);
			memcpy(colourIm->imageData, niImageMD.Data(), colourIm->imageSize); cvCvtColor(colourIm, colourIm, CV_RGB2BGR);
			cvFlip(colourIm, colourIm, 1);

			// Extract Depth Image
			depthIm = cvCreateImage(cvSize(niDepthMD.XRes(), niDepthMD.YRes()), IPL_DEPTH_16U, 1);
			memcpy(depthIm->imageData, niDepthMD.Data(), depthIm->imageSize);

			
			//niDepth.ConvertProjectiveToRealWorld(1);
		} else {
			colourIm = kinectColour->getFrame();
			depthIm = kinectDepth->getFrame();
		}
		
		cvShowImage("Colour Image", colourIm);
		
		switch (cvWaitKey(1)) {
			case 27:
				running = false;
				break;
			case ' ':
				calcKinectOpenGLTransform(colourIm, depthIm, &kinectTransform);
				break;
			case 13:
				if (kinectTransform!=0) {			
					CvFileStorage* fs = cvOpenFileStorage( "KinectTransform.yml", 0, CV_STORAGE_WRITE );
					if (fs!=0) {
						cvWriteInt(fs, "MarkerWidth", markerSize.width);
						cvWriteInt(fs, "MarkerHeight", markerSize.height);
						cvWrite(fs, "KinectTransform", kinectTransform);
						cvReleaseFileStorage( &fs );
					}
				}
				break;
		}
	
		IplImage *_arImage = capture->getFrame();
		IplImage *arImage = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3); cvResize(_arImage, arImage);
		cvShowImage("AR", arImage);
		double* transMat;
		
		if (getTransform(arImage, cParams, 0, &transMat)) {
			double* projMat = OPIRALibrary::calcProjection(cParams, 0, cvSize(320,240));

			draw(arImage, projMat, transMat, depthIm, kinectTransform);

		}
		cvReleaseImage(&arImage); cvReleaseImage(&_arImage);

		cvReleaseImage(&depthIm); cvReleaseImage(&colourIm);
	}


	return 0;

}

bool calcKinectOpenGLTransform(IplImage *colourIm, IplImage* depthIm, CvMat** transform) {
	bool found =false;
	vector<MarkerTransform> mt = kinectReg->performRegistration(colourIm, kinectParams, kinectDistort);
	if (mt.size()>0) {
		//Find the position of the corners on the image
		CvPoint2D32f *markerCorners = (CvPoint2D32f *)malloc(4*sizeof(CvPoint2D32f));
		markerCorners[0] = cvPoint2D32f(0,0); markerCorners[1] = cvPoint2D32f(mt.at(0).marker.size.width,0); 
		markerCorners[2] = cvPoint2D32f(mt.at(0).marker.size.width,mt.at(0).marker.size.height); markerCorners[3] = cvPoint2D32f(0,mt.at(0).marker.size.height);

		CvMat mCorners = cvMat(4,1,CV_32FC2, markerCorners);
		cvPerspectiveTransform(&mCorners, &mCorners, mt.at(0).homography);

		for (int i=0; i<4; i++) {
			if (markerCorners[i].x<0 || markerCorners[i].x>depthIm->width || markerCorners[i].y<0 || markerCorners[i].y>depthIm->height) {
				for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
				free(markerCorners);
				return false;
			}
		}

		//Find the position of the corners in the real world wrt kinect
		XnPoint3D xnCorner[4], xnNewCorner[4];

		for (int i=0; i<4; i++) {
			markCorn[i] = cvPoint3D32f(markerCorners[i].x, markerCorners[i].y, CV_IMAGE_ELEM(depthIm, unsigned short, (int)markerCorners[i].y, (int)markerCorners[i].x));
			xnCorner[i].X = markCorn[i].x; xnCorner[i].Y = markCorn[i].y; xnCorner[i].Z = markCorn[i].z;
		}
		niDepth.ConvertProjectiveToRealWorld(4, xnCorner, xnNewCorner);

		//Calculate width and height of marker in real world
		float width1 = sqrt((xnNewCorner[0].X - xnNewCorner[1].X)*(xnNewCorner[0].X - xnNewCorner[1].X) + (xnNewCorner[0].Y - xnNewCorner[1].Y)*(xnNewCorner[0].Y - xnNewCorner[1].Y) + (xnNewCorner[0].Z - xnNewCorner[1].Z)*(xnNewCorner[0].Z - xnNewCorner[1].Z));
		float width2 = sqrt((xnNewCorner[3].X - xnNewCorner[2].X)*(xnNewCorner[3].X - xnNewCorner[2].X) + (xnNewCorner[3].Y - xnNewCorner[2].Y)*(xnNewCorner[3].Y - xnNewCorner[2].Y) + (xnNewCorner[3].Z - xnNewCorner[2].Z)*(xnNewCorner[3].Z - xnNewCorner[2].Z));
		float height1 = sqrt((xnNewCorner[3].X - xnNewCorner[0].X)*(xnNewCorner[3].X - xnNewCorner[0].X) + (xnNewCorner[3].Y - xnNewCorner[0].Y)*(xnNewCorner[3].Y - xnNewCorner[0].Y) + (xnNewCorner[3].Z - xnNewCorner[0].Z)*(xnNewCorner[3].Z - xnNewCorner[0].Z));
		float height2 = sqrt((xnNewCorner[2].X - xnNewCorner[1].X)*(xnNewCorner[2].X - xnNewCorner[1].X) + (xnNewCorner[2].Y - xnNewCorner[1].Y)*(xnNewCorner[2].Y - xnNewCorner[1].Y) + (xnNewCorner[2].Z - xnNewCorner[1].Z)*(xnNewCorner[2].Z - xnNewCorner[1].Z));
		//markerSize.width = (width1+width2)/2.0; markerSize.height = markerSize.width * ((float)mt.at(0).marker.size.height/(float)mt.at(0).marker.size.width);
		markerSize.height = (height1+height2)/2.0; markerSize.width = markerSize.height * ((float)mt.at(0).marker.size.width/(float)mt.at(0).marker.size.height);
		printf("Marker Size %dx%d\n", markerSize.width, markerSize.height);

		//Render the marker corners
		IplImage *kinectMarker = cvCloneImage(colourIm);
		for (int i=0; i<4; i++) cvCircle(kinectMarker, cvPoint(markerCorners[i].x, markerCorners[i].y), 3, cvScalar(0,0,255), -1);
		cvShowImage("Kinect Found Marker", kinectMarker);
		cvReleaseImage(&kinectMarker);

		free(markerCorners);

		//Calculate Kinect to OpenGL Transform
		{
			vector <CvPoint3D32f> srcPoints3D, dstPoints3D; vector <CvPoint2D32f> srcPoints2D, dstPoints2D;
			srcPoints2D.resize(50); srcPoints3D.resize(50); dstPoints2D.resize(50); dstPoints3D.resize(50);
			float xStep = float(markerSize.width)/9.0; float yStep = float(markerSize.height)/4.0;
			float xStep1 = float(mt.at(0).marker.size.width)/9.0; float yStep1 = float(mt.at(0).marker.size.height)/4.0;
			for (int y=0; y<5; y++) {
				for (int x=0; x<10; x++) {
					int index = x+(y*10);
					srcPoints3D.at(index) = cvPoint3D32f(x*xStep, y*yStep, 0);
					srcPoints2D.at(index) = cvPoint2D32f(x*xStep1, y*yStep1);
				}
			}
			
			CvMat mSrcCorners = cvMat(50,1,CV_32FC2, &srcPoints2D[0]); CvMat mDstCorners = cvMat(50,1,CV_32FC2, &dstPoints2D[0]);
			cvPerspectiveTransform(&mSrcCorners, &mDstCorners, mt.at(0).homography);

			XnPoint3D _xnCorner[50], _xnNewCorner[50]; 
			for (int i=0; i<50; i++) {_xnCorner[i].X = dstPoints2D[i].x; _xnCorner[i].Y = dstPoints2D[i].y; _xnCorner[i].Z = CV_IMAGE_ELEM(depthIm, unsigned short, (int)_xnCorner[i].Y, (int)_xnCorner[i].X);}
			niDepth.ConvertProjectiveToRealWorld(50, _xnCorner, _xnNewCorner);
			for (int i=0; i<50; i++) {dstPoints3D[i] = cvPoint3D32f(_xnNewCorner[i].X, _xnNewCorner[i].Y, _xnNewCorner[i].Z);}

			*transform = findTransform(dstPoints3D, srcPoints3D);
			for (int y=0; y<4; y++) {
				for (int x=0; x<4; x++) {
					printf("%.2f\t", CV_MAT_ELEM((**transform), float, y,x));
				}
				printf("\n");
			}
		}

		/*{
			vector <CvPoint3D32f> srcPoints, dstPoints; srcPoints.resize(4); dstPoints.resize(4);
			srcPoints.at(0).x = 0; srcPoints.at(0).y = 0; srcPoints.at(0).z = 0;
			srcPoints.at(1).x = markerSize.width; srcPoints.at(1).y = 0; srcPoints.at(1).z = 0;
			srcPoints.at(2).x = markerSize.width; srcPoints.at(2).y = markerSize.height; srcPoints.at(2).z = 0;
			srcPoints.at(3).x = 0; srcPoints.at(3).y = markerSize.height; srcPoints.at(3).z = 0;

			for (int i=0; i<4; i++) {
				dstPoints.at(i).x = xnNewCorner[i].X; dstPoints.at(i).y = xnNewCorner[i].Y; dstPoints.at(i).z = xnNewCorner[i].Z;
			}
			*transform = findTransform(dstPoints, srcPoints);
			for (int y=0; y<4; y++) {
				for (int x=0; x<4; x++) {
					printf("%.2f\t", CV_MAT_ELEM((**transform), float, y,x));
				}
				printf("\n");
			}
		}*/

		//Load in the marker for registration
		arReg->removeMarker("Celica.bmp");
		arReg->addResizedScaledMarker("Celica.bmp", 400, markerSize.width);

		found = true;
		}

	for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
	return found;
}




void depthMouseFunc(int _event, int x, int y, int flags, void* param) {
	if (_event == CV_EVENT_MOUSEMOVE) {
		IplImage *depthIm = (IplImage*)param;
//		printf ("%d, %d, %d\n", x, y, CV_IMAGE_ELEM(depthIm, unsigned short, y, x));
	}

}

bool getTransform(IplImage *arImage, CvMat *capParams, CvMat *capDistortion, double **transform) {
	bool found = false;

	vector<MarkerTransform> mt = arReg->performRegistration(arImage, capParams, capDistortion);
	
	if (mt.size()>0) {
		found = true;

		markerSize = mt.at(0).marker.size;

		*transform = (double*)malloc(16*sizeof(double));
		memcpy(*transform, mt.at(0).transMat, 16*sizeof(double));
		
		CvPoint2D32f *markerCorners = (CvPoint2D32f *)malloc(4*sizeof(CvPoint2D32f));
		markerCorners[0] = cvPoint2D32f(0,0); markerCorners[1] = cvPoint2D32f(mt.at(0).marker.size.width,0); 
		markerCorners[2] = cvPoint2D32f(mt.at(0).marker.size.width,mt.at(0).marker.size.height); markerCorners[3] = cvPoint2D32f(0,mt.at(0).marker.size.height);

		CvMat mCorners = cvMat(4,1,CV_32FC2, markerCorners);
		cvPerspectiveTransform(&mCorners, &mCorners, mt.at(0).homography);

		IplImage *tmpIm = cvCloneImage(arImage);
		for (int i=0; i<4; i++) {
			cvCircle(tmpIm, cvPoint(markerCorners[i].x, markerCorners[i].y), 3, cvScalar(0,0,255), -1);
		}
		cvShowImage("AR Found Marker", tmpIm); cvReleaseImage(&tmpIm);
	}

	for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
	return found;
}




void draw(IplImage* frame_input, double *projectionMat, double* translationMat, IplImage* depthImage, CvMat* kinectTransform)
{
	//Clear the depth buffer 
	glClearDepth( 1.0 ); glClear(GL_DEPTH_BUFFER_BIT); glDepthFunc(GL_LEQUAL);

	//Set the viewport to the window size
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    //Set the Projection Matrix to an ortho slightly larger than the window
	glMatrixMode(GL_PROJECTION); glLoadIdentity();
	glOrtho(-0.5, WINDOW_WIDTH-0.5, WINDOW_HEIGHT-0.5, -0.5, 1.0, -1.0);
    //Set the modelview to the identity
	glMatrixMode(GL_MODELVIEW); glLoadIdentity();

	//Turn off Light and enable a texture
	glDisable(GL_LIGHTING);	glEnable(GL_TEXTURE_2D); glDisable(GL_DEPTH_TEST);

	glBindTexture(GL_TEXTURE_2D, GLTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, frame_input->width, frame_input->height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, frame_input->imageData);
	
	//Draw the background
	glPushMatrix();
        glColor3f(255, 255, 255);
        glBegin(GL_TRIANGLE_STRIP);
            glTexCoord2f(0.0, 0.0);	glVertex2f(0.0, 0.0);
            glTexCoord2f(1.0, 0.0);	glVertex2f(WINDOW_WIDTH, 0.0);
            glTexCoord2f(0.0, 1.0);	glVertex2f(0.0, WINDOW_HEIGHT);
            glTexCoord2f(1.0, 1.0);	glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT);
        glEnd();
	glPopMatrix();

	//Turn off Texturing
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D); glEnable(GL_DEPTH_TEST);

	//Loop through all the markers found

		//Set the Viewport Matrix
		glViewport(0,0,WINDOW_WIDTH,WINDOW_HEIGHT);

		//Load the Projection Matrix
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixd( projectionMat );

		//Load the camera modelview matrix 
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixd( translationMat );

		//Draw the boundry rectangle
		glColor3f(0,0,0);
		glBegin(GL_LINE_LOOP);
			glVertex3d(0,					0,0);
			glVertex3d(markerSize.width,	0,0);
			glVertex3d(markerSize.width,	markerSize.height,0);
			glVertex3d(0,					markerSize.height,0);
		glEnd();


		int display = 2;

		if (kinectTransform!=0) {

			glEnable(GL_DEPTH_TEST); glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
			
						vector<XnPoint3D> points;
						
						for (int y=0; y<480; y++) {
							for (int x=0; x<640; x++) {
								int z = CV_IMAGE_ELEM(depthImage, unsigned short, y, x);
								XnPoint3D _xnCorner; _xnCorner.X = x; _xnCorner.Y = y; _xnCorner.Z = z; 
								points.push_back(_xnCorner);
							}
						}
						
						vector<XnPoint3D> realPoints; realPoints.resize(points.size());
						niDepth.ConvertProjectiveToRealWorld(points.size(), &points[0], &realPoints[0]);

						float myDat[4]; CvMat myPoint = cvMat(4, 1, CV_32FC1, myDat);
						for (int i=0; i<realPoints.size(); i++) {
							myDat[0] = realPoints[i].X; myDat[1] = realPoints[i].Y; myDat[2] = realPoints[i].Z; myDat[3] = 1; 
							cvMatMul(kinectTransform, &myPoint, &myPoint);
							if (myDat[3]!=1) {	myDat[0] /=myDat[3]; myDat[1] /=myDat[3]; myDat[2] /=myDat[3]; }
							realPoints[i].X = myDat[0]; realPoints[i].Y = myDat[1]; realPoints[i].Z = -myDat[2];
						}

						IplImage *tmpIm = cvCreateImage(cvSize(640,480), IPL_DEPTH_8U, 1); cvZero(tmpIm);

						for (int y=0; y<480; y++) {
							for (int x=0; x<640; x++) {
								int i = x+(y*640);
								if (points.at(i).Z !=0 &&  realPoints.at(i).Z<-10) CV_IMAGE_ELEM(tmpIm, char, y, x) =255;
							}
						}

						cvErode(tmpIm, tmpIm,0,3);

						IplImage *myIm = cvCreateImage(cvGetSize(tmpIm), IPL_DEPTH_8U, 3);
						cvMerge(tmpIm, tmpIm, tmpIm, 0, myIm);

						CvMemStorage *storage = cvCreateMemStorage(0);
						CvSeq *contours;
						cvFindContours(tmpIm, storage, &contours);

						cvDrawContours(myIm, contours, cvScalar(0,0,255), cvScalarAll(0), 2);


						for (CvSeq *c=contours; c!=NULL; c=c->h_next) {
							glBegin(GL_QUADS);
							for (int i=0; i<c->total-1; i++) {
								CvPoint *p1 = (CvPoint*)cvGetSeqElem(c, i);
								CvPoint *p2 = (CvPoint*)cvGetSeqElem(c, i+1);
								int i1 = p1->x+(p1->y*640); int i2 = p2->x+(p2->y*640);
								//glVertex3f(realPoints[i1].X, realPoints[i1].Y, 0); glVertex3f(realPoints[i2].X, realPoints[i2].Y, 0);
								//glVertex3f(realPoints[i1].X, realPoints[i1].Y, realPoints[i1].Z); glVertex3f(realPoints[i2].X, realPoints[i2].Y, realPoints[i2].Z);
								glVertex3f(realPoints[i1].X, realPoints[i1].Y, 0); glVertex3f(realPoints[i1].X, realPoints[i1].Y, realPoints[i1].Z);
								glVertex3f(realPoints[i2].X, realPoints[i2].Y, realPoints[i2].Z); glVertex3f(realPoints[i2].X, realPoints[i2].Y, 0); 
							}
							glEnd();
						}

						cvReleaseMemStorage(&storage);
						cvShowImage("Depth", tmpIm); cvShowImage("Contours", myIm);
						cvReleaseImage(&tmpIm); cvReleaseImage(&myIm);

			glDisable(GL_DEPTH_TEST); glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		}

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
			glTranslatef(markerSize.width/2.0, markerSize.height/2.0,-25);
			glRotatef(-90, 1.0,0,0.0);
			glColor3f(1,0,0);
			glutSolidTeapot(50.0);
		glDisable(GL_LIGHTING);
	
	//Copy the OpenGL Graphics context into an IPLImage
	IplImage* outImage = cvCreateImage(cvSize(WINDOW_WIDTH,WINDOW_HEIGHT), IPL_DEPTH_8U, 3);
	glReadPixels(0,0,WINDOW_WIDTH,WINDOW_HEIGHT,GL_RGB, GL_UNSIGNED_BYTE, outImage->imageData);
	cvCvtColor( outImage, outImage, CV_BGR2RGB );
	cvFlip(outImage, outImage);

	cvNamedWindow("Registered Image"), cvShowImage("Registered Image", outImage);
	cvReleaseImage(&outImage);
}

bool loadKinectParams(char *filename, CvMat **params, CvMat **distortion) {
	CvFileStorage* fs = cvOpenFileStorage( filename, 0, CV_STORAGE_READ );
	if (fs==0) return false; 

	CvFileNode* fileparams;
	//Read the Camera Parameters
	fileparams = cvGetFileNodeByName( fs, NULL, "camera_matrix" );
	*params = (CvMat*)cvRead( fs, fileparams );

	//Read the Camera Distortion 
	fileparams = cvGetFileNodeByName( fs, NULL, "distortion_coefficients" );
	*distortion = (CvMat*)cvRead( fs, fileparams );
	cvReleaseFileStorage( &fs );

	return true;
}


CvPoint* rotateRect(CvRect r, CvPoint2D32f center, float rotAngle) {
	//Copy the corners
	CvPoint corners[4];
	corners[0].x = r.x; corners[0].y = r.y; 
	corners[1].x = r.x+r.width; corners[1].y = r.y; 
	corners[2].x = r.x+r.width; corners[2].y = r.y+r.height; 
	corners[3].x = r.x; corners[3].y = r.y+r.height; 

	//Call rotate points
	return rotatePoints(corners, 4, center, rotAngle);
}

CvPoint* rotatePoints(CvPoint* p, int cornCount, CvPoint2D32f center, float rotAngle) {
	CvPoint *rotCorners = (CvPoint*)malloc(sizeof(CvPoint)*cornCount);

	if (rotAngle == 0) {
		memcpy(rotCorners, p, sizeof(CvPoint)*cornCount);
		return rotCorners;
	}

	//Create the rotation matrix
	CvMat *rotMat = cvCreateMat(2,3,CV_32FC1);
	cv2DRotationMatrix(center, rotAngle, 1, rotMat);

	//Create the matrix of pre-rotated corners
	CvMat *corners = cvCreateMat(cornCount,1,CV_32FC2);
	for (int i=0; i<cornCount; i++) {
		corners->data.fl[i*2] = p[i].x; corners->data.fl[i*2+1] = p[i].y; 
	}

	//Perform the rotation
	CvMat *cornersTransformed = cvCreateMat(cornCount,1,CV_32FC2);
	cvTransform(corners, cornersTransformed, rotMat);

	//Copy the data out
	for (int i=0; i<cornCount; i++) {
		rotCorners[i].x = cornersTransformed->data.fl[i*2]; rotCorners[i].y = cornersTransformed->data.fl[i*2+1];
	}

	//Clean up
	cvReleaseMat(&corners);
	cvReleaseMat(&cornersTransformed);
	cvReleaseMat(&rotMat);

	return rotCorners;
}

void inpaintDepth(DepthMetaData *niDepthMD, bool halfSize) {
	IplImage *depthIm, *depthImFull;
	
	if (halfSize) {
		depthImFull = cvCreateImage(cvSize(niDepthMD->XRes(), niDepthMD->YRes()), IPL_DEPTH_16U, 1);
		depthImFull->imageData = (char*)niDepthMD->WritableData();
		depthIm = cvCreateImage(cvSize(depthImFull->width/2.0, depthImFull->height/2.0), IPL_DEPTH_16U, 1);
		cvResize(depthImFull, depthIm, 0);
	} else {
		depthIm = cvCreateImage(cvSize(niDepthMD->XRes(), niDepthMD->YRes()), IPL_DEPTH_16U, 1);
		depthIm->imageData = (char*)niDepthMD->WritableData();
	}
	
	IplImage *depthImMask = cvCreateImage(cvGetSize(depthIm), IPL_DEPTH_8U, 1);
	for (int y=0; y<depthIm->height; y++) {
		for (int x=0; x<depthIm->width; x++) {
			CV_IMAGE_ELEM(depthImMask, char, y, x)=CV_IMAGE_ELEM(depthIm, unsigned short,y,x)==0?255:0;
		}
	}

	IplImage *depthImMaskInv = cvCreateImage(cvGetSize(depthIm), IPL_DEPTH_8U, 1);
	cvNot(depthImMask, depthImMaskInv);

	double min, max; cvMinMaxLoc(depthIm, &min, &max, 0, 0, depthImMaskInv);
	
	IplImage *depthIm8 = cvCreateImage(cvGetSize(depthIm), IPL_DEPTH_8U, 1);
	float scale = 255.0/(max-min);
	cvConvertScale(depthIm, depthIm8, scale, -(min*scale));

	IplImage *depthPaint = cvCreateImage(cvGetSize(depthIm8), IPL_DEPTH_8U, 1);
	cvInpaint(depthIm8, depthImMask, depthPaint, 3, CV_INPAINT_NS);
	
	IplImage *depthIm16 = cvCreateImage(cvGetSize(depthIm), IPL_DEPTH_16U, 1);
	cvConvertScale(depthPaint, depthIm16, 1/scale, min);

	

	if (halfSize) {
		IplImage *depthPaintedFull = cvCreateImage(cvGetSize(depthImFull), IPL_DEPTH_16U, 1);
		cvResize(depthIm16, depthPaintedFull,0);
		IplImage *depthImMaskFull = cvCreateImage(cvGetSize(depthImFull), IPL_DEPTH_8U, 1);
		for (int y=0; y<depthImFull->height; y++) for (int x=0; x<depthImFull->width; x++)
			CV_IMAGE_ELEM(depthImMaskFull, char, y, x)=CV_IMAGE_ELEM(depthImFull, unsigned short,y,x)==0?255:0;
		cvCopy(depthPaintedFull, depthImFull, depthImMaskFull);
		cvReleaseImage(&depthPaintedFull); cvReleaseImage(&depthImMaskFull);
		cvReleaseImage(&depthImFull);
	} else {
		cvCopy(depthIm16, depthIm, depthImMask);
	}

	cvReleaseImage(&depthIm8); cvReleaseImage(&depthIm16);
	cvReleaseImage(&depthPaint);
	cvReleaseImage(&depthImMask); cvReleaseImage(&depthImMaskInv);
	cvReleaseImage(&depthIm);
}

#endif

