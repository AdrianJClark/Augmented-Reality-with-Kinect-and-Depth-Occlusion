bool getTransformChessboard(IplImage *arImage, CvMat *capParams, CvMat *capDistortion, double **transform) {
	CvPoint2D32f *imageCorners = (CvPoint2D32f *)malloc(6*8*sizeof(CvPoint2D32f)); int imageCornerCount;
	int wasFound = cvFindChessboardCorners(arImage, cvSize(8,6), imageCorners, &imageCornerCount);
	if (wasFound) {
		//Set up the OpenCV transformation matrix as an identity matrix
		CvMat *cvTransMat = cvCreateMat(4,4, CV_32FC1); cvSetIdentity(cvTransMat);

		cvDrawChessboardCorners(arImage, cvSize(8,6), imageCorners, imageCornerCount, wasFound);
		cvShowImage("AR Chessboard", arImage);

		CvPoint2D32f *markerCorners = (CvPoint2D32f *)malloc(6*8*sizeof(CvPoint2D32f));
		for(int y=0; y<6; y++) for (int x=0; x<8; x++) markerCorners[x+(y*8)] = cvPoint2D32f(x*29, y*29);

		CvMat mImage = cvMat(48,1,CV_32FC2, imageCorners);
		CvMat mMarker = cvMat(48,1,CV_32FC2, markerCorners);

		//Set the translation vector and rotation matrix to point inside the cvTransMat struct
		CvMat *mTranslation = cvCreateMatHeader(1, 3, CV_32F); cvGetSubRect(cvTransMat, mTranslation, cvRect(3,0,1,3));
		CvMat *mRotation = cvCreateMatHeader(3,3, CV_32FC1); cvGetSubRect(cvTransMat, mRotation, cvRect(0,0,3,3));

		CvMat *rotVector= cvCreateMat(1, 3, CV_32F); 
		cvFindExtrinsicCameraParams2(&mMarker, &mImage, capParams, capDistortion, rotVector, mTranslation);
		cvRodrigues2(rotVector, mRotation);
		
		cvReleaseMat(&rotVector);

		*transform = calcTransform(cvTransMat);

		cvReleaseMat(&mRotation); cvReleaseMat(&mTranslation);
		free(markerCorners);
	}
	free(imageCorners);
	return wasFound;
}


bool regGroundPlane(IplImage *colourIm, IplImage* depthIm) {
	bool found =false;
	vector<MarkerTransform> mt = kinectReg->performRegistration(colourIm, kinectParams, kinectDistort);
	if (mt.size()>0) {

		{
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

			IplImage *tmpIm = cvCloneImage(colourIm);
			for (int i=0; i<4; i++) {
				cvCircle(tmpIm, cvPoint(markerCorners[i].x, markerCorners[i].y), 3, cvScalar(0,0,255), -1);
				markCorn[i] = cvPoint3D32f(markerCorners[i].x, markerCorners[i].y, CV_IMAGE_ELEM(depthIm, unsigned short, (int)markerCorners[i].y, (int)markerCorners[i].x));
			}

			cvShowImage("Kinect Found Marker", tmpIm); cvReleaseImage(&tmpIm);
			free(markerCorners);
		}

		float _p[4]; _p[0] = _p[1] = _p[2] = 0; _p[3] = 1;
		CvMat p = cvMat(4,1, CV_32FC1, _p);

		float _po[4];
		CvMat po = cvMat(4,1, CV_32FC1, _po);

		float _tm[16]; for (int i=0; i<16; i++) _tm[i] = mt.at(0).transMat[i];

		CvMat tm = cvMat(4, 4, CV_32FC1, _tm);

		CvMat *reflect = cvCreateMat(4,4,CV_32FC1); cvZero(reflect);
		reflect->data.fl[0] = reflect->data.fl[15] = 1; reflect->data.fl[5] = reflect->data.fl[10] = -1;
		cvMatMul(&tm, reflect, &tm);
		cvTranspose(&tm, &tm);

		cvMatMul(&tm,&p, &po);

		printf("%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n%.2f, %.2f, %.2f, %.2f\n\n",
			_tm[0], _tm[1], _tm[2], _tm[3], _tm[4], _tm[5], _tm[6], _tm[7], _tm[8], _tm[9], _tm[10], _tm[11], _tm[12], _tm[13], _tm[14], _tm[15]);

		printf("%f, %f, %f, %f\n\n", _po[0], _po[1], _po[2], _po[3]);

		float _p3d[3]; _p3d[0] = _po[0]/_po[3]; _p3d[1] = _po[1]/_po[3]; _p3d[2] = _po[2]/_po[3];
		CvMat p3d = cvMat(1, 1, CV_32FC3, _p3d);

		float _p2d[2];
		CvMat p2d = cvMat(1,1,CV_32FC2, _p2d);

		CvMat *transVector = cvCreateMatHeader(1, 3, CV_32F); cvGetSubRect(&tm, transVector, cvRect(3,0,1,3));
		CvMat *rotMat = cvCreateMatHeader(3,3, CV_32FC1); cvGetSubRect(&tm, rotMat, cvRect(0,0,3,3));
		CvMat *rotVector= cvCreateMat(1, 3, CV_32F); cvRodrigues2(rotMat, rotVector);

		cvProjectPoints2(&p3d, rotVector, transVector, kinectParams, kinectDistort, &p2d);

		printf("%f, %f, %f - %f, %f\n\n\n", _p3d[0], _p3d[1], _p3d[2], _p2d[0], _p2d[1]);

		found = true;

	}

	for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
	return found;
}

bool calcGroundPlane(IplImage *colourIm, IplImage* depthIm, CvMat** homography) {
	bool found =false;
	vector<MarkerTransform> mt = kinectReg->performRegistration(colourIm, kinectParams, kinectDistort);
	if (mt.size()>0) {
		CvPoint2D32f *markerCorners = (CvPoint2D32f *)malloc(4*sizeof(CvPoint2D32f));
		markerCorners[0] = cvPoint2D32f(0,0); markerCorners[1] = cvPoint2D32f(mt.at(0).marker.size.width,0); 
		markerCorners[2] = cvPoint2D32f(mt.at(0).marker.size.width,mt.at(0).marker.size.height); markerCorners[3] = cvPoint2D32f(0,mt.at(0).marker.size.height);

//		printf("%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\n", markerCorners[0].x, markerCorners[0].y, markerCorners[1].x, markerCorners[1].y, markerCorners[2].x, markerCorners[2].y, markerCorners[3].x, markerCorners[3].y);
		CvMat mCorners = cvMat(4,1,CV_32FC2, markerCorners);
		cvPerspectiveTransform(&mCorners, &mCorners, mt.at(0).homography);
//		printf("%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\n", markerCorners[0].x, markerCorners[0].y, markerCorners[1].x, markerCorners[1].y, markerCorners[2].x, markerCorners[2].y, markerCorners[3].x, markerCorners[3].y);


		for (int i=0; i<4; i++) {
			if (markerCorners[i].x<0 || markerCorners[i].x>depthIm->width || markerCorners[i].y<0 || markerCorners[i].y>depthIm->height) {
				for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
				free(markerCorners);
				return false;
			}
		}

		IplImage *tmpIm = cvCloneImage(colourIm);
		for (int i=0; i<4; i++) {
			cvCircle(tmpIm, cvPoint(markerCorners[i].x, markerCorners[i].y), 3, cvScalar(0,0,255), -1);
			markCorn[i] = cvPoint3D32f(markerCorners[i].x, markerCorners[i].y, CV_IMAGE_ELEM(depthIm, unsigned short, (int)markerCorners[i].y, (int)markerCorners[i].x));
//			printf("%d, %d\n", i, markCorn[i].z);
		}

//		printf("\n");
		cvShowImage("Kinect Found Marker", tmpIm); cvReleaseImage(&tmpIm);
		free(markerCorners);

		found = true;
		*homography = cvCloneMat(mt.at(0).homography);

		}

	for (int i=0; i<mt.size(); i++) mt.at(i).clear(); mt.clear();
	return found;
}

bool calcGroundPlaneChessBoard(IplImage *colourIm, IplImage* depthIm, CvMat** homography) {
	CvPoint2D32f *imageCorners = (CvPoint2D32f *)malloc(6*8*sizeof(CvPoint2D32f)); int imageCornerCount;
		int wasFound = cvFindChessboardCorners(colourIm, cvSize(8,6), imageCorners, &imageCornerCount);
		if (wasFound) {

			CvPoint2D32f *markerCorners = (CvPoint2D32f *)malloc(6*8*sizeof(CvPoint2D32f));
			for(int y=0; y<6; y++) for (int x=0; x<8; x++) markerCorners[x+(y*8)] = cvPoint2D32f(x*29, y*29);

			*homography = cvCreateMat(3,3, CV_32FC1);
			CvMat mImage = cvMat(48,1,CV_32FC2, imageCorners); CvMat mMarker = cvMat(48,1,CV_32FC2, markerCorners);
			cvFindHomography(&mImage, &mMarker, *homography);
			free(markerCorners);

			for (int i=0; i<8*6; i++) {
				markCorn[i] = cvPoint3D32f(markerCorners[i].x, markerCorners[i].y, CV_IMAGE_ELEM(depthIm, unsigned short, (int)markerCorners[i].y, (int)markerCorners[i].x));
			}
			
			cvDrawChessboardCorners(colourIm, cvSize(8,6), imageCorners, imageCornerCount, wasFound);
			cvShowImage("Chessboard", colourIm);

		}
	free(imageCorners);
	return wasFound;
}

void draw() {
		//	glRotatef(180,0,0,1);
//			CvMat *invHomo = cvCreateMat(3,3,CV_32FC1);
//			cvInvert(homography, invHomo);

			//

			

			//printf("%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\n", markCorn[0].x, markCorn[0].y,markCorn[1].x, markCorn[1].y,markCorn[2].x, markCorn[2].y,markCorn[3].x, markCorn[3].y);
			/*CvPoint2D32f newCorn[4];
			CvMat mPoints = cvMat(4,1,CV_32FC2, markCorn);
			CvMat mPoints2 = cvMat(4,1,CV_32FC2, newCorn);
			cvPerspectiveTransform(&mPoints, &mPoints2, invHomo);
			printf("%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\t%.2f, %.2f\n", newCorn[0].x, newCorn[0].y,newCorn[1].x, newCorn[1].y,newCorn[2].x, newCorn[2].y,newCorn[3].x, newCorn[3].y);
			*/
			
			
/*
			for (int i=0; i<16; i++) {
				printf("%f\t", trans->data.fl[i]); if ((i+1)%4 ==0) printf("\n");
			}

			for (int i=0; i<4; i++) {
				float myDat[4]; myDat[0] = dstPoints.at(i).x; myDat[1] = dstPoints.at(i).y; myDat[2] = dstPoints.at(i).z; myDat[3] = 1; 
				CvMat myPoint = cvMat(4, 1, CV_32FC1, myDat);
				cvMatMul(trans, &myPoint, &myPoint);
				myDat[0] /=myDat[3]; myDat[1] /=myDat[3]; myDat[2] /=myDat[3];
				printf("%d, %f, %f, %f - %f, %f, %f\n\n", i, dstPoints.at(i).x, dstPoints.at(i).y, dstPoints.at(i).z, myDat[0], myDat[1], myDat[2]);
			}
*/
			//printf("

			glPointSize(10);
			glBegin(GL_POINTS);
/*
			printf("%d, %d, %d, %d\n", CV_IMAGE_ELEM(depthImage, unsigned short, (int)markCorn[0].y, (int)markCorn[0].x), CV_IMAGE_ELEM(depthImage, unsigned short, (int)markCorn[1].y, (int)markCorn[1].x), 
									 CV_IMAGE_ELEM(depthImage, unsigned short, (int)markCorn[2].y, (int)markCorn[2].x), CV_IMAGE_ELEM(depthImage, unsigned short, (int)markCorn[3].y, (int)markCorn[3].x));
			glColor3f(255,0,0); glVertex3f(newCorn[0].x, newCorn[0].y, 0);
			glColor3f(0,255,0); glVertex3f(newCorn[1].x, newCorn[1].y, 0);
			glColor3f(0,0,255); glVertex3f(newCorn[2].x, newCorn[2].y, 0);
			glColor3f(255,0,255); glVertex3f(newCorn[3].x, newCorn[3].y, 0);
*/

			for (int y=0; y<480; y++) {
				for (int x=0; x<640; x++) {
					int z = CV_IMAGE_ELEM(depthImage, unsigned short, y, x);
					if (z!=0) {

						XnPoint3D _xnCorner, _xnNewCorner;
						_xnCorner.X = x; _xnCorner.Y = y; _xnCorner.Z = z; 
						niDepth.ConvertProjectiveToRealWorld(1, &_xnCorner, &_xnNewCorner);

						float myDat[4]; myDat[0] = _xnNewCorner.X; myDat[1] = _xnNewCorner.Y; myDat[2] = _xnNewCorner.Z; myDat[3] = 1; 
						CvMat myPoint = cvMat(4, 1, CV_32FC1, myDat);
						cvMatMul(homography, &myPoint, &myPoint);
						myDat[0] /=myDat[3]; myDat[1] /=myDat[3]; myDat[2] /=myDat[3];

						if (-myDat[2]<-10)
							glVertex3f(myDat[0], myDat[1], -myDat[2]);

					}
				}
			}

/*
			CvPoint2D32f point;
			CvMat mPoint=cvMat(1,1,CV_32FC2, &point);

			cvSetZero(debugDepth);
*/
		/*	printf("%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n\n", markCorn[0].x, markCorn[0].y, markCorn[0].z,
				markCorn[1].x, markCorn[1].y, markCorn[1].z,
				markCorn[2].x, markCorn[2].y, markCorn[2].z,
				markCorn[3].x, markCorn[3].y, markCorn[3].z);

			point.x = markCorn[0].x; point.y = markCorn[0].y; cvPerspectiveTransform(&mPoint, &mPoint, invHomo); printf("%f, %f, ", point.x, point.y);
			{			//bilinear interpolation
						double dC0 = (((markerSize.width - point.x)/markerSize.width)*markCorn[0].z); //top left
						double dC1 = (((point.x)/markerSize.width)*markCorn[1].z); //top right
						double dC2 = (((point.x)/markerSize.width)*markCorn[2].z); //bottom right
						double dC3 = (((markerSize.width - point.x)/markerSize.width)*markCorn[3].z); //bottom left

						double r1 = dC0+dC1; double r2 = dC3+dC2;
						double p1 = ((markerSize.height-point.y)/markerSize.height)*r1;
						double p2 = ((point.y)/markerSize.height)*r2;
						double p = p1+p2;
						printf("%f\n", p);
			}
			point.x = markCorn[1].x; point.y = markCorn[1].y; cvPerspectiveTransform(&mPoint, &mPoint, invHomo); printf("%f, %f, ", point.x, point.y);
			{			//bilinear interpolation
						double dC0 = (((markerSize.width - point.x)/markerSize.width)*markCorn[0].z); //top left
						double dC1 = (((point.x)/markerSize.width)*markCorn[1].z); //top right
						double dC2 = (((point.x)/markerSize.width)*markCorn[2].z); //bottom right
						double dC3 = (((markerSize.width - point.x)/markerSize.width)*markCorn[3].z); //bottom left

						double r1 = dC0+dC1; double r2 = dC3+dC2;
						double p1 = ((markerSize.height-point.y)/markerSize.height)*r1;
						double p2 = ((point.y)/markerSize.height)*r2;
						double p = p1+p2;
						printf("%f\n", p);
			}
			point.x = markCorn[2].x; point.y = markCorn[2].y; cvPerspectiveTransform(&mPoint, &mPoint, invHomo); printf("%f, %f, ", point.x, point.y);
			{			//bilinear interpolation
						double dC0 = (((markerSize.width - point.x)/markerSize.width)*markCorn[0].z); //top left
						double dC1 = (((point.x)/markerSize.width)*markCorn[1].z); //top right
						double dC2 = (((point.x)/markerSize.width)*markCorn[2].z); //bottom right
						double dC3 = (((markerSize.width - point.x)/markerSize.width)*markCorn[3].z); //bottom left

						double r1 = dC0+dC1; double r2 = dC3+dC2;
						double p1 = ((markerSize.height-point.y)/markerSize.height)*r1;
						double p2 = ((point.y)/markerSize.height)*r2;
						double p = p1+p2;
						printf("%f\n", p);
			}
			point.x = markCorn[3].x; point.y = markCorn[3].y; cvPerspectiveTransform(&mPoint, &mPoint, invHomo); printf("%f, %f, ", point.x, point.y);
			{			//bilinear interpolation
						double dC0 = (((markerSize.width - point.x)/markerSize.width)*markCorn[0].z); //top left
						double dC1 = (((point.x)/markerSize.width)*markCorn[1].z); //top right
						double dC2 = (((point.x)/markerSize.width)*markCorn[2].z); //bottom right
						double dC3 = (((markerSize.width - point.x)/markerSize.width)*markCorn[3].z); //bottom left

						double r1 = dC0+dC1; double r2 = dC3+dC2;
						double p1 = ((markerSize.height-point.y)/markerSize.height)*r1;
						double p2 = ((point.y)/markerSize.height)*r2;
						double p = p1+p2;
						printf("%f\n\n", p);
			}
*/
/*			FILE *f = fopen("out.txt", "wb");
			for (int y=0; y<480; y++) {
				for (int x=0; x<640; x++) {
					int z = CV_IMAGE_ELEM(depthImage, unsigned short, y, x);
					if (z!=0) {
						point.x = x; point.y = y;
						cvPerspectiveTransform(&mPoint, &mPoint, invHomo);


						//bilinear interpolation
						double dC0 = (((markerSize.width - point.x)/markerSize.width)*markCorn[0].z); //top left
						double dC1 = (((point.x)/markerSize.width)*markCorn[1].z); //top right
						double dC2 = (((point.x)/markerSize.width)*markCorn[2].z); //bottom right
						double dC3 = (((markerSize.width - point.x)/markerSize.width)*markCorn[3].z); //bottom left

						double r1 = dC0+dC1; double r2 = dC3+dC2;
						double p1 = ((markerSize.height-point.y)/markerSize.height)*r1;
						double p2 = ((point.y)/markerSize.height)*r2;
						double p = p1+p2;

						//double p = float(markCorn[0].z+ markCorn[1].z+ markCorn[2].z+ markCorn[3].z)/4.0;

						if (p-z>5) {
							glVertex3f(point.x, point.y, -(p-z));
							CV_IMAGE_ELEM(debugDepth, unsigned short, y, x) = int(p-z);
						}
						//fprintf(f, "%.2f\n", p-z);
					}
				}
			}
			fclose(f);

			IplImage *depthDispImage = cvCreateImage(cvGetSize(debugDepth), IPL_DEPTH_16U, 1); 
			cvConvertScale(debugDepth, depthDispImage, 100);
			cvSetMouseCallback("Depth Image", depthMouseFunc, debugDepth);

			cvShowImage("Depth Image", depthDispImage);
			cvReleaseImage(&depthDispImage);
			cvWaitKey(1);
*/

			//cvReleaseImage(&debugDepth);
			/*	for (int y=0; y<480; y++) {
				for (int x=0; x<640; x++) {
					int z = CV_IMAGE_ELEM(depthImage, unsigned short, y, x);
					point.x = x; point.y = y;
					cvPerspectiveTransform(&mPoint, &mPoint, invHomo);
					if (z>10) glVertex3f(point.x, point.y, z);
				}
			}*/
			glEnd();
//			cvReleaseMat(&invHomo);
		}

		//Draw the infamous Teapot
		/*glEnable(GL_LIGHTING);
			glTranslatef(markerSize.width/2.0, markerSize.height/2.0,-25);
			glRotatef(-90, 1.0,0,0.0);
			glutSolidTeapot(50.0);
		glDisable(GL_LIGHTING);
*/
	
}
