#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

double THETA = 0.2;
double LOOPS = 10;

Mat mosaic_rgb(Mat X){

    int bs = 4;//border_size;
    int right_lim = (bs-1)/2+1;
    int left_lim = -(bs-1)/2;
    Mat X_mirrors;
    Mat R_mirrors;
    copyMakeBorder(X, X_mirrors, bs-1,bs-1,bs-1,bs-1, BORDER_REFLECT );
    Mat Result(X.rows, X.cols,  CV_8UC3, Scalar(0,0,0));

    for(int i=bs-1; i<X_mirrors.rows-bs+1; i++){
        for(int j=bs-1; j<X_mirrors.cols-bs+1; j++){

            int ii=i-bs+1;
            int jj=j-bs+1;
            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);
            Vec3b value;
            double p=0;
            double sum = 0;

            Mat conv = (Mat_<double>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
            for(int y=-1; y<2; y++){
                for(int x=-1; x<2; x++){
                    double strengh = 0;
                    for(int col=0;col<3;col++){
                        Vec3b pixy = X_mirrors.at<Vec3b>(i+x,j+y);
                        Vec3b pixy_t = X_mirrors.at<Vec3b>(i-x,j-y);
                        double fit = abs(double(pixy_t[col])-double(pixy[col]))/255.+abs(double(pixy_t[col])-double(pixel[col]))/255.+abs(double(pixel[col])-double(pixy[col]))/255.;
                        conv.at<double>(x+1,y+1) += (fit)/3. + 0.01;
                        sum++;
                    }
                }
            }
            for(int col=0;col<3;col++){
                double meanval = 0;
                double count = 0;
                for(int x=-1; x<2; x++){
                    for(int y=-1; y<2; y++){
                        Vec3b pixy = X_mirrors.at<Vec3b>(i+x,j+y);
                        double dist = abs(double(pixy[col])-double(pixel[col]))/255.;
                        if (dist<THETA || x*x+y*y==0){
                            meanval += pixy[col] * pow(1./conv.at<double>(x+1,y+1),4);
                            count += pow(1./conv.at<double>(x+1,y+1),4);
                        }
                    }
                }
                double val = meanval/count;
                double z = (exp((val-50)/205.)-exp(-50/205.)) / (exp(1)-exp(-50/205.));
                double t = 1-(exp((val-200)/205.)-exp(-200/205.)) / (exp(1)-exp(-200/205.));
                double random = (double)rand() / RAND_MAX;
                if (random < z && random >t){val+=5;}
                if(val<0){val=0;}
                if(val>255){val=255;}
                value[col] = val;
            }
            Result.at<Vec3b>(ii,jj) = value;
        }
    }
    return Result;
}

int main ( int argc, char** argv )
{
    Mat src;
    Mat last;

    /// Load an image
    src = imread( argv[1] );
    if( !src.data )  { return -1; }
    Mat grey;
    cvtColor(src,grey, CV_BGR2GRAY);
    Mat X = mosaic_rgb(src);
    Mat Y = mosaic_rgb(X);

    for (int i=0; i<LOOPS; i++){
        X = mosaic_rgb(Y);
        Y = mosaic_rgb(X);
        if (i%1==0){
            cout<<"."<<endl;
        }
    }
    imshow("sum",Y);
    imwrite(argv[2], Y);
    waitKey(0);
    return 0;
}
