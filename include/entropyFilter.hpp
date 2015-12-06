#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat entropy_window(Mat X, int window_size){

    int ws = window_size;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );
    Mat Result(X.rows, X.cols,  CV_8UC1);
    double Hmax = -0.5*log(0.5);

    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){
            auto pixel = X_mirrors.at<uchar>(i,j);
            double diff = 0.;
            for(int h=left_lim; h<right_lim; h++){
                for(int l=left_lim; l<right_lim; l++){
                    auto neighbor = X_mirrors.at<uchar>(i+h,j+l);
                    diff += abs(pixel - neighbor);
                }
            }
            uchar value;
            double P = diff/(8.*255.)+0.001;
            double H = -P*log(P)/Hmax*255;

            if(H<0){H=0;}
            if(H>255){H=255;}

            value = H;
            Result.at<uchar>(i-ws+1,j-ws+1) = value;
        }
    }

    //imshow( "sum", Result );
    return Result;
}


Mat entropy_window_rgb(Mat X, int window_size){

    int ws = window_size;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );
    Mat Result(X.rows, X.cols,  CV_8UC3, Scalar(0,0,0));
    double Hmax = -0.5*log(0.5);

    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){
            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);
            double diff_r = 0.;
            double diff_b = 0.;
            double diff_g = 0.;
            for(int h=left_lim; h<right_lim; h++){
                for(int l=left_lim; l<right_lim; l++){
                    Vec3b neighbor = X_mirrors.at<Vec3b>(i+h,j+l);
                    diff_r += abs(pixel[0] - neighbor[0]);
                    diff_b += abs(pixel[1] - neighbor[1]);
                    diff_g += abs(pixel[2] - neighbor[2]);
                }
            }
            Vec3b value;
            double P_r = diff_r/(8.*255.)+0.001;
            double P_b = diff_b/(8.*255.)+0.001;
            double P_g = diff_g/(8.*255.)+0.001;
            double H_r = -P_r*log(P_r)/Hmax*255;
            double H_b = -P_b*log(P_b)/Hmax*255;
            double H_g = -P_g*log(P_g)/Hmax*255;

            if(H_r<0){H_r=0;}
            if(H_b<0){H_b=0;}
            if(H_g<0){H_g=0;}
            if(H_r>255){H_r=255;}
            if(H_b>255){H_b=255;}
            if(H_g>255){H_g=255;}

            value[0] = H_r;
            value[1] = H_b;
            value[2] = H_g;
            Result.at<Vec3b>(i-ws+1,j-ws+1) = value;
        }
    }

    //imshow( "sum", Result );
    return Result;
}


Mat sliding_window(Mat X, int window_size){

    int ws = window_size;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );
    Mat Result(X.rows, X.cols,  CV_8UC3, Scalar(0,0,0));

    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){
            double sum = 0;
            for(int h=left_lim; h<right_lim; h++){
                for(int l=left_lim; l<right_lim; l++){
                    Vec3b neighbor = X_mirrors.at<Vec3b>(i+h,j+l);
                    sum += neighbor[0]/255.; 
                }
            }
            Vec3b value;
            double S = sum/(ws*ws)*255.;

            if(S<0){S=0;}
            if(S>255){S=255;}
            value[0] = S;
            value[1] = S;
            value[2] = S;
            Result.at<Vec3b>(i-ws+1,j-ws+1) = value;
        }
    }

    //imshow( "sum", Result );
    return Result;
}

int main ( int argc, char** argv )
{
    Mat src;
    Mat last;
   
    /// Load an image
    src = imread( "3.pgm" );
    if( !src.data )  { return -1; }
    Mat grey;
    cvtColor(src,grey, CV_BGR2GRAY);
    Mat X = entropy_window(grey,3);
    imshow("sum",X);
    waitKey(0);

    /// Load webcam
    /*
    VideoCapture cap(0);

    namedWindow( "sum", CV_WINDOW_AUTOSIZE );
    while(true){
        Mat src;
        Mat grey;
        cap >> src;
        cvtColor(src, grey, CV_BGR2GRAY);

        Mat X = entropy_window(grey,3);
        imshow("sum", X);
        waitKey(30);
    }*/
    
    return 0;
}
