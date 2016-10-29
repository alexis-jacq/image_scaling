#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

float attract(float x, float y, float z, float t, float  expose){
    //expose = 1.;
    //if(expose!=1.){expose=3.;}
    expose = 1./2.;
    return pow((-2*pow(x,expose) + 10*pow(y,expose) + 10*pow(z, expose) + -2*pow(t,expose))/16., 1./expose);
}

Mat interpolate_rgb(Mat X){

    int ws = 4;//window_size;
    double theta=1.15;
    int gama = 5;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    Mat R_mirrors;
    cout<<"test1"<<endl;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );
    Mat Result(2*X.rows, 2*X.cols,  CV_8UC3, Scalar(0,0,0));

    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){

            int ii=2*(i-ws+1);
            int jj=2*(j-ws+1);
            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);

            // midles """""""""""""""""""""""""""""""""
            Result.at<Vec3b>(ii,jj) = pixel;

        }
    }

    for(int i=ws-1; i<X_mirrors.rows-ws+1;i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){

            int ii=2*(i-ws+1)+1;
            int jj=2*(j-ws+1)+1;

            Vec3b value;

            for(int col=0;col<3;col++){
                // d1 and d2 strength
                double d1=0;
                double d2=0;
                for(int x=-1; x<2; x++){
                    for(int y=-1; y<2; y++){
                        //d1:
                        Vec3b pix11 = X_mirrors.at<Vec3b>(i+x+1,j+y);
                        Vec3b pix12 = X_mirrors.at<Vec3b>(i+x,j+y+1);
                        d1 += abs(double(pix11[col])-double(pix12[col]));
                        //d2:
                        Vec3b pix21 = X_mirrors.at<Vec3b>(i+x,j+y);
                        Vec3b pix22 = X_mirrors.at<Vec3b>(i+x+1,j+y+1);
                        d2 += abs(double(pix21[col])-double(pix22[col]));
                    }
                }
                Vec3b pix00 = X_mirrors.at<Vec3b>(i-1,j-1);
                Vec3b pix11 = X_mirrors.at<Vec3b>(i,j);
                Vec3b pix22 = X_mirrors.at<Vec3b>(i+1,j+1);
                Vec3b pix33 = X_mirrors.at<Vec3b>(i+2,j+2);
                Vec3b pix30 = X_mirrors.at<Vec3b>(i+2,j-1);
                Vec3b pix21 = X_mirrors.at<Vec3b>(i+1,j);
                Vec3b pix12 = X_mirrors.at<Vec3b>(i,j+1);
                Vec3b pix03 = X_mirrors.at<Vec3b>(i-1,j+2);

                if(100*(1+d1)>100*theta*(1+d2)){
                    double val = attract(double(pix00[col]),double(pix11[col]),double(pix22[col]),double(pix33[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value[col]=val;
                }
                if(100*(1+d2)>100*theta*(1+d1)){
                    double val = attract(double(pix30[col]),double(pix21[col]),double(pix12[col]),double(pix03[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value[col]=val;
                }
                if(100*(1+d1)<=100*theta*(1+d2) && 100*(1+d2)<=100*theta*(1+d1)){
                    double w1 = 1/(1+pow(d1,gama));
                    double w2 = 1/(1+pow(d2,gama));
                    double weight1 = w1/(w1+w2);
                    double weight2 = w2/(w1+w2);
                    double drval = attract(double(pix00[col]),double(pix11[col]),double(pix22[col]),double(pix33[col]),1./6.);
                    double urval = attract(double(pix30[col]),double(pix21[col]),double(pix12[col]),double(pix03[col]),1./6.);
                    double val=drval*weight1 + urval*weight2;
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value[col] = val;
                }
            }
            Result.at<Vec3b>(ii,jj) = value;
        }
    }


    copyMakeBorder(Result,R_mirrors, 4,4,4,4, BORDER_REFLECT );

    for(int i=ws-1; i<X_mirrors.rows-ws+1;i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){

            //left
            int ii1=2*(i-ws+1)+1;
            int jj1=2*(j-ws+1)+0;
            //down
            int ii2=2*(i-ws+1)+0;
            int jj2=2*(j-ws+1)+1;

            Vec3b value1;
            Vec3b value2;

            //left
            for(int col=0;col<3;col++){
                // d1 and d2 strength
                int x = ii1+4;
                int y = jj1+4;

                Vec3b pix1_2 = R_mirrors.at<Vec3b>(x+1,y-2);
                Vec3b pix_1_2 = R_mirrors.at<Vec3b>(x-1,y-2);
                Vec3b pix2_1 = R_mirrors.at<Vec3b>(x+2,y-1);
                Vec3b pix0_1 = R_mirrors.at<Vec3b>(x,y-1);
                Vec3b pix_2_1 = R_mirrors.at<Vec3b>(x-2,y-1);
                Vec3b pix30 = R_mirrors.at<Vec3b>(x+3,y);
                Vec3b pix10 = R_mirrors.at<Vec3b>(x+1,y);
                Vec3b pix_10 = R_mirrors.at<Vec3b>(x-1,y);
                Vec3b pix_30 = R_mirrors.at<Vec3b>(x-3,y);
                Vec3b pix21 = R_mirrors.at<Vec3b>(x+2,y+1);
                Vec3b pix01 = R_mirrors.at<Vec3b>(x,y+1);
                Vec3b pix_21 = R_mirrors.at<Vec3b>(x-2,y+1);
                Vec3b pix12 = R_mirrors.at<Vec3b>(x+1,y+2);
                Vec3b pix_12 = R_mirrors.at<Vec3b>(x-1,y+2);
                Vec3b pix03 = R_mirrors.at<Vec3b>(x,y+3);
                Vec3b pix0_3 = R_mirrors.at<Vec3b>(x,y-3);

                double d1= abs(double(pix1_2[col])-double(pix_1_2[col])) + \
                abs(double(pix2_1[col])-double(pix0_1[col])) + \
                abs(double(pix0_1[col])-double(pix_2_1[col])) + \
                abs(double(pix30[col])-double(pix10[col])) + \
                abs(double(pix10[col])-double(pix_10[col])) + \
                abs(double(pix_10[col])-double(pix_30[col])) + \
                abs(double(pix21[col])-double(pix01[col])) + \
                abs(double(pix01[col])-double(pix_21[col])) + \
                abs(double(pix12[col])-double(pix_12[col]));

                double d2= abs(double(pix_21[col])-double(pix_2_1[col])) + \
                abs(double(pix_12[col])-double(pix_10[col])) + \
                abs(double(pix_10[col])-double(pix_1_2[col])) + \
                abs(double(pix03[col])-double(pix01[col])) + \
                abs(double(pix01[col])-double(pix0_1[col])) + \
                abs(double(pix0_1[col])-double(pix0_3[col])) + \
                abs(double(pix12[col])-double(pix10[col])) + \
                abs(double(pix10[col])-double(pix1_2[col])) + \
                abs(double(pix21[col])-double(pix2_1[col]));

                if(100*(1+d1)>100*theta*(1+d2)){
                    double val = attract(double(pix0_3[col]),double(pix0_1[col]),double(pix01[col]),double(pix03[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value1[col]=val;
                }
                if(100*(1+d2)>100*theta*(1+d1)){
                    double val = attract(double(pix_30[col]),double(pix_10[col]),double(pix10[col]),double(pix30[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value1[col]=val;
                }
                if(100*(1+d1)<=100*theta*(1+d2) && 100*(1+d2)<=100*theta*(1+d1)){
                    double w1 = 1/(1+pow(d1,gama));
                    double w2 = 1/(1+pow(d2,gama));
                    double weight1 = w1/(w1+w2);
                    double weight2 = w2/(w1+w2);
                    double drval = attract(double(pix0_3[col]),double(pix0_1[col]),double(pix01[col]),double(pix03[col]),1./6.);
                    double urval = attract(double(pix_30[col]),double(pix_10[col]),double(pix10[col]),double(pix30[col]),1./6.);
                    double val=drval*weight1 + urval*weight2;
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value1[col] = val;
                }
            }
            //down
            for(int col=0;col<3;col++){
                // d1 and d2 strength
                int x = ii2+4;
                int y = jj2+4;

                Vec3b pix1_2 =R_mirrors.at<Vec3b>(x+1,y-2);
                Vec3b pix_1_2 = R_mirrors.at<Vec3b>(x-1,y-2);
                Vec3b pix2_1 = R_mirrors.at<Vec3b>(x+2,y-1);
                Vec3b pix0_1 = R_mirrors.at<Vec3b>(x,y-1);
                Vec3b pix_2_1 = R_mirrors.at<Vec3b>(x-2,y-1);
                Vec3b pix30 = R_mirrors.at<Vec3b>(x+3,y);
                Vec3b pix10 = R_mirrors.at<Vec3b>(x+1,y);
                Vec3b pix_10 = R_mirrors.at<Vec3b>(x-1,y);
                Vec3b pix_30 = R_mirrors.at<Vec3b>(x-3,y);
                Vec3b pix21 = R_mirrors.at<Vec3b>(x+2,y+1);
                Vec3b pix01 = R_mirrors.at<Vec3b>(x,y+1);
                Vec3b pix_21 = R_mirrors.at<Vec3b>(x-2,y+1);
                Vec3b pix12 = R_mirrors.at<Vec3b>(x+1,y+2);
                Vec3b pix_12 = R_mirrors.at<Vec3b>(x-1,y+2);
                Vec3b pix03 = R_mirrors.at<Vec3b>(x,y+3);
                Vec3b pix0_3 = R_mirrors.at<Vec3b>(x,y-3);

                double d1= abs(double(pix1_2[col])-double(pix_1_2[col])) + \
                abs(double(pix2_1[col])-double(pix0_1[col])) + \
                abs(double(pix0_1[col])-double(pix_2_1[col])) + \
                abs(double(pix30[col])-double(pix10[col])) + \
                abs(double(pix10[col])-double(pix_10[col])) + \
                abs(double(pix_10[col])-double(pix_30[col])) + \
                abs(double(pix21[col])-double(pix01[col])) + \
                abs(double(pix01[col])-double(pix_21[col])) + \
                abs(double(pix12[col])-double(pix_12[col]));

                double d2= abs(double(pix_21[col])-double(pix_2_1[col])) + \
                abs(double(pix_12[col])-double(pix_10[col])) + \
                abs(double(pix_10[col])-double(pix_1_2[col])) + \
                abs(double(pix03[col])-double(pix01[col])) + \
                abs(double(pix01[col])-double(pix0_1[col])) + \
                abs(double(pix0_1[col])-double(pix0_3[col])) + \
                abs(double(pix12[col])-double(pix10[col])) + \
                abs(double(pix10[col])-double(pix1_2[col])) + \
                abs(double(pix21[col])-double(pix2_1[col]));

                if(100*(1+d1)>100*theta*(1+d2)){
                    double val = attract(double(pix0_3[col]),double(pix0_1[col]),double(pix01[col]),double(pix03[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value2[col]=val;
                }
                if(100*(1+d2)>100*theta*(1+d1)){
                    double val = attract(double(pix_30[col]),double(pix_10[col]),double(pix10[col]),double(pix30[col]),1.);
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value2[col]=val;
                }
                if(100*(1+d1)<=100*theta*(1+d2) && 100*(1+d2)<=100*theta*(1+d1)){
                    double w1 = 1/(1+pow(d1,gama));
                    double w2 = 1/(1+pow(d2,gama));
                    double weight1 = w1/(w1+w2);
                    double weight2 = w2/(w1+w2);
                    double drval = attract(double(pix0_3[col]),double(pix0_1[col]),double(pix01[col]),double(pix03[col]),1./6.);
                    double urval = attract(double(pix_30[col]),double(pix_10[col]),double(pix10[col]),double(pix30[col]),1./6.);
                    double val=drval*weight1 + urval*weight2;
                    if(val<0){val=0;}
                    if(val>255){val=255;}
                    value2[col] = val;
                }
            }
            Result.at<Vec3b>(ii1,jj1) = value1;
            Result.at<Vec3b>(ii2,jj2) = value2;
        }
    }
    cout<<"test2"<<endl;
    //imshow( "sum", Result );
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
    Mat Y = interpolate_rgb(src);
    //Mat Y = interpolate_rgb(X);
    imshow("sum",Y);
    imwrite(argv[2], Y);
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
        Mat X = interpolate_rgb(src,3);
        imshow("sum", X);
        waitKey(30);
    }*/
    return 0;
}