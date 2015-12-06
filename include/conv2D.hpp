#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;

Mat best_conv(Mat X, Mat W)
{

    Mat convole;
    filter2D(X, convole, -1 , W, Point(-1,-1), 0, BORDER_DEFAULT );
    int ws = W.cols;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );


    double best_score=0;
    vector<int> best_i;
    vector<int> best_j;
 
    for(int i=0; i<convole.rows; i++){
        for(int j=0; j<convole.cols; j++){
            Scalar score =  convole.at<uchar>(i,j);
            if(score.val[0]>best_score){
                best_i.clear();
                best_i.push_back(i);
                best_j.push_back(j);
                best_score = score.val[0];
            }
            if(score.val[0]==best_score){
                best_i.push_back(i);
                best_j.push_back(j);
            }
        }
    }
    int index_i = rand() % best_i.size();
    int index_j = rand() % best_j.size();
    int i = best_i[index_i] + ws-1;
    int j = best_j[index_j] + ws-1;

    Mat new_W(W.cols,W.rows, CV_32F);
    double mean = 0;

    for(int h=left_lim; h<right_lim; h++){
        for(int l=left_lim; l<right_lim; l++){
            Scalar x = X_mirrors.at<uchar>(i-h,j-l);
            mean += x.val[0]/9.;
        }
    }

    for(int h=left_lim; h<right_lim; h++){
        for(int l=left_lim; l<right_lim; l++){
            Scalar x = X_mirrors.at<uchar>(i-h,j-l);
            new_W.at<float>(h+(ws-1)/2,l+(ws-1)/2) = (x.val[0]-mean)/255.;
        }
    }

    return W-new_W;
}

 
int main ( int argc, char** argv )
{
    Mat src;
    Mat grey;
    Mat Gx(3, 3,  CV_32F);
    Gx.at<float>(0,0)=-1;
    Gx.at<float>(0,1)=0;
    Gx.at<float>(0,2)=1;
    Gx.at<float>(1,0)=-2;
    Gx.at<float>(1,1)=0;
    Gx.at<float>(1,2)=2;
    Gx.at<float>(2,0)=-1;
    Gx.at<float>(2,1)=0;
    Gx.at<float>(2,2)=1;

    src = imread( "3.pgm" );
    if( !src.data )  { return -1; }
    cvtColor(src, grey, CV_BGR2GRAY);


    Mat Wx = best_conv(grey, Gx);
    Mat Wy = best_conv(grey, Gx.t());
    for(int i=0;i<20;i++){
        Wx = best_conv(grey,Wx);
        Wy = best_conv(grey,Wy);
    }
    Mat result0;
    Mat result00;
    filter2D(grey, result0, -1 , Wx, Point(-1,-1), 0, BORDER_DEFAULT );
    filter2D(grey, result00, -1 , Wy, Point(-1,-1), 0, BORDER_DEFAULT );
    imshow("test",result0+result00);

    Mat result1;
    Mat result2;
    filter2D(grey, result1, -1 , Gx, Point(-1,-1), 0, BORDER_DEFAULT );
    filter2D(grey, result2, -1 , Gx.t(), Point(-1,-1), 0, BORDER_DEFAULT );
    //Mat X = conv2d(grey,conv);
    imshow("sum",result1+result2);
    waitKey(0);

    return 0;
}
