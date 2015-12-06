#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


using namespace cv;
using namespace std;

Mat sub_sample(Mat X, int ws){
    //cout<<X.rows/ws<<" "<<X.cols/ws<<endl;
    Mat result(X.rows/ws, X.cols/ws, CV_8UC1);

    for(int i=0; i<result.rows; i++){
        for(int j=0; j<result.cols; j++){
            int val_max = 0;
            for(int h=0; h<ws; h++){
                for(int l=0; l<ws; l++){
                    if(ws*i+h<X.rows && ws*j+l<X.cols){
                        Scalar pix = X.at<uchar>(ws*i+h,ws*j+l);
                        if(pix.val[0]>val_max){
                            val_max = pix.val[0];
                        }
                    }
                }
            }
            result.at<uchar>(i,j)=val_max;
        }
    }
    //cout<<"test"<<endl;
    return result;
}


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
            Scalar pixel = X_mirrors.at<uchar>(i,j);
            double diff = 0.;
            for(int h=left_lim; h<right_lim; h++){
                for(int l=left_lim; l<right_lim; l++){
                    Scalar neighbor = X_mirrors.at<uchar>(i+h,j+l);
                    diff += abs(pixel.val[0] - neighbor.val[0]);
                }
            }
            uchar value;
            double P = diff/(ws*ws*255.)+0.001;
            double H = -P*log(P)/Hmax*255;

            if(H<0){H=0;}
            if(H>255){H=255;}

            value = H;
            Result.at<uchar>(i-ws+1,j-ws+1) = value;
        }
    }
    return Result;
}

Mat best_conv(Mat X, Mat H, Mat W)
{

    Mat convole;
    filter2D(H, convole, -1 , W, Point(-1,-1), 0, BORDER_DEFAULT );
    int ws = W.cols;
    int right_lim = (ws-1)/2+1;
    int left_lim = -(ws-1)/2;
    Mat X_mirrors;
    copyMakeBorder(X, X_mirrors, ws-1,ws-1,ws-1,ws-1, BORDER_REFLECT );
    double mean_conv = mean(convole).val[0];

    double best_score=0;
    vector<int> best_i;
    vector<int> best_j;
 
    for(int i=0; i<convole.rows; i++){
        for(int j=0; j<convole.cols; j++){
            Scalar conv_score =  convole.at<uchar>(i,j);
            double score = conv_score.val[0];
            if(score>best_score){
                best_i.clear();
                best_i.push_back(i);
                best_j.push_back(j);
                best_score = score;
            }
            if(score==best_score){
                best_i.push_back(i);
                best_j.push_back(j);
            }
        }
    }
    double relevance = (best_score-mean_conv)/255.;

    Mat new_W(ws, ws,  CV_32F);
    if(best_i.size()>0){
        int index_i = rand() % best_i.size();
        int index_j = rand() % best_j.size();
        int i = best_i[index_i] + ws-1;
        int j = best_j[index_j] + ws-1;

        double mean = 0;

        for(int h=left_lim; h<right_lim; h++){
            for(int l=left_lim; l<right_lim; l++){
                Scalar x = X_mirrors.at<uchar>(i-h,j-l);
                mean += x.val[0]/(ws*ws);
            }
        }

        for(int h=left_lim; h<right_lim; h++){
            for(int l=left_lim; l<right_lim; l++){
                Scalar x = X_mirrors.at<uchar>(i-h,j-l);
                new_W.at<float>(h+(ws-1)/2,l+(ws-1)/2) = relevance*(x.val[0]-mean)/255.;
            }
        }
        return W-new_W;
    }
    else{
        return W;
    }
}

/*int main ( int argc, char** argv )
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
}*/

int main ( int argc, char** argv )
{
    srand(time(NULL));;
    Mat src;
    Mat grey;
    int ws = 5;
    Mat R(ws, ws,  CV_32F);
    float mean = 0;
    for(int i=0;i<ws;i++){
        for(int j=0;j<ws;j++){
            R.at<float>(i,j) = rand()*2/4294967295.;
            mean+=R.at<float>(i,j)/25.;
        }
    }
    double sum = 0;
    for(int i=0;i<ws;i++){
        for(int j=0;j<ws;j++){
            R.at<float>(i,j) = 1*(R.at<float>(i,j)-mean);
        }
    }
    cout<<"sum :"<<sum<<endl;
    cout<<R<<endl;

    Mat W = R;

    /*for(int i=0; i<10; i++){ 
        src = imread( "BioID_000"+to_string(i)+".pgm" );
        if( !src.data )  { return -1; }
        cvtColor(src, grey, CV_BGR2GRAY);
        Mat H = entropy_window(grey,3);

        //for(int i=0;i<20;i++){
        W = best_conv(H,W);
        //}
    }

    for(int i=10; i<100; i++){ 
        src = imread( "BioID_00"+to_string(i)+".pgm" );
        if( !src.data )  { return -1; }
        cvtColor(src, grey, CV_BGR2GRAY);
        Mat H = entropy_window(grey,3);

        //for(int i=0;i<20;i++){
        W = best_conv(H,W);
        //}
    }*/

    for(int i=100; i<1000; i++){ 
        src = imread( "BioID_0"+to_string(i)+".pgm" );
        if( !src.data )  { return -1; }
        cvtColor(src, grey, CV_BGR2GRAY);
        Mat H = sub_sample(entropy_window(grey,3),3);

        //for(int i=0;i<20;i++){
        W = best_conv(src,H,W);
        //}
    }
    cout<<W<<endl;

    src = imread( "3.pgm" );
    if( !src.data )  { return -1; }
    cvtColor(src, grey, CV_BGR2GRAY);

    Mat result;
    filter2D(grey, result, -1 , W, Point(-1,-1), 0, BORDER_DEFAULT );
    imshow("test",result);
    waitKey(0);

    return 0;
}
