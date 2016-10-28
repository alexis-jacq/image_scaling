#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;



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

    // Angle maps:
    vector<double> angles_1;
    vector<double> angles_2;
    vector<Mat> angle_maps;
    Mat A_1_0(X_mirrors.rows, X_mirrors.cols, CV_32FC1, float(0)); angle_maps.push_back(A_1_0); angles_1.push_back(1); angles_2.push_back(0);
    Mat A_M1_0(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_M1_0); angles_1.push_back(-1); angles_2.push_back(0);
    Mat A_0_1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_0_1); angles_1.push_back(0); angles_2.push_back(1);
    Mat A_0_M1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_0_M1); angles_1.push_back(0); angles_2.push_back(-1);
    Mat A_1_1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_1_1); angles_1.push_back(1); angles_2.push_back(1);
    Mat A_1_M1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1,float(0)); angle_maps.push_back(A_1_M1); angles_1.push_back(1); angles_2.push_back(-1);
    Mat A_M1_M1(X_mirrors.rows, X_mirrors.cols, CV_32FC1, float(0)); angle_maps.push_back(A_M1_M1); angles_1.push_back(-1); angles_2.push_back(-1);
    Mat A_M1_1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_M1_1); angles_1.push_back(-1); angles_2.push_back(1);
    
    /*Mat A_1_2(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_1_2); angles_1.push_back(1); angles_2.push_back(2);
    Mat A_1_M2(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_1_M2); angles_1.push_back(1); angles_2.push_back(-2);

    Mat A_M1_2(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_M1_2); angles_1.push_back(-1); angles_2.push_back(2);
    Mat A_M1_M2(X_mirrors.rows, X_mirrors.cols, CV_32FC1, float(0)); angle_maps.push_back(A_M1_M2); angles_1.push_back(-1); angles_2.push_back(-2);

    Mat A_2_1(X_mirrors.rows, X_mirrors.cols,   CV_32FC1, float(0)); angle_maps.push_back(A_2_1); angles_1.push_back(2); angles_2.push_back(1);
    Mat A_2_M1(X_mirrors.rows, X_mirrors.cols,   CV_32FC1, float(0)); angle_maps.push_back(A_2_M1); angles_1.push_back(2); angles_2.push_back(-1);

    Mat A_M2_1(X_mirrors.rows, X_mirrors.cols,  CV_32FC1, float(0)); angle_maps.push_back(A_M2_1); angles_1.push_back(-2); angles_2.push_back(1);
    Mat A_M2_M1(X_mirrors.rows, X_mirrors.cols, CV_32FC1, float(0)); angle_maps.push_back(A_M2_M1); angles_1.push_back(-2); angles_2.push_back(-1);*/

    // Result:
    Mat Result(2*X.rows, 2*X.cols,  CV_8UC3, Scalar(0,0,0));
    Mat Counts(2*X.rows, 2*X.cols,  CV_32FC3, Scalar(0,0,0));
    Mat Values(2*X.rows, 2*X.cols,  CV_32FC3, Scalar(0,0,0));
    cout<<"test12"<<endl;


    // 1. fill angle maps
    //-------------------
    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){

            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);

            int angle_indice = 0;

            for(auto map : angle_maps){


                int a1 = angles_1[angle_indice];
                int a2 = angles_2[angle_indice];
                double score = 0;

                for(int x=1; x<3; x++){

                    int x1 = x*a1;
                    int x2 = x*a2;

                    Vec3b pix = X_mirrors.at<Vec3b>(i+x1,j+x2);

                    // friend match:
                    double d1 = 0;
                    for(int col=0;col<5;col++){
                        d1 += abs(pow((double(pixel[col])/255.-double(pix[col])/255.),2));
                    }

                    // enemy match:
                    double d2 = 0;
                    double count = 0;
                    for(int y=1; y<2; y++){

                        int y11 = x1 + y*a2;
                        int y12 = x2 - y*a1;
                        //---------------
                        int y21 = x1 - y*a2;
                        int y22 = x2 + y*a1;

                        Vec3b piy1 = X_mirrors.at<Vec3b>(i+y11,j+y12);
                        Vec3b piy2 = X_mirrors.at<Vec3b>(i+y21,j+y22);

                        for(int col=0;col<3;col++){
                            d2 += abs(pow((double(pix[col])/255.-double(piy1[col])/255.),2));
                            d2 += abs(pow((double(pix[col])/255.-double(piy2[col])/255.),2));
                            d2 += abs(pow((double(piy1[col])/255.-double(piy2[col])/255.),2));
                        }
                        count += 9.; // 3 col *3
                    }
                    d2 /= count;

                    score += exp(d2-d1);//*d2;
                }
                map.at<float>(i,j) = score;

                angle_indice += 1;
            }
            //cout<<"i"<<i<<endl;
        }
        //cout<<"j"<<i<<endl;
    }

    cout<<"done !!!"<<endl;

    // 2. fill frends
    //-----------------
    for(int i=ws-1; i<X_mirrors.rows-ws+1; i++){
        for(int j=ws-1; j<X_mirrors.cols-ws+1; j++){

            int ii=2*(i-ws+1);
            int jj=2*(j-ws+1);
            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);

            // midles """""""""""""""""""""""""""""""""
            Result.at<Vec3b>(ii,jj) = pixel;

            // looking for friends
            int angle_indice = 0;

            for(auto map : angle_maps){


                int a1 = angles_1[angle_indice];
                int a2 = angles_2[angle_indice];

                if(ii+a1>=0 && ii+a1<Result.rows && jj+a2>=0 && jj+a2<Result.cols){

                    Vec3f value = Values.at<Vec3f>(ii+a1,jj+a2);
                    Vec3f nb = Counts.at<Vec3f>(ii+a1,jj+a2);
                    double angle_score = double(map.at<float>(i,j));

                    double score = angle_score;
                    double theta = exp(0.61);

                    if(score>theta){

                        double d = exp(20*score);
                        //double d=0.1+(score/255.)*(score/255.);
                        //double d=1.;
                        // they are friends
                        for(int col=0;col<3;col++){

                            value[col]+=d*pixel[col];;
                            nb[col]+=d;
                        }
                        Values.at<Vec3f>(ii+a1,jj+a2)=value;
                        Counts.at<Vec3f>(ii+a1,jj+a2)=nb;
                    }
                }
                angle_indice += 1;
            }
        }
    }


    for(int i=0; i<Result.rows; i++){
        for(int j=0; j<Result.cols; j++){

            Vec3f nb = Counts.at<Vec3f>(i,j);
            if(nb[0]>0){
                Vec3f value = Values.at<Vec3f>(i,j);
                for(int col=0;col<3;col++){
                    value[col] /= nb[col];
                    //cout<<value[col]<<endl;
                }
                Result.at<Vec3b>(i,j) = value;
            }
        }
    }


    // treat friendless
    /*for(int i=ws-1; i<Result.rows-ws+1; i++){
        for(int j=ws-1; j<Result.cols-ws+1; j++){

            Vec3b nb = Count.at<Vec3b>(i,j);


            if(nb[0]<2 && nb[0]>=0){
                // you have no (or juste 1) friend
                Vec3b value;
                for(int col=0;col<3;col++){

                    double sum = 0;
                    double n = 0;

                    for(int x=0; x<2; x++){
                        for(int y=0; y<2; y++){

                            Vec3b value = Result.at<Vec3b>(i+x,j+y);
                            Vec3b nb = Count.at<Vec3b>(i+x,j+y);
                            double test = double(nb[0]);
                            if(nb[0]<2 && nb[0]>=0){test = 0;}
                            else{test = 1;}
                            sum += double(value[col]) * test * 1.1/(double(x*x+y*y)+0.1) * 1.1/(double(x*x+y*y)+0.1);
                            n += test * 1.1/(double(x*x+y*y)+0.1) * 1.1/(double(x*x+y*y)+0.1);
                            nb[col] += test;
                        }
                    }
                    sum = sum/n;
                    if(sum<0){sum=0;}
                    if(sum>255){sum=255;}
                    value[col] = sum;
                }
                Count.at<Vec3b>(i,j)=nb;
                Result.at<Vec3b>(i,j)=value;
            }
        }
    }*/

    cout<<"test2"<<endl;
    //imshow( "agnle1", Result );
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
    
    Mat X = interpolate_rgb(src);
    //Mat Y = interpolate_rgb(X);
    imshow("sum",X);
    imwrite(argv[2], X);
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
