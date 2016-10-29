#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;


Mat interpolate_rgb(Mat X, bool ok){

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
    Mat Cards(2*X.rows, 2*X.cols,  CV_32FC1, float(0));
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
                    for(int col=0;col<3;col++){
                        d1 += abs(pow((double(pixel[col])/255.-double(pix[col])/255.),2));
                    }
                    d1 /= 3;

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

                    score += exp(1*d2-10*d1);//*d2;
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
            Cards.at<float>(ii,jj) = 1;

            // looking for friends
            int angle_indice = 0;

            for(auto map : angle_maps){


                int a1 = angles_1[angle_indice];
                int a2 = angles_2[angle_indice];

                for(int s=1; s<5; s++){

                    a1*=s;
                    a2*=s;

                    if(ii+a1>=0 && ii+a1<Result.rows && jj+a2>=0 && jj+a2<Result.cols){

                        Vec3f value = Values.at<Vec3f>(ii+a1,jj+a2);
                        Vec3f nb = Counts.at<Vec3f>(ii+a1,jj+a2);
                        double angle_score = double(map.at<float>(i,j));

                        double score = angle_score;
                        double theta = exp(0.692);
                        //double theta = 0;

                        if(score>theta){

                            double d = exp(5*score + 2.3*(exp(1)-exp(s)));
                            //double d=0.1+(score/255.)*(score/255.);
                            //double d=1.;
                            // they are friends
                            for(int col=0;col<3;col++){

                                value[col]+=d*pixel[col];;
                                nb[col]+=d;
                            }
                            Values.at<Vec3f>(ii+a1,jj+a2)=value;
                            Counts.at<Vec3f>(ii+a1,jj+a2)=nb;
                            Cards.at<float>(ii+a1,jj+a2)+=1;

                            /*if(ii+2*a1>=0 && ii+2*a1<Result.rows && jj+2*a2>=0 && jj+2*a2<Result.cols){
                                Values.at<Vec3f>(ii+2*a1,jj+2*a2)=value;
                                Counts.at<Vec3f>(ii+2*a1,jj+2*a2)=nb;
                                Cards.at<float>(ii+2*a1,jj+2*a2)+=1;
                            }*/
                        }
                    }
                }
                angle_indice += 1;
            }
        }
    }


    for(int i=0; i<Result.rows; i++){
        for(int j=0; j<Result.cols; j++){

            Vec3f nb = Counts.at<Vec3f>(i,j);
            float card = Counts.at<float>(i,j);

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


    // 4. treat friendless
    //--------------------
    for(int i=0; i<Result.rows; i++){
        for(int j=0; j<Result.cols; j++){

            Vec3f nb = Counts.at<Vec3f>(i,j);
            float card = Cards.at<float>(i,j);

            //for(int col=0;col<3;col++){
            if(card<=0){
                // you have no friends
                Vec3b value;
                vector<Vec3b> results;

                if(i-1>=0){
                    float card2 = Cards.at<float>(i-1,j);
                    if(card2>0){results.push_back(Result.at<Vec3b>(i-1,j));}
                }
                if(j-1>=0){
                    float card2 = Cards.at<float>(i,j-1);
                    if(card2>0){results.push_back(Result.at<Vec3b>(i,j-1));}
                }
                if(i-1>=0 && j-1>=0){
                    float card2 = Cards.at<float>(i-1,j-1);
                    if(card2>0){results.push_back(Result.at<Vec3b>(i-1,j-1));}
                }
                if(i-1>=0 && j+1<Result.cols){
                    float card2 = Cards.at<float>(i-1,j+1);
                    if(card2>0){results.push_back(Result.at<Vec3b>(i-1,j+1));}
                }

                if(ok){
                    if(i+1<Result.rows){
                        float card2 = Cards.at<float>(i+1,j);
                        if(card2>0){results.push_back(Result.at<Vec3b>(i+1,j));}
                    }
                    if(j+1<Result.cols){
                        float card2 = Cards.at<float>(i,j+1);
                        if(card2>0){results.push_back(Result.at<Vec3b>(i,j+1));}
                    }
                    if(i+1<Result.rows && j+1<Result.cols){
                        float card2 = Cards.at<float>(i+1,j+1);
                        if(card2>0){results.push_back(Result.at<Vec3b>(i+1,j+1));}
                    }
                    if(i+1<Result.rows && j-1>=0){
                        float card2 = Cards.at<float>(i+1,j-1);
                        if(card2>0){results.push_back(Result.at<Vec3b>(i+1,j-1));}
                    }
                }
                

                for(int col=0;col<3;col++){
                    double sum = 0;
                    double count=exp(-15);
                    for(auto result : results){
                        sum+=result[col];
                        count+=1;
                    }
                    value[col] = sum/count;
                }
                Result.at<Vec3b>(i,j) = value;
            }
        }
    }

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
    
    Mat X = interpolate_rgb(src,true);
    Mat Y = interpolate_rgb(X,true);
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
