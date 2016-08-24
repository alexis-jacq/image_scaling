#include <algorithm>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

//double THETA = 0.;
double THETA = 0.1;
double HLEVEL = 800;

Mat entropy(Mat S_, int bs){
    Mat S__mirrors;
    copyMakeBorder(S_, S__mirrors, bs-1,bs-1,bs-1,bs-1, BORDER_REFLECT );
    Mat Result = Mat_<double>(S__mirrors.rows, S__mirrors.cols);

    for(int i=0; i<S__mirrors.rows; i++){
        for(int j=0; j<S__mirrors.cols; j++){
            Result.at<double>(i,j) = 0;
        }
    }

    for(int i=bs-1; i<S__mirrors.rows-bs+1; i++){
        for(int j=bs-1; j<S__mirrors.cols-bs+1; j++){
            Vec3b pixel = S__mirrors.at<Vec3b>(i,j);
            double p = 0;
            for(int y=-1; y<2; y++){
                for(int x=-1; x<2; x++){
                    for(int col=0;col<3;col++){
                        Vec3b pixy = S__mirrors.at<Vec3b>(i+x,j+y);
                        p += abs(double(pixy[col])-double(pixel[col]))/255.;
                    }
                }
            }
            p = p/(9*3);
            Result.at<double>(i,j) = -p*log(p);
        }
    }
    return Result;
}

Mat mosaic_rgb(Mat X, Mat S, int ws){

    int bs = ws+1;//border_size;
    int wright = (ws-1)/2+1;
    int wleft = (ws-1)/2;

    Mat X_mirrors;
    Mat S_mirrors;
    Mat Result;
    copyMakeBorder(X, X_mirrors, bs-1,bs-1,bs-1,bs-1, BORDER_REFLECT );
    copyMakeBorder(S, S_mirrors, bs-1,bs-1,bs-1,bs-1, BORDER_REFLECT );
    //copyMakeBorder(X, Result, bs-1,bs-1,bs-1,bs-1, BORDER_REFLECT );
    Result = X.clone();

    Mat Sj = Mat_<int>(X_mirrors.rows, X_mirrors.cols);
    Mat Si = Mat_<int>(X_mirrors.rows, X_mirrors.cols);

    Mat ss = Mat_<double>(X_mirrors.rows, X_mirrors.cols);
    Mat sw = Mat_<double>(X_mirrors.rows, X_mirrors.cols);
    Mat se = Mat_<double>(X_mirrors.rows, X_mirrors.cols);
    Mat ee = Mat_<double>(X_mirrors.rows, X_mirrors.cols);
    Mat tot = Mat_<double>(X_mirrors.rows, X_mirrors.cols);

    double total_size = X.rows*X.cols;
    double indice = 0;

    Mat H = entropy(S,bs);
    bool start = true;

    for(int i=bs-1; i<0.6*X_mirrors.rows-bs+1; i++){
        for(int j=bs-1; j<0.6*X_mirrors.cols-bs+1; j++){

            Vec3b pixel = X_mirrors.at<Vec3b>(i,j);
            int ii=i-bs+1;
            int jj=j-bs+1;

            Mat windowX = Mat_<double>(ws,ws);
            Mat windowS = Mat_<double>(ws,ws);

            for(int y=-wleft; y<wright; y++){
                for(int x=-wleft; x<wright; x++){
                    Vec3b pixy = X_mirrors.at<Vec3b>(i+x,j+y);
                    windowX.at<double>(x+wleft,y+wleft) = (pixy[0]+pixy[1]+pixy[2])/3;
                }
            }

            //if(start || tot.at<double>(i,j)<HLEVEL){
            if(true){//(start || tot.at<double>(i,j)<HLEVEL){

                start=false;
                double dist = 1000.;
                int i_s = 0;
                int j_s = 0;

                for(int is=bs-1; is<0.5*S_mirrors.rows-bs+1; is++){
                    for(int js=bs-1; js<0.5*S_mirrors.cols-bs+1; js++){

                        for(int y=-wleft; y<wright; y++){
                            for(int x=-wleft; x<wright; x++){
                                Vec3b pixy = S_mirrors.at<Vec3b>(is+x,js+y);
                                windowS.at<double>(x+wleft,y+wleft) = (pixy[0]+pixy[1]+pixy[2])/3;
                                tot.at<double>(i+x,j+y) += H.at<double>(is+x,js+y);
                            }
                        }
                        ee.at<double>(i+1,j) += H.at<double>(is+1,js);
                        se.at<double>(i+1,j+1) += H.at<double>(is+1,js+1);
                        sw.at<double>(i+1,j+1) += H.at<double>(is-1,js+1);
                        ss.at<double>(i,j+1) += H.at<double>(is,js+1);

                        if(dist>sum((windowS-windowX)*(windowS-windowX))[0]){
                            dist = sum((windowS-windowX)*(windowS-windowX))[0];
                            i_s = is;
                            j_s = js;
                        }
                        if(dist<THETA){
                            break;
                        }
                    }
                }

                Si.at<int>(i,j) = i_s;
                Sj.at<int>(i,j) = j_s;

                /*Vec3b pixelS = S_mirrors.at<Vec3b>(i_s,j_s);
                Result.at<Vec3b>(i,j) = pixelS;*/

                for(int y=-wleft; y<wright; y++){
                    for(int x=-wleft; x<wright; x++){
                        if(ii+x>=0 && ii+x<Result.rows && jj+y>0 && jj+y<Result.cols){
                            Vec3b pixelS = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            Result.at<Vec3b>(ii+x,jj+y) = pixelS;
                        }
                    }
                }
            }
            //if(tot.at<double>(i,j)>HLEVEL){
            if(false){//(tot.at<double>(i,j)>HLEVEL){
                if(ee.at<double>(i,j)>=se.at<double>(i,j) && ee.at<double>(i,j)>=sw.at<double>(i,j) && ee.at<double>(i,j)>=ss.at<double>(i,j)){
                    int i_s = Si.at<int>(i-1,j)+1;
                    int j_s = Sj.at<int>(i-1,j);

                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixy = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            tot.at<double>(i+x,j+y) += H.at<double>(i_s+x,j_s+y);
                        }
                    }
                    ee.at<double>(i+1,j) += H.at<double>(i_s+1,j_s);
                    se.at<double>(i+1,j+1) += H.at<double>(i_s+1,j_s+1);
                    sw.at<double>(i+1,j+1) += H.at<double>(i_s-1,j_s+1);
                    ss.at<double>(i,j+1) += H.at<double>(i_s,j_s+1);

                    Si.at<int>(i,j) = i_s;
                    Sj.at<int>(i,j) = j_s;
                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixelS = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            Result.at<Vec3b>(i+x,j+y) = pixelS;
                        }
                    }
                }
                //
                if(se.at<double>(i,j)>=ee.at<double>(i,j) && se.at<double>(i,j)>=sw.at<double>(i,j) && se.at<double>(i,j)>=ss.at<double>(i,j)){
                    int i_s = Si.at<int>(i-1,j)+1;
                    int j_s = Sj.at<int>(i-1,j)+1;

                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixy = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            tot.at<double>(i+x,j+y) += H.at<double>(i_s+x,j_s+y);
                        }
                    }
                    ee.at<double>(i+1,j) += H.at<double>(i_s+1,j_s);
                    se.at<double>(i+1,j+1) += H.at<double>(i_s+1,j_s+1);
                    sw.at<double>(i+1,j+1) += H.at<double>(i_s-1,j_s+1);
                    ss.at<double>(i,j+1) += H.at<double>(i_s,j_s+1);

                    Si.at<int>(i,j) = i_s;
                    Sj.at<int>(i,j) = j_s;
                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixelS = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            Result.at<Vec3b>(i+x,j+y) = pixelS;
                        }
                    }
                }
                //
                if(sw.at<double>(i,j)>=se.at<double>(i,j) && sw.at<double>(i,j)>=ee.at<double>(i,j) && sw.at<double>(i,j)>=ss.at<double>(i,j)){
                    int i_s = Si.at<int>(i-1,j)-1;
                    int j_s = Sj.at<int>(i-1,j)+1;
                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixy = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            tot.at<double>(i+x,j+y) += H.at<double>(i_s+x,j_s+y);
                        }
                    }
                    ee.at<double>(i+1,j) += H.at<double>(i_s+1,j_s);
                    se.at<double>(i+1,j+1) += H.at<double>(i_s+1,j_s+1);
                    sw.at<double>(i+1,j+1) += H.at<double>(i_s-1,j_s+1);
                    ss.at<double>(i,j+1) += H.at<double>(i_s,j_s+1);
                    Si.at<int>(i,j) = i_s;
                    Sj.at<int>(i,j) = j_s;
                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixelS = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            Result.at<Vec3b>(i+x,j+y) = pixelS;
                        }
                    }
                }
                //
                if(ss.at<double>(i,j)>=se.at<double>(i,j) && ss.at<double>(i,j)>=sw.at<double>(i,j) && ss.at<double>(i,j)>=ee.at<double>(i,j)){
                    int i_s = Si.at<int>(i-1,j);
                    int j_s = Sj.at<int>(i-1,j)+1;

                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixy = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            tot.at<double>(i+x,j+y) += H.at<double>(i_s+x,j_s+y);
                        }
                    }
                    ee.at<double>(i+1,j) += H.at<double>(i_s+1,j_s);
                    se.at<double>(i+1,j+1) += H.at<double>(i_s+1,j_s+1);
                    sw.at<double>(i+1,j+1) += H.at<double>(i_s-1,j_s+1);
                    ss.at<double>(i,j+1) += H.at<double>(i_s,j_s+1);

                    Si.at<int>(i,j) = i_s;
                    Sj.at<int>(i,j) = j_s;
                    for(int y=-wleft; y<wright; y++){
                        for(int x=-wleft; x<wright; x++){
                            Vec3b pixelS = S_mirrors.at<Vec3b>(i_s+x,j_s+y);
                            Result.at<Vec3b>(i+x,j+y) = pixelS;
                        }
                    }
                }
            }
            indice++;
            if(true){//(int(100*indice/total_size)%10<1){
                cout<<100*indice/total_size<<" %"<<endl;
            }
        }
    }
    cout<<"testin3"<<endl;
    return Result;
}

int main ( int argc, char** argv )
{
    Mat src;
    Mat style;

    /// Load an image
    src = imread( argv[1] );
    style = imread( argv[2] );
    cout<<"testout1"<<endl;
    if( !src.data || !style.data)  { return -1; }

    cout<<"testout2"<<endl;
    Mat X = mosaic_rgb(src, style, 7);

    imshow("sum",X);
    imwrite(argv[3], X);
    waitKey(0);
    return 0;
}
