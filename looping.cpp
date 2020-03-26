#include "opencv2/highgui.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace optflow;
using namespace std;

bool fexists(const char *filename) {
    std::ifstream ifile(filename);
    return (bool)ifile;
    }
    
float mag(Point2f vec) { //magnitude of vector
    return sqrt((vec.x*vec.x)+(vec.y*vec.y));
}

Mat DrawFlow(Mat flow, Mat background, Vec3b flowcolor, int sampling, float vscale, float motiontreshold) {
    int flowidth =flow.cols;
    int flowheight=flow.rows;
    int backwidth =background.cols;
    int backheight=background.rows;
    float ratio=(float)backwidth/(float)flowidth;
    Point2f getforce,pos;
    for (int i = 0; i < flowheight; i++) {
		for (int j = 0; j < flowidth; j++) {
            pos.x=j;
            pos.y=i;
            Point2f val=flow.at<Point2f>(i, j);
            if (!(i%sampling) && !(j%sampling)) {
                if (mag(val) > motiontreshold) {
                    circle(background,ratio*pos,1,flowcolor,1,LINE_AA);
                    line(background,ratio*pos,(ratio*pos)+(ratio*val*vscale),flowcolor,1,LINE_AA);
                    }
            }
        }
    }
    return background;
}

string flowmethodtext(int flowmethod) {
    if (flowmethod == 0) {return "no opticalflow";}
    if (flowmethod == 1) {return "ocv_deepflow";}
    if (flowmethod == 2) {return "farneback";}
    if (flowmethod == 3) {return "tvl1";}
    if (flowmethod == 4) {return "simpleflow";}
    if (flowmethod == 5) {return "sparsetodense";}
    if (flowmethod == 6) {return "rlof_epic";}
    if (flowmethod == 7) {return "rlof_ric";}
    if (flowmethod == 8) {return "pcaflow";}
    if (flowmethod == 9) {return "DISflow";}
    if (flowmethod == 10) {return "deepmatch/flow";}
}

int main(int argc, char **argv)
{
    cout << "usage : inputdir framename extension loop_start loop_end loop_half_window" << endl;
    cout << "> ./looping ../images lionwalk jpg 94 120 4" << endl;
    cout << endl;
    char *inputdir=             argv[1];
    char *framename=            argv[2];
    char *extension=            argv[3];
    int loop_start =            atoi(argv[4]);
    int loop_end =              atoi(argv[5]);
    int loop_half_window =      atoi(argv[6]);
    string input;
    int flowmethod=1;
    cout << "0: no opticalflow " << endl;
    cout << "1: opencv deepflow " << endl;
    cout << "2: farneback " << endl;
    cout << "3: tvl1 " << endl;
    cout << "4: simpleflow " << endl;
    cout << "5: sparsetodense " << endl;
    cout << "6: rlof_epic " << endl;
    cout << "7: rlof_ric " << endl;
    cout << "8: pcaflow " << endl;
    cout << "9: DISflow " << endl;
    cout << "10: deepmatching/deepflow " << endl;
    cout << "opticalflow method method[" << flowmethod << "] : ";
    getline(cin,input);
    if ( !input.empty() ) {
        istringstream stream(input);
        stream >> flowmethod;
    }
    int reuseflow=0;
    /*cout << "(0:no 1:yes)   use existing flow files[" << reuseflow << "] : ";
    getline(cin,input);
    if ( !input.empty() ) {
        istringstream stream(input);
        stream >> reuseflow;
    }
    */
    float flowscale =0.5;
    //downscale images for optical flow (.5 = half)
    cout << "downscale image for flow calculation[" << flowscale << "] : ";
    getline(cin,input);
    if ( !input.empty() ) {
        istringstream stream(input);
        stream >> flowscale;
    }
    
    //numbers
    int loop_cut            =loop_start+round((loop_end-loop_start)/2);
    int loop_out            =loop_end+(loop_cut-loop_start);
    int Boffset             =loop_start-loop_end;
    int transition_start    =loop_end-loop_half_window;
    int transition_end      =loop_end+loop_half_window;
    
    cout << "used frames: " << loop_start-loop_half_window+1 << "-" << loop_end+loop_half_window-1 << endl;
    cout << "final loop : " << loop_cut << "-" << loop_out << endl;
    cout << "offset     : " << Boffset << endl;
    cout << endl;
    
    //variables declaration
    char Aimage[1024];
    char Bimage[1024];
    char ASimage[1024];
    char BSimage[1024];
    char outputdir[1024];
    char outputimage[1024];
    char showflowimage[1024];
    char ABflowfile[1024];
    char BAflowfile[1024];
    char linuxcmd[1024];
    Mat imageA,imageB,simageA,simageB,bwA,bwB,mapx,mapy,warpA,warpB,showflowbg,finalframe;
    Mat_<Point2f> BAflow,ABflow;
    Ptr<DenseOpticalFlow> algorithm;
    double startTick,time;
    double averagetime=0;
    int timecount=0;
    
    //static flags
    int startatframe1=1; //output sequence starts at 1
    int useGpu = 1; //opencv deepflow uses gpu
    cv::ocl::setUseOpenCL(useGpu);
    int showflow=0;
    float showflowsampling=50.;
    int writeflo=1;
    int writetext=0;
    string comment;
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 2;
    int thickness = 1;
    //path to deepmatching and deepflow2 executables.adapt to your system.
    char deepmatching[] ="/shared/luluxxxx/video-loops/deepmatching-static";
    char deepflow[]     ="/shared/luluxxxx/video-loops/deepflow2-static";
    /*
     * you may need lipng12.so.0 to run the deep statics on Ubuntu 18.04+.
     * it's here : 
     * https://launchpad.net/~ubuntu-security/+archive/ubuntu/ppa/+build/15108504/+files/libpng12-0_1.2.54-1ubuntu1.1_amd64.deb
     * https://www.linuxuprising.com/2018/05/fix-libpng12-0-missing-in-ubuntu-1804.html
     */
    
    //create outputdir
    sprintf(outputdir,"./looping_%s_%d_%d",framename,loop_start,loop_end);
    sprintf(linuxcmd,"mkdir %s",outputdir);
    cout << linuxcmd << endl;
    system(linuxcmd);
    
    //no flow , just copy
    for (int i = loop_cut; i <= transition_start; i++) {
        int Aframe=i;
        sprintf(Aimage,"%s/%s.%04d.%s",inputdir,framename,Aframe,extension);
        cout << "processing loop frame : " << i-loop_cut+1 << endl;
        cout <<"reading : " << Aimage << endl;
        imageA= imread(Aimage,IMREAD_COLOR);
        //writing text
        if (writetext == 1)  {
            int baseline=0;
            comment=flowmethodtext(flowmethod);
            Size textSize=getTextSize(comment,fontFace,fontScale,thickness,&baseline);
            baseline += thickness;
            Point textOrg((imageA.cols-textSize.width)/2,(imageA.rows-textSize.height));
            rectangle(imageA,textOrg+Point(0,baseline),textOrg+Point(textSize.width,-textSize.height),Scalar(0,0,0),FILLED);
            putText(imageA,comment,textOrg,fontFace,fontScale,Scalar::all(255),thickness,LINE_AA);
        }
        //writing result
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,imageA);
        cout << endl;
    }
    //using flow
    for (int i = transition_start + 1; i < transition_end; i++) {
        int Aframe=i;
        int Bframe=Aframe+Boffset;
        sprintf(Aimage,"%s/%s.%04d.%s",inputdir,framename,Aframe,extension);
        sprintf(Bimage,"%s/%s.%04d.%s",inputdir,framename,Bframe,extension);
        cout << "processing loop frame : " << i-loop_cut+1 << endl;
        //read A/B image
        cout <<"readingA : " << Aimage << endl;
        imageA= imread(Aimage,IMREAD_COLOR);
        cout <<"readingB : " << Bimage << endl;
        imageB= imread(Bimage,IMREAD_COLOR);
        //compute fractional value
        float frac=(float((i-transition_start)/float((transition_end-transition_start))));
        cout << "fractional : " << frac << endl;
        
        if (flowmethod > 0) {
            startTick = (double) getTickCount(); // measure time
            if (flowmethod == 1) { //use opencv deep optical flow
                algorithm = createOptFlow_DeepFlow();
                cout << "using opencv deepflow algorithm" << endl;
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(bwA,bwA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(bwB,bwB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << bwA.size[1] << "," << bwA.size[0] << "]" << endl;
                }
                BAflow = Mat(bwA.size[0], bwA.size[1], CV_32FC2);
                ABflow = Mat(bwB.size[0], bwB.size[1], CV_32FC2);
                //process optical flow
                cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwB, bwA, BAflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwB, bwA, BAflow);
                cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwA, bwB, ABflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwA, bwB, ABflow);
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 2) { //use farneback optical flow
                algorithm = createOptFlow_Farneback();
                cout << "using farneback algorithm" << endl;
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(bwA,bwA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(bwB,bwB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << bwA.size[1] << "," << bwA.size[0] << "]" << endl;
                }
                BAflow = Mat(bwA.size[0], bwA.size[1], CV_32FC2);
                ABflow = Mat(bwB.size[0], bwB.size[1], CV_32FC2);
                //process optical flow
                cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwB, bwA, BAflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwB, bwA, BAflow);
                cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwA, bwB, ABflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwA, bwB, ABflow);
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 3) { //use tvl1 optical flow
                algorithm = createOptFlow_DualTVL1();
                cout << "using DualTVL1 algorithm" << endl;
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(bwA,bwA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(bwB,bwB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << bwA.size[1] << "," << bwA.size[0] << "]" << endl;
                }
                BAflow = Mat(bwA.size[0], bwA.size[1], CV_32FC2);
                ABflow = Mat(bwB.size[0], bwB.size[1], CV_32FC2);
                //process optical flow
                cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwB, bwA, BAflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwB, bwA, BAflow);
                cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwA, bwB, ABflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwA, bwB, ABflow);
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 4) { //use simpleflow optical flow
                algorithm = createOptFlow_SimpleFlow();
                cout << "using SimpleFlow algorithm" << endl;
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    BAflow = Mat(simageA.size[0], simageA.size[1], CV_32FC2);
                    ABflow = Mat(simageB.size[0], simageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageB, simageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageB, simageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageA, simageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageA, simageB, ABflow);
                }
                else {
                    BAflow = Mat(imageA.size[0], imageA.size[1], CV_32FC2);
                    ABflow = Mat(imageB.size[0], imageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageB, imageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageB, imageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageA, imageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageA, imageB, ABflow);
                }
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 5) { //use SparseToDense optical flow
                algorithm = createOptFlow_SparseToDense();
                cout << "using SparseToDense algorithm" << endl;
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    BAflow = Mat(simageA.size[0], simageA.size[1], CV_32FC2);
                    ABflow = Mat(simageB.size[0], simageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageB, simageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageB, simageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageA, simageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageA, simageB, ABflow);
                }
                else {
                    BAflow = Mat(imageA.size[0], imageA.size[1], CV_32FC2);
                    ABflow = Mat(imageB.size[0], imageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageB, imageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageB, imageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageA, imageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageA, imageB, ABflow);
                }
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 6) { //use DenseRLOF INTERP_EPIC optical flow
                algorithm = createOptFlow_DenseRLOF();
                Ptr<DenseRLOFOpticalFlow> rlof = algorithm.dynamicCast< DenseRLOFOpticalFlow>();
                rlof->setInterpolation(INTERP_EPIC);
                rlof->setForwardBackward(1.f);
                cout << "using DenseRLOF [EPIC] algorithm" << endl;
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    BAflow = Mat(simageA.size[0], simageA.size[1], CV_32FC2);
                    ABflow = Mat(simageB.size[0], simageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageB, simageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageB, simageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageA, simageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageA, simageB, ABflow);
                }
                else {
                    BAflow = Mat(imageA.size[0], imageA.size[1], CV_32FC2);
                    ABflow = Mat(imageB.size[0], imageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageB, imageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageB, imageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageA, imageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageA, imageB, ABflow);
                }
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 7) { //use DenseRLOF INTERP_RIC optical flow
                algorithm = createOptFlow_DenseRLOF();
                Ptr<DenseRLOFOpticalFlow> rlof = algorithm.dynamicCast< DenseRLOFOpticalFlow>();;
                rlof->setInterpolation(INTERP_RIC);
                rlof->setForwardBackward(1.f);
                cout << "using DenseRLOF [RIC] algorithm" << endl;
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    BAflow = Mat(simageA.size[0], simageA.size[1], CV_32FC2);
                    ABflow = Mat(simageB.size[0], simageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageB, simageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageB, simageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageA, simageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageA, simageB, ABflow);
                }
                else {
                    BAflow = Mat(imageA.size[0], imageA.size[1], CV_32FC2);
                    ABflow = Mat(imageB.size[0], imageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageB, imageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageB, imageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageA, imageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageA, imageB, ABflow);
                }
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 8) { //use PCAFlow optical flow
                algorithm = createOptFlow_PCAFlow();
                cout << "using PCAFlow algorithm" << endl;
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    BAflow = Mat(simageA.size[0], simageA.size[1], CV_32FC2);
                    ABflow = Mat(simageB.size[0], simageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageB, simageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageB, simageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(simageA, simageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(simageA, simageB, ABflow);
                }
                else {
                    BAflow = Mat(imageA.size[0], imageA.size[1], CV_32FC2);
                    ABflow = Mat(imageB.size[0], imageB.size[1], CV_32FC2);
                    //process optical flow
                    cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageB, imageA, BAflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageB, imageA, BAflow);
                    cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                    if (useGpu) algorithm->calc(imageA, imageB, ABflow.getUMat(ACCESS_RW));
                    else algorithm->calc(imageA, imageB, ABflow);
                }
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 9) { //use DISflow optical flow
                algorithm = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
                cout << "using DISflow [MEDIUM] algorithm" << endl;
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(bwA,bwA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(bwB,bwB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << bwA.size[1] << "," << bwA.size[0] << "]" << endl;
                }
                BAflow = Mat(bwA.size[0], bwA.size[1], CV_32FC2);
                ABflow = Mat(bwB.size[0], bwB.size[1], CV_32FC2);
                //process optical flow
                cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwB, bwA, BAflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwB, bwA, BAflow);
                cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwA, bwB, ABflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwA, bwB, ABflow);
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    cout << "writing flo files" << endl;
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            
            if (flowmethod == 10) {
                sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                sprintf(ASimage,"%s/tmpA.png",outputdir);
                sprintf(BSimage,"%s/tmpB.png",outputdir);
                if (flowscale != 1) {
                    cv::resize(imageA,simageA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(imageB,simageB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << simageA.size[1] << "," << simageA.size[0] << "]" << endl;
                    //write out rescaled image
                    cv::imwrite(ASimage,simageA);
                    cv::imwrite(BSimage,simageB);
                }
                else {
                    cv::imwrite(ASimage,imageA);
                    cv::imwrite(BSimage,imageB);
                }
                if (!fexists(BAflowfile) || reuseflow==0) {
                    sprintf(linuxcmd,"%s %s %s -nt 0 | %s %s %s %s -match -sintel >/dev/null",deepmatching,BSimage,ASimage,deepflow,BSimage,ASimage,BAflowfile);
                    cout << "processing B->A deepmatching/deepflow2 : " << BAflowfile << " [downscale : " << flowscale << "]" << endl;
                    //cout << linuxcmd << endl;
                    system(linuxcmd);
                }
                else {
                    cout << "reusing existing flow file : " << BAflowfile << endl;
                }
                if (!fexists(ABflowfile) || reuseflow==0) {
                    sprintf(linuxcmd,"%s %s %s -nt 0 | %s %s %s %s -match -sintel >/dev/null",deepmatching,ASimage,BSimage,deepflow,ASimage,BSimage,ABflowfile);
                    cout << "processing A->B deepmatching/deepflow2 : " << ABflowfile << " [downscale : " << flowscale << "]" << endl;
                    //cout << linuxcmd << endl;
                    system(linuxcmd);
                }
                else {
                    cout << "reusing existing flow file : " << ABflowfile << endl;
                }
                cout << "reading B->A  flow file :  : " << BAflowfile << endl;
                BAflow = readOpticalFlow(BAflowfile);
                cout << "reading A->B  flow file :  : " << ABflowfile << endl;
                ABflow     = readOpticalFlow(ABflowfile);
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
            }
            
            //warping with flow A -> B
            mapx = cv::Mat::zeros(ABflow.size(), CV_32FC1);
            mapy = cv::Mat::zeros(ABflow.size(), CV_32FC1);
            for (int y = 0; y < ABflow.rows; ++y)
                {
                for (int x = 0; x < ABflow.cols; ++x)
                    {
                    Vec2f of = ABflow.at<Vec2f>(y, x);
                    mapx.at<float>(y, x) = (float)x - (of.val[0]*frac);
                    mapy.at<float>(y, x) = (float)y - (of.val[1]*frac);
                    }
                }
            cout << "warping Aframe" << endl;
            cv::remap(imageA,warpA,mapx,mapy,INTER_CUBIC,BORDER_REPLICATE);
            //warping with flow B -> A
            mapx = cv::Mat::zeros(BAflow.size(), CV_32FC1);
            mapy = cv::Mat::zeros(BAflow.size(), CV_32FC1);
            for (int y = 0; y < BAflow.rows; ++y)
                {
                for (int x = 0; x < BAflow.cols; ++x)
                    {
                    Vec2f of = BAflow.at<Vec2f>(y, x);
                    mapx.at<float>(y, x) = (float)x - (of.val[0]*(1.0-frac));
                    mapy.at<float>(y, x) = (float)y - (of.val[1]*(1.0-frac));
                    }
                }
            cout << "warping Bframe" << endl;
            cv::remap(imageB,warpB,mapx,mapy,INTER_CUBIC,BORDER_REPLICATE);
            //final mix
            cv::addWeighted(warpA,1.0-frac,warpB,frac,0.0,finalframe,-1);
            cout << "mixing A/B" << endl;
            
            //pathetic attempt to understand what's going on
            if (showflow == 1) {
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                vector<Mat> channels;
                channels.push_back(bwA);
                channels.push_back(cv::Mat::zeros(bwA.size(), CV_8UC1));
                channels.push_back(bwB);
                merge(channels,showflowbg);
                DrawFlow(ABflow,showflowbg,Vec3b(0,0,255),bwA.cols/showflowsampling,1.,0);
                DrawFlow(BAflow,showflowbg,Vec3b(255,0,0),bwA.cols/showflowsampling,1.,0);
                sprintf(showflowimage,"%s/%s_showflow.%04d.%s",outputdir,framename,i-loop_cut+1,extension);
                cout << "writing showflow frame : " << showflowimage << endl;
                cv::imwrite(showflowimage,showflowbg);
            }
        }
        else {
            cout << "no optical flow : blending A/B" << endl;
            cv::addWeighted(imageA,1.0-frac,imageB,frac,0.0,finalframe,-1);
        }
        //writing text
        if (writetext == 1)  {
            int baseline=0;
            comment=flowmethodtext(flowmethod);
            Size textSize=getTextSize(comment,fontFace,fontScale,thickness,&baseline);
            baseline += thickness;
            Point textOrg((finalframe.cols-textSize.width)/2,(finalframe.rows-textSize.height));
            rectangle(finalframe,textOrg+Point(0,baseline),textOrg+Point(textSize.width,-textSize.height),Scalar(0,0,0),FILLED);
            putText(finalframe,comment,textOrg,fontFace,fontScale,Scalar::all(255),thickness,LINE_AA);
        }
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,finalframe);
        time = ((double) getTickCount() - startTick) / getTickFrequency();
        printf("Time [s]: %.3f\n", time);
        timecount++;
        averagetime+=time;
        cout << endl;
    }
    //no flow , just copy
    for (int i = transition_end; i < loop_out; i++) {
        int Bframe=i+Boffset;
        sprintf(Bimage,"%s/%s.%04d.%s",inputdir,framename,Bframe,extension);
        cout << "processing loop frame : " << i-loop_cut+1 << endl;
        cout <<"reading : " << Bimage << endl;
        imageB= imread(Bimage,IMREAD_COLOR);
        if (writetext == 1)  {
            int baseline=0;
            comment=flowmethodtext(flowmethod);
            Size textSize=getTextSize(comment,fontFace,fontScale,thickness,&baseline);
            baseline += thickness;
            Point textOrg((imageB.cols-textSize.width)/2,(imageB.rows-textSize.height));
            rectangle(imageB,textOrg+Point(0,baseline),textOrg+Point(textSize.width,-textSize.height),Scalar(0,0,0),FILLED);
            putText(imageB,comment,textOrg,fontFace,fontScale,Scalar::all(255),thickness,LINE_AA);
        }
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,imageB);
        cout << endl;
    }
    printf("\nDone! : OpticalFlow AverageTime [s]: %.3f\n", averagetime/timecount);
}
