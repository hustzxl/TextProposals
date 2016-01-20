#include "agglomerative_clustering.h"
#include<opencv/cv.h>
#define PI 3.1415926

void HierarchicalClustering::info_merge(HCluster& cluster,vector<float>& sample) {
    sample.push_back(0);
	Mat diameters      ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat strokes        ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat gradients      ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat fg_intensities ( cluster.elements.size(), 1, CV_32F, 1 );
	Mat bg_intensities ( cluster.elements.size(), 1, CV_32F, 1 );
	for (int i=cluster.elements.size()-1; i>=0; i--)
	{

	  diameters.at<float>(i,0)      = (float)max(regions.at(cluster.elements.at(i)).bbox_.width,
                                                   regions.at(cluster.elements.at(i)).bbox_.height);
	  strokes.at<float>(i,0)        = (float)regions.at(cluster.elements.at(i)).stroke_mean_;
	  gradients.at<float>(i,0)      = (float)regions.at(cluster.elements.at(i)).gradient_mean_;
	  fg_intensities.at<float>(i,0) = (float)regions.at(cluster.elements.at(i)).intensity_mean_;
	  bg_intensities.at<float>(i,0) = (float)regions.at(cluster.elements.at(i)).boundary_intensity_mean_;
    }

	Scalar mean,std;
	meanStdDev( diameters, mean, std );
	sample.push_back(std[0]/mean[0]);
	meanStdDev( strokes, mean, std );
	sample.push_back(std[0]/mean[0]); 
	meanStdDev( gradients, mean, std );
	sample.push_back(std[0]); 
	meanStdDev( fg_intensities, mean, std );
	sample.push_back(std[0]); 
	meanStdDev( bg_intensities, mean, std );
	sample.push_back(std[0]);    
//    for (vector<float>:: iterator iter = sample.begin(); iter!= sample.end();iter++)
//        std::cout<<*iter<<'\t';
//    std::cout<<std::endl;
}

double HierarchicalClustering::cal_angle(Point p1,Point p2){
    double angle = 0;
    double x_shift = p2.x - p1.x;
    double y_shift = p2.y - p1.y;
    double magnitude = sqrt( x_shift * x_shift + y_shift * y_shift );
    if (magnitude == 0) {
       angle = 0; 
    } else {
        angle = abs(x_shift) / magnitude;
    }
    angle = acos(angle);
}

int HierarchicalClustering::cal_angle_diff(Mat img,vector<int>& index,float& mean,float& stdev){
    int len = index.size();
    //std::cout<<"clusster size : "<<len<<std::endl;
    //for (int i = 0;i<len;i++) {
    //  std::cout<<"index: "<<index[i]<<std::endl;
    //}
    std::vector < double> angle;
    angle.clear();
    //for (int i = 0;i< len;i++) {
    //    Region re = regions[index[i]];
        //std::cout<<index[i]<<'\t'<<re.bbox_.x<<'\t'<<re.bbox_.y<<'\t'<<re.bbox_.width<<'\t'<<re.bbox_.height<<std::endl;
    //}
    for (int i = 0;i < len;i++){
        Region re = regions[ index[i] ];
        cv::Point p1 (re.bbox_.x + re.bbox_.width/2,re.bbox_.y + re.bbox_.height/2);
        //rectangle(img,re.bbox_,(0,0,255),2);
        //std::cout<<"location1: "<<index[i]<<" : "<<re.bbox_.x + re.bbox_.width/2<<" : "<<re.bbox_.y + re.bbox_.height/2<<std::endl;
        //location1 = location1 + char(re.bbox_.x) + " " + char(re.bbox_.width/2);
        //putText(img,location1,Point(p1.x,p1.y),font,2,(0,0,255));
        for (int j = 0;j < len ; j++){
            if (i == j){
                angle.push_back(0);
                continue;
            }
            Region re2 = regions[ index[j] ];
            cv::Point p2 (re2.bbox_.x + re2.bbox_.width/2,re2.bbox_.y + re2.bbox_.height/2);
            angle.push_back(cal_angle(p1,p2));
           // rectangle(img,re2.bbox_,(0,0,255),2);
           // std::cout<<"location2: "<<index[j]<<" : "<<re2.bbox_.x + re2.bbox_.width/2<<" : "<<re2.bbox_.y + re2.bbox_.height/2<<std::endl;
            //circle(img,p2,10,(0,0,255));
        }
    }
    imwrite("test_result.jpg",img);
    double sum = accumulate(angle.begin(),angle.end(),0);
    mean = sum / angle.size();
    double accum = 0;
    for (std::vector<double>::const_iterator iter = angle.begin();iter != angle.end();iter++){
        accum += (*iter-mean)*(*iter-mean);
    }
      
    stdev = sqrt(accum/(angle.size()-1));
    //std::cout<<"mean: "<<mean<<"stdev: "<<stdev<<std::endl;
    return 0;
}
HierarchicalClustering::HierarchicalClustering(vector<Region> &_regions,cv::Mat & _img): regions(_regions),img(_img)
{
    boost.load("./trained_boost_groups.xml", "boost");
}

//For feature space
void HierarchicalClustering::operator()(t_float *data, unsigned int num, int dim, unsigned char method, unsigned char metric, vector<HCluster> &merge_info)
{
    
    t_float *Z = (t_float*)malloc(((num-1)*4) * sizeof(t_float)); // we need 4 floats foreach merge
    linkage_vector(data, (int)num, dim, Z, method, metric,regions,img);
    build_merge_info(Z, data, (int)num, dim, merge_info);

    free(Z);
}


void HierarchicalClustering::build_merge_info(t_float *Z, t_float *X, int N, int dim, vector<HCluster> &merge_info)
{

    // walk the whole dendogram
    for (int i=0; i<(N-1)*4; i=i+4)
    {
        HCluster cluster;
        cluster.num_elem = Z[i+3]; //number of elements

        int node1  = Z[i];
        int node2  = Z[i+1];
        float dist = Z[i+2];
    
        if (node1<N) // child node1 is a single region
        {
            cluster.elements.push_back((int)node1);
            cluster.rect = regions.at(node1).bbox_;
            vector<float> point;
	    for (int n=0; n<dim; n++)
              point.push_back(X[node1*dim+n]);
            cluster.points.push_back(point);
        }
        else // child node1 is a cluster
        {
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node1-N).points.begin(),
                                  merge_info.at(node1-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node1-N).elements.begin(),
                                    merge_info.at(node1-N).elements.end());
            cluster.rect = merge_info.at(node1-N).rect;
        }
        if (node2<N) // child node2 is a single region
        {
            vector<float> point;
	        for (int n=0; n<dim; n++)
                  point.push_back(X[node2*dim+n]);
            cluster.points.push_back(point);
            cluster.elements.push_back((int)node2);
            cluster.rect = cluster.rect | regions.at(node2).bbox_; // min. area rect containing node 1 and node2
        }
        else // child node2 is a cluster
        {
            cluster.points.insert(cluster.points.end(),
                                  merge_info.at(node2-N).points.begin(),
                                  merge_info.at(node2-N).points.end());
            cluster.elements.insert(cluster.elements.end(),
                                    merge_info.at(node2-N).elements.begin(),
                                    merge_info.at(node2-N).elements.end());
            cluster.rect = cluster.rect | merge_info.at(node2-N).rect; // min. area rect containing node 1 and node2
        }
    
        cluster.node1 = node1;
        cluster.node2 = node2;

        Minibox mb;
        for (int i=0; i<cluster.points.size(); i++)
          mb.check_in(&cluster.points.at(i));	
        
        long double volume = mb.volume(); 
        if (volume >= 1) volume = 0.999999;
        if (volume == 0) volume = 0.000001; //TODO is this the minimum we can get?
		
        cluster.nfa = -1*(int)NFA( N, cluster.points.size(), (double) volume, 0); //this uses an approximation for the nfa calculations (faster)

        /* predict group class with boost */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//zxl
        float mean = 0,stdev = 0;
        if (cluster.elements.size() < 3) {
            cluster.constrain = false;
            merge_info.push_back(cluster);
        }else {
            //cal_angle_diff(img,cluster.elements,mean,stdev);
            if (mean < PI / 32 && stdev < PI / 16) {
                //std::cout<<"mean: "<<mean<<" stdev: "<<stdev<<std::endl;
                cluster.constrain = true;
            } else {
                std::cout<<"error"<<std::endl;
                cluster.constrain = false;
            }
            vector<float>sample; 
            sample.clear();
            info_merge(cluster,sample);
            float votes_group = boost.predict( Mat(sample), Mat(), Range::all(), false, true );
            cluster.probability = (double)1-(double)1/(1+exp(-2*votes_group));
            merge_info.push_back(cluster);
        }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	    
    }
}
