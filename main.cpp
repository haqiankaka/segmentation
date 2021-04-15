//一个并不清晰的参考文档：
//https://blog.csdn.net/wangyibo0201/article/details/51705966
//https://zhuanlan.zhihu.com/p/166369230

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <numeric>

using PointCloud  = std::vector<cv::Point2f>;

//int neighbor_size = 3; //邻居数量
float neibor_dist_min_threshold = 0.05;
float neibor_range_min_threshold = 0.05;
float density_threshold = 1; //lof算法，大于1 表示离群 ， 小于1 表示在点密集区域，等于1 表示相邻

struct PointWithRange
{
    float range;
    cv::Point2f pt;
};

float distance(cv::Point2f pt1,cv::Point2f pt2)
{
    float dist = std::sqrt(std::pow((pt1.x- pt2.x),2) + std::pow((pt1.y - pt2.y),2));
    return dist;
}
float range_dist(float range1,float range2)
{
    return range1 - range2;
}

//与它附近的点都很近，则通过检测 ，否则等待下一轮检测
bool IsPointNear(std::vector<PointWithRange> neibor_points,PointWithRange origin)
{
    size_t neighbor_size = neibor_points.size();
    for(int i = 0 ; i < neighbor_size ;i ++ )
    {
        if(distance(origin.pt,neibor_points[i].pt) > neibor_dist_min_threshold)
            return false;
        if(range_dist(origin.range,neibor_points[i].range) > neibor_range_min_threshold)
            return false;
    }
    return true;
}

float CalcLocalReachableDensity(std::vector<PointWithRange> neibor_points,PointWithRange origin)
{
    size_t neighbor_size = neibor_points.size();
    std::vector<std::pair<PointWithRange,float>> neighbor_info(neighbor_size);
    std::vector<float > distance_info(neighbor_size);

    for(size_t i = 0 ; i < neibor_points.size(); i ++)
    {
        std::pair<PointWithRange,float> neighbor_with_distance;
        neighbor_with_distance.second = distance(neibor_points[i].pt, origin.pt);
        neighbor_info[i] = neighbor_with_distance;
        distance_info[i] = neighbor_with_distance.second;
    }

    //求此时origin 点的k_distance; //周围邻居距离它的最大距离
    auto k_distance_ptr = std::max_element(distance_info.begin(),distance_info.end());
    float k_distance = *k_distance_ptr;

    //求origin 的局部可达距离,即周围邻居到达他的距离
    std::vector<float > reachable_distance(neighbor_size);
    for(size_t i = 0 ; i < neighbor_size;i++)
    {
        reachable_distance[i] = std::min(k_distance, neighbor_info[i].second);
    }

    float sum_reachable_dist = std::accumulate(reachable_distance.begin(),reachable_distance.end(),0.0f);
    std::cout<<"local reachable density:"<<float(neighbor_size) / sum_reachable_dist<<std::endl;
    return float(neighbor_size) / sum_reachable_dist;
}

float CalLocalOutlierFactor(std::vector<PointWithRange> neibor_points,PointWithRange origin)
{
    //目标点的local reachable density
    size_t neighbor_size = neibor_points.size();
    float origin_lrd = CalcLocalReachableDensity(neibor_points,origin);

    //求邻居点的local reachable density
    std::vector<float> neighbor_lrd(neighbor_size);
    for(size_t  i= 0; i < neighbor_size; i++)
    {
        //对于每个邻居点，生成它的邻居点
        std::vector<PointWithRange> tmp_neighbor;
        for(size_t j = 0 ; j < neighbor_size ; j++)
        {
            if(j != i )
            {
                tmp_neighbor.push_back(neibor_points[j]);
            }
        }
        tmp_neighbor.push_back(origin);
        neighbor_lrd[i] = CalcLocalReachableDensity(tmp_neighbor, neibor_points[i]);
    }
    float sum_neibor_lrd = std::accumulate(neighbor_lrd.begin(), neighbor_lrd.end(), 0.0f);
    float lof = (sum_neibor_lrd/float(neighbor_size)) / origin_lrd;
    std::cout<<"sum neibor lrd"<<sum_neibor_lrd<<"final lof value:"<<lof<<std::endl;
    return lof;
}

//领域的点一部分出现远距离，但在容忍范围内
bool IsPointInTorance(std::vector<PointWithRange> neibor_points,PointWithRange origin)
{
   float points_density =  CalLocalOutlierFactor(neibor_points,origin);
    if(points_density > density_threshold)
    {
        std::cout<<"origin point is local outlier !"<<std::endl;
        return false;
    }
    std::cout<<"origin point is inlier!"<<std::endl;
    return true;
}

//判断一个点是否和邻居相连
bool IsThisPointBelongToNeibor(std::vector<PointWithRange> neibor_points,PointWithRange origin)
{
    //与它附近的点都很近，则通过检测 ，否则等待下一轮检测
    if(IsPointNear(neibor_points,origin))
        return true;
    //虽然该点有些离群，但在容忍范围内，通过检测
    if(IsPointInTorance(neibor_points,origin))
        return true;

    return false;
}

std::vector<bool> Segmentation(std::vector<PointWithRange> pc)
{
    std::vector<bool> cluster_shape(pc.size(),0);
    cluster_shape.resize(pc.size());
    std::vector<size_t> neibor_index;
    const size_t neighbor_size = 3;
    std::vector<PointWithRange> neibor_points(neighbor_size);
    for(size_t i = 0 ; i < pc.size() ; i ++ )
    {
        //生成邻居的下标
        for(size_t j = 0; j < neighbor_size ; j ++)
        {
            neibor_index[j] = (i + j + 1) % pc.size();
            neibor_points[j] = pc[neibor_index[j]];
        }
        if(IsThisPointBelongToNeibor(neibor_points,pc[i]))
        {
            cluster_shape[i] = true;
        }
    }
    return cluster_shape;
}




int main()
{
    size_t neighbor_size = 3;
    std::vector<PointWithRange> neibor(neighbor_size);
    neibor[0] = {0.1,cv::Point2f(1,1),};
    neibor[1] = {0.1,cv::Point2f(1,2),};
    neibor[2] = {0.1,cv::Point2f(1.2,3),};
   //neibor[3] = {0.1,cv::Point2f(0.8,4),};
   // neibor[4] = {0.1,cv::Point2f(1,5),};
   //neibor[5] = {0.1,cv::Point2f(2.1,6),};
    PointWithRange origin = {0.1,cv::Point2f (1.2,3.2)};
    CalLocalOutlierFactor(neibor,origin);
}