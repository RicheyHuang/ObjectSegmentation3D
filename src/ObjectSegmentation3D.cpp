#define _CRT_SECURE_NO_WARNINGS

#include <pcl\point_cloud.h>
#include <pcl\io\ply_io.h>
#include <pcl\search\kdtree.h>

#include <pcl\features\normal_3d.h>
#include <pcl\features\moment_of_inertia_estimation.h>

#include <pcl\segmentation\extract_clusters.h>
#include <pcl\segmentation\region_growing.h>
#include <pcl\segmentation\supervoxel_clustering.h>
#include <pcl\segmentation\lccp_segmentation.h>
#include <pcl\segmentation\cpc_segmentation.h>
#include <pcl\segmentation\min_cut_segmentation.h>

#include <pcl\visualization\point_picking_event.h>
#include <pcl\visualization\mouse_event.h>
#include <pcl\visualization\keyboard_event.h>
#include <pcl\visualization\pcl_visualizer.h>

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <set>

class EuclideanClustering
{
public:
	EuclideanClustering() :
		_search_radius(0.03), 
		_cluster_tolerance(0.01), 
		_min_cluster_size(1500), 
		_max_cluster_size(5e5) {};
	void setInputCloudPath(std::string data_path)
	{
		_data_path = data_path;
	}
	void setRadiusSearch(double search_radius) 
	{
		_search_radius = search_radius;
	};
	void setClusterTolerance(double cluster_tolerance) 
	{
		_cluster_tolerance = cluster_tolerance;
	};
	void setMinClusterSize(int min_cluster_size) 
	{
		_min_cluster_size = min_cluster_size;
	};
	void setMaxClusterSize(int max_cluster_size)
	{
		_max_cluster_size = max_cluster_size;
	};
	int  segment()
	{
		/* INPUT DATA */
		pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud(obj);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(_search_radius);
		ne.compute(*normals);

		/* EUCLIDEAN CLUSTERING */
		std::vector<pcl::PointIndices> clusters;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance(_cluster_tolerance); // unit: 1cm
		ec.setMinClusterSize(_min_cluster_size);
		ec.setMaxClusterSize(_max_cluster_size);
		ec.setSearchMethod(tree);
		ec.setInputCloud(obj);
		ec.extract(clusters);

		if (clusters.empty())
		{
			std::cout << "No Cluster Extracted!!" << std::endl;
			return 1;
		}
		else
		{
			std::cout << clusters.size() << " Clusters Extracted!!" << std::endl;
		}

		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;

		/* VISUALIZATION */
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
		int view_port = 0;

		for (int i = 0; i < clusters.size(); i++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			temp_cloud->width = clusters[i].indices.size();
			temp_cloud->height = 1;
			std::vector<int>::iterator iter = clusters[i].indices.begin();
			while (iter != clusters[i].indices.end())
			{
				temp_cloud->push_back(obj->at(*iter));
				iter++;
			}
			clouds.push_back(temp_cloud);
		}

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
		pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
		Eigen::Matrix3f rotation_matrix_OBB;

		for (int i = 0; i < clouds.size(); i++)
		{
			bb.setInputCloud(clouds[i]);
			bb.compute();

			bb.getAABB(min_point_AABB, max_point_AABB);
			bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

			Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
			Eigen::Quaternionf quaternion(rotation_matrix_OBB);

			std::stringstream cloud_name_ss;
			cloud_name_ss << "cloud" << i << std::endl;
			std::string cloud_name = cloud_name_ss.str();

			pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(clouds[i]);
			viewer->addPointCloud(clouds[i], temp_handler, cloud_name, view_port);

			/* DRAWING ORIENTED BOUNDING BOXES */
			/*Eigen::Vector3f p1_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p2_OBB(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p3_OBB(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p4_OBB(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p5_OBB(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p6_OBB(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p7_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p8_OBB(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

			p1_OBB = rotation_matrix_OBB * p1_OBB + position;
			p2_OBB = rotation_matrix_OBB * p2_OBB + position;
			p3_OBB = rotation_matrix_OBB * p3_OBB + position;
			p4_OBB = rotation_matrix_OBB * p4_OBB + position;
			p5_OBB = rotation_matrix_OBB * p5_OBB + position;
			p6_OBB = rotation_matrix_OBB * p6_OBB + position;
			p7_OBB = rotation_matrix_OBB * p7_OBB + position;
			p8_OBB = rotation_matrix_OBB * p8_OBB + position;

			pcl::PointXYZ pt1_OBB(p1_OBB(0), p1_OBB(1), p1_OBB(2));
			pcl::PointXYZ pt2_OBB(p2_OBB(0), p2_OBB(1), p2_OBB(2));
			pcl::PointXYZ pt3_OBB(p3_OBB(0), p3_OBB(1), p3_OBB(2));
			pcl::PointXYZ pt4_OBB(p4_OBB(0), p4_OBB(1), p4_OBB(2));
			pcl::PointXYZ pt5_OBB(p5_OBB(0), p5_OBB(1), p5_OBB(2));
			pcl::PointXYZ pt6_OBB(p6_OBB(0), p6_OBB(1), p6_OBB(2));
			pcl::PointXYZ pt7_OBB(p7_OBB(0), p7_OBB(1), p7_OBB(2));
			pcl::PointXYZ pt8_OBB(p8_OBB(0), p8_OBB(1), p8_OBB(2));

			std::stringstream OBB_edge1_ss, OBB_edge2_ss,  OBB_edge3_ss,  OBB_edge4_ss,
			OBB_edge5_ss, OBB_edge6_ss,  OBB_edge7_ss,  OBB_edge8_ss,
			OBB_edge9_ss, OBB_edge10_ss, OBB_edge11_ss, OBB_edge12_ss;

			OBB_edge1_ss  << "OBB_edge1_"  << i << std::endl;
			OBB_edge2_ss  << "OBB_edge2_"  << i << std::endl;
			OBB_edge3_ss  << "OBB_edge3_"  << i << std::endl;
			OBB_edge4_ss  << "OBB_edge4_"  << i << std::endl;
			OBB_edge5_ss  << "OBB_edge5_"  << i << std::endl;
			OBB_edge6_ss  << "OBB_edge6_"  << i << std::endl;
			OBB_edge7_ss  << "OBB_edge7_"  << i << std::endl;
			OBB_edge8_ss  << "OBB_edge8_"  << i << std::endl;
			OBB_edge9_ss  << "OBB_edge9_"  << i << std::endl;
			OBB_edge10_ss << "OBB_edge10_" << i << std::endl;
			OBB_edge11_ss << "OBB_edge11_" << i << std::endl;
			OBB_edge12_ss << "OBB_edge12_" << i << std::endl;

			std::string OBB_edge1  = OBB_edge1_ss.str();
			std::string OBB_edge2  = OBB_edge2_ss.str();
			std::string OBB_edge3  = OBB_edge3_ss.str();
			std::string OBB_edge4  = OBB_edge4_ss.str();
			std::string OBB_edge5  = OBB_edge5_ss.str();
			std::string OBB_edge6  = OBB_edge6_ss.str();
			std::string OBB_edge7  = OBB_edge7_ss.str();
			std::string OBB_edge8  = OBB_edge8_ss.str();
			std::string OBB_edge9  = OBB_edge9_ss.str();
			std::string OBB_edge10 = OBB_edge10_ss.str();
			std::string OBB_edge11 = OBB_edge11_ss.str();
			std::string OBB_edge12 = OBB_edge12_ss.str();

			viewer->addLine (pt1_OBB, pt2_OBB, 0.0, 0.0, 1.0, OBB_edge1);
			viewer->addLine (pt1_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge2);
			viewer->addLine (pt1_OBB, pt5_OBB, 0.0, 0.0, 1.0, OBB_edge3);
			viewer->addLine (pt5_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge4);
			viewer->addLine (pt5_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge5);
			viewer->addLine (pt2_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge6);
			viewer->addLine (pt6_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge7);
			viewer->addLine (pt7_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge8);
			viewer->addLine (pt2_OBB, pt3_OBB, 0.0, 0.0, 1.0, OBB_edge9);
			viewer->addLine (pt4_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge10);
			viewer->addLine (pt3_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge11);
			viewer->addLine (pt3_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge12);*/

			/* DRAWING AXIS ALIGNED BOUNDING BOXES */
			pcl::PointXYZ pt1_AABB(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt2_AABB(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt3_AABB(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt4_AABB(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt5_AABB(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt6_AABB(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt7_AABB(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt8_AABB(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z);

			std::stringstream AABB_edge1_ss, AABB_edge2_ss, AABB_edge3_ss, AABB_edge4_ss,
				AABB_edge5_ss, AABB_edge6_ss, AABB_edge7_ss, AABB_edge8_ss,
				AABB_edge9_ss, AABB_edge10_ss, AABB_edge11_ss, AABB_edge12_ss;

			AABB_edge1_ss << "AABB_edge1_" << i << std::endl;
			AABB_edge2_ss << "AABB_edge2_" << i << std::endl;
			AABB_edge3_ss << "AABB_edge3_" << i << std::endl;
			AABB_edge4_ss << "AABB_edge4_" << i << std::endl;
			AABB_edge5_ss << "AABB_edge5_" << i << std::endl;
			AABB_edge6_ss << "AABB_edge6_" << i << std::endl;
			AABB_edge7_ss << "AABB_edge7_" << i << std::endl;
			AABB_edge8_ss << "AABB_edge8_" << i << std::endl;
			AABB_edge9_ss << "AABB_edge9_" << i << std::endl;
			AABB_edge10_ss << "AABB_edge10_" << i << std::endl;
			AABB_edge11_ss << "AABB_edge11_" << i << std::endl;
			AABB_edge12_ss << "AABB_edge12_" << i << std::endl;

			std::string AABB_edge1 = AABB_edge1_ss.str();
			std::string AABB_edge2 = AABB_edge2_ss.str();
			std::string AABB_edge3 = AABB_edge3_ss.str();
			std::string AABB_edge4 = AABB_edge4_ss.str();
			std::string AABB_edge5 = AABB_edge5_ss.str();
			std::string AABB_edge6 = AABB_edge6_ss.str();
			std::string AABB_edge7 = AABB_edge7_ss.str();
			std::string AABB_edge8 = AABB_edge8_ss.str();
			std::string AABB_edge9 = AABB_edge9_ss.str();
			std::string AABB_edge10 = AABB_edge10_ss.str();
			std::string AABB_edge11 = AABB_edge11_ss.str();
			std::string AABB_edge12 = AABB_edge12_ss.str();

			viewer->addLine(pt1_AABB, pt2_AABB, 1.0, 0.0, 0.0, AABB_edge1);
			viewer->addLine(pt1_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge2);
			viewer->addLine(pt1_AABB, pt5_AABB, 1.0, 0.0, 0.0, AABB_edge3);
			viewer->addLine(pt5_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge4);
			viewer->addLine(pt5_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge5);
			viewer->addLine(pt2_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge6);
			viewer->addLine(pt6_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge7);
			viewer->addLine(pt7_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge8);
			viewer->addLine(pt2_AABB, pt3_AABB, 1.0, 0.0, 0.0, AABB_edge9);
			viewer->addLine(pt4_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge10);
			viewer->addLine(pt3_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge11);
			viewer->addLine(pt3_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge12);

			std::stringstream obj_length_id_ss, obj_width_id_ss, obj_height_id_ss;

			obj_length_id_ss << "obj_length_" << i << std::endl;
			obj_width_id_ss << "obj_width_" << i << std::endl;
			obj_height_id_ss << "obj_height_" << i << std::endl;

			std::string obj_length_id = obj_length_id_ss.str();
			std::string obj_width_id = obj_width_id_ss.str();
			std::string obj_height_id = obj_height_id_ss.str();

			std::stringstream obj_length_val_ss, obj_width_val_ss, obj_height_val_ss;

			float obj_height = pcl::geometry::distance(pt1_AABB, pt5_AABB);
			float obj_length = pcl::geometry::distance(pt1_AABB, pt2_AABB);
			float obj_width = pcl::geometry::distance(pt1_AABB, pt4_AABB);

			obj_height_val_ss << obj_height << " m" << std::endl;
			obj_length_val_ss << obj_length << " m" << std::endl;
			obj_width_val_ss << obj_width << " m" << std::endl;

			std::string obj_height_txt = obj_height_val_ss.str();
			std::string obj_length_txt = obj_length_val_ss.str();
			std::string obj_width_txt = obj_width_val_ss.str();

			pcl::PointXYZ height_txt_position((pt1_AABB.x + pt5_AABB.x) / 2.0, (pt1_AABB.y + pt5_AABB.y) / 2.0, (pt1_AABB.z + pt5_AABB.z) / 2.0);
			pcl::PointXYZ length_txt_position((pt1_AABB.x + pt2_AABB.x) / 2.0, (pt1_AABB.y + pt2_AABB.y) / 2.0, (pt1_AABB.z + pt2_AABB.z) / 2.0);
			pcl::PointXYZ width_txt_position((pt1_AABB.x + pt4_AABB.x) / 2.0, (pt1_AABB.y + pt4_AABB.y) / 2.0, (pt1_AABB.z + pt4_AABB.z) / 2.0);

			float txt_scale_coeff = 0.04;
			float height_txt_scale = txt_scale_coeff * obj_height;
			float length_txt_scale = txt_scale_coeff * obj_length;
			float width_txt_scale = txt_scale_coeff * obj_width;

			viewer->addText3D(obj_height_txt, height_txt_position, height_txt_scale, 1.0, 1.0, 1.0, obj_height_id);
			viewer->addText3D(obj_length_txt, length_txt_position, length_txt_scale, 1.0, 1.0, 1.0, obj_length_id);
			viewer->addText3D(obj_width_txt, width_txt_position, width_txt_scale, 1.0, 1.0, 1.0, obj_width_id);

		}


		double coord_sys_scale = 0.3;
		viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		viewer->spin();

		return 0;
	};
private:
	std::string _data_path;
	double _search_radius;
	double _cluster_tolerance;
	int _min_cluster_size;
	int _max_cluster_size;
};

class RegionGrowing
{
public:
	RegionGrowing() :
		_search_radius(0.03), 
		_curvature_threshold(1.0), 
		_smoothness_threshold(M_PI * 5.0 / 180.0), 
		_max_cluster_size(1e5), 
		_min_cluster_size(50) {};
	void setInputCloudPath(std::string data_path) 
	{
		_data_path = data_path;
	};
	void setRadiusSearch(double search_radius)
	{
		_search_radius = search_radius;
	};
	void setCurvatureThreshold(float curvature_threshold)
	{
		_curvature_threshold = curvature_threshold;
	};
	void setSmoothnessThreshold(float smoothness_threshold)
	{
		_smoothness_threshold = smoothness_threshold;
	};
	void setMaxClusterSize(int max_cluster_size)
	{
		_max_cluster_size = max_cluster_size;
	};
	void setMinClusterSize(int min_cluster_size)
	{
		_min_cluster_size = min_cluster_size;
	};
	int segment()
	{
		/* INPUT DATA */
		pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud(obj);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(_search_radius);
		ne.compute(*normals);

		/* REGION GROWING */
		std::vector<pcl::PointIndices> clusters;
		pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> rg;
		rg.setInputCloud(obj);
		rg.setInputNormals(normals);
		rg.setCurvatureThreshold(_curvature_threshold);                  // 相邻两点的切线之间的夹角与两点间对应弧长的比率
		rg.setSmoothnessThreshold(_smoothness_threshold);  // 相邻两点的法线夹角
		rg.setMaxClusterSize(_max_cluster_size);
		rg.setMinClusterSize(_min_cluster_size);
		rg.extract(clusters);

		if (clusters.empty())
		{
			std::cout << "No Cluster Extracted!!" << std::endl;
			return 1;
		}
		else
		{
			std::cout << clusters.size() << " Clusters Extracted!!" << std::endl;
		}

		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;

		/* VISUALIZATION */
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
		int view_port = 0;

		for (int i = 0; i < clusters.size(); i++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			temp_cloud->width = clusters[i].indices.size();
			temp_cloud->height = 1;
			std::vector<int>::iterator iter = clusters[i].indices.begin();
			while (iter != clusters[i].indices.end())
			{
				temp_cloud->push_back(obj->at(*iter));
				iter++;
			}
			clouds.push_back(temp_cloud);
		}

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
		pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
		Eigen::Matrix3f rotation_matrix_OBB;

		for (int i = 0; i < clouds.size(); i++)
		{
			bb.setInputCloud(clouds[i]);
			bb.compute();

			bb.getAABB(min_point_AABB, max_point_AABB);
			bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

			Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
			Eigen::Quaternionf quaternion(rotation_matrix_OBB);

			std::stringstream cloud_name_ss;
			cloud_name_ss << "cloud" << i << std::endl;
			std::string cloud_name = cloud_name_ss.str();

			pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(clouds[i]);
			viewer->addPointCloud(clouds[i], temp_handler, cloud_name, view_port);

			/* DRAWING ORIENTED BOUNDING BOXES */
			/*Eigen::Vector3f p1_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p2_OBB(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p3_OBB(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p4_OBB(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p5_OBB(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p6_OBB(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p7_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p8_OBB(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

			p1_OBB = rotation_matrix_OBB * p1_OBB + position;
			p2_OBB = rotation_matrix_OBB * p2_OBB + position;
			p3_OBB = rotation_matrix_OBB * p3_OBB + position;
			p4_OBB = rotation_matrix_OBB * p4_OBB + position;
			p5_OBB = rotation_matrix_OBB * p5_OBB + position;
			p6_OBB = rotation_matrix_OBB * p6_OBB + position;
			p7_OBB = rotation_matrix_OBB * p7_OBB + position;
			p8_OBB = rotation_matrix_OBB * p8_OBB + position;

			pcl::PointXYZ pt1_OBB(p1_OBB(0), p1_OBB(1), p1_OBB(2));
			pcl::PointXYZ pt2_OBB(p2_OBB(0), p2_OBB(1), p2_OBB(2));
			pcl::PointXYZ pt3_OBB(p3_OBB(0), p3_OBB(1), p3_OBB(2));
			pcl::PointXYZ pt4_OBB(p4_OBB(0), p4_OBB(1), p4_OBB(2));
			pcl::PointXYZ pt5_OBB(p5_OBB(0), p5_OBB(1), p5_OBB(2));
			pcl::PointXYZ pt6_OBB(p6_OBB(0), p6_OBB(1), p6_OBB(2));
			pcl::PointXYZ pt7_OBB(p7_OBB(0), p7_OBB(1), p7_OBB(2));
			pcl::PointXYZ pt8_OBB(p8_OBB(0), p8_OBB(1), p8_OBB(2));

			std::stringstream OBB_edge1_ss, OBB_edge2_ss,  OBB_edge3_ss,  OBB_edge4_ss,
			OBB_edge5_ss, OBB_edge6_ss,  OBB_edge7_ss,  OBB_edge8_ss,
			OBB_edge9_ss, OBB_edge10_ss, OBB_edge11_ss, OBB_edge12_ss;

			OBB_edge1_ss  << "OBB_edge1_"  << i << std::endl;
			OBB_edge2_ss  << "OBB_edge2_"  << i << std::endl;
			OBB_edge3_ss  << "OBB_edge3_"  << i << std::endl;
			OBB_edge4_ss  << "OBB_edge4_"  << i << std::endl;
			OBB_edge5_ss  << "OBB_edge5_"  << i << std::endl;
			OBB_edge6_ss  << "OBB_edge6_"  << i << std::endl;
			OBB_edge7_ss  << "OBB_edge7_"  << i << std::endl;
			OBB_edge8_ss  << "OBB_edge8_"  << i << std::endl;
			OBB_edge9_ss  << "OBB_edge9_"  << i << std::endl;
			OBB_edge10_ss << "OBB_edge10_" << i << std::endl;
			OBB_edge11_ss << "OBB_edge11_" << i << std::endl;
			OBB_edge12_ss << "OBB_edge12_" << i << std::endl;

			std::string OBB_edge1  = OBB_edge1_ss.str();
			std::string OBB_edge2  = OBB_edge2_ss.str();
			std::string OBB_edge3  = OBB_edge3_ss.str();
			std::string OBB_edge4  = OBB_edge4_ss.str();
			std::string OBB_edge5  = OBB_edge5_ss.str();
			std::string OBB_edge6  = OBB_edge6_ss.str();
			std::string OBB_edge7  = OBB_edge7_ss.str();
			std::string OBB_edge8  = OBB_edge8_ss.str();
			std::string OBB_edge9  = OBB_edge9_ss.str();
			std::string OBB_edge10 = OBB_edge10_ss.str();
			std::string OBB_edge11 = OBB_edge11_ss.str();
			std::string OBB_edge12 = OBB_edge12_ss.str();

			viewer->addLine (pt1_OBB, pt2_OBB, 0.0, 0.0, 1.0, OBB_edge1);
			viewer->addLine (pt1_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge2);
			viewer->addLine (pt1_OBB, pt5_OBB, 0.0, 0.0, 1.0, OBB_edge3);
			viewer->addLine (pt5_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge4);
			viewer->addLine (pt5_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge5);
			viewer->addLine (pt2_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge6);
			viewer->addLine (pt6_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge7);
			viewer->addLine (pt7_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge8);
			viewer->addLine (pt2_OBB, pt3_OBB, 0.0, 0.0, 1.0, OBB_edge9);
			viewer->addLine (pt4_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge10);
			viewer->addLine (pt3_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge11);
			viewer->addLine (pt3_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge12);*/

			/* DRAWING AXIS ALIGNED BOUNDING BOXES */
			pcl::PointXYZ pt1_AABB(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt2_AABB(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt3_AABB(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt4_AABB(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt5_AABB(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt6_AABB(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt7_AABB(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt8_AABB(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z);

			std::stringstream AABB_edge1_ss, AABB_edge2_ss, AABB_edge3_ss, AABB_edge4_ss,
				AABB_edge5_ss, AABB_edge6_ss, AABB_edge7_ss, AABB_edge8_ss,
				AABB_edge9_ss, AABB_edge10_ss, AABB_edge11_ss, AABB_edge12_ss;

			AABB_edge1_ss << "AABB_edge1_" << i << std::endl;
			AABB_edge2_ss << "AABB_edge2_" << i << std::endl;
			AABB_edge3_ss << "AABB_edge3_" << i << std::endl;
			AABB_edge4_ss << "AABB_edge4_" << i << std::endl;
			AABB_edge5_ss << "AABB_edge5_" << i << std::endl;
			AABB_edge6_ss << "AABB_edge6_" << i << std::endl;
			AABB_edge7_ss << "AABB_edge7_" << i << std::endl;
			AABB_edge8_ss << "AABB_edge8_" << i << std::endl;
			AABB_edge9_ss << "AABB_edge9_" << i << std::endl;
			AABB_edge10_ss << "AABB_edge10_" << i << std::endl;
			AABB_edge11_ss << "AABB_edge11_" << i << std::endl;
			AABB_edge12_ss << "AABB_edge12_" << i << std::endl;

			std::string AABB_edge1 = AABB_edge1_ss.str();
			std::string AABB_edge2 = AABB_edge2_ss.str();
			std::string AABB_edge3 = AABB_edge3_ss.str();
			std::string AABB_edge4 = AABB_edge4_ss.str();
			std::string AABB_edge5 = AABB_edge5_ss.str();
			std::string AABB_edge6 = AABB_edge6_ss.str();
			std::string AABB_edge7 = AABB_edge7_ss.str();
			std::string AABB_edge8 = AABB_edge8_ss.str();
			std::string AABB_edge9 = AABB_edge9_ss.str();
			std::string AABB_edge10 = AABB_edge10_ss.str();
			std::string AABB_edge11 = AABB_edge11_ss.str();
			std::string AABB_edge12 = AABB_edge12_ss.str();

			viewer->addLine(pt1_AABB, pt2_AABB, 1.0, 0.0, 0.0, AABB_edge1);
			viewer->addLine(pt1_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge2);
			viewer->addLine(pt1_AABB, pt5_AABB, 1.0, 0.0, 0.0, AABB_edge3);
			viewer->addLine(pt5_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge4);
			viewer->addLine(pt5_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge5);
			viewer->addLine(pt2_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge6);
			viewer->addLine(pt6_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge7);
			viewer->addLine(pt7_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge8);
			viewer->addLine(pt2_AABB, pt3_AABB, 1.0, 0.0, 0.0, AABB_edge9);
			viewer->addLine(pt4_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge10);
			viewer->addLine(pt3_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge11);
			viewer->addLine(pt3_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge12);

			std::stringstream obj_length_id_ss, obj_width_id_ss, obj_height_id_ss;

			obj_length_id_ss << "obj_length_" << i << std::endl;
			obj_width_id_ss << "obj_width_" << i << std::endl;
			obj_height_id_ss << "obj_height_" << i << std::endl;

			std::string obj_length_id = obj_length_id_ss.str();
			std::string obj_width_id = obj_width_id_ss.str();
			std::string obj_height_id = obj_height_id_ss.str();

			std::stringstream obj_length_val_ss, obj_width_val_ss, obj_height_val_ss;

			float obj_height = pcl::geometry::distance(pt1_AABB, pt5_AABB);
			float obj_length = pcl::geometry::distance(pt1_AABB, pt2_AABB);
			float obj_width = pcl::geometry::distance(pt1_AABB, pt4_AABB);

			obj_height_val_ss << obj_height << " m" << std::endl;
			obj_length_val_ss << obj_length << " m" << std::endl;
			obj_width_val_ss << obj_width << " m" << std::endl;

			std::string obj_height_txt = obj_height_val_ss.str();
			std::string obj_length_txt = obj_length_val_ss.str();
			std::string obj_width_txt = obj_width_val_ss.str();

			pcl::PointXYZ height_txt_position((pt1_AABB.x + pt5_AABB.x) / 2.0, (pt1_AABB.y + pt5_AABB.y) / 2.0, (pt1_AABB.z + pt5_AABB.z) / 2.0);
			pcl::PointXYZ length_txt_position((pt1_AABB.x + pt2_AABB.x) / 2.0, (pt1_AABB.y + pt2_AABB.y) / 2.0, (pt1_AABB.z + pt2_AABB.z) / 2.0);
			pcl::PointXYZ width_txt_position((pt1_AABB.x + pt4_AABB.x) / 2.0, (pt1_AABB.y + pt4_AABB.y) / 2.0, (pt1_AABB.z + pt4_AABB.z) / 2.0);

			float txt_scale_coeff = 0.04;
			float height_txt_scale = txt_scale_coeff * obj_height;
			float length_txt_scale = txt_scale_coeff * obj_length;
			float width_txt_scale = txt_scale_coeff * obj_width;

			viewer->addText3D(obj_height_txt, height_txt_position, height_txt_scale, 1.0, 1.0, 1.0, obj_height_id);
			viewer->addText3D(obj_length_txt, length_txt_position, length_txt_scale, 1.0, 1.0, 1.0, obj_length_id);
			viewer->addText3D(obj_width_txt, width_txt_position, width_txt_scale, 1.0, 1.0, 1.0, obj_width_id);

		}


		double coord_sys_scale = 0.3;
		viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		viewer->spin();

		return 0;
	};
private:
	std::string _data_path;
	double _search_radius;
	float _curvature_threshold;
	float _smoothness_threshold;
	int _max_cluster_size;
	int _min_cluster_size;
};

class MinCut
{
public:
	MinCut() :
		_radius(0.5),
		_neighbors(15),
		_sigma(0.1),
		_source_weight(0.8),
		obj(new pcl::PointCloud<pcl::PointXYZ>), 
		selected_points(new pcl::PointCloud<pcl::PointXYZ>), 
		fg_points(new pcl::PointCloud<pcl::PointXYZ>), 
		bg_points(new pcl::PointCloud<pcl::PointXYZ>), 
		tree(new pcl::search::KdTree<pcl::PointXYZ>), 
		cloud_viewer(new pcl::visualization::PCLVisualizer){};
	void setInputCloudPath(std::string data_path)
	{
		_data_path = data_path;
	}
	void setRadius(double radius)
	{
		_radius = radius;
	};
	void setNumberOfNeighbours(int neighbors)
	{
		_neighbors = neighbors;
	};
	void setSigma(double sigma)
	{
		_sigma = sigma;
	};
	void setSourceWeight(double source_weight)
	{
		_source_weight = source_weight;
	};
	void KeyboardEventOccured(const pcl::visualization::KeyboardEvent &event, void* viewer_void)
	{
		// Foreground Cloud: 确定前景点
		if (event.getKeySym() == "f" && event.keyDown() && event.isCtrlPressed())
		{
			*fg_points = *selected_points;
			selected_points->clear();
			std::cout << "Foreground points confirmed!!" << std::endl;
		}

		// Background Cloud: 确定后景点
		if (event.getKeySym() == "b" && event.keyDown() && event.isCtrlPressed())
		{
			*bg_points = *selected_points;
			selected_points->clear();
			std::cout << "Background points confirmed!!" << std::endl;
		}

		// Cut: 点云分割
		if (event.getKeySym() == "x" && event.keyDown() && event.isCtrlPressed())
		{
			std::cout << "Starting segmentation..." << std::endl;
			pcl::MinCutSegmentation<pcl::PointXYZ> mc;
			mc.setInputCloud(obj);
			mc.setForegroundPoints(fg_points);

			if (!bg_points->empty())
			{
				mc.setBackgroundPoints(bg_points);
			}

			// 分割范围
			mc.setRadius(_radius);
			// 用于生成图的邻点数
			mc.setNumberOfNeighbours(_neighbors);
			mc.setSearchMethod(tree);
			// 连接两个vertex的edge的权重值中的sigma: w = e^(-(dist/sigma)^2)
			mc.setSigma(_sigma);
			// 设置与source vertex相连的edge的权值, 因为是最小割，所以权重越高的edge越不会被切割
			mc.setSourceWeight(_source_weight);
			mc.extract(clusters);

			if (clusters.empty())
			{
				std::cout << "No cluster extracted!!" << std::endl;
				return;
			}
			else
			{
				for (int i = 0; i < clusters.size(); i++)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
					temp_cloud->width = clusters[i].indices.size();
					temp_cloud->height = 1;
					std::vector<int>::iterator iter = clusters[i].indices.begin();
					while (iter != clusters[i].indices.end())
					{
						temp_cloud->push_back(obj->at(*iter));
						iter++;
					}
					clouds.push_back(temp_cloud);
				}
				std::cout << clusters.size() << " clusters extracted!!" << std::endl;
			}

			pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
			pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
			Eigen::Matrix3f rotation_matrix_OBB;

			for (int i = 0; i < clouds.size(); i++)
			{
				bb.setInputCloud(clouds[i]);
				bb.compute();

				bb.getAABB(min_point_AABB, max_point_AABB);
				bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

				Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quaternion(rotation_matrix_OBB);

				std::stringstream cloud_name_ss;
				cloud_name_ss << "cloud" << i << std::endl;
				std::string cloud_name = cloud_name_ss.str();

				pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(clouds[i]);

				// 类中使用viewer，直接使用数据成员global_viewer即可，不需要用callback函数中的局部变量local_viewer
				cloud_viewer->addPointCloud(clouds[i], temp_handler, cloud_name);

				/* DRAWING ORIENTED BOUNDING BOXES */
				//Eigen::Vector3f p1_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
				//Eigen::Vector3f p2_OBB(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
				//Eigen::Vector3f p3_OBB(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
				//Eigen::Vector3f p4_OBB(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
				//Eigen::Vector3f p5_OBB(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
				//Eigen::Vector3f p6_OBB(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
				//Eigen::Vector3f p7_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
				//Eigen::Vector3f p8_OBB(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

				//p1_OBB = rotation_matrix_OBB * p1_OBB + position;
				//p2_OBB = rotation_matrix_OBB * p2_OBB + position;
				//p3_OBB = rotation_matrix_OBB * p3_OBB + position;
				//p4_OBB = rotation_matrix_OBB * p4_OBB + position;
				//p5_OBB = rotation_matrix_OBB * p5_OBB + position;
				//p6_OBB = rotation_matrix_OBB * p6_OBB + position;
				//p7_OBB = rotation_matrix_OBB * p7_OBB + position;
				//p8_OBB = rotation_matrix_OBB * p8_OBB + position;

				//pcl::PointXYZ pt1_OBB(p1_OBB(0), p1_OBB(1), p1_OBB(2));
				//pcl::PointXYZ pt2_OBB(p2_OBB(0), p2_OBB(1), p2_OBB(2));
				//pcl::PointXYZ pt3_OBB(p3_OBB(0), p3_OBB(1), p3_OBB(2));
				//pcl::PointXYZ pt4_OBB(p4_OBB(0), p4_OBB(1), p4_OBB(2));
				//pcl::PointXYZ pt5_OBB(p5_OBB(0), p5_OBB(1), p5_OBB(2));
				//pcl::PointXYZ pt6_OBB(p6_OBB(0), p6_OBB(1), p6_OBB(2));
				//pcl::PointXYZ pt7_OBB(p7_OBB(0), p7_OBB(1), p7_OBB(2));
				//pcl::PointXYZ pt8_OBB(p8_OBB(0), p8_OBB(1), p8_OBB(2));

				//std::stringstream OBB_edge1_ss, OBB_edge2_ss, OBB_edge3_ss, OBB_edge4_ss,
				//	OBB_edge5_ss, OBB_edge6_ss, OBB_edge7_ss, OBB_edge8_ss,
				//	OBB_edge9_ss, OBB_edge10_ss, OBB_edge11_ss, OBB_edge12_ss;

				//OBB_edge1_ss << "OBB_edge1_" << i << std::endl;
				//OBB_edge2_ss << "OBB_edge2_" << i << std::endl;
				//OBB_edge3_ss << "OBB_edge3_" << i << std::endl;
				//OBB_edge4_ss << "OBB_edge4_" << i << std::endl;
				//OBB_edge5_ss << "OBB_edge5_" << i << std::endl;
				//OBB_edge6_ss << "OBB_edge6_" << i << std::endl;
				//OBB_edge7_ss << "OBB_edge7_" << i << std::endl;
				//OBB_edge8_ss << "OBB_edge8_" << i << std::endl;
				//OBB_edge9_ss << "OBB_edge9_" << i << std::endl;
				//OBB_edge10_ss << "OBB_edge10_" << i << std::endl;
				//OBB_edge11_ss << "OBB_edge11_" << i << std::endl;
				//OBB_edge12_ss << "OBB_edge12_" << i << std::endl;

				//std::string OBB_edge1 = OBB_edge1_ss.str();
				//std::string OBB_edge2 = OBB_edge2_ss.str();
				//std::string OBB_edge3 = OBB_edge3_ss.str();
				//std::string OBB_edge4 = OBB_edge4_ss.str();
				//std::string OBB_edge5 = OBB_edge5_ss.str();
				//std::string OBB_edge6 = OBB_edge6_ss.str();
				//std::string OBB_edge7 = OBB_edge7_ss.str();
				//std::string OBB_edge8 = OBB_edge8_ss.str();
				//std::string OBB_edge9 = OBB_edge9_ss.str();
				//std::string OBB_edge10 = OBB_edge10_ss.str();
				//std::string OBB_edge11 = OBB_edge11_ss.str();
				//std::string OBB_edge12 = OBB_edge12_ss.str();

				//viewer->addLine(pt1_OBB, pt2_OBB, 0.0, 0.0, 1.0, OBB_edge1);
				//viewer->addLine(pt1_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge2);
				//viewer->addLine(pt1_OBB, pt5_OBB, 0.0, 0.0, 1.0, OBB_edge3);
				//viewer->addLine(pt5_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge4);
				//viewer->addLine(pt5_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge5);
				//viewer->addLine(pt2_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge6);
				//viewer->addLine(pt6_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge7);
				//viewer->addLine(pt7_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge8);
				//viewer->addLine(pt2_OBB, pt3_OBB, 0.0, 0.0, 1.0, OBB_edge9);
				//viewer->addLine(pt4_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge10);
				//viewer->addLine(pt3_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge11);
				//viewer->addLine(pt3_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge12);

				/* DRAWING AXIS ALIGNED BOUNDING BOXES */
				pcl::PointXYZ pt1_AABB(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
				pcl::PointXYZ pt2_AABB(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
				pcl::PointXYZ pt3_AABB(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
				pcl::PointXYZ pt4_AABB(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
				pcl::PointXYZ pt5_AABB(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z);
				pcl::PointXYZ pt6_AABB(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
				pcl::PointXYZ pt7_AABB(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
				pcl::PointXYZ pt8_AABB(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z);

				std::stringstream AABB_edge1_ss, AABB_edge2_ss, AABB_edge3_ss, AABB_edge4_ss,
					AABB_edge5_ss, AABB_edge6_ss, AABB_edge7_ss, AABB_edge8_ss,
					AABB_edge9_ss, AABB_edge10_ss, AABB_edge11_ss, AABB_edge12_ss;

				AABB_edge1_ss << "AABB_edge1_" << i << std::endl;
				AABB_edge2_ss << "AABB_edge2_" << i << std::endl;
				AABB_edge3_ss << "AABB_edge3_" << i << std::endl;
				AABB_edge4_ss << "AABB_edge4_" << i << std::endl;
				AABB_edge5_ss << "AABB_edge5_" << i << std::endl;
				AABB_edge6_ss << "AABB_edge6_" << i << std::endl;
				AABB_edge7_ss << "AABB_edge7_" << i << std::endl;
				AABB_edge8_ss << "AABB_edge8_" << i << std::endl;
				AABB_edge9_ss << "AABB_edge9_" << i << std::endl;
				AABB_edge10_ss << "AABB_edge10_" << i << std::endl;
				AABB_edge11_ss << "AABB_edge11_" << i << std::endl;
				AABB_edge12_ss << "AABB_edge12_" << i << std::endl;

				std::string AABB_edge1 = AABB_edge1_ss.str();
				std::string AABB_edge2 = AABB_edge2_ss.str();
				std::string AABB_edge3 = AABB_edge3_ss.str();
				std::string AABB_edge4 = AABB_edge4_ss.str();
				std::string AABB_edge5 = AABB_edge5_ss.str();
				std::string AABB_edge6 = AABB_edge6_ss.str();
				std::string AABB_edge7 = AABB_edge7_ss.str();
				std::string AABB_edge8 = AABB_edge8_ss.str();
				std::string AABB_edge9 = AABB_edge9_ss.str();
				std::string AABB_edge10 = AABB_edge10_ss.str();
				std::string AABB_edge11 = AABB_edge11_ss.str();
				std::string AABB_edge12 = AABB_edge12_ss.str();

				cloud_viewer->addLine(pt1_AABB, pt2_AABB, 1.0, 0.0, 0.0, AABB_edge1);
				cloud_viewer->addLine(pt1_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge2);
				cloud_viewer->addLine(pt1_AABB, pt5_AABB, 1.0, 0.0, 0.0, AABB_edge3);
				cloud_viewer->addLine(pt5_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge4);
				cloud_viewer->addLine(pt5_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge5);
				cloud_viewer->addLine(pt2_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge6);
				cloud_viewer->addLine(pt6_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge7);
				cloud_viewer->addLine(pt7_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge8);
				cloud_viewer->addLine(pt2_AABB, pt3_AABB, 1.0, 0.0, 0.0, AABB_edge9);
				cloud_viewer->addLine(pt4_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge10);
				cloud_viewer->addLine(pt3_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge11);
				cloud_viewer->addLine(pt3_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge12);

				std::stringstream obj_length_id_ss, obj_width_id_ss, obj_height_id_ss;

				obj_length_id_ss << "obj_length_" << i << std::endl;
				obj_width_id_ss << "obj_width_" << i << std::endl;
				obj_height_id_ss << "obj_height_" << i << std::endl;

				std::string obj_length_id = obj_length_id_ss.str();
				std::string obj_width_id = obj_width_id_ss.str();
				std::string obj_height_id = obj_height_id_ss.str();

				std::stringstream obj_length_val_ss, obj_width_val_ss, obj_height_val_ss;

				float obj_height = pcl::geometry::distance(pt1_AABB, pt5_AABB);
				float obj_length = pcl::geometry::distance(pt1_AABB, pt2_AABB);
				float obj_width = pcl::geometry::distance(pt1_AABB, pt4_AABB);

				obj_height_val_ss << obj_height << " m" << std::endl;
				obj_length_val_ss << obj_length << " m" << std::endl;
				obj_width_val_ss << obj_width << " m" << std::endl;

				std::string obj_height_txt = obj_height_val_ss.str();
				std::string obj_length_txt = obj_length_val_ss.str();
				std::string obj_width_txt = obj_width_val_ss.str();

				pcl::PointXYZ height_txt_position((pt1_AABB.x + pt5_AABB.x) / 2.0, (pt1_AABB.y + pt5_AABB.y) / 2.0, (pt1_AABB.z + pt5_AABB.z) / 2.0);
				pcl::PointXYZ length_txt_position((pt1_AABB.x + pt2_AABB.x) / 2.0, (pt1_AABB.y + pt2_AABB.y) / 2.0, (pt1_AABB.z + pt2_AABB.z) / 2.0);
				pcl::PointXYZ width_txt_position((pt1_AABB.x + pt4_AABB.x) / 2.0, (pt1_AABB.y + pt4_AABB.y) / 2.0, (pt1_AABB.z + pt4_AABB.z) / 2.0);

				float txt_scale_coeff = 0.04;
				float height_txt_scale = txt_scale_coeff * obj_height;
				float length_txt_scale = txt_scale_coeff * obj_length;
				float width_txt_scale = txt_scale_coeff * obj_width;

				cloud_viewer->addText3D(obj_height_txt, height_txt_position, height_txt_scale, 1.0, 1.0, 1.0, obj_height_id);
				cloud_viewer->addText3D(obj_length_txt, length_txt_position, length_txt_scale, 1.0, 1.0, 1.0, obj_length_id);
				cloud_viewer->addText3D(obj_width_txt, width_txt_position, width_txt_scale, 1.0, 1.0, 1.0, obj_width_id);

			}
			std::cout << "Segmentation finished!!" << std::endl;
		}

		// Empty: 清空viewer
		if (event.getKeySym() == "e" && event.keyDown() && event.isCtrlPressed())
		{
			std::cout << "Cleaning up all the stuff..." << std::endl;
			cloud_viewer->removeAllShapes();
			cloud_viewer->removeAllPointClouds();
			selected_points->clear();
			fg_points->clear();
			bg_points->clear();
			clusters.clear();
			clouds.clear();
			//pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(obj);   // 刷新点云颜色
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> temp_handler(obj, 240, 60, 60);
			cloud_viewer->addPointCloud(obj, temp_handler);
		}

	};
	void PointPickEventOccured(const pcl::visualization::PointPickingEvent &event, void* viewer_void)
	{
		// 按H显示帮助菜单，SHIFT+鼠标左键选中点云中的数据点
		pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);
		float x, y, z;
		event.getPoint(x, y, z);
		pcl::PointXYZ selected_point;
		selected_point.x = x;
		selected_point.y = y;
		selected_point.z = z;
		selected_points->points.push_back(selected_point);
		std::cout << "Select point: (" << x << ", " << y << ", " << z << ")" << std::endl;
	};
	int segment()
	{
		/* INPUT DATA */
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		/* USAGE INSTRUCTION */
		std::cerr << "Pick    Point:             Press Shift + Left-click" << std::endl;
		std::cerr << "Confirm Foreground Points: Press Ctrl  + F" << std::endl;
		std::cerr << "Confirm Background Points: Press Ctrl  + B" << std::endl;
		std::cerr << "Segment Point Cloud:       Press Ctrl  + X" << std::endl;
		std::cerr << "Empty   Results:           Press Ctrl  + E" << std::endl;

		/* VISUALIZATION */
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(obj, 240, 60, 60);
		cloud_viewer->addPointCloud(obj, handler);

		// 在类中register callback函数时，所需的传入参数会增加一个，即第二个参数instance： 实现callback函数的类的实例
		// 传入callback函数时，需要通过取址符从类中获取成员函数
		cloud_viewer->registerPointPickingCallback(&MinCut::PointPickEventOccured, *this);
		cloud_viewer->registerKeyboardCallback(&MinCut::KeyboardEventOccured, *this);

		double coord_sys_scale = 0.3;
		cloud_viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		cloud_viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		// 用于交互并实时更新显示，静态显示则用spin()
		while (!cloud_viewer->wasStopped())
		{
			cloud_viewer->spinOnce(100);
		}

		return 0;
	};
private:
	double _radius;
	int _neighbors;
	double _sigma;
	double _source_weight;
	std::string _data_path;
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj;
	pcl::PointCloud<pcl::PointXYZ>::Ptr selected_points;
	pcl::PointCloud<pcl::PointXYZ>::Ptr fg_points;
	pcl::PointCloud<pcl::PointXYZ>::Ptr bg_points;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
	pcl::visualization::PCLVisualizer::Ptr cloud_viewer;
	std::vector<pcl::PointIndices> clusters;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
};

class SupervoxelClustering
{
public:
	SupervoxelClustering() :
		_search_radius(0.03),
		_voxel_resolution(0.01),
		_seed_resolution(0.05),
		_use_single_cam_transform(true),
		_normal_importance(0.5),
		_spatial_importance(0.5) {};
	void setInputCloudPath(std::string data_path)
	{
		_data_path = data_path;
	};
	void setRadiusSearch(double search_radius)
	{
		_search_radius = search_radius;
	};
	void setVoxelResolution(float voxel_resolution)
	{
		_voxel_resolution = voxel_resolution;
	};
	void setSeedResolution(float seed_resolution)
	{
		_seed_resolution = seed_resolution;
	};
	void setUseSingleCameraTransform(bool use_single_cam_transform)
	{
		_use_single_cam_transform = use_single_cam_transform;
	};
	void setNormalImportance(float normal_importance)
	{
		_normal_importance = normal_importance;
	};
	void setSpatialImportance(float spatial_importance)
	{
		_spatial_importance = spatial_importance;
	};
	int segment()
	{
		/* INPUT DATA */
		pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud(obj);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(_search_radius);
		ne.compute(*normals);

		/* SUPERVOXEL CLUSTERING */
		std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> supervoxels;

		pcl::SupervoxelClustering<pcl::PointXYZ> sv(_voxel_resolution, _seed_resolution);
		sv.setUseSingleCameraTransform(_use_single_cam_transform);
		sv.setInputCloud(obj);
		sv.setNormalCloud(normals);
		sv.setNormalImportance(_normal_importance);
		sv.setSpatialImportance(_spatial_importance);
		sv.extract(supervoxels);

		if (supervoxels.empty())
		{
			std::cout << "No Supervoxels Extracted!!" << std::endl;
			return 1;
		}
		else
		{
			std::cout << supervoxels.size() << " Supervoxels Extracted!!" << std::endl;
		}

		/* VISUALIZATION */
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
		int view_port = 0;

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
		pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
		Eigen::Matrix3f rotation_matrix_OBB;

		std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr>::iterator iter = supervoxels.begin();
		int i = 0;
		while (iter != supervoxels.end())
		{
			bb.setInputCloud(iter->second->voxels_);
			bb.compute();

			bb.getAABB(min_point_AABB, max_point_AABB);
			bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

			Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
			Eigen::Quaternionf quaternion(rotation_matrix_OBB);

			std::stringstream cloud_name_ss;
			cloud_name_ss << "cloud" << i << std::endl;
			std::string cloud_name = cloud_name_ss.str();

			pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(iter->second->voxels_);
			viewer->addPointCloud(iter->second->voxels_, temp_handler, cloud_name, view_port);

			iter++;
			i++;
		}

		double coord_sys_scale = 0.3;
		viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		viewer->spin();

		return 0;
	};
private:
	std::string _data_path;
	double _search_radius;
	float _voxel_resolution;
	float _seed_resolution;
	bool _use_single_cam_transform;
	float _normal_importance;
	float _spatial_importance;
};

class LocallyConvexConnectedPatches
{
public:
	LocallyConvexConnectedPatches() :
		_search_radius(0.03),
		_voxel_resolution(0.01),
		_seed_resolution(0.05),
		_use_single_cam_transform(true),
		_normal_importance(0.5),
		_spatial_importance(0.5),
		_concavity_tolerance_threshold(10),
		_k_factor(10),
		_min_segment_size(50),
		_use_sanity_check(true),
		_use_smoothness_check(true) {};
	void setInputCloudPath(std::string data_path)
	{
		_data_path = data_path;
	};
	void setRadiusSearch(double search_radius)
	{
		_search_radius = search_radius;
	};
	void setVoxelResolution(float voxel_resolution)
	{
		_voxel_resolution = voxel_resolution;
	};
	void setSeedResolution(float seed_resolution)
	{
		_seed_resolution = seed_resolution;
	};
	void setUseSingleCameraTransform(bool use_single_cam_transform)
	{
		_use_single_cam_transform = use_single_cam_transform;
	};
	void setNormalImportance(float normal_importance)
	{
		_normal_importance = normal_importance;
	};
	void setSpatialImportance(float spatial_importance)
	{
		_spatial_importance = spatial_importance;
	};
	void setConcavityToleranceThreshold(float concavity_tolerance_threshold) 
	{
		_concavity_tolerance_threshold = concavity_tolerance_threshold;
	};
	void setKFactor(int k_factor)
	{
		_k_factor = k_factor;
	};
	void setMinSegmentSize(int min_segment_size)
	{
		_min_segment_size = min_segment_size;
	};
	void setSanityCheck(bool use_sanity_check)
	{
		_use_sanity_check = use_sanity_check;
	};
	void setSmoothnessCheck(bool use_smoothness_check)
	{
		_use_smoothness_check = use_smoothness_check;
	};
	int segment()
	{
		/* INPUT DATA */
		pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud(obj);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(_search_radius);
		ne.compute(*normals);

		/* SUPERVOXEL CLUSTERING */
		std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> supervoxels;
		std::multimap<uint32_t, uint32_t> supervoxel_adjacency;

		pcl::SupervoxelClustering<pcl::PointXYZ> sv(_voxel_resolution, _seed_resolution);
		// supervoxel内部使用了octree，该变换可使得octree bin大小随着深度的增加而指数增大，避免因点云密度渐小而找不到octree bin的临近关系
		//如果点云是有序的，需要设置为false，如果点云是无序的，则需要设置为true
		sv.setUseSingleCameraTransform(_use_single_cam_transform);
		sv.setInputCloud(obj);
		sv.setNormalCloud(normals);
		sv.setNormalImportance(_normal_importance);
		sv.setSpatialImportance(_spatial_importance);
		sv.extract(supervoxels);
		sv.getSupervoxelAdjacency(supervoxel_adjacency);

		/* LOCALLY CONVEX CONNECTED PATCHES */
		std::map<uint32_t, std::set<uint32_t>> segments;
		pcl::LCCPSegmentation<pcl::PointXYZ> lccpseg;
		lccpseg.setInputSupervoxels(supervoxels, supervoxel_adjacency);
		// 单位是度，即CC中的beta_thresh，如果两个patch的法线夹角（数值上等于两个patch的夹角）小于设定的角度阈值，则认为这种程度的凹面是可以接受作为整体的一部分的
		lccpseg.setConcavityToleranceThreshold(_concavity_tolerance_threshold);
		// 用于校验的第三方patch的点数
		lccpseg.setKFactor(_k_factor);
		lccpseg.setMinSegmentSize(_min_segment_size);
		lccpseg.setSanityCheck(_use_sanity_check);
		// voxel_resolution与seed_resolution应与前面进行supervoxel clustering时设定的值一致
		lccpseg.setSmoothnessCheck(_use_smoothness_check, _voxel_resolution, _seed_resolution);
		lccpseg.segment();
		// 返回一个map, 映射关系为：一个分割部分的ID对应一组supervoxel的ID
		lccpseg.getSegmentToSupervoxelMap(segments);

		if (segments.empty())
		{
			std::cout << "No Segment Extracted!!" << std::endl;
			return 1;
		}
		else
		{
			std::cout << segments.size() << " Segments Extracted!!" << std::endl;
		}

		/* VISUALIZATION */
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
		int view_port = 0;

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
		pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
		Eigen::Matrix3f rotation_matrix_OBB;

		std::map<uint32_t, std::set<uint32_t>>::iterator outer_iter = segments.begin();
		int i = 0;
		while (outer_iter != segments.end())
		{
			std::set<uint32_t>::iterator inner_iter = outer_iter->second.begin();
			pcl::PointCloud<pcl::PointXYZ>::Ptr seg_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			while (inner_iter != outer_iter->second.end())
			{
				*seg_cloud += *supervoxels[*inner_iter]->voxels_;
				inner_iter++;
			}
			bb.setInputCloud(seg_cloud);
			bb.compute();

			bb.getAABB(min_point_AABB, max_point_AABB);
			bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

			Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
			Eigen::Quaternionf quaternion(rotation_matrix_OBB);

			std::stringstream cloud_name_ss;
			cloud_name_ss << "cloud" << i << std::endl;
			std::string cloud_name = cloud_name_ss.str();

			pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(seg_cloud);
			viewer->addPointCloud(seg_cloud, temp_handler, cloud_name, view_port);

			/* DRAWING ORIENTED BOUNDING BOXES */
			/*Eigen::Vector3f p1_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p2_OBB(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p3_OBB(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p4_OBB(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p5_OBB(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p6_OBB(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p7_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p8_OBB(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

			p1_OBB = rotation_matrix_OBB * p1_OBB + position;
			p2_OBB = rotation_matrix_OBB * p2_OBB + position;
			p3_OBB = rotation_matrix_OBB * p3_OBB + position;
			p4_OBB = rotation_matrix_OBB * p4_OBB + position;
			p5_OBB = rotation_matrix_OBB * p5_OBB + position;
			p6_OBB = rotation_matrix_OBB * p6_OBB + position;
			p7_OBB = rotation_matrix_OBB * p7_OBB + position;
			p8_OBB = rotation_matrix_OBB * p8_OBB + position;

			pcl::PointXYZ pt1_OBB(p1_OBB(0), p1_OBB(1), p1_OBB(2));
			pcl::PointXYZ pt2_OBB(p2_OBB(0), p2_OBB(1), p2_OBB(2));
			pcl::PointXYZ pt3_OBB(p3_OBB(0), p3_OBB(1), p3_OBB(2));
			pcl::PointXYZ pt4_OBB(p4_OBB(0), p4_OBB(1), p4_OBB(2));
			pcl::PointXYZ pt5_OBB(p5_OBB(0), p5_OBB(1), p5_OBB(2));
			pcl::PointXYZ pt6_OBB(p6_OBB(0), p6_OBB(1), p6_OBB(2));
			pcl::PointXYZ pt7_OBB(p7_OBB(0), p7_OBB(1), p7_OBB(2));
			pcl::PointXYZ pt8_OBB(p8_OBB(0), p8_OBB(1), p8_OBB(2));

			std::stringstream OBB_edge1_ss, OBB_edge2_ss,  OBB_edge3_ss,  OBB_edge4_ss,
			OBB_edge5_ss, OBB_edge6_ss,  OBB_edge7_ss,  OBB_edge8_ss,
			OBB_edge9_ss, OBB_edge10_ss, OBB_edge11_ss, OBB_edge12_ss;

			OBB_edge1_ss  << "OBB_edge1_"  << i << std::endl;
			OBB_edge2_ss  << "OBB_edge2_"  << i << std::endl;
			OBB_edge3_ss  << "OBB_edge3_"  << i << std::endl;
			OBB_edge4_ss  << "OBB_edge4_"  << i << std::endl;
			OBB_edge5_ss  << "OBB_edge5_"  << i << std::endl;
			OBB_edge6_ss  << "OBB_edge6_"  << i << std::endl;
			OBB_edge7_ss  << "OBB_edge7_"  << i << std::endl;
			OBB_edge8_ss  << "OBB_edge8_"  << i << std::endl;
			OBB_edge9_ss  << "OBB_edge9_"  << i << std::endl;
			OBB_edge10_ss << "OBB_edge10_" << i << std::endl;
			OBB_edge11_ss << "OBB_edge11_" << i << std::endl;
			OBB_edge12_ss << "OBB_edge12_" << i << std::endl;

			std::string OBB_edge1  = OBB_edge1_ss.str();
			std::string OBB_edge2  = OBB_edge2_ss.str();
			std::string OBB_edge3  = OBB_edge3_ss.str();
			std::string OBB_edge4  = OBB_edge4_ss.str();
			std::string OBB_edge5  = OBB_edge5_ss.str();
			std::string OBB_edge6  = OBB_edge6_ss.str();
			std::string OBB_edge7  = OBB_edge7_ss.str();
			std::string OBB_edge8  = OBB_edge8_ss.str();
			std::string OBB_edge9  = OBB_edge9_ss.str();
			std::string OBB_edge10 = OBB_edge10_ss.str();
			std::string OBB_edge11 = OBB_edge11_ss.str();
			std::string OBB_edge12 = OBB_edge12_ss.str();

			viewer->addLine (pt1_OBB, pt2_OBB, 0.0, 0.0, 1.0, OBB_edge1);
			viewer->addLine (pt1_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge2);
			viewer->addLine (pt1_OBB, pt5_OBB, 0.0, 0.0, 1.0, OBB_edge3);
			viewer->addLine (pt5_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge4);
			viewer->addLine (pt5_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge5);
			viewer->addLine (pt2_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge6);
			viewer->addLine (pt6_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge7);
			viewer->addLine (pt7_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge8);
			viewer->addLine (pt2_OBB, pt3_OBB, 0.0, 0.0, 1.0, OBB_edge9);
			viewer->addLine (pt4_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge10);
			viewer->addLine (pt3_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge11);
			viewer->addLine (pt3_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge12);*/

			/* DRAWING AXIS ALIGNED BOUNDING BOXES */
			pcl::PointXYZ pt1_AABB(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt2_AABB(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt3_AABB(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt4_AABB(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt5_AABB(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt6_AABB(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt7_AABB(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt8_AABB(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z);

			std::stringstream AABB_edge1_ss, AABB_edge2_ss, AABB_edge3_ss, AABB_edge4_ss,
				AABB_edge5_ss, AABB_edge6_ss, AABB_edge7_ss, AABB_edge8_ss,
				AABB_edge9_ss, AABB_edge10_ss, AABB_edge11_ss, AABB_edge12_ss;

			AABB_edge1_ss << "AABB_edge1_" << i << std::endl;
			AABB_edge2_ss << "AABB_edge2_" << i << std::endl;
			AABB_edge3_ss << "AABB_edge3_" << i << std::endl;
			AABB_edge4_ss << "AABB_edge4_" << i << std::endl;
			AABB_edge5_ss << "AABB_edge5_" << i << std::endl;
			AABB_edge6_ss << "AABB_edge6_" << i << std::endl;
			AABB_edge7_ss << "AABB_edge7_" << i << std::endl;
			AABB_edge8_ss << "AABB_edge8_" << i << std::endl;
			AABB_edge9_ss << "AABB_edge9_" << i << std::endl;
			AABB_edge10_ss << "AABB_edge10_" << i << std::endl;
			AABB_edge11_ss << "AABB_edge11_" << i << std::endl;
			AABB_edge12_ss << "AABB_edge12_" << i << std::endl;

			std::string AABB_edge1 = AABB_edge1_ss.str();
			std::string AABB_edge2 = AABB_edge2_ss.str();
			std::string AABB_edge3 = AABB_edge3_ss.str();
			std::string AABB_edge4 = AABB_edge4_ss.str();
			std::string AABB_edge5 = AABB_edge5_ss.str();
			std::string AABB_edge6 = AABB_edge6_ss.str();
			std::string AABB_edge7 = AABB_edge7_ss.str();
			std::string AABB_edge8 = AABB_edge8_ss.str();
			std::string AABB_edge9 = AABB_edge9_ss.str();
			std::string AABB_edge10 = AABB_edge10_ss.str();
			std::string AABB_edge11 = AABB_edge11_ss.str();
			std::string AABB_edge12 = AABB_edge12_ss.str();

			viewer->addLine(pt1_AABB, pt2_AABB, 1.0, 0.0, 0.0, AABB_edge1);
			viewer->addLine(pt1_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge2);
			viewer->addLine(pt1_AABB, pt5_AABB, 1.0, 0.0, 0.0, AABB_edge3);
			viewer->addLine(pt5_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge4);
			viewer->addLine(pt5_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge5);
			viewer->addLine(pt2_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge6);
			viewer->addLine(pt6_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge7);
			viewer->addLine(pt7_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge8);
			viewer->addLine(pt2_AABB, pt3_AABB, 1.0, 0.0, 0.0, AABB_edge9);
			viewer->addLine(pt4_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge10);
			viewer->addLine(pt3_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge11);
			viewer->addLine(pt3_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge12);

			std::stringstream obj_length_id_ss, obj_width_id_ss, obj_height_id_ss;

			obj_length_id_ss << "obj_length_" << i << std::endl;
			obj_width_id_ss << "obj_width_" << i << std::endl;
			obj_height_id_ss << "obj_height_" << i << std::endl;

			std::string obj_length_id = obj_length_id_ss.str();
			std::string obj_width_id = obj_width_id_ss.str();
			std::string obj_height_id = obj_height_id_ss.str();

			std::stringstream obj_length_val_ss, obj_width_val_ss, obj_height_val_ss;

			float obj_height = pcl::geometry::distance(pt1_AABB, pt5_AABB);
			float obj_length = pcl::geometry::distance(pt1_AABB, pt2_AABB);
			float obj_width = pcl::geometry::distance(pt1_AABB, pt4_AABB);

			obj_height_val_ss << obj_height << " m" << std::endl;
			obj_length_val_ss << obj_length << " m" << std::endl;
			obj_width_val_ss << obj_width << " m" << std::endl;

			std::string obj_height_txt = obj_height_val_ss.str();
			std::string obj_length_txt = obj_length_val_ss.str();
			std::string obj_width_txt = obj_width_val_ss.str();

			pcl::PointXYZ height_txt_position((pt1_AABB.x + pt5_AABB.x) / 2.0, (pt1_AABB.y + pt5_AABB.y) / 2.0, (pt1_AABB.z + pt5_AABB.z) / 2.0);
			pcl::PointXYZ length_txt_position((pt1_AABB.x + pt2_AABB.x) / 2.0, (pt1_AABB.y + pt2_AABB.y) / 2.0, (pt1_AABB.z + pt2_AABB.z) / 2.0);
			pcl::PointXYZ width_txt_position((pt1_AABB.x + pt4_AABB.x) / 2.0, (pt1_AABB.y + pt4_AABB.y) / 2.0, (pt1_AABB.z + pt4_AABB.z) / 2.0);

			float txt_scale_coeff = 0.04;
			float height_txt_scale = txt_scale_coeff * obj_height;
			float length_txt_scale = txt_scale_coeff * obj_length;
			float width_txt_scale = txt_scale_coeff * obj_width;

			viewer->addText3D(obj_height_txt, height_txt_position, height_txt_scale, 1.0, 1.0, 1.0, obj_height_id);
			viewer->addText3D(obj_length_txt, length_txt_position, length_txt_scale, 1.0, 1.0, 1.0, obj_length_id);
			viewer->addText3D(obj_width_txt, width_txt_position, width_txt_scale, 1.0, 1.0, 1.0, obj_width_id);


			outer_iter++;
			i++;
		}

		double coord_sys_scale = 0.3;
		viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		viewer->spin();

		return 0;
	}
private:
	std::string _data_path;
	double _search_radius;
	float _voxel_resolution;
	float _seed_resolution;
	bool _use_single_cam_transform;
	float _normal_importance;
	float _spatial_importance;
	float _concavity_tolerance_threshold;
	int _k_factor;
	int _min_segment_size;
	bool _use_sanity_check;
	bool _use_smoothness_check;
};

class ConstrainedPlanarCuts
{
public:
	ConstrainedPlanarCuts() :
		_search_radius(0.03),
		_voxel_resolution(0.01),
		_seed_resolution(0.05),
		_use_single_cam_transform(true),
		_normal_importance(0.5),
		_spatial_importance(0.5),
		_use_sanity_check(true),
		_use_smoothness_check(true),
		_concavity_tolerance_threshold(10),
		_k_factor(10),
		_min_segment_size(50),
		_ransac_iterations(200),
		_max_cuts(20),
		_cutting_min_segments(0),
		_cutting_min_score(0.16),
		_use_locally_constraints(true),
		_use_directed_cutting(true),
		_use_clean_cutting(false) {};
	void setInputCloudPath(std::string data_path)
	{
		_data_path = data_path;
	};
	void setRadiusSearch(double search_radius)
	{
		_search_radius = search_radius;
	};
	void setVoxelResolution(float voxel_resolution)
	{
		_voxel_resolution = voxel_resolution;
	};
	void setSeedResolution(float seed_resolution)
	{
		_seed_resolution = seed_resolution;
	};
	void setUseSingleCameraTransform(bool use_single_cam_transform)
	{
		_use_single_cam_transform = use_single_cam_transform;
	};
	void setNormalImportance(float normal_importance)
	{
		_normal_importance = normal_importance;
	};
	void setSpatialImportance(float spatial_importance)
	{
		_spatial_importance = spatial_importance;
	};
	void setSanityCheck(bool use_sanity_check)
	{
		_use_sanity_check = use_sanity_check;
	};
	void setSmoothnessCheck(bool use_smoothness_check)
	{
		_use_smoothness_check = use_smoothness_check;
	};
	void setConcavityToleranceThreshold(float concavity_tolerance_threshold)
	{
		_concavity_tolerance_threshold = concavity_tolerance_threshold;
	};
	void setKFactor(int k_factor)
	{
		_k_factor = k_factor;
	};
	void setMinSegmentSize(int min_segment_size)
	{
		_min_segment_size = min_segment_size;
	};
	void setRANSACIterations(int ransac_iterations)
	{
		_ransac_iterations = ransac_iterations;
	};
	void setMaxCuts(int max_cuts) 
	{
		_max_cuts = max_cuts;
	};
	void setCuttingMinSegments(int cutting_min_segments) 
	{
		_cutting_min_segments = cutting_min_segments;
	};
	void setCuttingMinScore(float cutting_min_score) 
	{
		_cutting_min_score = cutting_min_score;
	};
	void setLocallyConstrained(bool use_locally_constraints) 
	{
		_use_locally_constraints = use_locally_constraints;
	};
	void setDirectedCutting(bool use_directed_cutting) 
	{
		_use_directed_cutting = use_directed_cutting;
	};
	void setCleanCutting(bool use_clean_cutting) 
	{
		_use_clean_cutting = use_clean_cutting;
	};
	int segment()
	{
		/* INPUT DATA */
		pcl::PointCloud<pcl::PointXYZ>::Ptr obj(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile(_data_path, *obj);

		if (obj->empty())
		{
			std::cout << "Wrong File Path!" << std::endl;
			return 1;
		}

		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		ne.setInputCloud(obj);
		ne.setSearchMethod(tree);
		ne.setRadiusSearch(_search_radius);
		ne.compute(*normals);

		/* SUPERVOXEL CLUSTERING */
		std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZ>::Ptr> supervoxels;
		std::multimap<uint32_t, uint32_t> supervoxel_adjacency;

		pcl::SupervoxelClustering<pcl::PointXYZ> sv(_voxel_resolution, _seed_resolution);
		// supervoxel内部使用了octree，该变换可使得octree bin大小随着深度的增加而指数增大，避免因点云密度渐小而找不到octree bin的临近关系
		// 如果点云是有序的，需要设置为false，如果点云是无序的，则需要设置为true
		sv.setUseSingleCameraTransform(_use_single_cam_transform);
		sv.setInputCloud(obj);
		sv.setNormalCloud(normals);
		sv.setNormalImportance(_normal_importance);
		sv.setSpatialImportance(_spatial_importance);
		sv.extract(supervoxels);
		sv.getSupervoxelAdjacency(supervoxel_adjacency);

		/* CONSTRAINED PLANAR CUTS */
		std::map<uint32_t, std::set<uint32_t>> segments;
		pcl::CPCSegmentation<pcl::PointXYZ> cpcseg;
		cpcseg.setInputSupervoxels(supervoxels, supervoxel_adjacency);
		cpcseg.setSanityCheck(_use_sanity_check);
		cpcseg.setSmoothnessCheck(_use_smoothness_check, _voxel_resolution, _seed_resolution);
		cpcseg.setConcavityToleranceThreshold(_concavity_tolerance_threshold);
		cpcseg.setKFactor(_k_factor);
		cpcseg.setRANSACIterations(_ransac_iterations);
		cpcseg.setMinSegmentSize(_min_segment_size);
		// clean_cutting: clean: supervoxel级别的切割， not-clean: sub-supervoxel级别的切割
		// cutting_min_score: 由权重决定的切割分数（正比），超过设定的阈值即进行切割
		cpcseg.setCutting(_max_cuts, _cutting_min_segments, _cutting_min_score, _use_locally_constraints, _use_directed_cutting, _use_clean_cutting);
		cpcseg.segment();
		cpcseg.getSegmentToSupervoxelMap(segments);

		if (segments.empty())
		{
			std::cout << "No Segment Extracted!!" << std::endl;
			return 1;
		}
		else
		{
			std::cout << segments.size() << " Segments Extracted!!" << std::endl;
		}

		/* VISUALIZATION */
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
		int view_port = 0;

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> bb;
		pcl::PointXYZ min_point_AABB, max_point_AABB, min_point_OBB, max_point_OBB, position_OBB;
		Eigen::Matrix3f rotation_matrix_OBB;

		std::map<uint32_t, std::set<uint32_t>>::iterator outer_iter = segments.begin();
		int i = 0;
		while (outer_iter != segments.end())
		{
			std::set<uint32_t>::iterator inner_iter = outer_iter->second.begin();
			pcl::PointCloud<pcl::PointXYZ>::Ptr seg_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			while (inner_iter != outer_iter->second.end())
			{
				*seg_cloud += *supervoxels[*inner_iter]->voxels_;
				inner_iter++;
			}
			bb.setInputCloud(seg_cloud);
			bb.compute();

			bb.getAABB(min_point_AABB, max_point_AABB);
			bb.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotation_matrix_OBB);

			Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
			Eigen::Quaternionf quaternion(rotation_matrix_OBB);

			std::stringstream cloud_name_ss;
			cloud_name_ss << "cloud" << i << std::endl;
			std::string cloud_name = cloud_name_ss.str();

			pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> temp_handler(seg_cloud);
			viewer->addPointCloud(seg_cloud, temp_handler, cloud_name, view_port);

			/* DRAWING ORIENTED BOUNDING BOXES */
			/*Eigen::Vector3f p1_OBB(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p2_OBB(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p3_OBB(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p4_OBB(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p5_OBB(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
			Eigen::Vector3f p6_OBB(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p7_OBB(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
			Eigen::Vector3f p8_OBB(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

			p1_OBB = rotation_matrix_OBB * p1_OBB + position;
			p2_OBB = rotation_matrix_OBB * p2_OBB + position;
			p3_OBB = rotation_matrix_OBB * p3_OBB + position;
			p4_OBB = rotation_matrix_OBB * p4_OBB + position;
			p5_OBB = rotation_matrix_OBB * p5_OBB + position;
			p6_OBB = rotation_matrix_OBB * p6_OBB + position;
			p7_OBB = rotation_matrix_OBB * p7_OBB + position;
			p8_OBB = rotation_matrix_OBB * p8_OBB + position;

			pcl::PointXYZ pt1_OBB(p1_OBB(0), p1_OBB(1), p1_OBB(2));
			pcl::PointXYZ pt2_OBB(p2_OBB(0), p2_OBB(1), p2_OBB(2));
			pcl::PointXYZ pt3_OBB(p3_OBB(0), p3_OBB(1), p3_OBB(2));
			pcl::PointXYZ pt4_OBB(p4_OBB(0), p4_OBB(1), p4_OBB(2));
			pcl::PointXYZ pt5_OBB(p5_OBB(0), p5_OBB(1), p5_OBB(2));
			pcl::PointXYZ pt6_OBB(p6_OBB(0), p6_OBB(1), p6_OBB(2));
			pcl::PointXYZ pt7_OBB(p7_OBB(0), p7_OBB(1), p7_OBB(2));
			pcl::PointXYZ pt8_OBB(p8_OBB(0), p8_OBB(1), p8_OBB(2));

			std::stringstream OBB_edge1_ss, OBB_edge2_ss,  OBB_edge3_ss,  OBB_edge4_ss,
			OBB_edge5_ss, OBB_edge6_ss,  OBB_edge7_ss,  OBB_edge8_ss,
			OBB_edge9_ss, OBB_edge10_ss, OBB_edge11_ss, OBB_edge12_ss;

			OBB_edge1_ss  << "OBB_edge1_"  << i << std::endl;
			OBB_edge2_ss  << "OBB_edge2_"  << i << std::endl;
			OBB_edge3_ss  << "OBB_edge3_"  << i << std::endl;
			OBB_edge4_ss  << "OBB_edge4_"  << i << std::endl;
			OBB_edge5_ss  << "OBB_edge5_"  << i << std::endl;
			OBB_edge6_ss  << "OBB_edge6_"  << i << std::endl;
			OBB_edge7_ss  << "OBB_edge7_"  << i << std::endl;
			OBB_edge8_ss  << "OBB_edge8_"  << i << std::endl;
			OBB_edge9_ss  << "OBB_edge9_"  << i << std::endl;
			OBB_edge10_ss << "OBB_edge10_" << i << std::endl;
			OBB_edge11_ss << "OBB_edge11_" << i << std::endl;
			OBB_edge12_ss << "OBB_edge12_" << i << std::endl;

			std::string OBB_edge1  = OBB_edge1_ss.str();
			std::string OBB_edge2  = OBB_edge2_ss.str();
			std::string OBB_edge3  = OBB_edge3_ss.str();
			std::string OBB_edge4  = OBB_edge4_ss.str();
			std::string OBB_edge5  = OBB_edge5_ss.str();
			std::string OBB_edge6  = OBB_edge6_ss.str();
			std::string OBB_edge7  = OBB_edge7_ss.str();
			std::string OBB_edge8  = OBB_edge8_ss.str();
			std::string OBB_edge9  = OBB_edge9_ss.str();
			std::string OBB_edge10 = OBB_edge10_ss.str();
			std::string OBB_edge11 = OBB_edge11_ss.str();
			std::string OBB_edge12 = OBB_edge12_ss.str();

			viewer->addLine (pt1_OBB, pt2_OBB, 0.0, 0.0, 1.0, OBB_edge1);
			viewer->addLine (pt1_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge2);
			viewer->addLine (pt1_OBB, pt5_OBB, 0.0, 0.0, 1.0, OBB_edge3);
			viewer->addLine (pt5_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge4);
			viewer->addLine (pt5_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge5);
			viewer->addLine (pt2_OBB, pt6_OBB, 0.0, 0.0, 1.0, OBB_edge6);
			viewer->addLine (pt6_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge7);
			viewer->addLine (pt7_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge8);
			viewer->addLine (pt2_OBB, pt3_OBB, 0.0, 0.0, 1.0, OBB_edge9);
			viewer->addLine (pt4_OBB, pt8_OBB, 0.0, 0.0, 1.0, OBB_edge10);
			viewer->addLine (pt3_OBB, pt4_OBB, 0.0, 0.0, 1.0, OBB_edge11);
			viewer->addLine (pt3_OBB, pt7_OBB, 0.0, 0.0, 1.0, OBB_edge12);*/

			/* DRAWING AXIS ALIGNED BOUNDING BOXES */
			pcl::PointXYZ pt1_AABB(min_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt2_AABB(min_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt3_AABB(max_point_AABB.x, min_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt4_AABB(max_point_AABB.x, min_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt5_AABB(min_point_AABB.x, max_point_AABB.y, min_point_AABB.z);
			pcl::PointXYZ pt6_AABB(min_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt7_AABB(max_point_AABB.x, max_point_AABB.y, max_point_AABB.z);
			pcl::PointXYZ pt8_AABB(max_point_AABB.x, max_point_AABB.y, min_point_AABB.z);

			std::stringstream AABB_edge1_ss, AABB_edge2_ss, AABB_edge3_ss, AABB_edge4_ss,
				AABB_edge5_ss, AABB_edge6_ss, AABB_edge7_ss, AABB_edge8_ss,
				AABB_edge9_ss, AABB_edge10_ss, AABB_edge11_ss, AABB_edge12_ss;

			AABB_edge1_ss << "AABB_edge1_" << i << std::endl;
			AABB_edge2_ss << "AABB_edge2_" << i << std::endl;
			AABB_edge3_ss << "AABB_edge3_" << i << std::endl;
			AABB_edge4_ss << "AABB_edge4_" << i << std::endl;
			AABB_edge5_ss << "AABB_edge5_" << i << std::endl;
			AABB_edge6_ss << "AABB_edge6_" << i << std::endl;
			AABB_edge7_ss << "AABB_edge7_" << i << std::endl;
			AABB_edge8_ss << "AABB_edge8_" << i << std::endl;
			AABB_edge9_ss << "AABB_edge9_" << i << std::endl;
			AABB_edge10_ss << "AABB_edge10_" << i << std::endl;
			AABB_edge11_ss << "AABB_edge11_" << i << std::endl;
			AABB_edge12_ss << "AABB_edge12_" << i << std::endl;

			std::string AABB_edge1 = AABB_edge1_ss.str();
			std::string AABB_edge2 = AABB_edge2_ss.str();
			std::string AABB_edge3 = AABB_edge3_ss.str();
			std::string AABB_edge4 = AABB_edge4_ss.str();
			std::string AABB_edge5 = AABB_edge5_ss.str();
			std::string AABB_edge6 = AABB_edge6_ss.str();
			std::string AABB_edge7 = AABB_edge7_ss.str();
			std::string AABB_edge8 = AABB_edge8_ss.str();
			std::string AABB_edge9 = AABB_edge9_ss.str();
			std::string AABB_edge10 = AABB_edge10_ss.str();
			std::string AABB_edge11 = AABB_edge11_ss.str();
			std::string AABB_edge12 = AABB_edge12_ss.str();

			viewer->addLine(pt1_AABB, pt2_AABB, 1.0, 0.0, 0.0, AABB_edge1);
			viewer->addLine(pt1_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge2);
			viewer->addLine(pt1_AABB, pt5_AABB, 1.0, 0.0, 0.0, AABB_edge3);
			viewer->addLine(pt5_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge4);
			viewer->addLine(pt5_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge5);
			viewer->addLine(pt2_AABB, pt6_AABB, 1.0, 0.0, 0.0, AABB_edge6);
			viewer->addLine(pt6_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge7);
			viewer->addLine(pt7_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge8);
			viewer->addLine(pt2_AABB, pt3_AABB, 1.0, 0.0, 0.0, AABB_edge9);
			viewer->addLine(pt4_AABB, pt8_AABB, 1.0, 0.0, 0.0, AABB_edge10);
			viewer->addLine(pt3_AABB, pt4_AABB, 1.0, 0.0, 0.0, AABB_edge11);
			viewer->addLine(pt3_AABB, pt7_AABB, 1.0, 0.0, 0.0, AABB_edge12);

			std::stringstream obj_length_id_ss, obj_width_id_ss, obj_height_id_ss;

			obj_length_id_ss << "obj_length_" << i << std::endl;
			obj_width_id_ss << "obj_width_" << i << std::endl;
			obj_height_id_ss << "obj_height_" << i << std::endl;

			std::string obj_length_id = obj_length_id_ss.str();
			std::string obj_width_id = obj_width_id_ss.str();
			std::string obj_height_id = obj_height_id_ss.str();

			std::stringstream obj_length_val_ss, obj_width_val_ss, obj_height_val_ss;

			float obj_height = pcl::geometry::distance(pt1_AABB, pt5_AABB);
			float obj_length = pcl::geometry::distance(pt1_AABB, pt2_AABB);
			float obj_width = pcl::geometry::distance(pt1_AABB, pt4_AABB);

			obj_height_val_ss << obj_height << " m" << std::endl;
			obj_length_val_ss << obj_length << " m" << std::endl;
			obj_width_val_ss << obj_width << " m" << std::endl;

			std::string obj_height_txt = obj_height_val_ss.str();
			std::string obj_length_txt = obj_length_val_ss.str();
			std::string obj_width_txt = obj_width_val_ss.str();

			pcl::PointXYZ height_txt_position((pt1_AABB.x + pt5_AABB.x) / 2.0, (pt1_AABB.y + pt5_AABB.y) / 2.0, (pt1_AABB.z + pt5_AABB.z) / 2.0);
			pcl::PointXYZ length_txt_position((pt1_AABB.x + pt2_AABB.x) / 2.0, (pt1_AABB.y + pt2_AABB.y) / 2.0, (pt1_AABB.z + pt2_AABB.z) / 2.0);
			pcl::PointXYZ width_txt_position((pt1_AABB.x + pt4_AABB.x) / 2.0, (pt1_AABB.y + pt4_AABB.y) / 2.0, (pt1_AABB.z + pt4_AABB.z) / 2.0);

			float txt_scale_coeff = 0.04;
			float height_txt_scale = txt_scale_coeff * obj_height;
			float length_txt_scale = txt_scale_coeff * obj_length;
			float width_txt_scale = txt_scale_coeff * obj_width;

			viewer->addText3D(obj_height_txt, height_txt_position, height_txt_scale, 1.0, 1.0, 1.0, obj_height_id);
			viewer->addText3D(obj_length_txt, length_txt_position, length_txt_scale, 1.0, 1.0, 1.0, obj_length_id);
			viewer->addText3D(obj_width_txt, width_txt_position, width_txt_scale, 1.0, 1.0, 1.0, obj_width_id);


			outer_iter++;
			i++;
		}

		double coord_sys_scale = 0.3;
		viewer->addCoordinateSystem(coord_sys_scale);
		double bg_color[3] = { 0.0, 0.0, 0.0 };
		viewer->setBackgroundColor(bg_color[0], bg_color[1], bg_color[2]);

		viewer->spin();

		return 0;
	}
private:
	std::string _data_path;
	double _search_radius;
	float _voxel_resolution;
	float _seed_resolution;
	bool _use_single_cam_transform;
	float _normal_importance;
	float _spatial_importance;
	bool _use_sanity_check;
	bool _use_smoothness_check;
	float _concavity_tolerance_threshold;
	int _k_factor;
	int _min_segment_size;
	int _ransac_iterations;
	int _max_cuts;
	int _cutting_min_segments;
	float _cutting_min_score;
	bool _use_locally_constraints;
	bool _use_directed_cutting;
	bool _use_clean_cutting;
};


int main()
{
	MinCut seg;
	seg.setInputCloudPath("..//data//hands.ply");
	seg.segment();
	return 0;
}
