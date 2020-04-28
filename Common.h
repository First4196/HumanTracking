#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <limits>
#include <math.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace std;
using namespace cv;

string dir = "D:\\Workspace\\Sensor Tech\\RPLidarData\\processed\\";
vector<string> filenames = {
	"g1_still.txt",
	"aq1.txt",
	"aq2.txt",
	"aq3.txt",
	"aq4.txt",
	"pr1_room.txt",
	"pr2_straight.txt",
	"pr3_curve.txt",
	"pr4_obstacle.txt",
	"pw1_room.txt",
	"pw2_straight.txt",
	"pw3_leftnright.txt",
	"pw4_behindobject.txt",
};
string map_dir = "D:\\Workspace\\Sensor Tech\\RPLidarData\\map\\";
vector<string> map_filenames = {
	"aq.png",
	"pr.png",
	"pw.png",
};
double max_dist = 8.0f;
int window_size = 800;

double angles_mean(const vector<double> &angles, const vector<double> &weights) {
	assert(angles.size() == weights.size());
	double sum_x = 0.0f;
	double sum_y = 0.0f;
	double sum_weights = 0.0f;
	for (int i = 0; i < angles.size(); i++) {
		sum_x += weights[i] * cos(angles[i] * M_PI / 180.0f);
		sum_y += weights[i] * sin(angles[i] * M_PI / 180.0f);
		sum_weights += weights[i];
	}
	double mean = atan2(sum_y / sum_weights, sum_x / sum_weights) * 180.0f / M_PI;
	return mean >= 0.0f ? mean : mean + 360.0f;
}

double angles_mean(const vector<double>& angles) {
	vector<double> weights(angles.size(), 1.0f);
	return angles_mean(angles, weights);
}

double angle_wrap(double angle) {
	angle = fmod(angle, 360.0f);
	return angle >= 0.0f ? angle : angle + 360.f;
}

double norm(Point2d point) {
	return sqrt(point.x * point.x + point.y * point.y);
}

Point2d rotate_around_origin(Point2d point, double angle_deg) {
	double angle_rad = angle_deg * M_PI / 180.f;
	double x = point.x * cos(angle_rad) - point.y * sin(angle_rad);
	double y = point.x * sin(angle_rad) + point.y * cos(angle_rad);
	return Point2d(x, y);
}

double guassian_pdf(double x, double mean, double sd) {
	double z = (x - mean) / sd;
	return exp(-0.5f * z * z / sqrt(2.0f * M_PI));
}

vector<Point2d> ellipse_ray_intersection(double a, double b, Point2d p, Point2d q) {
	double eps = 1e-6;
	vector<Point2d> intersections;
	bool flip = false;
	if (abs(p.x - q.x) < eps) {
		if (abs(p.y - q.y) < eps) {
			return intersections;
		}
		flip = true;
		swap(a, b);
		swap(p.x, p.y);
		swap(q.x, q.y);
	}
	double m = (p.y - q.y) / (p.x - q.x);
	double c = p.y - m * p.x;
	double t = 4 * a * a * b * b * (m * m * a * a + b * b - c * c);
	if (t > eps) {
		{
			double x = (-2 * m * c * a * a + sqrt(t)) / (2 * m * m * a * a + 2 * b * b);
			Point2d s(x, m * x + c);
			if ((q - p).dot(s - p) > eps) {
				intersections.push_back(s);
			}
		}
		{
			double x = (-2 * m * c * a * a - sqrt(t)) / (2 * m * m * a * a + 2 * b * b);
			Point2d s(x, m * x + c);
			if ((q - p).dot(s - p) > eps) {
				intersections.push_back(s);
			}
		}
	}
	else if (t > -eps) {
		double x = (-2 * m * c * a * a) / (2 * m * m * a * a + 2 * b * b);
		Point2d s(x, m * x + c);
		if ((q - p).dot(s - p) > eps) {
			intersections.push_back(s);
		}
	}
	if (flip) {
		for (int i = 0; i < intersections.size(); i++) {
			swap(intersections[i].x, intersections[i].y);
		}
	}
	return intersections;
}

void polar_to_euclidean(double angle, double dist, double& x, double& y) {
	x = dist * cos(angle * M_PI / 180.f);
	y = dist * sin(angle * M_PI / 180.f);
}

void euclidean_to_pixel(double x, double y, int& px, int& py) {
	px = (window_size / 2) * (1.0f + x / max_dist);
	py = (window_size / 2) * (1.0f + y / max_dist);
}

void euclidean_to_pixel_size(double x, double y, int& sx, int& sy) {
	sx = (window_size / 2) * (x / max_dist);
	sy = (window_size / 2) * (y / max_dist);
}

void draw_grid(Mat& m) {
	{
		int px1, py1, px2, py2;
		euclidean_to_pixel(0.0f, -max_dist, px1, py1);
		euclidean_to_pixel(0.0f, max_dist, px2, py2);
		line(m, Point(px1, py1), Point(px2, py2), Scalar(0, 100, 0));
	}
	{
		int px1, py1, px2, py2;
		euclidean_to_pixel(-max_dist, 0.0f, px1, py1);
		euclidean_to_pixel(max_dist, 0.0f, px2, py2);
		line(m, Point(px1, py1), Point(px2, py2), Scalar(0, 0, 100));
	}
	for (double i = 1.0f; i < max_dist; i += 1.0f) {
		for (int k = 0; k < 4; k++) {
			Point2d point1, point2;
			if (k == 0) {
				point1 = Point2d(i, -max_dist);
				point2 = Point2d(i, max_dist);
			}
			else if (k == 1) {
				point1 = Point2d(-i, -max_dist);
				point2 = Point2d(-i, max_dist);
			}
			else if (k == 2) {
				point1 = Point2d(-max_dist, i);
				point2 = Point2d(max_dist, i);
			}
			else {
				point1 = Point2d(-max_dist, -i);
				point2 = Point2d(max_dist, -i);
			}
			int px1, py1, px2, py2;
			euclidean_to_pixel(point1.x, point1.y, px1, py1);
			euclidean_to_pixel(point2.x, point2.y, px2, py2);
			line(m, Point(px1, py1), Point(px2, py2), Scalar(50, 50, 50));
		}
	}
}
