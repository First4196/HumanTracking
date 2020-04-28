#include "HumanTracking.h"
#include <functional>

void test_sampling(int n, function<double ()> f) {
	double sum = 0, mn = 1000000, mx = -1000000;
	for (int i = 0; i < n; i++) {
		double x = f();
		sum += x;
		if (x < mn) mn = x;
		if (x > mx) mx = x;
	}
	cout << sum / n << " " << mn << " " << mx << endl;
}

void test_rotate_around_origin(Point2d point, double angle) {
	Point2d rotated = rotate_around_origin(point, angle);
	cout << rotated.x << " " << rotated.y << endl;
}

void test_ellipse_ray_intersection(double a, double b, Point2d p, Point2d q) {
	vector<Point2d> intersections = ellipse_ray_intersection(a, b, p, q);
	cout << intersections.size() << " ";
	for (int i = 0; i < intersections.size(); i++) {
		cout << "(" << intersections[i].x << "," << intersections[i].y << ") ";
	}
	cout << endl;
}

void test_guassian_pdf(double x, double mean, double sd) {
	cout << guassian_pdf(x, mean, sd) << endl;
}

void test_guassian_likelihood(State state, Point2d point) {
	ParticleFilter pf(1, 1, 1);
	cout << pf.guassian_likelihood(state, point) << endl;
}

void test_angles_mean(vector<double> angles) {
	cout << angles_mean(angles) << endl;
}

void test_angles_mean(vector<double> angles, vector<double> weights) {
	cout << angles_mean(angles, weights) << endl;
}

void test_angle_wrap(double angle) {
	cout << angle_wrap(angle) << endl;
}

int main() {
	test_sampling(1000000, EXYT);
	test_sampling(1000000, EAngleT);
	test_sampling(1000000, EVT);
	cout << endl;

	test_rotate_around_origin(Point2d(1.0f, 1.0f), 90.0f);
	test_rotate_around_origin(Point2d(1.0f, 0.0f), 45.0f);
	test_rotate_around_origin(Point2d(0.0f, 1.0f), 180.0f);
	cout << endl;

	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(-10.0f, 10.0f), Point2d(10.0f, 10.0f)); // 0
	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(-10.0f, 2.0f), Point2d(10.0f, 2.0f)); // 1
	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(-10.0f, 0.0f), Point2d(10.0f, 0.0f)); // 2
	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(5.0f, 0.0f), Point2d(10.0f, 0.0f)); // 0
	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(-5.0f, -4.0f), Point2d(5.0f, 4.0f)); // 2
	test_ellipse_ray_intersection(1.5f, 2.0f, Point2d(0.0f, -10.0f), Point2d(0.0f, 10.0f)); // 2 flip
	cout << endl;

	for (double x = 0.8f; x < 1.2f; x += 0.01f) {
		cout << x << " ";
		test_guassian_pdf(x, 1.0f, 0.1f);
	}
	cout << endl;

	for (double x = 0.8f; x < 1.2f; x += 0.01f) {
		cout << x << " ";
		test_guassian_likelihood(State(1.0f, 0.0f, 0.0f, 0.0f), Point2d(x, 0.0f));
	}
	cout << endl;

	test_angles_mean({ 10, 30, 50, 70, 90 });
	test_angles_mean({ -30, -10, 10, 30, 50 });
	test_angles_mean({ 330, 350, 10, 30, 50 });
	test_angles_mean({ 330, 330, 330, 330, 330, 350, 350, 350, 350, 10, 10, 10, 30, 30, 50 });
	test_angles_mean({ 330, 350, 10, 30, 50 }, { 5, 4, 3, 2, 1 });
	test_angles_mean({ 0, 180 });
	cout << endl;

	test_angle_wrap(-360.0f);
	test_angle_wrap(-315.0f);
	test_angle_wrap(-270.0f);
	test_angle_wrap(-225.0f);
	test_angle_wrap(-180.0f);
	test_angle_wrap(-135.0f);
	test_angle_wrap(-90.0f);
	test_angle_wrap(-45.0f);
	test_angle_wrap(0.0f);
	test_angle_wrap(45.0f);
	test_angle_wrap(90.0f);
	test_angle_wrap(135.0f);
	test_angle_wrap(180.0f);
	test_angle_wrap(225.0f);
	test_angle_wrap(270.0f);
	test_angle_wrap(315.0f);
	test_angle_wrap(360.0f);
	cout << endl;

	return 0;
}