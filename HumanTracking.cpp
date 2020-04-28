#include "HumanTracking.h"

string filename = filenames[2];
string window_name = "HumanTracking";
bool use_map = true;
string map_filename = map_filenames[0];
bool draw_particles = true;

int main() {
	namedWindow(window_name);
	Mat map;
	if (use_map) {
		map = imread(map_dir + map_filename, IMREAD_GRAYSCALE);
	}
	ifstream input_file(dir + filename);
	int n_frames = 0;
	ParticleFilter pf(n_particles_init, n_particles_min, n_particles_decay_rate);
	while (!input_file.eof()) {
		Mat frame(window_size, window_size, CV_8UC3, Scalar(0, 0, 0));
		draw_grid(frame);
		vector<Point2d> points;
		int n_points;
		input_file >> n_points;
		n_frames += 1;
		for (int i = 0; i < n_points; i++) {
			double angle, dist, x, y;
			int px, py;
			input_file >> angle >> dist;
			polar_to_euclidean(angle, dist, x, y);
			euclidean_to_pixel(x, y, px, py);
			if (use_map && map.at<uchar>(Point(px, py)) > 0) {
				frame.at<Vec3b>(Point(px, py)) = Vec3b(255, 255, 0);
			}
			else {
				points.push_back(Point2d(x, y));
				frame.at<Vec3b>(Point(px, py)) = Vec3b(255, 255, 255);
			}
		}
		pf.step(points);
		pf.draw(frame, draw_particles);
		putText(frame, to_string(n_frames), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		putText(frame, to_string(pf.n_particles), Point(5, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow(window_name, frame);
		waitKey(10);
	}
	return 0;
}