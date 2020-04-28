#include "Common.h"

string filename = filenames[1];
string window_name = "DataViewer";

int main(){
	namedWindow(window_name);
	ifstream input_file(dir + filename);
	int n_frames = 0;
	Mat map(window_size, window_size, CV_8UC3, Scalar(0, 0, 0));
	while (!input_file.eof()) {
		Mat frame(window_size, window_size, CV_8UC3, Scalar(0, 0, 0));
		draw_grid(frame);
		int n_points;
		input_file >> n_points;
		n_frames += 1;
		for (int i = 0; i < n_points; i++) {
			double angle, dist, x, y;
			int px, py;
			input_file >> angle >> dist;
			polar_to_euclidean(angle, dist, x, y);
			euclidean_to_pixel(x, y, px, py);
			frame.at<Vec3b>(Point(px, py)) = Vec3b(255, 255, 255);
			map.at<Vec3b>(Point(px, py)) = Vec3b(255, 255, 255);
		}
		putText(frame, to_string(n_frames), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		imshow(window_name, frame);
		waitKey(10);
	}
	imshow(window_name, map);
	imwrite("D:\\Workspace\\Sensor Tech\\RPLidarData\\processed\\" + filename + ".png", map);
	waitKey();
	return 0;
}