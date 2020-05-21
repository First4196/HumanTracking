#include "Common.h"

int n_particles_init = 5000;
int n_particles_min = 1000;
double n_particles_decay_rate = 0.95f;
double brand_new_particle_rate = 0.01f;
bool use_velocity = false;
bool use_particles_average = true;
bool merge_opposite_angle = true;

double human_a = 0.1f;
double human_b = 0.2f;

bool use_beam_model = true;
double dummy_point_likelihood = 0.1f;
double beam_likelihood_sd = 0.1f;
double point_likelihood_sd = 0.1f;

double min_x = -4.0f;
double max_x = 4.0f;
double min_y = -4.0f;
double max_y = 4.0f;
double min_v = -1.0f;
double max_v = 1.0f;
default_random_engine rng;
uniform_real_distribution<double> R(0.0f, 1.0f);
uniform_real_distribution<double> X(min_x, max_x);
uniform_real_distribution<double> Y(min_y, max_y);
uniform_real_distribution<double> Angle(0.0f, 360.0f);
uniform_real_distribution<double> V(min_v, max_v);

double sd_exy = 0.1f;
double min_exy = -1.0f;
double max_exy = 1.0f;
normal_distribution<double> EXY(0, sd_exy);
double EXYT() {
	while (true){
		double exy = EXY(rng);
		if (min_exy <= exy && exy < max_exy) {
			return exy;
		}
	}
}

double sd_eangle = 5.0f;
double min_eangle = -45.0f;
double max_eangle = 45.0f;
normal_distribution<double> EAngle(0, sd_eangle);
double EAngleT() {
	while (true) {
		double eangle = EAngle(rng);
		if (min_eangle <= eangle && eangle < max_eangle) {
			return eangle;
		}
	}
}

double sd_ev = 0.1f;
double min_ev = -1.0f;
double max_ev = 1.0f;
normal_distribution<double> EV(0, sd_ev);
double EVT() {
	while (true) {
		double ev = EV(rng);
		if (min_ev <= ev && ev < max_ev) {
			return ev;
		}
	}
}

class State {
public:
	State(double x, double y, double angle, double v)
		: x(x), y(y), angle(angle), v(v) {

	}
	double x, y, angle, v;
};

class Particle {
public:
	Particle(double x, double y, double angle, double v, double logp = 1.0f)
		: state(x, y, angle, v), logp(logp){

	}
	Particle(State state, double logp = 1.0f)
		: state(state), logp(logp) {

	}
	State state;
	double logp;
};

class ParticleFilter {
public:
	ParticleFilter(int n_particles_init, int n_particles_min, double n_particles_decay_rate)
		: n_particles(n_particles_init), n_particles_init(n_particles_init), n_particles_min(n_particles_min), n_particles_decay_rate(n_particles_decay_rate) {
		particles = vector<Particle>();
		for (int i = 0; i < n_particles; i++) {
			particles.push_back(brand_new_particle());
		}
		normalize();
	}
	Particle brand_new_particle() {
		double x = X(rng);
		double y = Y(rng);
		double angle = Angle(rng);
		if (merge_opposite_angle && angle >= 180.f) {
			angle -= 180.0f;
		}
		double v = use_velocity ? V(rng) : 0.0f;
		return Particle(x, y, angle, v);
	}
	void step(const vector<Point2d>& points) {
		prediction();
		correction(points);
		normalize();
	}
	State guassian_transition(State state) {
		double newx = state.x + state.v * cos(state.angle * M_PI / 180.0) + EXYT();
		double newy = state.y + state.v * sin(state.angle * M_PI / 180.0) + EXYT();
		double newangle = angle_wrap(state.angle + EAngleT());
		if (merge_opposite_angle && newangle >= 180.f) {
			newangle -= 180.0f;
		}
		double newv = state.v + (use_velocity ? EVT() : 0.0f);
		return State(newx, newy, newangle, newv);
	}
	void prediction() {
		vector<double> logps;
		for (int i = 0; i < n_particles; i++) {
			logps.push_back(exp(particles[i].logp));
		}
		discrete_distribution<int> I(logps.begin(), logps.end());
		int new_n_particles = max((int)(n_particles * n_particles_decay_rate), n_particles_min);
		vector<Particle> new_particles;
		while(new_particles.size() < new_n_particles) {
			if (R(rng) < brand_new_particle_rate) {
				new_particles.push_back(brand_new_particle());
			}
			else {
				Particle particle = particles[I(rng)];
				particle.state = guassian_transition(particle.state);
				particle.logp = 1.0f;
				if (norm(Point2d(particle.state.x, particle.state.y)) < max_dist) {
					new_particles.push_back(particle);
				}
			}
		}
		n_particles = new_n_particles;
		particles = new_particles;
	}
	double beam_likelihood(State state, Point2d point) {
		Point2d center = Point2d(state.x, state.y);
		Point2d pointT = point - center;
		Point2d originT = -center;
		Point2d pointTR = rotate_around_origin(pointT, -state.angle);
		Point2d originTR = rotate_around_origin(originT, -state.angle);
		vector<Point2d> intersectionsTR = ellipse_ray_intersection(human_a, human_b, originTR, pointTR);
		vector<Point2d> intersectionsT;
		for (int i = 0; i < intersectionsTR.size(); i++) {
			intersectionsT.push_back(rotate_around_origin(intersectionsTR[i], state.angle));
		}
		vector<Point2d> intersections;
		for (int i = 0; i < intersectionsT.size(); i++) {
			intersections.push_back(intersectionsT[i] + center);
		}
		if (intersections.size() == 1) {
			double dist = norm(point) - norm(intersections[0]);
			return guassian_pdf(dist, 0, beam_likelihood_sd);
		}
		else if(intersections.size() == 2){
			double dist = norm(point) - min(norm(intersections[0]), norm(intersections[1]));
			return guassian_pdf(dist, 0, beam_likelihood_sd);
		}
		else {
			return -1.0f;
		}
	}
	double point_likelihood(State state, Point2d point) {
		// point ellipse distance is hard, use origin-point intersection with ellipse instead
		Point2d center = Point2d(state.x, state.y);
		Point2d pointT = point - center;
		Point2d pointTR = rotate_around_origin(pointT, -state.angle);
		vector<Point2d> intersectionsTR = ellipse_ray_intersection(human_a, human_b, Point2d(0.0f, 0.0f), pointTR);
		assert(intersectionsTR.size() == 1);
		double dist = norm(pointTR) - norm(intersectionsTR[0]);
		return guassian_pdf(dist, 0, point_likelihood_sd);
	}
	void correction(const vector<Point2d>& points) {
		for (int i = 0; i < n_particles; i++) {
			int n_valid_points = 0;
			double logscore = log(dummy_point_likelihood);
			for (int j = 0; j < points.size(); j++) {
				double likelihood;
				if (use_beam_model) {
					likelihood = beam_likelihood(particles[i].state, points[j]);
				}
				else {
					likelihood = point_likelihood(particles[i].state, points[j]);
				}
				if (likelihood >= 0) {
					n_valid_points++;
					logscore += log(likelihood);
				}
			}
			logscore /= (double)n_valid_points + 1;
			particles[i].logp += logscore;
		}
	}
	void normalize() {
		double maxlogp = numeric_limits<double>::min();
		for (int i = 0; i < n_particles; i++) {
			if (particles[i].logp > maxlogp) {
				maxlogp = particles[i].logp;
			}
		}
		double sumexp = 0.0f;
		for (int i = 0; i < n_particles; i++) {
			sumexp += exp(particles[i].logp - maxlogp);
		}
		double logsumexp = maxlogp + log(sumexp);
		for (int i = 0; i < n_particles; i++) {
			particles[i].logp -= logsumexp;
		}
	}
	void draw(Mat& frame, bool draw_particles = true) {
		if (draw_particles) {
			for (int i = 0; i < n_particles; i++) {
				Point2d point1 = Point2d(particles[i].state.x, particles[i].state.y);
				Point2d point2 = point1 + rotate_around_origin(Point2d(0.1f, 0.0f), particles[i].state.angle);
				int px1, py1, px2, py2;
				euclidean_to_pixel(point1.x, point1.y, px1, py1);
				euclidean_to_pixel(point2.x, point2.y, px2, py2);
				arrowedLine(frame, Point(px1, py1), Point(px2, py2), Scalar(255, 0, 0));
			}
		}
		State estimated_state = particles[0].state;
		if (use_particles_average) {
			double sum_x = 0.0f;
			double sum_y = 0.0f;
			double sum_v = 0.0f;
			double sum_p = 0.0f;
			vector<double> angles;
			vector<double> ps;
			for (int i = 1; i < n_particles; i++) {
				double p = exp(particles[i].logp);
				sum_x += p * particles[i].state.x;
				sum_y += p * particles[i].state.y;
				sum_v += p * particles[i].state.v;
				sum_p += p;
				angles.push_back(particles[i].state.angle * (merge_opposite_angle ? 2.0f : 1.0f));
				ps.push_back(p);
			}
			double avg_angle = angles_mean(angles, ps) / (merge_opposite_angle ? 2.0f : 1.0f);
			estimated_state = State(sum_x / sum_p, sum_y / sum_p, avg_angle, sum_v / sum_p);
		}
		else {
			double maxlogp = particles[0].logp;
			for (int i = 1; i < n_particles; i++) {
				if (particles[i].logp > maxlogp) {
					maxlogp = particles[i].logp;
					estimated_state = particles[i].state;
				}
			}
		}
		{
			Point2d point1 = Point2d(estimated_state.x, estimated_state.y);
			Point2d point2 = point1 + rotate_around_origin(Point2d(0.2f, 0.0f), estimated_state.angle);
			int px1, py1, px2, py2, sx, sy;
			euclidean_to_pixel(point1.x, point1.y, px1, py1);
			euclidean_to_pixel(point2.x, point2.y, px2, py2);
			euclidean_to_pixel_size(human_a, human_b, sx, sy);
			ellipse(frame, Point(px1, py1), Size(sx, sy), estimated_state.angle, 0, 360, Scalar(0, 255, 0));
			arrowedLine(frame, Point(px1, py1), Point(px2, py2), Scalar(0, 255, 0));
			if (merge_opposite_angle) {
				Point2d point3 = point1 + rotate_around_origin(Point2d(-0.2f, 0.0f), estimated_state.angle);
				int px3, py3;
				euclidean_to_pixel(point3.x, point3.y, px3, py3);
				arrowedLine(frame, Point(px1, py1), Point(px3, py3), Scalar(0, 255, 0));
			}
		}
	}
	int n_particles;
	int n_particles_init;
	int n_particles_min;
	double n_particles_decay_rate;
	vector<Particle> particles;
};