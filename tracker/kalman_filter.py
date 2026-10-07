import numpy as np
import scipy.linalg


class KalmanFilter:
    """TrackTacular-style constant-velocity Kalman filter for BEV points.

    State layout is [x, y, vx, vy]. Measurements are current model detections
    in [x, y]. Box dimensions are not used.

    Kalman notation used in this file:
        x: state mean [x, y, vx, vy] -> code variable `mean`
        P: state covariance -> code variable `covariance`
        F: state transition matrix -> `self._motion_mat`
        H: measurement matrix -> `self._update_mat`
        z: detection measurement [x, y] -> code variable `measurement`
        Q: process/motion noise covariance -> code variable `motion_cov`
        R: measurement noise covariance
        S: projected/innovation covariance HPH^T + R -> `projected_cov`
        K: Kalman gain -> `kalman_gain`
    """

    def __init__(self):
        ndim, dt = 2, 1.0

        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    @staticmethod
    def _xy_measurement(measurement):
        measurement = np.asarray(measurement, dtype=np.float32).reshape(-1)
        if measurement.size < 2:
            raise ValueError("measurement must contain at least x and y")
        return measurement[:2]

    @staticmethod
    def _xy_measurements(measurements):
        measurements = np.asarray(measurements, dtype=np.float32)
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)
        if measurements.shape[1] < 2:
            raise ValueError("measurements must contain at least x and y")
        return measurements[:, :2]

    @staticmethod
    def _symmetrize(covariance):
        return (0.5 * (covariance + covariance.T)).astype(np.float32)

    @staticmethod
    def _symmetrize_state_covariance(covariance):
        return (0.5 * (covariance + covariance.T)).astype(np.float32)

    def _default_measurement_covariance(self):
        # Default R used when the detector does not provide measurement noise.
        std = [self._std_weight_position, self._std_weight_position]
        return np.diag(np.square(std)).astype(np.float32)

    def initiate(self, measurement):
        # z initializes the position part of x; velocity starts unknown at zero.
        mean_pos = self._xy_measurement(measurement)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        position_std = [2 * self._std_weight_position, 2 * self._std_weight_position]
        position_covariance = np.diag(np.square(position_std)).astype(np.float32)
        velocity_std = [10 * self._std_weight_velocity, 10 * self._std_weight_velocity]
        covariance = np.zeros((4, 4), dtype=np.float32)
        covariance[:2, :2] = position_covariance
        covariance[2:, 2:] = np.diag(np.square(velocity_std)).astype(np.float32)
        return mean.astype(np.float32), covariance

    def predict(self, mean, covariance):
        # Q is the process/motion noise. It says how much the constant-velocity
        # model can drift between frames.
        std_pos = [self._std_weight_position, self._std_weight_position]
        std_vel = [self._std_weight_velocity, self._std_weight_velocity]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        # Kalman predict:
        #   x' = F x
        #   P' = F P F^T + Q
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        covariance = self._symmetrize_state_covariance(covariance)
        return mean.astype(np.float32), covariance

    def multi_predict(self, mean, covariance):
        if len(mean) == 0:
            return mean, covariance

        ones = np.ones_like(mean[:, 0])
        std_pos = [self._std_weight_position * ones, self._std_weight_position * ones]
        std_vel = [self._std_weight_velocity * ones, self._std_weight_velocity * ones]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        # Batched Q, one process-noise covariance per track.
        motion_cov = np.asarray([np.diag(sqr_i) for sqr_i in sqr], dtype=np.float32)

        # Batched Kalman predict:
        #   x' = F x
        #   P' = F P F^T + Q
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        covariance = 0.5 * (covariance + covariance.transpose((0, 2, 1)))
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean, covariance):
        innovation_cov = self._default_measurement_covariance()
        # Project state into measurement space:
        #   z_hat = H x
        #   S = H P H^T + R
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        projected_covariance = self._symmetrize(covariance + innovation_cov)
        return mean.astype(np.float32), projected_covariance

    def update(self, mean, covariance, measurement):
        # z is the current detection. projected_mean is z_hat, and
        # projected_cov is S = H P H^T + R.
        measurement = self._xy_measurement(measurement)
        innovation_cov = self._default_measurement_covariance()
        projected_mean, projected_cov = self.project(mean, covariance)

        # K = P H^T S^-1
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        # y = z - z_hat, the innovation/residual.
        innovation = measurement - projected_mean

        # Kalman update:
        #   x_new = x + K y
        #   P_new uses Joseph form for better numerical symmetry/PSD behavior.
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        identity = np.eye(covariance.shape[0], dtype=np.float32)
        update_factor = identity - np.dot(kalman_gain, self._update_mat)
        new_covariance = (
            np.linalg.multi_dot((update_factor, covariance, update_factor.T))
            + np.linalg.multi_dot((kalman_gain, innovation_cov, kalman_gain.T))
        )
        new_covariance = self._symmetrize_state_covariance(new_covariance)
        return new_mean.astype(np.float32), new_covariance
