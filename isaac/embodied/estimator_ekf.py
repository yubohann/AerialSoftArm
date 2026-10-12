"""误差状态 EKF 估计器（纯 numpy），面向四旋翼 IMU+GNSS+气压融合，估计 IMU 偏置。

与 estimator.py 的 StateEstimator 完全相同的接口契约（drop-in 替换）：
    update_imu(acc, gyro, dt) / update_gnss(meas, dt) / update_baro(z, dt)
    correct_position(meas, dt_meas, alpha) / predict(acc, gyro, dt) /
    correct_attitude(acc) / out(omega) -> {"pos", "vel", "quat", "omega"}
符号与坐标系与现有栈一致：pos/vel 在世界系（z 向上），quat wxyz 为 body->world，
acc 为 body 系比力（悬停水平时 ≈ [0,0,g]），gyro 为 body 系角速度。

状态（标称 16 维，误差 15 维）
    x   = [p, v, q, b_a, b_g]        p,v ∈ R^3, q 四元数, b_a/b_g ∈ R^3
    δx  = [δp, δv, δθ, δb_a, δb_g]   15 维
    误差约定（world/global 系，q_true = exp(δθ/2) ⊗ q，δθ 固定在世界系）:
        p_true = p + δp,  v_true = v + δv,  b_true = b + δb
    采用世界系误差使不可观测的偏航方向始终为世界 z 轴：无磁强计时偏航
    不可观测 [3]，body 系误差会让不可观测方向随 yaw 旋转而与可观子空间
    混合，在多段平移中持续产生伪偏航修正（离线实测偏航 RMS 6.3° vs 1.6°）；
    世界系约定下重力更新对 δθ_z 的雅可比严格为零。

IMU 传播（1 kHz，dt 默认取自构造参数）
    ω̂ = gyro - b_g
    a_b = acc - b_a
    a_w = R(q) a_b - [0,0,g]
    p ← p + v dt + 0.5 a_w dt²
    v ← v + a_w dt
    q ← normalize(q ⊗ exp(ω̂ dt/2))
    b_a, b_g 保持（随机游走只进 Q）

误差动力学 F（离散一阶，见文献 [1][2][5] 的标准 ESKF 线性化；R=R(q)）
    f_w = R (acc - b_a)                 比力世界系向量
    δp' = δp + δv dt - 0.5 [f_w]x δθ dt² - 0.5 R δb_a dt²
    δv' = δv - [f_w]x δθ dt - R δb_a dt
    δθ' = δθ - R δb_g dt
    δb' = δb
    P ← F P Fᵀ + Q

过程噪声 Q（每步，按离散样本噪声量级建模；σ_a=ACC_SIGMA, σ_g=GYRO_SIGMA）
    Q_vv = (σ_a² + σ_p²) dt² I,  Q_pv = 0.5 Q_vv dt,  Q_pp = 0.25 Q_vv dt²
    Q_θθ = σ_g² dt² I
    Q_ba = σ_ba,rw² dt I,  Q_bg = σ_bg,rw² dt I   （偏置随机游走，保持可自适应）
    σ_p 为未建模加速度（机架振动/软臂反力），默认 0.03 m/s²

量测更新（误差注入后立即清零）
    GNSS 位置 (10 Hz, R=σ_gnss² I, 3 维，软 NIS 门控)
    z = p_meas;  H = [I 0 0 0 0];  Joseph 形式 P ← (I-KH)P(I-KH)ᵀ + K R Kᵀ
    气压高度 (50 Hz, R=σ_baro², 标量)
    z = z_meas;  H = [0 0 e_zᵀ 0 0]（作用于 δp_z）
    重力方向（可选，随 IMU 1 kHz，三重门控，见下）
    z = acc/|acc|（已减 b_a）;  h(q) = R(q)ᵀ [0,0,1]
    H_θ = Rᵀ [e_z]x（world 系误差；对 δθ_z 严格为零，不污染偏航）
    trust = g_mag * g_rate * g_dyn，低于 att_trust_min 直接跳过
    R_att = (σ_att/trust)² I,  σ_att 默认 0.08 rad, NIS 门控
    比力模长 |acc-b_a|-g、角速率、世界系水平加速度低通（tau=0.05 s, 门 0.08）
    分别门控；机动/振动时不把非重力比力当重力方向。

偏航冻结（freeze_yaw=True，默认）
    无磁强计/航向参考时偏航不可观测 [6]：GNSS 位置、气压、重力方向都不含
    航向信息。误差状态更新中把世界 z 姿态修正行置零，并把陀螺偏置修正中
    会改变世界 z 角速率的分量（沿 Rᵀe_z）投影掉，使偏航纯由陀螺积分决定、
    不被运动段噪声诱导的伪修正污染。离线 A/B：清洁数据偏航 RMS 1.36°→0.001°，
    噪声数据 2.15°→1.67°。需要航向参考（磁/视觉）时可关。

鲁棒性
    - GNSS/气压 NIS 超限时按 nis/gate 放大 R（软门控，不丢信息）；
    - 偏置限幅 |b_a|<=0.5 m/s², |b_g|<=0.05 rad/s；
    - P 对称化、对角线限幅 [1e-12, 1e4]，非有限时按默认 P 重建；
    - 速度不额外限幅（估计器不做反 windup，交给控制器）。

参数（构造函数可覆盖，默认对齐 spec.py）
    acc_sigma=ACC_SIGMA=0.08, gyro_sigma=GYRO_SIGMA=0.02,
    gnss_sigma=GNSS_SIGMA=0.05, baro_sigma=BARO_SIGMA=0.25,
    acc_bias_rw=0.002 m/s²/√s, gyro_bias_rw=1e-4 rad/s/√s,
    acc_process=0.03 m/s², use_gravity_update=True, freeze_yaw=True。
    P0: pos 0.5², vel 1.0², att 0.1², b_a=ACC_BIAS², b_g=0.006²。

参考文献（详见 docs/research/01_estimator_ekf.md）
    [1] Abeywardena et al., arXiv:1607.01486（四旋翼 EKF，含 acc/gyro 偏置）
    [2] Markovic et al., arXiv:2109.04908（ES-EKF 多传感器融合）
    [3] Cohen & Forbes, arXiv:2403.00076（IEKF，偏置+风速增广）
    [4] Lopez, arXiv:2303.02777（层级收缩观测器：姿态+陀螺偏置 / 位置+acc 偏置）
    [5] Chang, arXiv:2003.12978（SE(3)-EKF 姿态+陀螺偏置）
    [6] Garcia de Marina et al., arXiv:1609.07436（UKF/TRIAD 姿态，观测性讨论）
"""
import numpy as np

import spec as S

I3 = np.eye(3)


class EkfEstimator:
    """15 维误差状态 EKF，接口与 StateEstimator 一致。"""

    def __init__(self, pos0, dt, acc_sigma=S.ACC_SIGMA, gyro_sigma=S.GYRO_SIGMA,
                 gnss_sigma=S.GNSS_SIGMA, baro_sigma=S.BARO_SIGMA,
                 acc_bias_rw=0.002, gyro_bias_rw=1e-4, acc_process=0.03,
                 use_gravity_update=True, freeze_yaw=True):
        self.dt = float(dt)

        self.pos = np.asarray(pos0, dtype=float).copy()
        self.vel = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        self.acc_sigma = float(acc_sigma)
        self.gyro_sigma = float(gyro_sigma)
        self.gnss_sigma = float(gnss_sigma)
        self.baro_sigma = float(baro_sigma)
        self.acc_bias_rw = float(acc_bias_rw)
        self.gyro_bias_rw = float(gyro_bias_rw)
        self.acc_process = float(acc_process)
        self.use_gravity_update = bool(use_gravity_update)
        self.freeze_yaw = bool(freeze_yaw)

        self.acc_bias_limit = 0.5
        self.gyro_bias_limit = 0.05
        self.gnss_gate = 30.0
        self.baro_gate = 9.0
        self.p_diag_min = 1e-12
        self.p_diag_max = 1e4

        self.p0_pos = 0.25
        self.p0_vel = 1.0
        self.p0_att = 0.01
        self.p0_ba = S.ACC_BIAS ** 2
        self.p0_bg = (0.006 ** 2)
        self.P = self._default_P()

        self.acc_gate_sigma = 0.4
        self.rate_gate = 2.0
        self.dyn_accel_gate = 0.08
        self.dyn_accel_tau = 0.05
        self.att_sigma = 0.08
        self.att_trust_floor = 0.02
        self.att_trust_min = 0.4
        self.att_gate = 30.0
        self.aw_lp = 0.0

        self.last_gnss_t = -1.0
        self.last_baro_t = -1.0
        self.last_nis_gnss = 0.0
        self.last_nis_baro = 0.0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def update_imu(self, acc, gyro, dt=None):
        dt = self.dt if dt is None else float(dt)
        acc = np.asarray(acc, dtype=float)
        gyro = np.asarray(gyro, dtype=float)

        a_b = acc - self.acc_bias
        r_mat = self.quat_to_mat(self.quat)
        a_w = r_mat @ a_b - np.array([0.0, 0.0, S.G])
        self.pos = self.pos + self.vel * dt + 0.5 * a_w * dt * dt
        self.vel = self.vel + a_w * dt

        w = gyro - self.gyro_bias
        self.quat = self._quat_normalize(
            self.quat_mul(self.quat, self.quat_from_rotvec(w * dt)))

        f_w = r_mat @ a_b
        f_mat = np.eye(15)
        f_mat[0:3, 3:6] = I3 * dt
        f_mat[0:3, 6:9] = -0.5 * self.skew(f_w) * dt * dt
        f_mat[0:3, 9:12] = -0.5 * r_mat * dt * dt
        f_mat[3:6, 6:9] = -self.skew(f_w) * dt
        f_mat[3:6, 9:12] = -r_mat * dt
        f_mat[6:9, 12:15] = -r_mat * dt
        self.P = f_mat @ self.P @ f_mat.T + self._q_discrete(dt)
        self.P = self._sanitize_p(self.P)

        al = 1.0 - np.exp(-dt / self.dyn_accel_tau)
        self.aw_lp += al * (float(np.linalg.norm(a_w[:2])) - self.aw_lp)

        if self.use_gravity_update:
            self._gravity_update(acc, gyro)

    def predict(self, acc, gyro, dt=None):
        self.update_imu(acc, gyro, self.dt if dt is None else dt)

    def correct_attitude(self, acc):
        pass

    def correct_position(self, meas, dt_meas, alpha=None):
        self.update_gnss(meas, dt_meas)

    def update_gnss(self, meas, dt=None):
        meas = np.asarray(meas, dtype=float).ravel()
        n = min(meas.size, 3)
        if n < 2:
            return
        h = np.zeros((n, 15))
        h[:, 0:n] = np.eye(n)
        r_cov = (self.gnss_sigma ** 2) * np.eye(n)
        self._kf_update(meas[:n], self.pos[:n], h, r_cov, self.gnss_gate)
        self.last_gnss_t = 0.0 if dt is None else float(dt)

    def update_baro(self, z, dt=None):
        z = float(z)
        if not np.isfinite(z):
            return
        h = np.zeros((1, 15))
        h[0, 2] = 1.0
        self._kf_update(np.array([z]), np.array([self.pos[2]]), h,
                        np.array([[self.baro_sigma ** 2]]), self.baro_gate)
        self.last_baro_t = 0.0 if dt is None else float(dt)

    def out(self, omega):
        return {"pos": self.pos.copy(), "vel": self.vel.copy(),
                "quat": self.quat.copy(),
                "omega": np.asarray(omega, dtype=float).copy()}

    # ------------------------------------------------------------------
    # 重力方向更新（门控）
    # ------------------------------------------------------------------
    def att_trust(self, acc, gyro):
        a_b = np.asarray(acc, dtype=float) - self.acc_bias
        a_norm = float(np.linalg.norm(a_b)) + 1e-9
        g_mag = float(np.exp(-0.5 * ((a_norm - S.G) / self.acc_gate_sigma) ** 2))
        w_norm = float(np.linalg.norm(np.asarray(gyro, dtype=float) - self.gyro_bias))
        g_rate = 1.0 / (1.0 + (w_norm / self.rate_gate) ** 2)
        g_dyn = float(np.exp(-0.5 * (self.aw_lp / self.dyn_accel_gate) ** 2))
        return float(np.clip(g_mag * g_rate * g_dyn, self.att_trust_floor, 1.0))

    def _gravity_update(self, acc, gyro):
        a_b = np.asarray(acc, dtype=float) - self.acc_bias
        a_norm = float(np.linalg.norm(a_b)) + 1e-9
        trust = self.att_trust(acc, gyro)
        z = a_b / a_norm
        u = self.quat_rotate(self.quat_conj(self.quat),
                             np.array([0.0, 0.0, 1.0]))
        h = np.zeros((3, 15))
        h[:, 6:9] = self.quat_to_mat(self.quat).T @ self.skew(
            np.array([0.0, 0.0, 1.0]))
        if trust < self.att_trust_min:
            return
        r_cov = ((self.att_sigma / trust) ** 2) * I3
        r = z - u
        s_mat = h @ self.P @ h.T + r_cov
        nis = float(r @ np.linalg.solve(s_mat, r))
        if nis > self.att_gate:
            return
        self._kf_update(z, u, h, r_cov, None)

    # ------------------------------------------------------------------
    # 通用卡尔曼更新
    # ------------------------------------------------------------------
    def _kf_update(self, z, h_pred, h_mat, r_cov, gate):
        z = np.asarray(z, dtype=float).ravel()
        h_pred = np.asarray(h_pred, dtype=float).ravel()
        r = z - h_pred
        s_mat = h_mat @ self.P @ h_mat.T + r_cov
        if gate is not None:
            nis = float(r @ np.linalg.solve(s_mat, r))
            if nis > gate:
                r_cov = r_cov * (nis / gate)
                s_mat = h_mat @ self.P @ h_mat.T + r_cov
                if gate == self.gnss_gate:
                    self.last_nis_gnss = nis
                elif gate == self.baro_gate:
                    self.last_nis_baro = nis
        k = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        if self.freeze_yaw:
            k[8, :] = 0.0
            axis = self.quat_to_mat(self.quat).T @ np.array([0.0, 0.0, 1.0])
            k[12:15, :] -= np.outer(axis, axis @ k[12:15, :])
        dx = k @ r
        self._inject(dx)
        i_kh = np.eye(15) - k @ h_mat
        self.P = i_kh @ self.P @ i_kh.T + k @ r_cov @ k.T
        self.P = self._sanitize_p(self.P)

    def _inject(self, dx):
        self.pos = self.pos + dx[0:3]
        self.vel = self.vel + dx[3:6]
        self.acc_bias = np.clip(self.acc_bias + dx[9:12],
                                -self.acc_bias_limit, self.acc_bias_limit)
        self.gyro_bias = np.clip(self.gyro_bias + dx[12:15],
                                 -self.gyro_bias_limit, self.gyro_bias_limit)
        self.quat = self._quat_normalize(
            self.quat_mul(self.quat_from_rotvec(dx[6:9]), self.quat))

    # ------------------------------------------------------------------
    # 噪声与数值工具
    # ------------------------------------------------------------------
    def _q_discrete(self, dt):
        q = np.zeros((15, 15))
        sig_v = (self.acc_sigma ** 2 + self.acc_process ** 2) * dt * dt
        q[0:3, 0:3] = 0.25 * sig_v * dt * dt * I3
        q[0:3, 3:6] = 0.5 * sig_v * dt * I3
        q[3:6, 0:3] = 0.5 * sig_v * dt * I3
        q[3:6, 3:6] = sig_v * I3
        q[6:9, 6:9] = (self.gyro_sigma ** 2) * dt * dt * I3
        q[9:12, 9:12] = (self.acc_bias_rw ** 2) * dt * I3
        q[12:15, 12:15] = (self.gyro_bias_rw ** 2) * dt * I3
        return q

    def _default_P(self):
        return np.diag(np.concatenate([
            np.full(3, self.p0_pos), np.full(3, self.p0_vel),
            np.full(3, self.p0_att), np.full(3, self.p0_ba),
            np.full(3, self.p0_bg)]))

    def _sanitize_p(self, p):
        if not np.isfinite(p).all():
            return self._default_P()
        p = 0.5 * (p + p.T)
        d = np.diag(p).copy()
        d = np.clip(d, self.p_diag_min, self.p_diag_max)
        p[np.diag_indices(15)] = d
        return p

    @staticmethod
    def skew(v):
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    @staticmethod
    def _quat_normalize(q):
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

    @staticmethod
    def quat_mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    @staticmethod
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_rotate(q, v):
        qv = np.array([0.0, v[0], v[1], v[2]])
        r = EkfEstimator.quat_mul(
            EkfEstimator.quat_mul(q, qv), EkfEstimator.quat_conj(q))
        return r[1:]

    @staticmethod
    def quat_from_rotvec(rv):
        ang = float(np.linalg.norm(rv))
        if ang < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / ang
        s = np.sin(ang / 2.0)
        return np.array([np.cos(ang / 2.0), axis[0] * s,
                         axis[1] * s, axis[2] * s])

    @staticmethod
    def quat_to_mat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
