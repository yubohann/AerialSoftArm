"""融合状态估计器（纯 numpy），面向四旋翼：姿态 Mahony + 高度互补 + 水平 alpha-beta/Kalman。

与 estimator.py 的 StateEstimator 相同的输出契约与 duck-type 方法：
    out(omega) -> {"pos", "vel", "quat", "omega"}
    update_imu(acc, gyro, dt) / correct_position(meas, dt_meas) / predict(...)
因此任何按该接口调用估计器的闭环（如 fused_tests.py 的 runner）可无缝换用
本类；plant.run_controller 本身硬编码了 StateEstimator，需要一行改动才能注入。

坐标系与符号
    pos, vel : world (z 向上)
    quat     : wxyz, body -> world
    acc      : body 系比力 (specific force, m/s^2)；悬停水平时 acc ≈ [0, 0, g]
    gyro     : body 系角速度 (rad/s)

============================================================================
1) 姿态：Mahony 互补滤波 + 自适应加速度信任（三重门控）
============================================================================
    u        = R(q)^T e_z                      (估计重力方向在 body 系)
    e_raw    = f_hat x u,  f_hat = acc/|acc|   (方向误差, body 系)
    g_a      = exp(-0.5 * ((|acc| - g)/sigma_a)^2)          <- 比力幅值门控
    g_w      = 1 / (1 + (|gyro - b_g|/w_gate)^2)            <- 角速率门控
    g_d      = exp(-0.5 * (a_h_lp / a_gate)^2)              <- 动态加速度门控
    trust    = clip(g_a * g_w * g_d, trust_floor, 1)
    e        = trust * e_raw
    b_g_dot  = -k_i * e                      (陀螺零偏, 限幅 |b_g| <= b_max)
    omega_c  = gyro - b_g + k_p * e
    q_dot    = 0.5 * q ⊗ [0, omega_c]
    q       <- normalize(q + q_dot * dt)

    a_h_lp 为世界系水平加速度模长 |(R(q)acc - g)_xy| 的一阶低通 (tau=0.5 s)。
    比力模长偏离 g（机动/振动/外力）、机身高速旋转（离心加速度污染加速度
    计）或持续水平加速度（比力方向不再等于重力方向）时 trust 下降，避免把
    非重力比力当成重力修姿态；陀螺零偏积分同时被 trust 缩放，机动时不 windup。
    实测整定 k_p=0.6, k_i=0.15（对比原 estimator.py 的 0.9/0.03，机动场景姿态
    RMSE 明显更低）。

2) 位置/速度预测（IMU 积分）
============================================================================
    a_w = R(q) * (acc - [0,0,b_z]) - [0, 0, g]      (b_z: 垂直加速度计偏置)
    v  <- clamp(v + a_w * dt)        (|v| <= v_max, 反 windup)
    p  <- p + v * dt

3) 高度：IMU + 气压计一阶互补 + GNSS-z 低增益 + 气压辅助的加速度计偏置
============================================================================
    以临界阻尼离散 alpha-beta 实现"时间常数 tau_baro 的一阶互补"：
      h   = 气压计采样间隔,  lam = exp(-h/tau_baro)
      a_b = 1 - lam^2                    (位置增益)
      b_b = (1 - lam)^2 / h              (速度增益, 1/s)
      r   = clip(z_baro - p_z, ±res_clamp)
      p_z <- p_z + a_b * r
      v_z <- v_z + b_b * r               (位置修正同时推进速度，避免双计)

    规格给的时间常数是 ~1.5 s；气压计 sigma=0.25 m 是 GNSS(0.05 m) 的 5 倍，
    取 tau_baro=2.5 s 换取更低的垂直噪声底（0.029 -> 0.021 m），绝对慢漂由
    GNSS-z 与 b_z 偏置估计兜底。稳态对常值垂直加速度偏差 d 的估计偏差
    ≈ d*tau_baro^2，被下面的偏置估计削减。

    GNSS-z 低增益 alpha-beta：
      r = clip(z_gnss - p_z, ±gnss_z_clamp)
      p_z += gnss_z_alpha * r ;  v_z += gnss_z_beta * r

    垂直加速度计偏置慢估计（baro 残差驱动，仅在气压采样更新）：
      b_z <- clip(b_z - k_b * r, ±b_max)
    常值偏差下 b_z -> d，消除 d*tau^2 稳态偏移；20 s 静止 (GNSS 关闭) 漂移
    从 ~0.42 m 降到 ~0.24 m，留有 0.5 m 阈值裕量。

4) 水平：2 状态 (p, v) alpha-beta，增益由稳态 Kalman/Riccati 推导 + 自适应
============================================================================
    每轴状态 x = [p, v]^T，GNSS 位置量测方差 r = sigma_gnss^2，过程噪声采用
    白加速度模型 Q = q * [[h^3/3, h^2/2], [h^2/2, h]]，q 由"IMU 速度随机游走
    噪声"参数化（默认 0.0707 m/s per sqrt(s)，q_lo = 0.005 m^2/s^3；实测这是
    噪声底与偏置抑制的折中）。迭代 Riccati 至稳态得到 K = [alpha, beta]^T，
    一步更新：
      rho  = clip(z_gnss - p, ±res_clamp)
      p   <- p + alpha * rho
      v   <- v + beta  * rho             (修正速度, 杜绝旧实现只用含噪差分测速
                                          且位置/速度不一致的双计错误)

    对常值水平加速度偏差 d，稳态位置偏差 ≈ d*h*(1-alpha)/beta，q=q_lo 时约
    1 cm (d=0.05)，既能跟随机动，也不被加速度计零偏拖走。

    自适应增益调度（机动时信任 GNSS，平稳时压低噪声）：
      e_lp  <- (1-gamma) e_lp + gamma * ||z_gnss_xy - p_xy||   (gamma=0.1)
      s      = min((e_lp / e_ref)^shape, 1)                    (e_ref=0.15, shape=8)
      alpha  = alpha_lo + pos_boost * s * (alpha_hi - alpha_lo) (pos_boost=0.5)
      beta   = beta_lo  + s * (beta_hi - beta_lo)
    q_hi=0.3 对应的大增益只在持续残差（软臂伸出扰动、强机动）时接管；残差
    主要来自 GNSS 白噪声的平稳悬停仍用低增益，避免把量测噪声灌进状态。

5) 抗积分饱和与限幅
============================================================================
    - 陀螺零偏限幅 + 自适应 trust 缩放；
    - 垂直加速度计偏置慢估计限幅 ±b_max；
    - GNSS/气压残差限幅（异常值不灌进状态）；
    - 速度范数限幅 v_max=10 m/s；
    - 气压互补增益对 dt 限幅，数值稳定。
"""

import numpy as np

import spec as S


class FusionEstimator:
    """Mahony(三重自适应门控) + 高度互补/偏置估计 + 水平自适应 2 态 Kalman。"""

    def __init__(self, pos0, dt, gnss_sigma=S.GNSS_SIGMA, baro_sigma=S.BARO_SIGMA,
                 gnss_rate=S.GNSS_RATE, imu_vel_noise=0.0707):
        # ---------------------------------------------------------- state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.pos = np.array(pos0, dtype=float).copy()
        self.vel = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.dt = float(dt)

        # ------------------------------------------------------- attitude
        self.kp_att = 0.6          # Mahony 比例增益 (rad/s per rad err)
        self.ki_att = 0.15         # 陀螺零偏积分增益 (1/s per rad err)
        self.gyro_bias_limit = 0.05
        self.acc_gate_sigma = 0.18 * S.G   # |acc|-g 高斯门控尺度
        self.rate_gate = 2.0               # rad/s, 角速率门控尺度
        self.att_trust_floor = 0.05        # 门控下限（保留慢收敛能力）
        self.dyn_accel_gate = 1.5          # m/s^2, 动态水平加速度门控尺度
        self.dyn_accel_tau = 0.5           # s, a_h 低通时间常数
        self.aw_lp = 0.0                   # |a_world_xy| 低通值

        # ------------------------------------------------- vertical/baro
        self.baro_tau = 2.5        # 气压互补时间常数 (s)，规格 ~1.5，见 docstring
        self.baro_res_clamp = 2.0  # m
        self.baro_accel_trim = 0.0002   # b_z 慢估计增益 (每 baro 采样)
        self.accel_bias_limit = 0.2     # m/s^2, b_z 限幅
        self.accel_bias_z = 0.0         # 垂直加速度计偏置估计
        self.gnss_z_alpha = 0.15   # GNSS-z 低增益位置修正
        self.gnss_z_beta = 0.15 / (1.0 / gnss_rate) * 0.30   # v_z 修正
        self.gnss_z_res_clamp = 1.5

        # ----------------------------------------------------- horizontal
        self.gnss_sigma = float(gnss_sigma)
        self.baro_sigma = float(baro_sigma)
        self.gnss_rate = float(gnss_rate)
        self.imu_vel_noise = float(imu_vel_noise)
        self.gnss_res_clamp = 1.5  # m, 残差限幅（异常值保护）
        self.vel_limit = 10.0      # m/s, 反 windup
        # 稳态 Kalman 增益（每轴相同），暴露为类参数便于整定/测试
        self.gnss_h = 1.0 / self.gnss_rate
        self.gnss_q_lo = self.imu_vel_noise ** 2 / 1.0   # m^2/s^3
        self.gnss_q_hi = 0.3                             # 机动时过程噪声上界
        self.gnss_alpha_lo, self.gnss_beta_lo = self._steady_state_gains(
            self.gnss_q_lo, self.gnss_h, self.gnss_sigma ** 2)
        self.gnss_alpha_hi, self.gnss_beta_hi = self._steady_state_gains(
            self.gnss_q_hi, self.gnss_h, self.gnss_sigma ** 2)
        # 自适应调度参数
        self.gnss_innov_ref = 0.15     # m, 残差参考尺度
        self.gnss_innov_shape = 8.0    # 调度锐度
        self.gnss_innov_smooth = 0.10  # e_lp 一阶平滑 (每 GNSS 采样)
        self.gnss_pos_boost = 0.5      # 位置增益只释放一半，速度增益全释放
        self.e_lp = 0.0
        # 当前生效增益（初始为平稳档）
        self.gnss_alpha = self.gnss_alpha_lo
        self.gnss_beta = self.gnss_beta_lo

    # =================================================================
    # Riccati: 标量双积分器的稳态 Kalman 增益
    # =================================================================
    @staticmethod
    def _steady_state_gains(q_psd, h, r, n_iter=2000):
        """迭代离散 Riccati 方程求 x=[p,v] 双积分器的稳态增益 K=[alpha,beta]。

            P <- F P F^T + Q
            K  = P H^T / (H P H^T + r)
            P <- (I - K H) P
        F = [[1,h],[0,1]], H = [1,0], Q = q*[[h^3/3,h^2/2],[h^2/2,h]]。
        返回 alpha (位置增益), beta (速度增益, v += beta*r)。
        """
        f = np.array([[1.0, h], [0.0, 1.0]])
        q = q_psd * np.array([[h ** 3 / 3.0, h ** 2 / 2.0],
                              [h ** 2 / 2.0, h]])
        h_mat = np.array([[1.0, 0.0]])
        p = np.eye(2) * 1.0
        k = np.zeros(2)
        for _ in range(n_iter):
            p = f @ p @ f.T + q
            s = float((h_mat @ p @ h_mat.T).ravel()[0]) + r
            k = (p @ h_mat.T / s).ravel()
            p = p - np.outer(k, h_mat @ p)
        return float(k[0]), float(k[1])

    # =================================================================
    # 姿态：Mahony adaptive
    # =================================================================
    def att_trust(self, acc, gyro):
        """自适应加速度信任度 in [trust_floor, 1]。

        g_a: | |acc|-g | 高斯衰减；g_w: 角速率（去偏后）平方衰减；
        g_d: 低通水平加速度平方衰减（持续加速度使比力方向≠重力方向）。
        """
        a_norm = float(np.linalg.norm(acc)) + 1e-9
        mag_err = abs(a_norm - S.G)
        g_a = np.exp(-0.5 * (mag_err / self.acc_gate_sigma) ** 2)
        w_norm = float(np.linalg.norm(np.asarray(gyro, dtype=float) - self.gyro_bias))
        g_w = 1.0 / (1.0 + (w_norm / self.rate_gate) ** 2)
        g_d = np.exp(-0.5 * (self.aw_lp / self.dyn_accel_gate) ** 2)
        return float(np.clip(g_a * g_w * g_d, self.att_trust_floor, 1.0))

    def update_imu(self, acc, gyro, dt=None):
        """1 kHz IMU 步：自适应 Mahony 姿态 + 世界系加速度积分。"""
        dt = self.dt if dt is None else float(dt)
        acc = np.asarray(acc, dtype=float)
        gyro = np.asarray(gyro, dtype=float)

        # ---- Mahony + adaptive gates (姿态用原始比力方向) -----------
        a_norm = float(np.linalg.norm(acc)) + 1e-9
        a_hat = acc / a_norm
        u = self.quat_rotate(self.quat_conj(self.quat),
                             np.array([0.0, 0.0, 1.0]))
        trust = self.att_trust(acc, gyro)
        e = np.cross(a_hat, u) * trust
        # 陀螺零偏积分（trust 缩放防 windup）+ 限幅
        self.gyro_bias = np.clip(self.gyro_bias - self.ki_att * e * dt,
                                 -self.gyro_bias_limit, self.gyro_bias_limit)
        w = gyro - self.gyro_bias + self.kp_att * e
        dq = np.array([1.0, w[0] * dt / 2.0, w[1] * dt / 2.0, w[2] * dt / 2.0])
        qn = self.quat_mul(self.quat, dq)
        self.quat = qn / (np.linalg.norm(qn) + 1e-12)

        # ---- world acceleration / dead reckoning (减去垂直偏置) -----
        acc_c = acc.copy()
        acc_c[2] -= self.accel_bias_z
        a_world = self.quat_rotate(self.quat, acc_c) - np.array([0.0, 0.0, S.G])
        self.vel = self.vel + a_world * dt
        self.vel = self._clamp_vel(self.vel)
        self.pos = self.pos + self.vel * dt

        # 动态加速度门控的低通量（用于下一步姿态信任）
        a_h = float(np.linalg.norm(a_world[:2]))
        al = 1.0 - np.exp(-dt / self.dyn_accel_tau)
        self.aw_lp += al * (a_h - self.aw_lp)

    # =================================================================
    # 高度：气压互补 + GNSS-z 低增益 + 垂直加速度计偏置慢估计
    # =================================================================
    def update_baro(self, z, dt):
        """气压高度修正：临界阻尼 alpha-beta，等效一阶互补 tau=baro_tau。"""
        z = float(z)
        if not np.isfinite(z):
            return
        h = float(np.clip(dt, 1e-3, 0.2))
        lam = float(np.exp(-h / self.baro_tau))
        a_b = 1.0 - lam * lam
        b_b = (1.0 - lam) ** 2 / h
        r = float(np.clip(z - self.pos[2], -self.baro_res_clamp,
                          self.baro_res_clamp))
        self.pos[2] += a_b * r
        self.vel[2] += b_b * r
        self.vel = self._clamp_vel(self.vel)
        # 加速度计 z 偏置慢估计：残差为正（baro 高于估计）说明估计偏低
        if self.baro_accel_trim > 0.0:
            self.accel_bias_z = float(np.clip(
                self.accel_bias_z - self.baro_accel_trim * r,
                -self.accel_bias_limit, self.accel_bias_limit))

    # =================================================================
    # 水平：2 态 Kalman(alpha-beta) 自适应增益，2D/3D 输入皆可
    # =================================================================
    def update_gnss(self, meas, dt):
        """GNSS 修正：pos 与 vel 同步推进（禁止只挪 pos 的双计错误）。

        meas: 长度 2 只修 xy；长度 3 时 z 轴走低增益通道（baro 管快漂）。
        dt:   距上次 GNSS 的间隔（用于 z 通道归一，可不用）。
        """
        meas = np.asarray(meas, dtype=float).ravel()
        # ---- 自适应增益调度：残差越大越信任 GNSS --------------------
        e = float(np.linalg.norm(meas[:2] - self.pos[:2]))
        self.e_lp += self.gnss_innov_smooth * (e - self.e_lp)
        s = min((self.e_lp / self.gnss_innov_ref) ** self.gnss_innov_shape, 1.0)
        self.gnss_alpha = self.gnss_alpha_lo + self.gnss_pos_boost * s * (
            self.gnss_alpha_hi - self.gnss_alpha_lo)
        self.gnss_beta = self.gnss_beta_lo + s * (
            self.gnss_beta_hi - self.gnss_beta_lo)
        r_xy = np.clip(meas[:2] - self.pos[:2], -self.gnss_res_clamp,
                       self.gnss_res_clamp)
        self.pos[:2] += self.gnss_alpha * r_xy
        self.vel[:2] += self.gnss_beta * r_xy
        if meas.size >= 3:
            r_z = float(np.clip(meas[2] - self.pos[2], -self.gnss_z_res_clamp,
                                self.gnss_z_res_clamp))
            self.pos[2] += self.gnss_z_alpha * r_z
            self.vel[2] += self.gnss_z_beta * r_z
        self.vel = self._clamp_vel(self.vel)

    # -----------------------------------------------------------------
    # 与 StateEstimator 的 duck-type 兼容（plant.run_controller 直接调用）
    # -----------------------------------------------------------------
    def predict(self, acc, gyro, dt=None):
        self.update_imu(acc, gyro, self.dt if dt is None else dt)

    def correct_attitude(self, acc):
        pass

    def correct_position(self, meas, dt_meas, alpha=None):
        """旧接口兼容：等价于 update_gnss(meas, dt_meas)。

        alpha 参数保留但不使用（本类用自带的稳态 Kalman 增益）。
        """
        self.update_gnss(meas, dt_meas)

    def out(self, omega):
        return {"pos": self.pos.copy(), "vel": self.vel.copy(),
                "quat": self.quat.copy(),
                "omega": np.asarray(omega, dtype=float).copy()}

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    def _clamp_vel(self, v):
        n = float(np.linalg.norm(v))
        if n > self.vel_limit:
            return v * (self.vel_limit / n)
        return v

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
        r = FusionEstimator.quat_mul(
            FusionEstimator.quat_mul(q, qv), FusionEstimator.quat_conj(q))
        return r[1:]

    @staticmethod
    def quat_from_rotvec(rv):
        ang = np.linalg.norm(rv)
        if ang < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rv / ang
        s = np.sin(ang / 2.0)
        return np.array([np.cos(ang / 2.0), axis[0] * s,
                         axis[1] * s, axis[2] * s])
