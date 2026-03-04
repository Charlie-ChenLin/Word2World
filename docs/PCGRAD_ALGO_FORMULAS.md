# PCGrad Env Feedback 算法公式说明（数学定义版）

本文档只做一件事：给出我们当前讨论并实现的算法的**完整数学公式**。  
不讲开发过程，不讲提交历史。

---

## 0. 重点速览：PCGrad 冲突消解 + 有界融合

这部分是本算法的核心。设

$$
d=\langle g_{\text{pg}},g_{\text{env}}\rangle,\quad
n_{\text{pg}}^2=\|g_{\text{pg}}\|^2,\quad
n_{\text{env}}^2=\|g_{\text{env}}\|^2.
$$

### 0.1 冲突消解（Conflict Fix）

$$
g_{\text{pg}}^{\text{pc}}=g_{\text{pg}}-\min(0,d)\frac{g_{\text{env}}}{n_{\text{env}}^2+\epsilon}.
$$

等价写法：
$$
d^-=\max(0,-d),\quad
g_{\text{pg}}^{\text{pc}}=g_{\text{pg}}+\frac{d^-}{n_{\text{env}}^2+\epsilon}g_{\text{env}}.
$$

### 0.2 有界正向融合（Bounded Aligned Fusion）

先取 env 在 PG 方向上的正分量：
$$g_{\mathrm{env}}^{+}=\frac{\max(0,d)}{n_{\mathrm{pg}}^{2}+\epsilon}\cdot g_{\mathrm{pg}}.
$$

再定义有界门控系数：
$$
\lambda=
\min\!\left(
\lambda_{\max},
\lambda_{\text{norm}}\frac{\|g_{\mathrm{pg}}\|}{\|g_{\mathrm{env}}^{+}\|+\epsilon}
\right).
$$

若 $\max(0,d)=0$ 或范数退化，则 $\lambda=0$。

### 0.3 最终梯度

$$
g_{\text{final}}=g_{\text{pg}}^{\text{pc}}+g_{\text{other}}+\lambda g_{\text{env}}^+.
$$

### 0.4 两个关键性质

1. 去冲突：当 $d<0$ 时，$\langle g_{\text{pg}}^{\text{pc}},g_{\text{env}}\rangle\approx 0$（数值误差内）。  
2. 注入有上界：注入比例满足
$$
\frac{\|\lambda g_{\text{env}}^+\|}{\|g_{\text{pg}}\|+\epsilon}
\lesssim
\lambda_{\text{norm}}.
$$

---

## 1. 记号与掩码定义

对每个样本，设 response 长度为 $T$，token 索引 $t=1,\dots,T$。

- $\log \pi_\theta(a_t|s_t)$：当前策略在第 $t$ 个 response token 的对数概率
- $\log \pi_{\text{old}}(a_t|s_t)$：旧策略对数概率
- $A_t$：优势（advantage）
- $r_t(\theta)=\exp(\log \pi_\theta-\log \pi_{\text{old}})$

掩码：

- $m_t^{\text{resp}}\in\{0,1\}$：response 有效 token 掩码（`response_mask`）
- $m_t^{\text{env}}\in\{0,1\}$：env feedback token 掩码（`env_feedback_mask`）

记归一化 masked mean：

$$
\operatorname{Mean}_{m}(x)=
\frac{\sum_{t=1}^{T} m_t x_t}{\sum_{t=1}^{T} m_t + \varepsilon_m}.
$$

---

## 2. Loss 公式（token 层）

### 2.1 PPO policy 项（response mask 上）

$$L_{\text{pg}}=
\operatorname{Mean}_{m^{\text{resp}}}
\left[
\max\left(
-A_t r_t(\theta),\;
-A_t \cdot \operatorname{clip}(r_t(\theta),1-\epsilon_{\text{clip}},1+\epsilon_{\text{clip}})
\right)
\right].
$$

这对应代码中的 `compute_policy_loss(..., eos_mask=response_mask, cliprange=clip_ratio)`。

### 2.2 Entropy 项（response mask 上）

设 token entropy 为 $H_t(\theta)$，则

$$
L_{\text{ent}} = \operatorname{Mean}_{m^{\text{resp}}}[H_t(\theta)].
$$

在总 loss 中以负号进入（鼓励探索）：

$$
L_{\text{other,ent}} = -\beta_{\text{ent}} \, L_{\text{ent}}.
$$

### 2.3 KL 项（response mask 上）

令
$$
D_t^{\text{KL}} = \text{KL-penalty}\big(\log\pi_\theta,\log\pi_{\text{ref}}\big)_t,
$$
则
$$
L_{\text{kl}} = \operatorname{Mean}_{m^{\text{resp}}}[D_t^{\text{KL}}],
\quad
L_{\text{other,kl}} = c_{\text{kl}} L_{\text{kl}}.
$$

### 2.4 Env feedback 项（env mask 上）

$$L_{\text{env,raw}}=-\operatorname{Mean}_{m^{\text{env}}}\left[\log \pi_\theta(a_t|s_t)\right],\quad L_{\text{env}}=
c_{\text{env}} L_{\text{env,raw}}.
$$

---

## 3. 梯度分解

每个 micro-batch 引入 scale $s$（梯度累积缩放）：

$$
\tilde L = s \cdot L.
$$

定义三路梯度（都在参数空间中）：

$$
g_{\text{pg}}=\nabla_\theta(sL_{\text{pg}}),\quad
g_{\text{other}}=\nabla_\theta\!\left(s(L_{\text{other,ent}}+L_{\text{other,kl}})\right),\quad
g_{\text{env}}=\nabla_\theta(sL_{\text{env}}).
$$

若不做任何投影/交互，则标准梯度是：

$$
g_{\text{plain}} = g_{\text{pg}} + g_{\text{other}} + g_{\text{env}}.
$$

---

## 4. 旧方案：`alpha_scale`（当前仍保留）

设 target 梯度为 $g_{\text{tar}}$：

- `target=pg` 时 $g_{\text{tar}}=g_{\text{pg}}$
- `target=pg_plus_other` 时 $g_{\text{tar}}=g_{\text{pg}}+g_{\text{other}}$

定义
$$
\alpha_{\text{raw}}=\frac{\langle g_{\text{tar}},g_{\text{env}}\rangle}{\|g_{\text{tar}}\|^2+\epsilon},
\quad
\alpha=\max(0,\alpha_{\text{raw}})\;\;(\text{alphaclamp})
$$

最终梯度：
$$
g_{\text{final}}^{\alpha}=
(1+\alpha)g_{\text{tar}}
+\mathbf{1}_{\text{target=pg}}\,g_{\text{other}}.
$$

解释：

- `target=pg`：$(1+\alpha)g_{\text{pg}}+g_{\text{other}}$
- `target=pg_plus_other`：$(1+\alpha)(g_{\text{pg}}+g_{\text{other}})$

---

## 5. 新方案：`pcgrad`（PG-anchored + bounded fusion）

当前实现固定锚点 `pg`，即只对 $g_{\text{pg}}$ 与 $g_{\text{env}}$ 做交互，$g_{\text{other}}$ 直接加回。

定义：

$$
d=\langle g_{\text{pg}},g_{\text{env}}\rangle,\quad
n_{\text{pg}}^2=\|g_{\text{pg}}\|^2,\quad
n_{\text{env}}^2=\|g_{\text{env}}\|^2.
$$

### 5.1 冲突消解（Conflict Fix）

$$
d^-=\max(0,-d),\quad
\gamma_{\text{conf}}=\frac{d^-}{n_{\text{env}}^2+\epsilon}.
$$

$$
g_{\text{pg}}^{\text{pc}}=g_{\text{pg}}+\gamma_{\text{conf}}g_{\text{env}}.
$$

当 $d<0$ 时，上式等价于经典 PCGrad 的“去掉冲突分量”写法。

### 5.2 正向有界融合（Aligned Injection）

先取 env 在 PG 方向上的正分量：

$$
d^{+}=\max(0,d),\quad
\gamma_{\text{pos}}=\frac{d^{+}}{n_{\text{pg}}^2+\epsilon},
\quad
g_{\mathrm{env}}^{+}=\gamma_{\text{pos}}\cdot g_{\mathrm{pg}}.
$$

注入系数 $\lambda$ 采用双上界：

$$
\lambda_{\text{norm}}=\rho\frac{\|g_{\mathrm{pg}}\|}{\|g_{\mathrm{env}}^{+}\|+\epsilon},\quad\lambda=
\min(\lambda_{\max},\lambda_{\text{norm}}).
$$

且当 $d^+=0$ 或范数退化时，强制 $\lambda=0$。

### 5.3 最终梯度

$$
g_{\text{final}}^{\text{pcgrad}}=g_{\text{pg}}^{\text{pc}}+g_{\text{other}}+ \lambda g_{\mathrm{env}}^{+}.
$$

写成显式组合：

$$
g_{\text{final}}^{\text{pcgrad}}=\left(1+\lambda\frac{d^+}{n_{\text{pg}}^2+\epsilon}\right)g_{\text{pg}}+g_{\text{other}}
+\frac{d^-}{n_{\text{env}}^2+\epsilon}g_{\mathrm{env}}.
$$

这就是代码里 `pg_scale + other + conflict_scale*env` 的数学形式。

---

## 6. 与代码指标一一对应的公式

### 6.1 基础几何量

$$
\texttt{dot\_pg\_env\_before}=d,\quad
\texttt{cos\_pg\_env\_before}=
\frac{d}{\|g_{\text{pg}}\|\|g_{\text{env}}\|+\epsilon}.
$$

### 6.2 冲突修复后 dot

$$
\texttt{dot\_pg\_env\_after\_conflict\_fix}=\langle g_{\text{pg}}^{\text{pc}},g_{\text{env}}\rangle=d+\gamma_{\text{conf}}n_{\text{env}}^2.
$$

### 6.3 注入比率

注入项为 $\lambda g_{\mathrm{env}}^{+}$，故
$$
\texttt{env\_inject\_ratio}=\frac{\|\lambda g_{\mathrm{env}}^{+}\|}{\|g_{\mathrm{pg}}\|+\epsilon}.
$$

### 6.4 最终梯度与 PG 的夹角

$$
\texttt{final\_vs\_pg\_cosine}=\frac{\langle g_{\text{final}}^{\text{pcgrad}},g_{\text{pg}}\rangle}
{\|g_{\text{final}}^{\text{pcgrad}}\|\|g_{\text{pg}}\|+\epsilon}.
$$

---

## 7. 参数语义（数学）

- `lambda_max`：$\lambda$ 的绝对上界
- `lambda_norm_ratio`：$\rho$，控制注入项相对 PG 范数的比例上界
- `env_feedback_pcgrad_eps`：PCGrad 数值稳定项 $\epsilon$
- `env_feedback_grad_proj_eps`：投影相关稳定项，实际实现里取两者较大者参与 PCGrad 计算

---

## 8. 关键性质（从公式直接可见）

1. 冲突时（$d<0$）：

$$
\gamma_{\text{conf}}>0,\quad
\langle g_{\text{pg}}^{\text{pc}},g_{\text{env}}\rangle=
d+\frac{-d}{n_{\text{env}}^2+\epsilon}n_{\text{env}}^2
\approx 0.
$$

即 env 的负向分量不会继续拉偏 PG。

2. 对齐时（$d>0$）：

$$
\gamma_{\text{conf}}=0,\quad
g_{\mathrm{env}}^{+}\parallel g_{\mathrm{pg}},
$$

只沿 PG 方向做有界增强，不引入额外偏转方向。

3. `other` 始终加性保留：$g_{\text{other}}$ 不参与冲突投影，仅进入最终和。

---

## 9. 当前版本的明确边界

1. `pcgrad` 首版仅支持锚点 `pg`（不支持 `pg_plus_other`）。
2. 所有公式均针对 micro-batch 梯度；step 级行为是 micro-batch 聚合后再 optimizer step。
3. 训练中若触发 OOM fallback，会跳过受影响 micro-batch 的投影路径，该 batch 不满足上述理想公式。

---

## 10. 一页总结公式（可直接引用）

$$
\boxed{
\begin{aligned}
g_{\text{pg}}&=\nabla_\theta(sL_{\text{pg}}),\;
g_{\text{other}}=\nabla_\theta\big(s(-\beta_{\text{ent}}L_{\text{ent}}+c_{\text{kl}}L_{\text{kl}})\big),\;
g_{\text{env}}=\nabla_\theta(s c_{\text{env}}L_{\text{env,raw}}),\\
d&=\langle g_{\text{pg}},g_{\text{env}}\rangle,\;
d^-=\max(0,-d),\;
d^+=\max(0,d),\\
g_{\text{pg}}^{\text{pc}}&=g_{\text{pg}}+\frac{d^-}{\|g_{\text{env}}\|^2+\epsilon}g_{\text{env}},\\
g_{\mathrm{env}}^{+}&=\frac{d^+}{\|g_{\mathrm{pg}}\|^2+\epsilon}g_{\mathrm{pg}},\\
\lambda&=\min\!\left(\lambda_{\max},\rho\frac{\|g_{\mathrm{pg}}\|}{\|g_{\mathrm{env}}^{+}\|+\epsilon}\right),\quad
\lambda=0\;\text{if }d^+=0\text{ or degenerate},\\
g_{\text{final}}&=g_{\text{pg}}^{\text{pc}}+g_{\text{other}}+\lambda g_{\mathrm{env}}^{+}.
\end{aligned}
}
$$
