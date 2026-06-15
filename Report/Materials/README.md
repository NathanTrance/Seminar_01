# Seminar Paper Summaries

This document contains structured summaries of the papers in the `Materials` folder: Abstract, main contributions, methods summary, and pseudocode implementation.

---

## 1. Rectified Flow: Flow Straight and Fast (2209.03003)

### Abstract
We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions $\pi_0$ and $\pi_1$, hence providing a unified solution to generative modeling and domain transfer. The idea of rectified flow is to learn the ODE to follow the straight paths connecting the points drawn from $\pi_0$ and $\pi_1$ as much as possible. This is achieved by solving a straightforward nonlinear least squares optimization problem. The straight paths are special because they are the shortest paths between two points, and can be simulated exactly without time discretization. We show that the procedure of learning a rectified flow from data, called rectification, turns an arbitrary coupling of $\pi_0$ and $\pi_1$ to a new deterministic coupling with provably non-increasing convex transport costs. In addition, recursively applying rectification allows us to obtain a sequence of flows with increasingly straight paths, which can be simulated accurately with coarse time discretization. On image generation and translation, our method yields nearly straight flows that give high quality results even with a single Euler discretization step.

### Main Contributions
- **Straight-path ODEs**: Learn ODEs to follow straight paths between distributions, minimizing transport cost.
- **Rectification**: A procedure that turns an arbitrary coupling into a deterministic coupling with non-increasing convex transport cost.
- **Recursive refinement**: Applying rectification repeatedly yields flows with increasingly straight paths.
- **Unified framework**: Applicable to generative modeling, image-to-image translation, and domain adaptation.
- **Single-step inference**: With sufficiently straight paths, a single Euler step gives high-quality transport.

### Methods Summary
Given an initial coupling of $x_0 \sim \pi_0$ and $x_1 \sim \pi_1$ (e.g., independent coupling), the straight-line path is $x_t = (1-t) x_0 + t x_1$ with conditional velocity $u_t = x_1 - x_0$. The rectified flow is learned by regressing a neural network $v_\theta(x_t, t)$ to match this velocity:
$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2.$$
This gives the **1-rectified flow**. After training, one simulates the ODE from $t=0$ to $t=1$ to obtain new endpoint pairs $(x_0, \hat{x}_1)$, forming a new deterministic coupling $\pi_{01}^{(1)}$. Training on this new coupling yields the **2-rectified flow**, whose paths are straighter. Recursively applying this gives $k$-rectified flows. When the flow becomes perfectly straight (constant velocity), a single Euler step gives exact transport.

### Pseudocode Implementation
```python
# Training a rectified flow (same as OT flow matching)
def train_rectified_flow(x0, x1, v_theta, optimizer):
    optimizer.zero_grad()
    t = torch.rand(x0.shape[0])
    
    # Straight-line path (OT interpolation)
    xt = (1 - t) * x0 + t * x1
    ut = x1 - x0
    
    # Predict velocity
    v_pred = v_theta(xt, t)
    
    loss = F.mse_loss(v_pred, ut)
    loss.backward()
    optimizer.step()
    return loss

# Rectification: simulate ODE to get a new deterministic coupling
def rectify(v_theta, x0, num_steps=100):
    x = x0
    ts = torch.linspace(0, 1, num_steps)
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        v = v_theta(x, ts[i])
        x = x + dt * v
    return x  # new x1

# Recursive rectification
def recursive_rectification(v_theta, x0, x1, k=3):
    for level in range(k):
        # Train on current coupling
        for epoch in range(epochs):
            train_rectified_flow(x0, x1, v_theta, optimizer)
        # Get new coupling by simulation
        with torch.no_grad():
            x1 = rectify(v_theta, x0)
    return v_theta
```

---

## 2. Flow Matching for Generative Modeling (2210.02747)

### Abstract
We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples — which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization.

### Main Contributions
- **Flow Matching (FM) framework**: A simulation-free method for training CNFs by regressing vector fields of conditional probability paths.
- **General probability paths**: Compatible with any Gaussian path, subsuming diffusion and flow matching paths as special cases.
- **Optimal Transport (OT) paths**: Using OT displacement interpolation yields straighter paths, faster training, and better generalization.
- **Scalability**: Trains CNFs at large scale (e.g., ImageNet) with better likelihood and sample quality than diffusion-based methods.

### Methods Summary
Given data $x_1 \sim p_1$ and noise $x_0 \sim p_0$, define a conditional path $x_t = \alpha_t x_0 + \beta_t x_1$ with velocity $u_t = \dot{\alpha}_t x_0 + \dot{\beta}_t x_1$. The marginal velocity field $v(t, x_t)$ is the expectation of $u_t$ over all $(x_0, x_1)$ pairs consistent with $x_t$. Since this marginal is intractable, FM uses the **conditional flow matching** objective, which has the same gradient but is tractable:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - u_t \|^2.$$
The minimizer of this loss equals the true marginal velocity field. For OT paths, $x_t = (1-t)x_0 + t x_1$ and $u_t = x_1 - x_0$.

### Pseudocode Implementation
```python
# Flow Matching Training Loop
def train_step(x1, x0, v_theta, optimizer):
    optimizer.zero_grad()
    t = torch.rand(x1.shape[0])
    
    # Conditional path (e.g., OT interpolation)
    xt = (1 - t) * x0 + t * x1
    
    # Conditional velocity (constant for OT)
    ut = x1 - x0
    
    # Predict velocity
    v_pred = v_theta(xt, t)
    
    # Loss
    loss = F.mse_loss(v_pred, ut)
    loss.backward()
    optimizer.step()
    return loss

# Sampling (ODE integration)
def sample(v_theta, x0, num_steps=50):
    x = x0
    ts = torch.linspace(0, 1, num_steps)
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        v = v_theta(x, ts[i])
        x = x + dt * v
    return x
```

## 3. Consistency Flow Matching: Defining Straight Flows with Velocity Consistency (2407.02398)

### Abstract
Flow matching (FM) is a general framework for defining probability paths via ODEs to transform between noise and data samples. Recent approaches attempt to straighten these flow trajectories to generate high-quality samples with fewer function evaluations. We introduce Consistency Flow Matching (Consistency-FM), a novel FM method that explicitly enforces self-consistency in the velocity field. Consistency-FM directly defines straight flows starting from different times to the same endpoint, imposing constraints on their velocity values. Additionally, we propose a multi-segment training approach for Consistency-FM to enhance expressiveness, achieving a better trade-off between sampling quality and speed.

### Main Contributions
- **Velocity consistency**: Enforces self-consistency in the velocity field space (not sample space), enabling faster training than consistency models.
- **Straight flows without OT approximation**: Defines straight flows directly without computing expensive optimal transport plans.
- **Multi-segment training**: Divides the time interval into segments to construct piece-wise linear trajectories for complex distributions.
- **Efficiency**: Converges 4.4x faster than consistency models and 1.7x faster than rectified flow models.

### Methods Summary
Given a flow $\gamma_x(t)$ with velocity $v(t, \gamma_x(t))$, **velocity consistency** requires the velocity to be constant along the trajectory:
$$v(t, \gamma_x(t)) = v(s, \gamma_x(s)) \quad \forall t, s \in [0,1].$$
This is equivalent to the trajectory condition:
$$\gamma_x(t) + (1-t) v(t, \gamma_x(t)) = \gamma_x(s) + (1-s) v(s, \gamma_x(s)).$$

**Training Loss**:
$$\mathcal{L}_\theta = \mathbb{E}_{t, x_t, x_{t+\Delta t}} \left[ \| f_\theta(t, x_t) - f_{\theta^-}(t+\Delta t, x_{t+\Delta t}) \|^2 + \alpha \| v_\theta(t, x_t) - v_{\theta^-}(t+\Delta t, x_{t+\Delta t}) \|^2 \right]$$
where $f_\theta(t, x_t) = x_t + (1-t) v_\theta(t, x_t)$ and $\theta^-$ is an EMA of parameters.

**Multi-segment**: For $K$ segments, the time interval $[0,1]$ is divided into $[i/K, (i+1)/K]$, and a consistent velocity $v_\theta^i$ is learned within each segment.

### Pseudocode Implementation
```python
# Consistency Flow Matching Training
def train_consistency_fm(x0, x1, v_theta, optimizer, ema_model, alpha=0.1):
    optimizer.zero_grad()
    t = torch.rand(x0.shape[0]) * (1 - delta_t)
    
    # Sample along path (e.g., OT)
    eps = torch.randn_like(x0)
    xt = (1 - t) * x0 + t * eps
    xtd = (1 - (t + delta_t)) * x0 + (t + delta_t) * eps
    
    # Predict velocities
    v_t = v_theta(xt, t)
    v_td = v_theta(xtd, t + delta_t)
    
    # Consistency functions f(t, x) = x + (1-t) * v(t, x)
    f_t = xt + (1 - t) * v_t
    f_td = xtd + (1 - (t + delta_t)) * v_td
    
    # EMA targets
    with torch.no_grad():
        v_t_ema = ema_model(xt, t)
        v_td_ema = ema_model(xtd, t + delta_t)
        f_t_ema = xt + (1 - t) * v_t_ema
        f_td_ema = xtd + (1 - (t + delta_t)) * v_td_ema
    
    # Loss: consistency on f and v
    loss_f = F.mse_loss(f_t, f_td_ema) + F.mse_loss(f_td, f_t_ema)
    loss_v = F.mse_loss(v_t, v_td_ema) + F.mse_loss(v_td, v_t_ema)
    loss = loss_f + alpha * loss_v
    
    loss.backward()
    optimizer.step()
    update_ema(ema_model, v_theta)
    return loss

# One-step sampling
def sample_consistency_fm(v_theta, x0):
    t = torch.ones(x0.shape[0])
    v = v_theta(x0, t)
    x1 = x0 + (1 - t) * v  # for t=1, this is x0 + 0, but for general t:
    return x1
```

---

## 4. Mean Flows for One-step Generative Modeling (2505.13447)

### Abstract
We propose a principled and effective framework for one-step generative modeling. We introduce the notion of **average velocity** to characterize flow fields, in contrast to instantaneous velocity modeled by Flow Matching methods. A well-defined identity between average and instantaneous velocities is derived and used to guide neural network training. Our method, termed the MeanFlow model, is self-contained and requires no pre-training, distillation, or curriculum learning. MeanFlow demonstrates strong empirical performance: it achieves an FID of 3.43 with a single function evaluation (1-NFE) on ImageNet 256x256 trained from scratch.

### Main Contributions
- **Average velocity field**: Defines $u(z_t, r, t) = \frac{1}{t-r} \int_r^t v(z_\tau, \tau) d\tau$ as the ground-truth target for one-step generation.
- **MeanFlow Identity**: Derives $u = v - (t-r) \frac{d}{dt} u$, which provides a tractable training target without computing integrals.
- **Self-contained training**: Trains from scratch without distillation or curriculum.
- **State-of-the-art 1-NFE results**: FID 3.43 on ImageNet 256x256.

### Methods Summary
The average velocity $u(z_t, r, t)$ is defined as the displacement over a time interval divided by the duration. Differentiating the definition $(t-r)u = \int_r^t v \, d\tau$ with respect to $t$ yields the **MeanFlow Identity**:
$$u(z_t, r, t) = v(z_t, t) - (t-r) \frac{d}{dt} u(z_t, r, t).$$
The total derivative is expanded as:
$$\frac{d}{dt} u = v \cdot \partial_z u + \partial_t u,$$
which is computed efficiently as a Jacobian-Vector Product (JVP).

**Training Objective**:
$$\mathcal{L}(\theta) = \mathbb{E} \| u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \|^2$$
where $u_{\text{tgt}} = v(z_t, t) - (t-r) (v(z_t, t) \cdot \partial_z u_\theta + \partial_t u_\theta)$, and `sg` denotes stop-gradient.

### Pseudocode Implementation
```python
# MeanFlow Training
def train_meanflow(x, eps, u_theta, optimizer):
    optimizer.zero_grad()
    r, t = torch.rand(2)
    
    # Interpolant
    zt = (1 - t) * x + t * eps
    
    # Conditional instantaneous velocity
    v = eps - x
    
    # Predict average velocity
    u_pred = u_theta(zt, r, t)
    
    # Compute target via MeanFlow Identity using JVP
    # du/dt = v * dz/dt + du/dt = v * d_u + d_t_u
    def u_func(z, tau):
        return u_theta(z, r, tau)
    
    # JVP: (v, 1) dot (grad_z, grad_t)
    _, d_u = torch.autograd.functional.jvp(
        u_func, (zt, t), (v, torch.ones_like(t)), create_graph=True
    )
    
    u_target = v - (t - r) * d_u
    
    loss = F.mse_loss(u_pred, u_target.detach())
    loss.backward()
    optimizer.step()
    return loss

# One-step sampling
def sample_meanflow(u_theta, eps):
    u = u_theta(eps, 0.0, 1.0)
    return eps + (1.0 - 0.0) * u
```

---

## 5. Align Your Flow: Scaling Continuous-Time Flow Map Distillation (2506.14603)

### Abstract
Diffusion- and flow-based models require many sampling steps. Consistency models can distill these into efficient one-step generators; however, their performance inevitably degrades when increasing the number of steps, which we show both analytically and empirically. Flow maps generalize these approaches by connecting any two noise levels in a single step and remain effective across all step counts. We introduce two new continuous-time objectives for training flow maps, along with additional training techniques. We demonstrate that autoguidance can improve performance, and adversarial finetuning further boosts quality with minimal loss in diversity. We achieve state-of-the-art few-step generation performance on ImageNet 64x64 and 512x512.

### Main Contributions
- **Analytical flaw of CMs**: Prove that consistency models inherently suffer from error accumulation in multi-step sampling.
- **Two new flow map objectives**: AYF-Eulerian Map Distillation (AYF-EMD) and AYF-Lagrangian Map Distillation (AYF-LMD).
- **Autoguidance for distillation**: Use a low-quality guidance model during distillation to improve sample quality.
- **Adversarial finetuning**: Short finetuning with adversarial loss boosts sharpness with minimal diversity loss.
- **State-of-the-art results**: Best few-step generation on ImageNet 64x64 and 512x512.

### Methods Summary
**Flow map** $f_\theta(x_t, t, s)$ maps from time $t$ to time $s$ along the teacher's PF-ODE. It satisfies $f_\theta(x_t, t, t) = x_t$.

**AYF-EMD (Eulerian)**: Ensures that for fixed $s$, the flow map output remains constant as $(x_t, t)$ moves along the PF-ODE. The gradient converges to:
$$\nabla_\theta \mathbb{E}_{x_t, t, s} \left[ w(t,s) \| f_\theta(x_t, t, s) - f_{\theta^-}(x_{t'}, t', s) \|^2 \right]$$
where $x_{t'}$ is obtained by one Euler step on the teacher ODE.

**AYF-LMD (Lagrangian)**: Ensures the tangent of the flow map matches the teacher velocity. The gradient converges to:
$$\nabla_\theta \mathbb{E}_{x_t, t, s} \left[ w(t,s) \| \partial_t f_\theta(x_t, t, s) - v_{\theta^-}(f_\theta(x_t, t, s), t, s) \|^2 \right]$$
where $v$ is the average velocity parameterized as $v_{s,t}(x) = (f_{s,t}(x) - x) / (t-s)$.

**Parameterization**: $f_\theta(x_t, t, s) = c_{\text{skip}}(t,s) x_t + c_{\text{out}}(t,s) F_\theta(x_t, t, s)$. In practice, $c_{\text{skip}} = 1$ and $c_{\text{out}} = (s-t)$.

### Pseudocode Implementation
```python
# Align Your Flow (AYF) Distillation Training
def train_ayf(x1, x0, v_teacher, f_theta, optimizer):
    optimizer.zero_grad()
    t, s = torch.rand(2)  # s < t usually
    
    # Sample noisy state at t
    xt = (1 - t) * x0 + t * x1
    
    # Teacher ODE step (Euler) to get x_{t'}
    dt = epsilon * (s - t)
    vt = v_teacher(xt, t)
    xt_next = xt + dt * vt
    t_next = t + dt
    
    # Flow map predictions
    f_st = f_theta(xt, t, s)
    f_st_next = f_theta(xt_next, t_next, s)
    
    # AYF-EMD loss: consistency along trajectory
    loss_emd = F.mse_loss(f_st, f_st_next.detach())
    
    # AYF-LMD loss: tangent matching
    # Compute time derivative of f_theta at t
    df_dt = torch.autograd.functional.jvp(
        lambda tau: f_theta(xt, tau, s), t, create_graph=True
    )[1]
    
    # Average velocity target from teacher
    v_target = (f_st.detach() - xt) / (s - t)
    loss_lmd = F.mse_loss(df_dt, v_target)
    
    loss = loss_emd + loss_lmd
    loss.backward()
    optimizer.step()
    return loss

# Multi-step sampling with flow map
def sample_ayf(f_theta, x_noise, num_steps=4):
    x = x_noise
    ts = torch.linspace(1, 0, num_steps + 1)
    for i in range(num_steps):
        x = f_theta(x, ts[i], ts[i+1])
    return x
```

---

## 6. Terminal Velocity Matching (2511.19797)

### Abstract
We propose Terminal Velocity Matching (TVM), a generalization of flow matching that enables high-fidelity one- and few-step generative modeling. TVM models the transition between any two diffusion timesteps and regularizes its behavior at its **terminal time** rather than at the initial time. We prove that TVM provides an upper bound on the 2-Wasserstein distance between data and model distributions when the model is Lipschitz continuous. Since Diffusion Transformers lack this property, we introduce minimal architectural changes (RMSNorm QK-Norm, time embedding normalization) that achieve stable, single-stage training. We develop a fused attention kernel supporting backward passes on Jacobian-Vector Products. On ImageNet-256x256, TVM achieves 3.29 FID with 1 NFE and 1.99 FID with 4 NFEs.

### Main Contributions
- **Terminal Velocity Matching**: Matches the time derivative of the displacement map at the terminal time, providing a distribution-level guarantee (Wasserstein upper bound).
- **Architectural stability**: RMSNorm QK-Norm and normalized AdaLN ensure Lipschitzness and training stability for DiT architectures.
- **Efficient JVP kernel**: Fused Flash Attention kernel supporting backward passes on JVP, achieving 65% speedup.
- **Scaled CFG parameterization**: Network output scales with CFG weight $w$ naturally, enabling stable training across guidance scales.
- **State-of-the-art**: 3.29 FID (1-NFE) and 1.99 FID (4-NFE) on ImageNet-256x256.

### Methods Summary
Define the displacement map $f(x_t, t, s) = \int_t^s u(x_r, r) \, dr$. The **terminal velocity condition** states:
$$\frac{d}{ds} f(x_t, t, s) = u(\psi(x_t, t, s), s),$$
where $\psi$ is the flow map. Differentiating the displacement map and using the network as a proxy yields the training objective:
$$\mathcal{L}_{\text{TVM}}^{t,s} = \mathbb{E} \left[ \underbrace{\left\| \frac{d}{ds} f_\theta(x_t, t, s) - u_{\theta_{\text{sg}}^*}(x_t + f_{\theta_{\text{sg}}}(x_t, t, s), s) \right\|^2}_{\text{terminal velocity}} + \underbrace{\| u_\theta(x_s, s) - v_s \|^2}_{\text{flow matching}} \right].$$

The displacement map is parameterized as $f_\theta(x_t, t, s) = (s-t) F_\theta(x_t, t, s)$. Its time derivative is:
$$\frac{d}{ds} f_\theta = F_\theta + (s-t) \partial_s F_\theta,$$
where $\partial_s F_\theta$ is computed via JVP.

For CFG, the model is conditioned on weight $w$ and class $c$, and the loss is scaled by $1/w^2$ to prevent gradient explosion.

### Pseudocode Implementation
```python
# Terminal Velocity Matching Training
def train_tvm(x, eps, F_theta, u_theta, optimizer):
    optimizer.zero_grad()
    t, s = torch.rand(2)
    
    # Interpolants
    xt = (1 - t) * x + t * eps
    xs = (1 - s) * x + s * eps
    vs = eps - x  # conditional velocity
    
    # Displacement map f(xt, t, s) = (s-t) * F(xt, t, s)
    f_st = (s - t) * F_theta(xt, t, s)
    
    # Terminal velocity: d/ds f = F + (s-t) * d/ds F
    # Compute JVP for d/ds F
    _, dF_ds = torch.autograd.functional.jvp(
        lambda tau: F_theta(xt, t, tau), s, create_graph=True
    )
    
    term_vel = F_theta(xt, t, s) + (s - t) * dF_ds
    
    # Proxy target: u at the arrived point (with stop-grad)
    x_arrived = (xt + f_st).detach()
    u_target = u_theta(x_arrived, s).detach()
    
    # Terminal velocity loss
    loss_tv = F.mse_loss(term_vel, u_target)
    
    # Flow matching loss at s
    u_pred_s = u_theta(xs, s)
    loss_fm = F.mse_loss(u_pred_s, vs)
    
    # Combined
    loss = loss_tv + loss_fm
    loss.backward()
    optimizer.step()
    return loss

# Sampling (n-step or 1-step)
def sample_tvm(F_theta, eps, n_steps=1):
    x = eps
    ts = torch.linspace(1, 0, n_steps + 1)
    for i in range(n_steps):
        t, s = ts[i], ts[i+1]
        x = x + (s - t) * F_theta(x, t, s)
    return x
```

---

## 7. Meta Flow Maps enable scalable reward alignment (2601.14430)

### Abstract
Controlling generative models is computationally expensive because optimal alignment with a reward function requires estimating the value function, which demands access to the conditional posterior $p_{1|t}(x_1 | x_t)$. This typically requires costly trajectory simulations. We introduce Meta Flow Maps (MFMs), a framework extending consistency models and flow maps into the **stochastic regime**. MFMs are trained to perform stochastic one-step posterior sampling, generating arbitrarily many i.i.d. draws of clean data $x_1$ from any intermediate state. These samples provide a differentiable reparametrization that unlocks efficient value function estimation. We enable inference-time steering without inner rollouts, and unbiased off-policy fine-tuning to general rewards.

### Main Contributions
- **Stochastic Flow Maps**: Generalizes deterministic flow maps to stochastic maps that can represent the full conditional posterior $p_{1|t}(\cdot | x_t)$.
- **Meta Flow Maps (MFMs)**: An amortized model that acts as a "meta" flow map over the infinite family of posterior-targeting flow maps.
- **Efficient steering**: Provides Monte Carlo estimators of the value function gradient without trajectory rollouts.
- **Unbiased fine-tuning**: Enables off-policy fine-tuning to general rewards using unbiased objectives.
- **Empirical validation**: Outperforms Best-of-1000 baselines on ImageNet across multiple rewards.

### Methods Summary
A **stochastic flow map** is a transformation $X_{s,u}(\cdot; t, x)$ that maps exogenous noise $\epsilon \sim p_0$ and an intermediate state $(t, x)$ to a sample $x_1 \sim p_{1|t}(\cdot | x)$. By varying $\epsilon$, the map generates arbitrarily many i.i.d. draws from the posterior in one step.

**Training Objectives**:
1. **Diagonal loss** (standard flow matching): Enforces the correct instantaneous drift along the diagonal $s=u$:
   $$\mathcal{L}_{\text{diag}} = \int_0^1 \mathbb{E} \| \hat{v}_{u,u}(I_u) - \dot{I}_u \|^2 \, du.$$
2. **Consistency loss**: Ensures the flow map satisfies the semigroup property globally:
   $$\mathcal{L}_{\text{cons}} = \mathbb{E} \| \hat{X}_{w,u}(\hat{X}_{s,w}(x)) - \hat{X}_{s,u}(x) \|^2.$$

For **reward alignment**, the value function is $V_t(x) = \log \mathbb{E}[e^{r(X_1)} | X_t = x]$. The optimal controlled drift is $b_t^* = b_t + \frac{\sigma_t^2}{2} \nabla V_t$. MFMs enable efficient estimation of $\nabla V_t$ via:
- **Gradient-Free estimator**: Uses posterior samples $x_1^{(i)} \sim p_{1|t}(\cdot | x)$ and reward evaluations $r(x_1^{(i)})$.
- **Gradient-Based estimator**: Uses the reparametrization trick through the differentiable stochastic flow map.

### Pseudocode Implementation
```python
# Meta Flow Map (MFM) Training
def train_mfm(x0, x1, X_mfm, optimizer):
    optimizer.zero_grad()
    s, u, w = torch.rand(3)  # s < w < u
    
    # Stochastic interpolant
    z = torch.randn_like(x0)
    I_s = alpha_s * x0 + beta_s * x1 + gamma_s * z
    I_w = alpha_w * x0 + beta_w * x1 + gamma_w * z
    I_u = alpha_u * x0 + beta_u * x1 + gamma_u * z
    I_dot_u = alpha_dot_u * x0 + beta_dot_u * x1 + gamma_dot_u * z
    
    # Diagonal loss: instantaneous velocity matching
    v_uu = X_mfm(I_u, u, u)  # returns instantaneous velocity at u
    loss_diag = F.mse_loss(v_uu, I_dot_u)
    
    # Consistency loss: composition of flow maps
    # Map from s to w, then w to u
    x_w = X_mfm(I_s, s, w)
    x_u_composed = X_mfm(x_w, w, u)
    
    # Direct map from s to u
    x_u_direct = X_mfm(I_s, s, u)
    
    loss_cons = F.mse_loss(x_u_composed, x_u_direct)
    
    loss = loss_diag + loss_cons
    loss.backward()
    optimizer.step()
    return loss

# Inference-time steering with MFM
def steer_mfm(X_mfm, x_t, reward_fn, n_samples=16):
    # Generate posterior samples in one step
    eps = torch.randn(n_samples, *x_t.shape)
    samples = [X_mfm(x_t, t, 1.0, eps_i) for eps_i in eps]
    
    # Evaluate rewards
    rewards = torch.stack([reward_fn(s) for s in samples])
    
    # Gradient-free value gradient estimate
    weights = F.softmax(rewards, dim=0)
    weighted_mean = sum(w * s for w, s in zip(weights, samples))
    
    # Compute gradient correction (simplified)
    grad_V = (weighted_mean - x_t) / sigma_t**2
    return grad_V

# Sampling with MFM (one-step posterior)
def sample_mfm(X_mfm, x_t, t, eps):
    return X_mfm(x_t, t, 1.0, eps)
```

---

*End of README.*
