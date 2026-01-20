1. Adam
	1. $$\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}$$
	2. Intuitive understanding: $$\frac{|mean|}{variance} ~= \frac{signal}{noise}$$
	3. Why need bias correction?
		1. Because the initial m and v will be highly rely on g since m0=0 & v0=0.  
		2. At the beginning, the effective step is $$\frac{(1-\beta_1)g}{\sqrt{(1-\beta_2)g^2}} = \frac{(1-\beta_1)sign(g)}{\sqrt{(1-\beta_2)}}$$
		3. Outcome 1: step is too large: with default parma $$\beta_1 = 0.9, \beta_2 = 0.999$$, the value would be 3.16*sign(g) --> much larger than expected learning_rate
		4. Outcome 2: step is too small: 
			1. mt corrects the bounce between negative and positive(it's signed)
			2. vt is unsigned, so if g is large for one time, it will remain large for a long time. 
			3. So if mt corrects the bounce and vt is affected by one time large gradient, then step will be very small. 
		5. Outcome 3: more rely on warmup(lower learning rate in the beginning)
2. Adam W
	1. $$\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{align}$$
3. Adam W + L2 Regularization
	1. $$\begin{align}
g_t &= \nabla_\theta L(\theta_{t-1}) + \lambda_{L2} \theta_{t-1} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda_{WD} \theta_{t-1}\right)
\end{align}$$
