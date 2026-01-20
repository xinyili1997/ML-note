1. **Warmup**:
	1. Goal: solve the instability when  training starts by lowering the LR
	2. $$\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}} \quad \text{for } t \leq T_{warmup}$$
	3. Pros:
		1. Avoid spikes due to random params, gradients, unstable m and v for Adam
		2. Give adam a soft start period
		3. More stable activation 
	4. Cons:
		1. More costly
		2. Need hyperparms tuning --> usually 1-5% of total steps
		3. Could hide real problem like bad initialization, large learning rate, bad data
	5. Warmup improves stability early in training by preventing overly large updates before the optimizer’s statistics and activations settle, but it slows down early learning and introduces extra hyperparameters; too much warmup can waste budget or hurt convergence
2. **Cosine decay**:
	1. Goal: improve late-stage convergence
	2. $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$
	3. Later in training, you don’t want the learning rate to stay large because it can:
		1. keep overshooting the optimum
		2. cause oscillations around a good solution
		3. hurt final performance/generalization
	3. Cosine decay gradually decreases the learning rate, helping the model make smaller, fine-grained updates and converge better.
3. 