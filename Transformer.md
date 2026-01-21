1. ![[Pasted image 20260120190611.png]]
2. Self-attention
	1. Every token is trying to define:
		1. Q: what I am looking for?
		2. K: what property it has?
		3. V: what information I can provide
	2. Attention = $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
	3. why dK: if Q and K are high dimension, the value would be too large. 
		1. Say if Q and K are nominal distribution whose std is 1, then $$QK^T$$ 's standard deviation would be sqrt(d_k)
		2. This is **scaling**
		3. Difference between scaling and layer/batch normalization:
			1. Scaling is applying a constant factor and not rely on the input values. The goal is to avoid overflow and softmax saturation.
			2. Normalization is calculating std and mean. The goal is to ensure stable activation distribution. 
		4. Outcome 1: one bit would be close to 1, and all others would be close to 0, making the gradient shape.
			1. The model could focus on a single token, which could be randomly chosen in the beginning.
		5. Outcome 2: $$e^x$$ could be huge. Meaning potentially overflow.
2. Multi-head attention
3. Block（residual + layernorm + FFN）
	1. Feed-forward network
		1. ![[Pasted image 20260120185646.png]]
	2. Step by step:
		1. h = W₁ · x + b₁ // Linear transformation (expand) 
			1. x: [d_model] e.g., [768]
			2. W₁: [d_ff, d_model] e.g., [3072, 768]
			3. h: [d_ff] e.g., [3072]
		2. a = GELU(h) // Non-linear activation
		3. y = W₂ · a + b₂ // Linear transformation (project back)
			1. W₂: [d_model, d_ff] e.g., [768, 3072]
			2. y: [d_model] e.g., [768]
		4. Expanding to higher dimension:
			1. The 4x expansion (d_model → 4×d_model) is an empirical finding. It provides enough capacity for complex transformations while being computationally manageable.
			2. FFN contains ~2/3 of the parameters in each transformer block! 

5. Decoder causal mask（为什么不能看未来）
6. KV caching（推理性能）