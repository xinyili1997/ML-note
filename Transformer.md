1. Self-attention
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
4. Decoder causal mask（为什么不能看未来）
5. KV caching（推理性能）