1. Why we want to stable activation distribution?
	1. Activation is the output of each layer. Activation distribution refers to its output distribution including mean, std, max, min etc.
	2. Stable activation distribution refers to the std of each layer output is within a range.
	3. Why does it matter?
		1. Exploding activation/gradient: --> output are larger and larger
			1. Outcome:
				1. loss becomes Nan
				2. gradients norm is too large
				3. overflow
			2. Why output are very likely to grow?
				1. As long as the ||W|| is large than 1, then the outcome is expanding.
				2. bias could make the output grow
				3. ReLu/SiLu will cancel the negatives
				4. Residual is accmulating at every layer 
			3. 
		2. Vanishing activation/gradient: 
			1. loss decrease too slow
			2. softmax/sigmoid's outputs are too extreme
	4. 
2. $$sqrt(d_k)$$ is logits scaling/normalization
	1. Solved: softmax saturation, overflow, gradient is too small
3. **LayerNorm**
	1. Normalized the token, and make it back to normal distritbution
	2. Solved: The std and mean of each hidden layer, avoiding activation drift.
	3. Why does activation drift matter?
		1. Avoid overflow and loss becomes nan
		2. Avoid slow converge
		3. If layer 0 is output distribution is within a range, it's easier for layer 1 to learn the correct mapping. --> internal covariate shift
	  4. Post layer norm vs pre layer norm:
		  1. Post-LN: x = LN(x + Attn(x))
			  1. $$ \begin{align} y &= x + \text{Attention}(x) \\ z &= \text{LayerNorm}(y) \end{align} $$
			  2. $$ \frac{\partial z}{\partial x} = \frac{\partial \text{LN}}{\partial y}*(1+\frac{\partial \text{Attention}}{\partial x})$$
				It can ensure the final output has the normal distribution, which could bring better outcome. But it relies on careful hyper-params tuning. 
				It could also ensure model less likely to over-relay some inputs to overfit. 
	
		  3. Pre-LN: x = x + Attn(LN(x))
			  1. $$ \begin{align} y &= x + \text{Attention}(\text{LayerNorm}(x))\end{align} $$
			  2. $$\frac{\partial L}{\partial x} = (1+\frac{\partial \text{Attention}}{\partial LN} \cdot \frac{\partial LN}{\partial x})$$
			  3. So it can ensure the gradient is less likely to disappear, which is good for deep learning training. 
			  
		 4. Chatgpt uses pre-Ln and apply a final LN:
			 1. Pre-Ln is more robust and easy to train. 
			 2. Pre-Ln cannot ensure the final output is normalized. Final LN could do that which can help stabilize logits/softmax. 
4. Residual Connection
	1. $$ y = x + F(x) $$
	2. Solved:
		1. Gradient disappear as it always have 1
		2. Easier to optimize:
			1. Could start with y ~=x, which is identity,   making loss/gradient smoother and easier to use SGD/Adam.
		3. Residual connections add a skip path that preserves the input and helps gradients propagate through many layers, reducing vanishing/exploding issues. This encourages layers to make small, incremental updates, which typically leads to faster convergence and more stable training.
5. Gradient clipping
	1. Solved gradient clipping, usually needed as engineering practice
	2. ```python
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
	optimizer.step()
	optimizer.zero_grad()
	   ```
    3. Used when:
	    1. Potential loss spike/Nan/Inf 
	    2. Long-tail cases
	    3. Deep network
	    4. Mixed precision --> fp16 range is small

6. Weight decay
	1. $$ L_{\text{total}}(W) = L_{\text{data}}(W) + \frac{\lambda}{2}\|W\|^2 $$
	2. Goal: avoid overfitting to penalize param weight is too large. It could also reduce the effect of weight gradients including :
		1. activation is too large 
		2. gradient is too large or too flaky
		3. loss spike
	3. Weight decay is a form of L2 regularization that discourages large weights, improving generalization and preventing weight growth that can destabilize training. In practice, AdamW decouples weight decay from the adaptive gradient update and is widely used for Transformers.
7. Xaiver/He