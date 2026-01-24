- Cross entropy: penalize being confidently wrong 
	- $$L = -log(p_y)$$               y is the correct label
	- $$% Softmax
p_k = \mathrm{softmax}(\mathbf{z})_k
= \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}
$$
	- $$% Cross entropy loss (single example, class y)
\mathcal{L} = -\log p_y$$
	- $$% Substitute softmax into cross entropy
\mathcal{L}
= -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{C} e^{z_j}}\right)$$
	- $$% Simplified softmax loss form
\mathcal{L}
= -z_y + \log\left(\sum_{j=1}^{C} e^{z_j}\right)$$
