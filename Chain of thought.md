**Why it work**
	- More computation
	- Produce the explicit anchors that the model could use as reference later, instead of keeping everything internal. Each intermediate result becomes a concrete attention target 
		Before: [Question tokens] → [magical hidden state compression] → [Answer]
		 After: [Question tokens] → [Intermediate result tokens] → [More intermediate tokens] → [Answer]

**How to train**
1. Data collection
	1. Human annotated reasoning process
	2. Synthetic generation from existing strong models
	3. Programmatic labeling
2. Supervised fine-tuning
	1. **When:** After pre-training, before RL 
	2. **Data:** Question + full reasoning + answer 
	3. **Goal:** Model learns to generate reasoning format
	4. 
3. 