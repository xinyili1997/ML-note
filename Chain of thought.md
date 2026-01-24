**Why it work**
	- More computation
	- Produce the explicit anchors that the model could use as reference later, instead of keeping everything internal. Each intermediate result becomes a concrete attention target 
		Before: [Question tokens] → [magical hidden state compression] → [Answer]
		 After: [Question tokens] → [Intermediate result tokens] → [More intermediate tokens] → [Answer]
	- 