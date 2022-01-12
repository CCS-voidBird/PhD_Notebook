https://redditech.github.io/team-fast-tabulous/jupyter/2021/07/21/Regressor-Versus-Classifier.html

Notes:

The Random Forest algorithm is a typical technical instance of the Bootstrap aggregating in ML fields.

https://scikit-learn.org/stable/modules/ensemble.html#id6

The computational theory of decision tree:



The information gain:

+ Entropy
  + for $X$ with a distribution: $$P(X = x_i) = p_i,i\in N$$
  + The entropy of $X$ is $H(X) = -\sum p_ilog(p_i)$
  + The higher entropy, the higher indeterminacy 
  + while the $X \in {1,2}$ , the $H(X) = -plog(p) - (1-p)log(1-p)$
  + Then the $\frac {\partial H(X)}{\partial p} = -log \frac{p}{1-p}$

+ The definition of information gain ($g(D,A)$)
  + The indeterminacy decrease by acknowledged $X$ information 
  + $g(D,A) = H(D) - H(D|A)$

+ The information gain ratio ($g_R(D|A)$)
  + A metric of feature-selection
  + $$g_R (D|A) = \frac{g(D,A)}{H_A(D)}$$ 
  + and the $$H_A(D)  = - \sum \frac{\abs{D_i}}{\abs{D}} log \frac{\abs{D_i}}{\abs{D}}$$
+ The algorithm of building decision tree
  + ID3
  + ID4.5
  + CART
  + Minimal MSE
  + Gini
