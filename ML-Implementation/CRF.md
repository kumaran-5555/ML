# Linear CRF

## Definition

Labels ${S = \{s_0, s_1, s_2 ... s_n-1}\}, s_0$ is a special start state.

Input sequence ${X_i = \{x_1, x_1, x_2 ... x_m\}, x_i \in \mathbb{R}^d}$

Input label sequence ${Y_i = \{y_1, y_1, y_2...y_m}\}, y_i \in \{S-s_o\}, y_0 = s_0$

All possible label sequence of length $m, S^m ; Y_k \in S^m$ is one of the permutations in $S^m$ and $\{{y_1^k, y_2^k, ... y_m^k}\}$ is corresponding label sequence, $y_i^k \in \{S-s_o\}$.

Edge weights $E_{n X n X m}, E[i, j] \in \mathbb{R}^d$

Label weights $W_{n X m}, W[i] \in \mathbb{R}^d$

Feature function $\phi ( y_{i-1}, y_i, x_i) = \exp(E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i)$

$score(Y_i,X_i) = \prod_{i=1}^m \phi (y_{i-1}, y_i, x_i)$

$Z(X_i) = \sum_{Y_k \in S^m} score(Y_k, X_i)$

$P(Y_i|X_i) = \frac{ score(Y_i, X_i)}{ Z(X_i)}$


## Features

Any sequence inputs like text can be convered to features in $\mathbb{R}^d$ to use in CRF. Features can be sparse for example, trichar prefix, suffix for each token. 

## Training

Training data $T = {\{ \{X_1, Y_1\}, \{X_2, Y_2\}, ... \{X_n, Y_n\} \}}$

Use maximum log likelyhood for training parameters, 

$L =  \log(\prod_{i=1}^n) P(Y_i|X_i)$

$E, W = \underset{E, W}{\operatorname{argmin}} -L$

## Computing $Z(X)$

$Z$ is sum of exponential number of terms from sequence permutation $S^m$. This can be computed efficiently using dynamic programing with following structure.

1. $s_0$ is special start state, we assume all sequences are implicitly starting with label $y_0  = s_0$.
2. 
$\begin{aligned}
Z(X_i) &= \sum_{Y_k \in S^m} score(Y_k, X_i)\\
   score(Y_i, X_i) &= \prod_{i=1}^m \phi (y_{i-1}, y_i, x_i), Y_i \in S^m\\
    &= \phi (y_0, y_1, x_1) * \phi (y_1, y_2, x_2) ... \phi (y_{m-1}, y_m, x_m)\\    
\end{aligned}  
$

3. This form has prefix structure $score(Y_{1...j}, X_{1...j})$ can be computed only using elements of labels and features from position $1...j$.
   
4. And $score(Y_{1...j}, X_{1...j}) = score(Y_{1...j-1}, X_{1...j-1}) * \phi (y_{j-1}, y_j, x_j)$
   
5. Define

$\begin{aligned}
\alpha[s,1] & = \phi (s_0, s, x_1)\\
\alpha[s,i] & = \sum_{Y_k \in S^i} score(Y_k, X_i) , y_i^k = s\\
\alpha[s,i] & = \sum_{Y_k \in S^{i-1}} score(Y_k, X_{1...i-1}) * \phi (y_{i-1}^k, s, x_i)\\
&= \sum_{s' \in \{S-s_0\}} \alpha[s', i_1] * \phi (s',s, x_i)\\
\end{aligned}
$
    




