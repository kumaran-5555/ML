# Linear CRF

# Definition

Labels ${S = \{s_0, s_1, s_2 ... s_n-1}\}, s_0$ is a special start state.

Input sequence length $m$

Input sequence ${X = \{x_1, x_1, x_2 ... x_m\}, x_i \in \mathbb{R}^d}$

Input label sequence ${Y = \{y_1, y_1, y_2...y_m}\}, y_i \in \{S-s_o\}, y_0 = s_0$

All possible label sequence of length $m, S^m$ . $Y^k$ is one of the permutations in $S^m$ and $\{{y_1^k, y_2^k, ... y_m^k}\}$ is corresponding label sequence, $y_i^k \in \{S-s_o\}$. Subsequence $Y_{1...j} = \{y_1, y_2 ... y_j\}$ , $X_{1...j} = \{x_1, x_2 ... x_j\}$

Edge weights $E_{n X n X m}, E[i, j] \in \mathbb{R}^d$

Label weights $W_{n X m}, W[i] \in \mathbb{R}^d$

Feature function $\phi ( y_{i-1}, y_i, x_i) = \exp(E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i)$

$score(Y, X) = \prod_{i=1}^m \phi (y_{i-1}, y_i, x_i)$

$score(Y, X) = \prod_{i=1}^m \exp(E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i))$

$score(Y, X) = \exp(\sum_{i=1}^m E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i)$

$Z(X) = \sum_{Y^k \in S^m} score(Y^k, X)$

$P(Y|X) = \frac{ score(Y, X)}{ Z(X)}$


# Features

Any sequence inputs like text can be convered to features in $\mathbb{R}^d$ to use in CRF. Features can be sparse for example, trichar prefix, suffix for each token. 

# Training

Training data $T = {\{ \{X^1, Y^1\}, \{X^2, Y^2\}, ... \{X^n, Y^n\} \}}$

Use maximum log likelyhood for training parameters, 

$L =  \log(\prod_{i=1}^n P(Y^i|X^i))$

$L = \sum_1^n \log(P(Y^i|X^i))$

$E, W = \underset{E, W}{\operatorname{argmin}} -L$

# Computing $Z(X)$

$Z$ is sum of exponential number of terms from sequence permutation $S^m$. This can be computed efficiently using dynamic programing with following structure.

1. $s_0$ is special start state, we assume all sequences are implicitly starting with label $y_0  = s_0$.
2. 
$\begin{aligned}
Z(X) &= \sum_{Y^k \in S^m} score(Y^k, X)\\
   score(Y^k, X) &= \prod_{i=1}^m \phi (y_{i-1}, y_i, x_i), Y^k \in S^m\\
    &= \phi (y_0, y_1, x_1) * \phi (y_1, y_2, x_2) ... \phi (y_{m-1}, y_m, x_m)\\    
\end{aligned}  
$

3. This form has prefix structure $score(Y_{1...j}, X_{1...j})$ can be computed only using elements of labels and features from position $1...j$.
   
4. And $score(Y_{1...j}, X_{1...j}) = score(Y_{1...j-1}, X_{1...j-1}) * \phi (y_{j-1}, y_j, x_j)$
   
5. Define

$\begin{aligned}
\alpha[s,1] & = \phi (s_0, s, x_1)\\
\alpha[s,i] & = \sum_{Y^k \in S^i} score(Y^k, X_i) , y_i^k = s\\
\alpha[s,i] & = \sum_{Y^k \in S^{i-1}} score(Y^k, X_{1...i-1}) * \phi (y_{i-1}^k, s, x_i)\\
&= \sum_{s' \in \{S-s_0\}} \alpha[s', i_1] * \phi (s',s, x_i)\\
\end{aligned}
$
    
6. $Z(X) = \sum_{s \in \{S-s_0\}} \alpha[s, m]$


# Computing Gradient of $L$

We learn $E,W$ which optimizes $L$. 

$E, W = \underset{E, W}{\operatorname{argmin}} -L$

Use stochostic gradient descent with batch size 1, i.e., each input sequence is used to update gradients. 

$
\begin{aligned}
L & = log(P(Y|X))\\
\frac {\partial L} {\partial E,W} & = \frac {\partial } {\partial E,W} {log(P(Y|X))}\\
& =  \frac {\partial } {\partial E,W} log(\frac{ score(Y, X)}{ Z(X)})\\
& = \frac {\partial } {\partial E,W} (log(score(Y, X)) - log(Z(X)))\\
& = \frac {\partial } {\partial E,W} log(score(Y, X)) - \frac {\partial } {\partial E,W}log(Z(X)) \\
\end{aligned}$


Partial derivative of first term 

$
\begin{aligned}
\frac {\partial } {\partial E,W} log(score(Y, X)) & = \frac {\partial } {\partial E,W} log(\prod_{i=1}^m \phi (y_{i-1}, y_i, x_i)) \\
& = \frac {\partial } {\partial E,W} \sum_{i=1}^m log(\phi (y_{i-1}, y_i, x_i))\\
& = \frac {\partial } {\partial E,W} \sum_{i=1}^m log(\exp(E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i))\\
& = \frac {\partial } {\partial E,W} \sum_{i=1}^m E[y_{i-1}, y_i]^T x_i + W[y_i]^T x_i\\
\end{aligned}$

Parital derivative with respect $E, W$

$
\begin{aligned}
\frac {\partial } {\partial E} log(score(Y, X)) & = \frac {\partial } {\partial E} \sum_{i=1}^m E[y_{i-1}, y_i]^T x_i \\
 & = \sum_{i=1}^m \frac {\partial} {\partial E[y_{i-1}, y_i]} E[y_{i-1}, y_i]^T x_i \\
{\partial E[y_{i-1}, y_i]} & = x_i\\
\frac {\partial } {\partial W} log(score(Y, X)) & = \frac {\partial } {\partial E} \sum_{i=1}^m  W
[y_i]^T x_i \\
& = \sum_{i=1}^m \frac {\partial} {\partial W[y_i]} W[y_i]^T x_i \\
{\partial W[y_i]} & = x_i\\
\end{aligned}
$


Partial derivate of second term

$
\begin{aligned}
\frac {\partial } {\partial E,W} log(Z(X)) & = \frac {\partial } {\partial E,W} log(\sum_{Y^k \in S^m} score(Y^k, X))\\
& = \frac {\partial } {\partial E,W}  log(\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i))\\
& = \frac {1} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)} * \frac {\partial } 
{\partial E,W}  \sum_{Y^j \in S^m} \prod_{i=1}^m (\phi (y_{i-1}^j, y_i^j, x_i))\\
\\
& = \frac { \sum_{Y^j \in S^m} \frac {\partial } {\partial E,W} \prod_{i=1}^m (\phi (y_{i-1}^j, y_i^j, x_i))} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)}\\
\\
& = \frac { \sum_{Y^j \in S^m} \frac {\partial } {\partial E,W} \prod_{i=1}^m (\exp(E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i))} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)}\\
\\
& = \frac { \sum_{Y^j \in S^m} \frac {\partial } {\partial E,W} \exp( \sum_{i=1}^m (E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i)} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)}\\
\\
& = \frac { \sum_{Y^j \in S^m} \exp( \sum_{i=1}^m (E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i)} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)} * \frac {\partial } {\partial E,W}  \sum_{i=1}^m E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i\\
\\
& = \sum_{Y^j \in S^m}  \frac { \exp( \sum_{i=1}^m (E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i)} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)} * \frac {\partial } {\partial E,W}  \sum_{i=1}^m E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i\\
\\
& = \sum_{Y^j \in S^m}  \frac {score(Y^j, X)} {\sum_{Y^k \in S^m} \prod_{i=1}^m \phi (y_{i-1}^k, y_i^k, x_i)} * \frac {\partial } {\partial E,W}  \sum_{i=1}^m E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i\\
\\
& = \sum_{Y^j \in S^m} P(Y^j|X) * \frac {\partial } {\partial E,W}  \sum_{i=1}^m E[y_{i-1}^j, y_i^j]^T x_i + W[y_i^j]^T x_i\\
\end{aligned}
$