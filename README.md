
[![Integrate](https://github.com/directed-graph/at_least_n/actions/workflows/integrate.yaml/badge.svg)](https://github.com/directed-graph/at_least_n/actions/workflows/integrate.yaml)

Given a list of $N$ probabilities, what is the probability that at least $n \le
N$ of them resolves to true?

The program in this repository attempts to answer this question. The solution
is based on <https://math.stackexchange.com/a/4695443/197924>.

The first part of the solution is pretty straight forward. The second part may
need some explanation. For clarity, instead of 6, suppose we want at least $n$
elements. We will use the notation $g(m, n)$ here. If $g(m, n)$ is the
probability of $n$ or more resolving to true, then:

$g(m, n) = P(m)g(m - 1, n - 1) + (1 - P(m))g(m - 1, n)$

But $g(m - 1, n - 1) = f(m - 1, n - 1) + g(m - 1, n)$ (i.e. the probability of
items $0$ through $m - 1$ having at least $n - 1$ resolving to true is equal to
the probability of exactly $n - 1$ resolving to true, plus the probability of
$n$ or more resolving to true).

Thus:

$$\begin{aligned}
  g(m, n) &= P(m)g(m - 1, n - 1) + (1 - P(m))g(m - 1, n) \\
          &= P(m)(f(m - 1, n - 1) + g(m - 1, n)) + (1 - P(m))g(m - 1, n) \\
          &= P(m)f(m - 1, n - 1) + P(m)g(m - 1, n) + g(m - 1, n) - P(m)g(m - 1, n) \\
          &= P(m)f(m - 1, n - 1) + g(m - 1, n)
\end{aligned}$$

