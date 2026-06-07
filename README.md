# bbo-challenge
Bayesian optimisation of unknown black-box functions using Gaussian Processes, ARD diagnostics and DBSCAN-based ensemble candidate selection.
1. Project Overview
This project aims to optimise eight unknown black-box functions under a limited query budget. The true functional forms are hidden, so the objective is to learn from a small number of observations and propose increasingly better query locations over time.

The challenge is highly relevant to real-world machine learning because many practical optimisation problems involve expensive evaluations and incomplete knowledge. 

From a professional perspective, this project is closely related to geopolitical tail-risk analysis and portfolio hedging. In practice, protective hedges are expensive and cannot be purchased every time a new piece of information appears. The underlying probability distribution is unknown and must be inferred from incomplete evidence. This creates a problem similar to black-box optimisation: each new observation provides information about an unknown objective function, but observations are limited and costly. The challenge therefore provides a useful framework for learning how to update beliefs, compare competing models and decide when additional information is valuable enough to justify action.

2. Inputs and Outputs
Each function receives a vector of inputs:

x = [x₁, x₂, ..., xₙ]
where each dimension is constrained to:

0 ≤ xᵢ < 1
The dimensionality varies by function:

Function	Dimensions
1	2
2	2
3	3
4	4
5	4
6	5
7	6
8	8
For each submitted query, the portal returns a scalar response value:

y = f(x)
The objective is to maximise the unknown function value using a limited number of sequential queries.

Example query:

0.581903-0.003846-0.310088
Example output:

y = -0.0259
3. Challenge Objectives
The objective is to maximise each unknown function while operating under several constraints.

Only one new query may be submitted per function each week, meaning information arrives slowly and decisions cannot be revised immediately. In addition, the true functional forms are completely hidden. The response surfaces may be smooth, highly non-linear, multi-modal, or contain local optima and other structural features that are not directly observable from the available data.

As a result, the challenge is not simply about finding the highest predicted value. It is about deciding where to allocate a limited number of observations in order to learn the unknown function as efficiently as possible. Each query therefore represents a trade-off between exploiting regions that already appear promising and exploring regions where uncertainty remains high.

4. Technical Approach
Week 1
The initial approach used Gaussian Process (GP) surrogate models with Matérn kernels. Candidate points were selected primarily through distance-based exploration and posterior uncertainty analysis.

Week 2
After receiving additional observations, the strategy shifted towards analysing posterior updates and experimenting with different acquisition-function parameters. The goal was to understand how exploration and exploitation affected query selection and whether promising regions were beginning to emerge.

Several functions showed substantial posterior updates after new observations, suggesting that model assumptions were becoming increasingly important.

Week 3
The current approach is significantly more systematic.

For each function, I perform a grid search across:

Matérn kernels with different smoothness assumptions (ν = 0.5, 1.5 and 2.5)

RBF kernels

Multiple acquisition-function parameter settings

This generates 36 candidate points per function.

Rather than selecting a single model manually, I use DBSCAN clustering to identify regions where multiple Bayesian Optimisation models independently agree. This effectively creates an ensemble voting system. Candidate points are first grouped by consensus, then ranked using predicted performance and uncertainty measures.

I also introduced Automatic Relevance Determination (ARD) diagnostics to identify which dimensions appear influential. For example, some functions appear to depend heavily on one or two dimensions, while others exhibit more balanced behaviour across all dimensions. Although ARD is not used directly for candidate selection, it provides useful interpretability and helps identify potentially irrelevant dimensions.

This workflow balances robustness and performance by combining Bayesian optimisation, ensemble agreement and feature-relevance diagnostics rather than relying on a single model configuration.
