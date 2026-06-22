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
### Week 4

Week 4 focused on testing alternative exploration mechanisms and introducing neural-network diagnostics.

#### Thompson Sampling

In addition to EI and UCB, I implemented Thompson Sampling as an alternative acquisition strategy. Unlike EI and UCB, which optimise a deterministic acquisition function, Thompson Sampling draws a sample from the Gaussian Process posterior and selects candidates that appear optimal under that sampled function. This naturally encourages exploration in regions where uncertainty remains high.

Interestingly, Thompson Sampling was selected by the ensemble framework for Function 3, one of the most uncertain functions in the challenge. This result was consistent with other diagnostics, suggesting that Function 3 remains poorly understood and may benefit from additional exploration rather than aggressive exploitation.

#### Function 1: Problem Interpretation

Week 4 also highlighted the importance of understanding the problem description rather than relying solely on model outputs.

Function 1 was described as a radiation-detection problem where only proximity to a contamination source generates a meaningful signal. This interpretation suggests a highly localised response surface rather than a smooth global optimisation problem. As a result, I modified the search strategy from broad exploration toward local exploitation around the strongest observed signal region. The Gaussian Process length scales and observed outputs were consistent with this interpretation.

#### Neural-Network Diagnostics

I introduced a TensorFlow/Keras neural network as a diagnostic tool rather than a replacement for Bayesian Optimisation. The network used two hidden layers with ReLU activations and a sigmoid output layer. Hyperparameters included hidden-unit sizes of 8 and 16, learning rates of 0.01 and 0.05, and early stopping with a maximum of 75 epochs.

To evaluate separability, I converted each optimisation problem into a binary classification task by labelling the top 25% of observed outputs as "good" and the remaining observations as "bad".

The results revealed significant differences between functions:

* Functions 5 and 8 achieved strong classification performance and low validation loss, suggesting clear decision boundaries and concentrated high-performance regions.
* Functions 6 and 7 showed moderate classification performance but displayed increasing validation loss despite falling training loss, indicating mild overfitting.
* Function 3 performed poorly, with classification performance close to random guessing and clear signs of overfitting. This result independently supported the Thompson Sampling recommendation that additional exploration may be required.

The convergence behaviour of Functions 3, 6 and 7 proved particularly informative. In all three cases, training loss continued to decrease while validation loss eventually deteriorated, suggesting that model uncertainty remains significant despite increasing model complexity.

#### Current Framework

The current workflow combines:

* Gaussian Process surrogate models
* Matérn and RBF kernel ensembles
* EI, UCB and Thompson Sampling acquisition strategies
* DBSCAN consensus clustering

## Week 5 Update – Neural Network Experiments

This week I extended the optimisation framework with two neural-network components.

First, I added a PyTorch surrogate model that directly approximates each black-box function using the accumulated observations. After training, gradient-based optimisation was performed on the network inputs to identify candidate query locations. Validation loss was tracked separately from training loss to assess potential overfitting.

Second, I expanded the TensorFlow/Keras diagnostic classifier by introducing an additional hidden layer. The classifier labels the top 25% of observed outcomes as “high-performing” regions and evaluates whether a neural network can distinguish promising areas of the search space. I compared one-hidden-layer and two-hidden-layer architectures using validation loss and AUC metrics.

Several observations emerged from these experiments. In a number of functions, adding a second hidden layer improved validation performance, suggesting that some non-linear structure exists in the accumulated observations. However, the benefits were inconsistent across functions, highlighting the trade-off between model complexity and limited data availability.

The PyTorch surrogate also revealed an important behaviour: when trained on small datasets, neural networks frequently pushed candidate solutions towards boundary values (0 or 1). This behaviour proved highly successful for Function 5, where both the neural-network surrogate and the Gaussian-process framework pointed towards the upper corner of the search space, resulting in a substantial improvement in the objective value. In contrast, Function 8 demonstrated that strong neural-network confidence does not necessarily translate into better optimisation performance, as boundary-seeking behaviour did not improve upon the current best result.

Overall, these experiments suggest that neural networks are currently more useful as diagnostic and exploratory tools than as primary optimisation engines. Gaussian-process models remain the main optimisation framework, while neural networks provide additional insight into non-linear structure, separability, and potential overfitting.

* ARD feature-relevance diagnostics
* Neural-network separability diagnostics

The neural network is not currently used for query generation. Instead, it serves as an independent diagnostic tool to assess whether meaningful structure exists within each optimisation problem and to cross-check conclusions drawn from the Bayesian Optimisation framework.
