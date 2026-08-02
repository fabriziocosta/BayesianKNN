# Bayesian Model Averaging

## Executive summary

This project implements Bayesian-style model averaging over k-nearest-neighbours,
linear, and Gaussian predictive models rather than hyperparameter optimization.

Traditional k-NN requires several modelling decisions, including the number of neighbours, the representation of the feature space, and often the amount of data used to construct the neighbourhood. These choices are typically treated as hyperparameters and selected through cross-validation or other optimization procedures, resulting in a single model that is then used for prediction.

The central idea of this work is fundamentally different.

Instead of searching for one “best” model, every reasonable k-NN configuration is regarded as a plausible explanation of the data. Each configuration defines a simple predictive model, and the final prediction is obtained by averaging over many such models according to their support from the observed data.

The implementation therefore replaces optimization with integration.

For the default k-NN family, three independent sources of modelling uncertainty
are explicitly represented:

- **Representation scale**, describing how the original feature space is viewed through random projections.
- **Data scale**, describing how much of the training data participates in constructing an individual local model.
- **Prediction scale**, describing the size of the neighbourhood used for local prediction.

Rather than fixing any of these quantities, they are treated as latent random variables and sampled from common probabilistic priors. A single reusable scale-prior mechanism governs all ordered scale variables, ensuring that the implementation remains conceptually simple and internally consistent.

The estimator can also integrate across predictive model families. In
`model_family="mixed"`, k-NN, linear, and Gaussian models are sampled from a
uniform family prior and averaged using the same cross-validated
pseudo-posterior weights. Gaussian classification draws additionally integrate
over isotropic, diagonal, and full covariance structures under a simplicity
prior.

Each sampled model is evaluated using a cross-validated predictive pseudo-likelihood. These predictive scores determine the contribution of each model to the final Bayesian model average. Consequently, models that make better predictions naturally receive greater influence, while models that are less predictive contribute proportionally less.

The resulting algorithm has several attractive properties:

1. It avoids committing to a single arbitrary choice of neighbourhood size, projection dimension, or subset size. Instead, it integrates over uncertainty in these quantities.
2. It preserves the simplicity and interpretability of the component models. Every sampled model is a standard estimator from one of the supported families; only the averaging procedure is new.
3. It is naturally parallel. Each Monte Carlo model is completely independent of every other model, allowing straightforward parallel execution across multiple processor cores.
4. It is intentionally modular. Representation learning, scale priors, prediction, scoring, convergence diagnostics, and Bayesian averaging are independent components with clearly defined interfaces.

Conceptually, the framework shifts machine learning from selecting a single optimal model to quantifying uncertainty over an entire family of simple local models. Predictions emerge from Bayesian averaging across scales rather than from committing to one particular representation or one particular notion of locality.

## Goal and design principles

This repository provides production-quality, modular, scikit-learn-compatible
Bayesian model-averaging classifiers and regressors.

The philosophy of the algorithm is:

- do **not** optimize hyperparameters;
- do **not** select a single model;
- instead **integrate over an entire family of simple predictive models**.

Every uncertain modelling choice is treated as a latent variable and integrated out by Monte Carlo sampling.

The implementation should be clean, modular, well documented, fully typed, and suitable for open-source release.

Each component should have a single responsibility. Avoid monolithic files.

## High-level architecture

The algorithm has four independent modules.

### 1. Representation module

This module transforms the feature space. The default implementation is a **mixed representation family** that includes random projections and the identity representation.

The representation module is independent of the prediction module. Its purpose is simply to generate alternative representations of the input space.

It should expose an interface such as:

~~~
fit(X)

transform(X)

sample_parameters(random_state)
~~~

The design must allow future projection families to be added without modifying the prediction code.

Initially implement:

- Gaussian random projection;
- sparse random projection;
- identity projection (no projection).

The estimator parameter `representation` accepts `"gaussian"`, `"sparse"`,
`"identity"`, or `"mixed"`. The fixed-family modes use the selected family for
every Monte Carlo draw. The default `"mixed"` mode samples uniformly from all
three families, with family probability `1/3` recorded in every model draw.
Cross-validated pseudo-likelihoods determine the model weights, so the identity
family can retain its full feature space when compression is not predictive.

### 2. Prediction module

For k-NN draws, prediction is standard weighted k-nearest neighbours.

The prediction module knows nothing about random projections. It receives
transformed data only and supports classification and regression estimators
selected by the sampled model family.

### 3. Model-family module

The estimator parameter `model_family` accepts:

- `"knn"`: weighted k-NN, with the sampled neighbourhood size and
  representation machinery;
- `"linear"`: `LogisticRegression` for classification and `Ridge` for
  regression;
- `"gaussian"`: a generative Gaussian classifier or `BayesianRidge` for
  regression;
- `"mixed"`: a uniform prior over the three families.

The default is `"mixed"`, so the renamed estimator averages all three families
without requiring special configuration. Linear and Gaussian draws use the
identity representation and do not sample k. Their family probability and any
applicable covariance probability are retained in the model record and in
`log_prior`.

For Gaussian classification, covariance structure is itself a discrete latent
variable sampled for every Gaussian draw, just as k is sampled for every k-NN
draw. The structures are ordered by complexity:

$$
q_{\mathrm{iso}}=1,\qquad
q_{\mathrm{diag}}=d,\qquad
q_{\mathrm{full}}=\frac{d(d+1)}{2}.
$$

With simplicity parameter `lambda`, the prior is

$$
P(s)\propto\exp\{-\lambda\log(1+q_s)\}.
$$

The default `lambda=1` gives isotropic covariance the greatest prior mass,
then diagonal, then full covariance when `d > 1`. The selected structure and
its conditional probability are stored with the draw. Gaussian regression
uses `BayesianRidge`; covariance structure is a classification-only choice.

### 4. Bayesian integration module

This module performs Monte Carlo sampling over complete models.

Each Monte Carlo draw samples:

- a model family when `model_family="mixed"`;
- a representation family when `representation="mixed"`;
- representation parameters within the selected representation family;
- subset size;
- subset;
- neighbourhood size for k-NN draws;
- Gaussian covariance structure for Gaussian classification draws.

It fits the corresponding model, computes its cross-validated pseudo-likelihood, and contributes to the Bayesian model average.

## Package structure

~~~
bayesian_model_averaging/

    __init__.py

    classifier.py
    regressor.py

    representation/
        base.py
        gaussian_projection.py
        sparse_projection.py
        identity.py

    priors.py
    model_families.py
    sampling.py
    scoring.py
    convergence.py

    models.py
    utils.py
~~~

## Installation and quick start

Install the package and its runtime dependencies with:

~~~bash
python -m pip install .
~~~

The estimators use mixed model families, a mixed representation family,
distance-weighted Euclidean k-NN components, five-fold cross-validation, and
automatic Monte Carlo growth by default. Set `model_family="knn"` to use only
the k-NN family.
For a small deterministic run:

~~~python
from bayesian_model_averaging import BayesianModelAveragingClassifier

model = BayesianModelAveragingClassifier(n_estimators=20, random_state=7, n_jobs=-1)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
~~~

`BayesianModelAveragingRegressor` exposes the corresponding `fit`, `predict`, `score`, and `get_model_draws` methods. It intentionally does not expose `predict_proba`.

## Unified logistic scale prior

### Conceptual role

The package uses one common modelling assumption for ordered scales:

> Ordered model scales should be sampled from a monotone decreasing prior, with smaller scales preferred a priori, while uncertainty about the strength and location of that preference is integrated out.

Do not implement separate prior classes for projection dimension, training-subset size, and neighbourhood size. All three must use the same generic sampler.

The class must be independent of k-NN, random projections, classification, and regression. It should operate only on an ordered collection of allowable values.

### Required class and API

Implement a class such as:

~~~
LogisticScalePrior
~~~

in:

~~~
bayesian_model_averaging/priors.py
~~~

Provide a method such as:

~~~
draw(
    values: Sequence[int],
    rng: np.random.Generator,
) -> ScalePriorDraw
~~~

The returned object must be an immutable dataclass containing:

~~~
value
index
beta
cutoff
probability
log_probability
probabilities
~~~

These fields mean:

- value is the sampled allowable value;
- index is its position in values;
- beta is the sampled positive logistic slope;
- cutoff is the sampled cutoff value;
- probability is the normalized probability assigned to the sampled value;
- log_probability is its logarithm;
- probabilities is the complete normalized discrete distribution over all allowable values.

### Statistical definition

Given ordered allowable values:

$$
v_1,\ldots,v_J,
$$

map their positions to the normalized interval:

$$
u_j =
\begin{cases}
0, & J=1,\\
\frac{j}{J-1}, & J>1,
\end{cases}
\qquad j=0,\ldots,J-1.
$$

For each draw, sample:

$$
\beta \sim \mathrm{Gamma}(a_\beta,s_\beta),
$$

and

$$
c\sim\mathrm{Uniform}(0,1).
$$

Use the defaults:

~~~
beta_shape=2.0
beta_scale=1.0
~~~

Define the unnormalized probability of each allowable value as:

$$
q_j
=
\frac{1}
{1+\exp(\beta(u_j-c))}.
$$

Normalize:

$$
p_j=\frac{q_j}{\sum_r q_r}.
$$

Then sample one allowable value according to the normalized probabilities. Because the sampled logistic slope is positive, this prior is monotone decreasing over the ordered values and therefore prefers smaller scales.

For a single allowable value, return that value with probability one while still sampling and recording beta and cutoff.

### Numerical stability and validation

Compute the logistic weights stably. Prefer a formulation such as:

~~~
log_weights = -np.logaddexp(0.0, beta * (u - cutoff))
~~~

Then normalize in log space using:

~~~
scipy.special.logsumexp
~~~

Do not compute unstable exponentials directly.

Validate that:

- values is one-dimensional;
- values is non-empty;
- values is ordered;
- beta_shape > 0;
- beta_scale > 0.

## Reuse of the common prior

The exact same LogisticScalePrior instance or configuration must be reusable for all three model components.

### Projection dimension

For Gaussian and sparse random projections, use:

~~~
values = range(1, n_features + 1)
~~~

For identity, use `values = [n_features]`. The representation module asks the
scale prior for one projected dimension using the family-specific allowable
values below. Identity never truncates or otherwise changes the feature space.

When `representation="mixed"`, first sample the family uniformly from
`("gaussian", "sparse", "identity")`. Store that family probability and add
its log probability to the model's recorded prior. Because models are sampled
directly from this declared prior, the default self-normalized weights still
use the cross-validated pseudo-likelihood alone; do not multiply the family
probability into the weights a second time.

The projection dimension is a latent variable. For normalized position `u`, the representation-specific form is:

$$
u=\begin{cases}
0,&d=1,\\
\frac{d'-1}{d-1},&d>1.
\end{cases}
$$

Sample:

$$
\beta_d\sim\mathrm{Gamma}(2,1)
$$

and

$$
c_d\sim\mathrm{Uniform}(0,1).
$$

Define:

$$
P(u)
\propto
\frac{1}
{1+\exp(\beta_d(u-c_d))}.
$$

Discretize this over all allowable projection dimensions, sample one projection dimension, and then sample the projection matrix according to the selected representation family.

### Training-subset size

Use:

~~~
values = range(min_subset_size, max_subset_size + 1)
~~~

The Monte Carlo model sampler asks for one subset size. For normalized position `u`, the subset-specific form is:

$$
u=\frac{m-m_{\min}}
{m_{\max}-m_{\min}}.
$$

Sample:

$$
\beta_m\sim\mathrm{Gamma}(2,1)
$$

and

$$
c_m\sim\mathrm{Uniform}(0,1).
$$

Construct the logistic distribution, sample one subset size, and then sample the subset uniformly from the CV-admissible subsets of that size.

Validate `1 <= min_subset_size <= max_subset_size <= n_samples`. Let `n_splits` equal the integer `cv` value, or `cv.get_n_splits()` for a supplied splitter. A subset must be large enough for that splitter. For regression, require `m >= n_splits`. For classification, only subsets containing at least `n_splits` observations from every global class are admissible. Sample uniformly from the admissible subsets for the selected `m`; do not silently score an invalid subset. If no admissible subset exists, fail during `fit` with a clear validation error.

### Neighbourhood size

Use:

~~~
values = range(1, k_max + 1)
~~~

where the minimum CV training-fold size is:

~~~
n_train_min = min(
    len(train_indices)
    for train_indices, _ in cv_splitter.split(X_subset, y_subset)
)
~~~

Then:

~~~
k_max = min(n_train_min, subset_size)
~~~

when max_neighbors=None, otherwise:

~~~
k_max = min(max_neighbors, n_train_min, subset_size)
~~~

The prediction module asks for one neighbourhood size. For normalized position `u`, the neighbourhood-specific form is:

$$
u=\frac{k-1}
{k_{\max}-1}.
$$

Sample beta and cutoff using the same generic prior, generate the logistic distribution, and sample one neighbourhood size.

The default `max_neighbors=None` means:

$$
k\le n_{\mathrm{train,min}}\le m.
$$

The neighbourhood prior must be drawn only after the subset and its CV splitter are known. If `k_max < 1`, reject the draw or fail validation rather than constructing an invalid model.

The prior implementation must not contain special-case logic for any of these uses. It receives only ordered values and returns one sampled scale.

## Dependency injection and architecture

Use dependency injection. The top-level estimator should accept a scale-prior object or scale-prior configuration, for example:

~~~
scale_prior=LogisticScalePrior(
    beta_shape=2.0,
    beta_scale=1.0,
)
~~~

The same prior object should be passed to:

- the representation sampler;
- the subset sampler;
- the neighbourhood sampler.

Do not duplicate the logistic sampling code anywhere else in the package.

This unified scale-selection principle treats projection complexity, dataset complexity, and prediction locality as ordered scales integrated using the same prior mechanism.

The component should be designed so that another prior family can later implement the same interface, for example:

~~~
BetaScalePrior
PowerLawScalePrior
ExponentialPowerScalePrior
~~~

without changing the representation, sampling, scoring, or prediction modules.

## One Monte Carlo model draw

One draw samples:

1. model family when using the mixed mode;
2. representation family for a k-NN draw when using the mixed mode;
3. projection dimension within the selected representation family;
4. projection matrix or identity transform;
5. Gaussian covariance structure for a Gaussian classifier;
6. subset logistic parameters;
7. subset size;
8. subset indices;
9. neighbourhood logistic parameters and size for a k-NN draw.

The transformed subset is then fitted with the selected estimator. Every
sampled parameter is retained.

## Cross-validated pseudo-likelihood

Use the default:

~~~
cv=5
~~~

`cv` must be an integer at least 2 or a scikit-learn splitter. For an integer, construct `StratifiedKFold` for classification and `KFold` for regression, using the estimator's deterministic seed. For a supplied splitter, clone it and validate it for every sampled subset. Classification requires at least `n_splits` samples of every global class in an admissible subset. Regression requires at least `n_splits` observations. The selected `k` is bounded by the smallest training-fold size.

For classification, expand every fold's probability vector to the estimator's global `classes_` ordering before averaging. If `p` is the k-neighbour class frequency vector and `C` is the number of global classes, Dirichlet smoothing uses a documented positive `alpha` parameter:

$$
p_{\mathrm{smooth},j}=\frac{k p_j+\alpha}{k+C\alpha}.
$$

Apply this to all global classes, including classes absent from an individual fold.

For each classification fold:

- fit on the training fold;
- predict probabilities on the validation fold;
- apply Dirichlet smoothing;
- accumulate

$$
\log P(y_i|x_i).
$$

Average over every validation observation. This is a cross-validated scoring utility, not a literal Bayesian likelihood.

Non-k-NN classification draws use the same global class alignment and
Dirichlet smoothing with unit predictive concentration, namely
`(p + alpha) / (1 + C * alpha)`, because linear and Gaussian classifiers do
not have a neighbourhood count.

For regression, define a Gaussian pseudo-likelihood from validation residuals:

$$
\log \tilde P(y_i\mid x_i,\theta)
=-\frac12\left[\log(2\pi\sigma_f^2)
+\frac{(y_i-\hat y_i)^2}{\sigma_f^2}\right].
$$

Estimate `sigma_f^2` from the training fold only as `max(var(y_train), epsilon**2)`, where `epsilon > 0` is an estimator parameter. Average these values over validation observations. This defines a leakage-free regression scoring utility while leaving point prediction as the weighted average of the selected regression estimators.

## Prior probability of a complete model

Each Monte Carlo model draw must store separate prior draws for:

~~~
projection_scale_draw
subset_scale_draw
neighbor_scale_draw
~~~

Each of these records must expose:

~~~
value
beta
cutoff
probability
log_probability
~~~

Mixed-family draws additionally store the selected model family and its prior
probability. Gaussian classification draws store the selected covariance
structure, its probability, and the complete covariance-prior draw.

The complete model record must therefore make it possible to reconstruct:

- the selected model family and, where applicable, covariance structure;
- the selected projection dimension;
- the selected subset size;
- the selected neighbourhood size;
- the prior probability of each selection;
- the latent logistic parameters that generated each selection.

The model sampler should compute:

$$
\log p(\theta)
=
\log p(f)
+ \log p(s_{\mathrm{cov}}\mid f)
+ \log p(d')
+
\log p(m)
+
\log p(k)
+
\log p(S\mid m)
+
\log p(R\mid d'),
$$

where applicable.

For a k-NN draw, store the sum of the three scale-selection log probabilities:

$$
\log p_{\mathrm{scale}}(\theta)
=
\log p(d')
+
\log p(m)
+
\log p(k).
$$

Because subsets are restricted to CV-admissible subsets, store the actual conditional subset probability. For a fixed `m`, let `A_m` be the set of admissible subsets:

$$
\log p(S\mid m)
=-\log |A_m|,\qquad S\in A_m.
$$

Projection-matrix densities may be omitted from posterior weighting if the projection matrix is sampled directly from its prior and the proposal equals the prior, but document this clearly.

## Bayesian model weights

When models are sampled directly from the declared prior, prior probabilities are already represented by sampling frequency. Therefore, the default self-normalized pseudo-posterior weights should be based on the cross-validated pseudo-likelihood alone:

$$
w_i
\propto
\tilde{L}_{\mathrm{CV}}(\theta_i).
$$

Compute the stored `posterior_weight` by applying a numerically stable softmax to the sampled models' `log_importance_weight` values, so the weights are finite and sum to one.

Do not multiply by the prior probability again in this default case, because that would count the prior twice. These are pseudo-posterior weights, not exact Bayesian posterior probabilities, because the scoring utility is cross-validated and averaged.

Nevertheless, store the log prior probabilities for:

- diagnostics;
- reproducibility;
- possible future importance sampling;
- verifying that the sampler behaves as expected.

The implementation should clearly distinguish:

~~~
log_prior
log_proposal
cv_log_pseudo_likelihood
log_importance_weight
posterior_weight
~~~

For ordinary prior sampling:

~~~
log_proposal == log_prior
~~~

and therefore:

~~~
log_importance_weight = cv_log_pseudo_likelihood
~~~

up to an additive normalization constant.

## Prediction by Bayesian model averaging

Prediction is Bayesian model averaging.

For classification:

$$
P(y|x)
=
\sum_i
w_i
P_i(y|x).
$$

For regression:

$$
\hat y
=
\sum_i
w_i
\hat y_i.
$$

## Automatic Monte Carlo and convergence

Support:

~~~
n_estimators="auto"
~~~

Start with 20 models and double the number of models as follows:

~~~
20
40
80
160
320
...
~~~

Reuse previously fitted models and only fit newly required models. Require a positive finite `max_estimators`, defaulting to `1280`. If the tolerance is not reached by `max_estimators`, stop and set `converged_ = False`; never loop indefinitely.

Choose a fixed, reproducibly selected convergence subset. For classifiers, compare the averaged class-probability matrices. For regressors, compare the averaged prediction vectors. In both cases compute:

- maximum absolute change;
- mean absolute change;
- median absolute change.

Do not use the undefined term “support” as a convergence quantity.

Stop when:

~~~
difference <= tolerance
~~~

`difference` is the configured convergence metric, defaulting to maximum absolute change. `tolerance` must be finite and strictly positive.

Store:

- convergence history;
- number of estimators used;
- convergence flag.

## Parallelization

Each Monte Carlo draw is completely independent. Derive one deterministic child seed from `(random_state, draw_index)` and use it for that draw. Never share or mutate one RNG across parallel workers; this guarantees identical results for repeated fits with the same seed and stable reuse during automatic estimator growth.

Use joblib.Parallel for:

- model construction;
- cross-validation;
- final fitting.

Prediction should also be parallelized across sampled models.

Avoid nested parallelism.

Outer layer:

~~~
n_jobs=-1
~~~

Inner k-NN:

~~~
n_jobs=1
~~~

## Stored model information

Each sampled model stores:

- model family and model-family probability;
- representation family;
- representation family probability;
- projection dimension;
- projection parameters;
- subset size;
- subset indices;
- neighbourhood size;
- beta and cutoff for projection dimension;
- beta and cutoff for subset size;
- beta and cutoff for neighbourhood size;
- Gaussian covariance structure and its prior probability when applicable;
- pseudo-log-likelihood;
- posterior weight.

Expose:

~~~
get_model_draws()
~~~

returning a list of dictionaries.

Also expose:

~~~
get_model_masses()
~~~

This aggregates the fitted model weights into diagnostics. The
`model_family` mapping reports global mass for k-NN, linear, and Gaussian
families and sums to one. Under `by_family.knn`, `neighborhood_size` reports
the joint mass of each selected k and sums to the k-NN family mass. Under
`by_family.gaussian`, `covariance_structure` reports the joint mass of
isotropic, diagonal, and full covariance and sums to the Gaussian family mass.
Each section also provides a conditional within-family mapping that sums to
one whenever that family has positive mass. The same structure is available
as the fitted `model_masses_` attribute.

The 2D experiment result exposes the same report as `result.model_masses` for
plotting or tabular diagnostics.

## Public estimator API

Implement:

~~~
BayesianModelAveragingClassifier

BayesianModelAveragingRegressor
~~~

The classifier must support:

~~~
fit()

predict()

predict_proba()

score()

get_model_draws()
~~~

The regressor supports `fit()`, `predict()`, `score()`, and `get_model_draws()`. It does not expose `predict_proba()`, consistent with the scikit-learn estimator API.

The estimator constructors must expose, validate, and preserve through cloning the parameters that affect these rules, including `cv`, `alpha`, `epsilon`, `max_neighbors`, `n_estimators`, `max_estimators`, `tolerance`, `convergence_metric`, `n_jobs`, and `random_state`.

The implementation must comply with the scikit-learn estimator API and support cloning, pipelines, and GridSearchCV. It must be organized so that new representation modules, priors, convergence criteria, or prediction engines can be added without changing the core Bayesian integration logic.

## Tests

Add unit tests covering:

1. probabilities sum to one;
2. all probabilities are finite and positive;
3. the distribution is monotone non-increasing for positive beta;
4. the same random seed produces the same draw;
5. a single allowable value is handled correctly;
6. sampled values always belong to the supplied values;
7. smaller values are sampled more frequently than larger values over many draws;
8. projection, subset, and neighbourhood modules all use the same sampler;
9. no duplicated logistic sampling implementation exists elsewhere;
10. stored log_probability equals the logarithm of the sampled probability;
11. sampled `k` never exceeds the smallest CV training-fold size;
12. invalid subset sizes and class-inadequate subsets fail validation;
13. classifier fold probabilities align to global classes and remain normalized after smoothing;
14. regression pseudo-likelihood scores are finite and use training-fold-only variance estimates;
15. identity projection always returns the original feature dimension;
16. automatic convergence stops at `max_estimators` when tolerance is not met;
17. repeated seeded parallel fits produce identical model draws and weights.

Also include statistical tests over many draws showing that the empirical frequency of sampled values approximates the marginal distribution induced by sampling both the logistic slope and cutoff.

## Documentation and extensibility

Document the conceptual role of LogisticScalePrior as a generic prior over ordered scales.

Emphasize that the package has one unified scale-selection principle:

- projection complexity;
- dataset complexity;
- prediction locality.

All three are treated as ordered scales and integrated using the same prior mechanism. Alternative prior families should be able to implement the same interface without changing the representation, sampling, scoring, or prediction modules.
