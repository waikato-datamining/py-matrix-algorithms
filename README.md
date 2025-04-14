# py-matrix-algorithms

Python library of 2-dimensional matrix algorithms.

### Algorithms

Available:

* [PCA](https://web.archive.org/web/20160630035830/http://statmaster.sdu.dk:80/courses/ST02/module05/module.pdf)
* [CCAFilter](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.30.16)
* [GLSW](http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering#GLSW_Algorithm)
* [EPO](http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering#External_Parameter_Orthogonalization_.28EPO.29) (External Parameter Orthogonalization)
* [YGradientGLSW](http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering#GLSW_Algorithm)
* [YGradientEPO](http://wiki.eigenvector.com/index.php?title=Advanced_Preprocessing:_Multivariate_Filtering#External_Parameter_Orthogonalization_.28EPO.29)
* [FastICA](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf)
* [CCARegression]()
* [DIPLS](https://pubs.acs.org/doi/10.1021/acs.analchem.8b00498)
* [KernelPLS](http://www.plantbreeding.wzw.tum.de/fileadmin/w00bdb/www/kraemer/icml_kernelpls.pdf)
* [NIPALS](http://www.statsoft.com/textbook/partial-least-squares/#NIPALS)
* [OPLS](https://www.r-bloggers.com/evaluation-of-orthogonal-signal-correction-for-pls-modeling-osc-pls-and-opls/) (orthogonal signal correction)
* [PLS1](https://web.archive.org/web/20081001154431/http://statmaster.sdu.dk:80/courses/ST02/module07/module.pdf)
* [PRM]()
* [SIMPLS](http://www.statsoft.com/textbook/partial-least-squares/#SIMPLS)
* [SparsePLS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2810828/)
* [VCPLS](http://or.nsfc.gov.cn/bitstream/00001903-5/485833/1/1000013952154.pdf)

Planned:

* [rPLS](https://www.researchgate.net/publication/259536250_Recursive_weighted_partial_least_squares_rPLS_An_efficient_variable_selection_method_using_PLS)
* [iPLS](https://www.researchgate.net/publication/247776629_Interval_Partial_Least-Squares_Regression_iPLS_A_Comparative_Chemometric_Study_with_an_Example_from_Near-Infrared_Spectroscopy)
* [PLS2](https://web.archive.org/web/20160702070233/http://statmaster.sdu.dk/courses/ST02/module08/module.pdf)
* [mwPLS]()
* [biPLS](https://www.academia.edu/14468430/Sequential_application_of_backward_interval_partial_least_squares_and_genetic_algorithms_for_the_selection_of_relevant_spectral_regions)

### Filters

* [Downsample]()

### Transformations

* [Center]()
* [Log]()
* [MultiplicativeScatterCorrection]()
* [Passthrough]()
* [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
* [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
* [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
* [RowNorm]()
* [Savitzky-Golay]()
* [Savitzky-Golay 2]()
* [Standardize]()
* [FFT]()

## Installation

Install from the PyPI index:

```bash
pip install wai_ma
```

## Examples

To use an algorithm:

```python
# For example, the SIMPLS algorithm
from wai.ma.algorithm.pls import SIMPLS

# Create the algorithm object
simpls: SIMPLS = SIMPLS()
simpls.num_coefficients = 5  # Set any parameters to desired values

# Get some training data
from wai.ma.core.matrix import Matrix, helper
train_predictors: Matrix = helper.read("train_predictors.csv", False, ",")
train_response: Matrix = helper.read("train_response.csv", False, ",")

# Train the algorithm
simpls.initialize(train_predictors, train_response)

# Get some test data
test_predictors: Matrix = helper.read("test_predictors.csv", False, ",")

# Get the predicted response
test_predicted_response: Matrix = simpls.predict(test_predictors)

# Compare with actual response, or use in some other way...
```

To use a transformation:

```python
# For example, the Savitzky-Golay transformation
from wai.ma.transformation import SavitzkyGolay

# Create the transformation object
sg: SavitzkyGolay = SavitzkyGolay()
sg.num_points_right = sg.num_points_left = 4  # Set any parameters to desired values

# Get some configuration data
from wai.ma.core.matrix import Matrix, helper
train: Matrix = helper.read("train.csv", False, ",")

# Configure the transformation (if not configured, will automatically
# do so with the first set of data given to transform). 
sg.configure(train)

# Get some test data
test: Matrix = helper.read("test.csv", False, ",")

# Transform the test data
test_transformed: Matrix = sg.transform(test)

# Further use of transformed data...
```
