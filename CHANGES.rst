Changelog
=========

0.0.9 (2025-04-14)
------------------

- project name uses underscores now
- lifted `numpy<1.22` restriction
- `np.INF`/`np.NINF` to `np.inf`/`-np.inf`
- updated regression files


0.0.8 (2023-12-19)
------------------

- Updated/relaxed dependencies
- Using `int` instead of `np.int` now

0.0.7 (2020-06-29)
------------------

- Fixed Downsample so it filters columns instead of rows.

0.0.6 (2019-11-18)
------------------

- Added tranformations based on scikit-learn: PowerTransformer, QuantileTranformer and RobustScalar.

0.0.5 (2019-11-01)
------------------

- Added Log transformation.
- Updated to wai.common v0.0.17

0.0.4 (2019-10-04)
-------------------

- Standardize now implements Serialisable interface.
- Serialisable can now serialise strings.

0.0.3 (2019-08-21)
-------------------

- Added FFT as a transformation.
- Fixed bug in initialize(Matrix, Matrix) for YGradientEPO and OPLS where any error message wasn't being returned.
- Removed utilities to wai.common, and import them back to here.
- read/write in matrix.helper can now accept open file handles as well as file names.
- Calling inverseTransform on SavitzkyGolay now raises a NotImplementedError, instead of silently passing.
- Testing infrastructure is now imported from wai.test.
- Added a Serialisable interface, which allows objects to write their state to a binary stream. SavitzkyGolay
  and SIMPLS implement this.
- Added Equidistance resampling filter.

0.0.2 (2019-07-25)
-------------------

- Modified RowNorm to reconfigure itself for every matrix it sees.

0.0.1 (2019-07-15)
-------------------

- Initial release