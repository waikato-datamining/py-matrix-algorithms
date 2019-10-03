Changelog
=========

0.0.1 (2019-07-15)
-------------------

- Initial release

0.0.2 (2019-07-25)
-------------------

- Modified RowNorm to reconfigure itself for every matrix it sees.

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

0.0.4 (????-??-??)
-------------------

- Standardize now implements Serialisable interface.
- Serialisable can now serialise strings.
