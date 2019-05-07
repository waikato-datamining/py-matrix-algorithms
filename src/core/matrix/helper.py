from numbers import Number
from typing import List, Set

import numpy as np

from core.error import InvalidShapeError
from core.matrix import Matrix, real


def delete_col(data: Matrix, col: int) -> Matrix:
    return delete_cols(data, [col])


def delete_cols(data: Matrix, cols: List[int]) -> Matrix:
    cols = [col for col in range(data.num_columns()) if col not in cols]
    return data.get_sub_matrix((0, data.num_rows()), cols)


def row_as_vector(data: Matrix, row: int) -> Matrix:
    return data.get_row(row)


def mean(mat: Matrix, index: int, column: bool = True) -> real:
    if column:
        return column_mean(mat, index)
    else:
        return row_mean(mat, index)


def stdev(mat: Matrix, index: int, column: bool = True) -> real:
    if column:
        return column_stdev(mat, index)
    else:
        return row_stdev(mat, index)


def column_as_vector(m: Matrix, column_index: int) -> Matrix:
    return m.get_column(column_index)


def euclidean_distance(x: Matrix, y: Matrix, squared: bool) -> Matrix:
    xx: Matrix = row_norms(x, True)
    yy: Matrix = row_norms(y, True)

    distances: Matrix = x.mul(y.transpose())
    distances = distances.mul(-2)
    distances = distances.add(xx)
    distances = distances.add(yy)

    if x == y:
        fill_diagonal(distances, 0)

    if squared:
        return distances
    else:
        return distances.sqrt()


def row_norms(x: Matrix, squared: bool) -> Matrix:
    # TODO
    raise NotImplementedError


def equal(m1: Matrix, m2: Matrix, epsilon: real = real(0)) -> bool:
    if not m1.same_shape_as(m2):
        return False

    for row in range(m1.num_rows()):
        for col in range(m1.num_columns()):
            if np.abs(m1.get(row, col) - m2.get(row, col)) > epsilon:
                return False

    return True


def read(filename: str, header: bool, separator: str) -> Matrix:
    # TODO
    raise NotImplementedError


def to_lines(data: Matrix, header: bool, separator: str, num_dec: int, scientific: bool = False) -> List[str]:
    result: List[str] = []

    if header:
        result.append(separator.join(['col' + str(j + 1) for j in range(data.num_columns())]))

    data_to_string(data, separator, num_dec, result, scientific)

    return result


def data_to_string(data: Matrix, separator: str, num_dec: int, result: List[str], scientific: bool):
    # TODO
    raise NotImplementedError


def write(data: Matrix, filename: str, header: bool, separator: str, num_dec: int, scientific: bool = False):
    # TODO
    raise NotImplementedError


def to_string(data: Matrix, header: bool = True, separator: str = '\t', num_dec: int = 6) -> str:
    return '\n'.join(to_lines(data, header, separator, num_dec))


def dim(m: Matrix) -> str:
    return str(m.num_rows()) + ' x ' + str(m.num_columns())


def inverse(m: Matrix) -> Matrix:
    return m.inverse()


def throw_invalid_shapes(m1: Matrix, m2: Matrix):
    raise InvalidShapeError("Invalid matrix multiplication. Shapes " +
                            m1.shape_string() +
                            " and " +
                            m2.shape_string() +
                            " do not match.")


def row_mean(mat: Matrix, index: int) -> real:
    return row_means(mat).get(index, 0)


def row_means(mat: Matrix) -> Matrix:
    result = Matrix(np.mean(mat.data, 1, real))
    result = result.transpose()
    return result


def column_mean(mat: Matrix, index: int) -> real:
    return column_means(mat).get(0, index)


def column_means(mat: Matrix) -> Matrix:
    return Matrix(np.mean(mat.data, 0, real))


def row_stdev(mat: Matrix, index: int) -> real:
    return row_stdevs(mat).get(index, 0)


def row_stdevs(mat: Matrix) -> Matrix:
    stdevs = Matrix(np.std(mat.data, 1, real,ddof=1))
    stdevs = stdevs.transpose()
    return stdevs


def column_stdev(mat: Matrix, index: int) -> real:
    return column_stdevs(mat).get(0, index)


def column_stdevs(mat: Matrix) -> Matrix:
    return Matrix(np.std(mat.data, 0, real, ddof=1))


def fill_diagonal(mat: Matrix, value: Number):
    i = 0
    while i < mat.num_rows() and i < mat.num_columns():
        mat.data[i][i] = real(value)
        i += 1


def must_be_row_vector(vector: 'Matrix'):
    if not vector.is_row_vector():
        raise ValueError('Must be a row vector')


def must_be_column_vector(vector: 'Matrix'):
    if not vector.is_column_vector():
        raise ValueError('Must be a column vector')


def must_be_vector(vector: 'Matrix'):
    if not vector.is_vector():
        raise ValueError('Must be a vector')


def must_be_same_shape(m1: Matrix, m2: Matrix):
    dimensions_must_match(m1, m2, rows_to_rows=True, columns_to_columns=True)


def dimensions_must_match(m1: Matrix, m2: Matrix, *,
                          rows_to_rows: bool = False,
                          rows_to_columns: bool = False,
                          columns_to_rows: bool = False,
                          columns_to_columns: bool = False):
    checks: Set[bool] = {not rows_to_rows or m1.num_rows() == m2.num_rows(),
                         not rows_to_columns or m1.num_rows() == m2.num_columns(),
                         not columns_to_rows or m1.num_columns() == m2.num_rows(),
                         not columns_to_columns or m1.num_columns() == m2.num_columns()}

    if False in checks:
        raise InvalidShapeError('', m1, m2)