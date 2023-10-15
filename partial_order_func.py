import numpy as np

def random_dag(num_actors:int, edge_probability:float, dag:bool=True)->np.ndarray:
  '''generates a random DAG (Brightwell distribution).

  Args:
    num_actors: The number of actors in DAG.
    edge_probability: The probability of having an edge, in [0,1].
    dag: Whether to output a DAG. Default: True.

  Returns:
    An adjacency matrix.
  '''
  probabilities = np.random.uniform(size=num_actors*num_actors)
  elements = (probabilities<edge_probability).astype(int)
  binary_matrix = elements.reshape((num_actors,num_actors))
  if dag:
    return np.triu(binary_matrix,k=1)
  else:
    np.fill_diagonal(binary_matrix,0)
    return binary_matrix

def _matrix_exp_iterativ(matrix:np.ndarray)->np.ndarray:
  '''Iterator over a matrix.'''
  num_row = matrix.shape[0]
  matrix_temp1 = matrix
  matrix_temp2 = matrix
  idx = 1
  while idx < num_row:
    matrix_temp2 = np.matmul(matrix_temp2, matrix)
    matrix_temp1 = matrix_temp1 + matrix_temp2
    idx = idx + 1
  return matrix_temp1

def transitive_closure(partial_order:np.ndarray)->np.ndarray:
  '''Computes the transitive closure of an adjacency matrix.'''
  partial_order_updated = _matrix_exp_iterativ(partial_order)
  partial_order_updated = (partial_order_updated > 0).astype(int)
  np.fill_diagonal(partial_order_updated,0)
  return partial_order_updated

def transitive_reduction(partial_order:np.ndarray)->np.ndarray:
  '''Computes the transitive reduction of an adjacency matrix.'''
  num_row = partial_order.shape[0]
  partial_order_closure = transitive_closure(partial_order)
  for y in range(num_row):
    for x in range(num_row):
      if partial_order_closure[x,y]!=0:
        for j in range(num_row):
          if (partial_order_closure[y,j]!=0):
            partial_order_closure[x,j]=0
  return partial_order_closure