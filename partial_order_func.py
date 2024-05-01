import math
import numpy as np
from more_itertools import one

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


def depth(tr:np.ndarray)->int:
  '''Calculates the depth of a partial order.
  Args:
    tr: The transitive reduction representation of partial order. 
  
  Returns:
    The DAG depth of the partial order.
  '''
  num_items = tr.shape[0] 
  if num_items==1:
    return 1
  col_sum = tr.sum(axis=0)
  if col_sum.sum()==0:
    return 1
  col_sum_indicator = (col_sum==0)
  row_sum = tr.sum(axis=1)
  row_sum_indicator = (row_sum==0)
  free_items = one(np.nonzero(row_sum_indicator & col_sum_indicator))
  num_free_items = len(free_items)
  if num_free_items == num_items:
    return 1
  if num_free_items > 0:
    tr = np.delete(tr,free_items,0)
    tr = np.delete(tr,free_items,1)
    col_sum = tr.sum(axis=0)
    col_sum_indicator = (col_sum==0)
    row_sum = tr.sum(axis=1)
    row_sum_indicator = (row_sum==0)
  tops = one(np.nonzero(col_sum_indicator))
  bottoms = one(np.nonzero(row_sum_indicator))
  if len(bottoms)>len(tops):
    tops = bottoms
  tr = np.delete(tr,tops,0)
  tr = np.delete(tr,tops,1)
  return 1 + depth(tr)


def nle(tr:np.ndarray)->int:
  '''Counts the number of linear extensions of a partial order. 

  Args: 
    tr: The transitive reduction representation of partial order. 

  Returns: 
    The number of linear extension of the partial order. 
  '''
  num_items = tr.shape[0]
  if num_items==1:
    return 1
  col_sum = tr.sum(axis=0)
  if col_sum.sum()==0:
    return int(math.factorial(num_items))
  col_sum_indicator = (col_sum==0)
  row_sum = tr.sum(axis=1)
  row_sum_indicator = (row_sum==0)
  free_items = one(np.nonzero(row_sum_indicator & col_sum_indicator))
  num_free_items = len(free_items)
  if num_free_items == num_items:
    return int(math.factorial(num_items))
  if num_free_items > 0:
    tr = np.delete(tr,free_items,0)
    tr = np.delete(tr,free_items,1)
    col_sum = tr.sum(axis=0)
    col_sum_indicator = (col_sum==0)
    row_sum = tr.sum(axis=1)
    row_sum_indicator = (row_sum==0)
    factor = math.factorial(num_items)/math.factorial(num_items-num_free_items)
  else: 
    factor = 1
  if (num_items - num_free_items) == 2:
    return int(factor)
  tops = one(np.nonzero(col_sum_indicator))
  bottoms = one(np.nonzero(row_sum_indicator))
  if ((len(tops)==1) & (len(bottoms)==1)):
    used_items = np.concatenate((tops, bottoms))
    tr_new = np.delete(tr,used_items,0)
    tr_new = np.delete(tr_new,used_items,1)
    return int(factor*nle(tr_new))
  if len(bottoms)<len(tops):
    tops = bottoms
  count = 0
  for item in tops:
    trr = np.delete(tr,item,0)
    trr = np.delete(trr,item,1)
    count = count + nle(trr)
  return int(factor*count)
