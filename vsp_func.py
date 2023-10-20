import numpy as np
import pandas as pd
import graphviz
from more_itertools import one
from math import comb

class Node:
  '''A node in class BinaryDecompositionTree.

  Attributes:
    index: The node index.
    actor: The actor associated. It is a str for leaf nodes and 'None' for internal nodes. Default: None.
    parent: Its parent node. Default: None.
    left_child: It's left child. Default: None.
    right_child: It's right child. Default: None.
    node_type: The operation type. It takes values 'S' (series-operation) and 'P' (parallel-operation) for internal nodes or 'None' for leaf nodes. Default: None.
    order: The order in series (S) operation. The node is on top if order='+'. The node is at the bottom of the series operation if order='-'. Otherwise, 'None'. Default: None.
    num_children: The number of children below (inclusive). Default: None.
    zombie: Whehter this is a zombie node. Default: False.
  '''
  def __init__(self, index:int=None, actor:str=None, node_type:str=None, order:str=None, num_children:int=None, zombie:bool=False):
    self.index = index
    self.actor = actor
    self.parent = None
    self.left_child = None
    self.right_child = None
    self.node_type = node_type
    self.order = order
    self.num_children = num_children
    self.zombie = zombie
  def show(self)->str:
    '''Prints node information.'''
    parent_index = None if self.parent is None else self.parent.index
    left_child_index = None if self.left_child is None else self.left_child.index
    right_child_index = None if self.right_child is None else self.right_child.index
    print(f'node(index={self.index}, actor={self.actor}, parent_index={parent_index}, left_child_index={left_child_index}, right_child_index={right_child_index}, node_type={self.node_type}, order={self.order}, num_children={self.num_children})')
  def is_leaf(self)->bool:
    '''Tests whether this is a leaf node.'''
    return (not self.zombie) & (self.left_child is None)
  def is_root(self)->bool:
    '''Tests whether this is a root.'''
    return (not self.zombie) & (self.parent is None)
  def is_internal(self)->bool:
    '''Tests whether this is an internal node.'''
    return (not self.zombie) & (self.left_child is not None)

class BinaryDecompositionTree:
  '''A binary decomposition tree class with both internal and leaf nodes.'''
  def __init__(self):
    self.nodes = []
  def add_node(self, index:int=None, actor:int=None, node_type:str=None, order:str=None, parent:Node=None, left_child:Node=None, right_child:Node=None):
    '''Adds a node to the binary decomposition tree (BDT).'''
    node = Node(index=index, actor=actor,node_type=node_type,order=order)
    node.parent=parent
    node.left_child=left_child
    node.right_child=right_child
    self.nodes.append(node)
  def get_root(self)->Node:
    '''Fetches the root in the BDT.'''
    return one([node for node in self.nodes if node.is_root()])
  def show(self, node_index:bool):
    '''Visualises the binary decomposition tree.
    Args: 
      node_index: Whether to show the node index or the actor & node type. 
    '''
    dot = graphviz.Digraph()
    root = self.get_root()
    if node_index:
      dot.node(str(root.index))
    else:
      root_key = str(root.node_type if root.is_internal() else root.actor)+str('' if root.order is None else root.order)
      dot.node(str(root.index),label=root_key)
    def _add_nodes_edges(node):
      if node.is_internal():
        if node_index:
          dot.node(str(node.left_child.index))
          dot.node(str(node.right_child.index))
        else: 
          key_left = str(node.left_child.node_type if node.left_child.is_internal() else node.left_child.actor)+str('' if node.left_child.order is None else node.left_child.order)
          key_right = str(node.right_child.node_type if node.right_child.is_internal() else node.right_child.actor)+str('' if node.right_child.order is None else node.right_child.order)
          dot.node(str(node.left_child.index),label=key_left)
          dot.node(str(node.right_child.index),label=key_right)
        dot.edge(str(node.index), str(node.left_child.index))
        dot.edge(str(node.index), str(node.right_child.index))
        _add_nodes_edges(node.left_child)
        _add_nodes_edges(node.right_child)
    _add_nodes_edges(root)
    return(dot)

def _get_children_count(tree: BinaryDecompositionTree, node: Node=None)->dict:
  '''Calculates the children count on each tree-nodes under node given.'''
  if node is None:
    node = tree.get_root()
  if node.is_internal():
    left_children_count = _get_children_count(tree, node.left_child)
    right_children_count = _get_children_count(tree, node.right_child)
    node_children_count = {node.index: left_children_count[node.left_child.index]+right_children_count[node.right_child.index]}
    children_count = {**node_children_count, **left_children_count, **right_children_count}
  else:
    children_count = {node.index: 1}
  return children_count


def random_tree(actors:list, probability_sndoe:float=0.5)->BinaryDecompositionTree:
  '''Builds a random BDT with actors and S-node probability.
  
  Args: 
    actors: The actors in leaf nodes.
    probability_snode: The probability of obtaining an S-internal node.

  Returns: 
    A random binary decomposition tree.
  '''
  num_actors = len(actors)
  
  tree = BinaryDecompositionTree()
  # define root.
  tree.add_node(index=0)
  # add first two actors.
  tree.add_node(index=1, actor=actors[0], parent=tree.nodes[0])
  tree.add_node(index=2, actor=actors[1], parent=tree.nodes[0])
  tree.nodes[0].left_child=tree.nodes[1]
  tree.nodes[0].right_child=tree.nodes[2]
  # randomly add remaining actors. 
  num_nodes = 3
  if num_actors > 2:
    for actor in actors[2:]:
      node_idx = np.random.choice(num_nodes)
      new_internal_idx = num_nodes
      new_leaf_idx = num_nodes+1
      tree.add_node(index=new_internal_idx, left_child=tree.nodes[node_idx])
      tree.add_node(index=new_leaf_idx, actor=actor, parent=tree.nodes[new_internal_idx])
      tree.nodes[new_internal_idx].right_child = tree.nodes[new_leaf_idx]
      
      if not tree.nodes[node_idx].is_root():
        parent_node = tree.nodes[node_idx].parent
        if parent_node.left_child == tree.nodes[node_idx]:
          parent_node.left_child = tree.nodes[new_internal_idx]
        else: 
          parent_node.right_child = tree.nodes[new_internal_idx]
        tree.nodes[new_internal_idx].parent = parent_node

      tree.nodes[node_idx].parent = tree.nodes[new_internal_idx]
      num_nodes += 2
  # assign node types to internal nodes.
  for node in tree.nodes:
    if node.is_internal():
      node_type = np.random.choice(a=['S','P'],replace=False,p=[probability_sndoe,1-probability_sndoe])
      node.node_type=node_type
      if node_type=='S':
        left_child = np.random.choice([True, False])
        node.left_child.order = '+' if left_child else '-'
        node.right_child.order = '-' if left_child else '+'
  # add children counts
  children_count = _get_children_count(tree)
  for idx, count in children_count.items():
    tree.nodes[idx].num_children = count
  return tree


def _tree2vsp(tree: BinaryDecompositionTree, node: Node=None)->pd.DataFrame:
  '''Generates the adjacency matrix (transitive closure) for a VSP from a binary decomposition tree.'''
  if node is None:
    node = tree.get_root()
  if node.is_leaf():
    vsp = pd.DataFrame([0], index=[node.actor], columns=[node.actor])
    return vsp

  if not node.left_child.order=='-':
    upper_vsp1 = _tree2vsp(tree, node.left_child)
  else:
    upper_vsp1 = _tree2vsp(tree, node.right_child)
  if not node.right_child.order=='+':
    lower_vsp2 = _tree2vsp(tree, node.right_child)
  else:
    lower_vsp2 = _tree2vsp(tree, node.left_child)

  num_actors_upper = upper_vsp1.shape[0]
  num_actors_lower = lower_vsp2.shape[0]
  is_snode = int(node.node_type == 'S')
  upper_vsp2 = pd.DataFrame(np.full((num_actors_upper,num_actors_lower),is_snode))
  lower_vsp1 = upper_vsp2.transpose()*0
  upper_vsp = pd.concat([upper_vsp1.reset_index(drop=True), upper_vsp2.reset_index(drop=True)], axis=1)
  lower_vsp = pd.concat([lower_vsp1.reset_index(drop=True), lower_vsp2.reset_index(drop=True)], axis=1)
  lower_vsp.columns = upper_vsp.columns
  vsp = pd.concat([upper_vsp, lower_vsp], axis=0)

  names = upper_vsp1.index.append(lower_vsp2.index)
  vsp.columns = names
  vsp.index = names
  vsp = vsp.sort_index(axis=0)
  vsp = vsp.sort_index(axis=1)

  return vsp


def tree2vsp(tree:BinaryDecompositionTree) -> np.ndarray:
  '''Generates the adjacency matrix (transitive closure) for a VSP from a binary decomposition tree.

  Args:
    tree: The binary decomposition tree.

  Returns:
    A transitively-closed adjacency matrix.
  '''
  vsp = _tree2vsp(tree)
  return vsp.to_numpy()


def nle_tree(tree:BinaryDecompositionTree, node:Node=None)->int:
  '''Counts the linear extensions for a VSP represented by the binary decomposition tree.

  Args:
    tree: The binary decomposition tree representing a VSP. 
    node: The node of interest. Default: None (root).

  Returns:
    The number of linear extensions of the VSP. 
  '''
  if node is None: 
    node = tree.get_root()
  if node.is_leaf():
    return 1
  
  nle_left = nle_tree(tree, node.left_child)
  nle_right = nle_tree(tree, node.right_child)
  total = nle_left * nle_right
  if node.node_type=='P':
    total = total * comb(node.left_child.num_children + node.right_child.num_children, node.left_child.num_children)
  
  return total
