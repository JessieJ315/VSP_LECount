import numpy as np
from more_itertools import one

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


def get_children_count(tree: BinaryDecompositionTree, node: Node=None)->dict:
  '''Calculates the children count on each tree-nodes under node given.'''
  if node is None:
    node = tree.get_root()
  if node.is_internal():
    left_children_count = get_children_count(tree, node.left_child)
    right_children_count = get_children_count(tree, node.right_child)
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
  children_count = get_children_count(tree)
  for idx, count in children_count.items():
    tree.nodes[idx].num_children = count
  return tree
