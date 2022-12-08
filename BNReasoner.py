from typing import Union
from BayesNet import BayesNet
import networkx as nx
import pandas as pd
from itertools import combinations

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

        # Get the structure of the BN
        self.structure = self.bn.structure
        print(f'structure: {self.structure}')

        # Get a list of all variables in the BN
        self.all_variables = self.bn.get_all_variables()
        print(f'all_variables: {self.all_variables}')

        # Get the interaction graph
        self.G = self.bn.get_interaction_graph()
        print(f'interaction_graph: {self.G}')

        # Get a figure of the structure of the BN
        # self.drawed_structure = self.bn.draw_structure()
        # print(f'drawed_structure: {self.drawed_structure}')

        # Get the conditional probability tables of all the variables in the BN
        # self.all_cpts = self.bn.get_all_cpts()
        # print(f'all_cpts: {self.all_cpts}')

    def EdgePruning(self, e):
        # print("Edge pruning:")

        for var, val in e.items():
            # print(f"Var: {var}")
            cpt = self.bn.get_cpt(var)
            cpt_update = self.bn.get_compatible_instantiations_table(pd.Series({var: val}), cpt)
            # print(cpt_update)
            # print(cpt)
            self.bn.update_cpt(var, cpt_update)
            # print(len(e.items()))
            # print(e.values(), e.keys())
            print("Prune all edges from", var, "to all other nodes")
            if not self.bn.get_children(var):
                pass
            else:
                for child in self.bn.get_children(var):
                    # prune edge between evident node and its child
                    self.bn.del_edge((var, child))

                    # update CPT
                    cpt = self.bn.get_cpt(child)
                    cpt_update = self.bn.get_compatible_instantiations_table(pd.Series({var: val}), cpt)
                    self.bn.update_cpt(child, cpt_update)

    def NodePruning(self, Q, e):
        # node pruning
        # print("Node pruning:")
        pruning = True
        while pruning:
            pruning = False
            for variable in self.all_variables:
                # print("Variable:", variable)

            # remove leaf node when it is not influencing Q or e
                if not self.bn.get_children(variable):
                    print(set(Q))
                    print(variable)
                    if variable not in set(Q) and variable not in set(e.keys()):
                        # delete leaf node and check if new leaf nodes are created
                        print("Delete leaf node", variable)
                        self.bn.del_var(variable)
                        pruning = True

    def NetworkPruning(self, Q: list, e: dict) -> None:
        """
        Given a set of query variables Q and evidence e, node- and edge-prunes the Bayesian network so that queries of
        the form P(Q|E) can still be correctly calculated.
        :param Q: list of query variables
        :param e: a dictionary of evidence variables with their respective values
        :return: node- and edge-pruned Bayesian network
        """
        self.EdgePruning(e)
        #if evidence is given, prune the nodes
        if e:
            self.NodePruning(Q, e)

    def d_Seperation(self, X, Y, Z) -> bool:
        """
        Given three sets of variables X, Y, and Z, determines whether X is d-separated of Y given Z.

        :param X: set of variables X
        :param Y: set of variables Y
        :param Z: set of variables Z
        :return: True/False
        """
        if nx.is_directed_acyclic_graph(self.G) == False:
            print("The test only works on DAG's.")
            return

        else:
            d_seperated = nx.d_separated(self.G, X, Y, Z)
            return d_seperated

    def Independence(self):
        """
        Given three sets of variables X, Y, and Z, determines whether X is independent of Y given Z.

        :param X: set of variables X
        :param Y: set of variables Y
        :param Z: set of variables Z
        :return: True/False
        """
        return

    def SumOutVar(self, X, cpt):
        """
        Given a factor and a variable X, computes the CPT in which X is summed-out.

        :param X: set of variables X
        :param cpt: cpt to be filtered
        :return: True/False
        """
        if X not in cpt.columns:
            return f"The variable '{X}' is not in the cpt and can therefor not be summed-out."

        else:
            self.new_cpt = cpt.copy()

            # Delete the column of the variable X that is summed out
            del self.new_cpt[X]

            # Get a list of all variables in the cpt
            self.variables_in_cpt = [value for value in list(self.new_cpt.columns) if value in self.all_variables]

            # Sum out
            self.new_cpt = self.new_cpt.groupby(self.variables_in_cpt)['p'].sum().reset_index()

            return self.new_cpt

    def MaxOutVar(self, X, cpt):
        """
        Given a factor and a variable X, computes the CPT in which X is maximized-out. Also keeps track of which
        instantiation of X led to the maximized value

        :param X: set of variables X
        :param cpt: cpt to be filtered
        :return: True/False
        """
        if X not in cpt.columns:
            return f"The variable '{X}' is not in the cpt and can therefor not be maximized-out."

        else:
            self.new_cpt = cpt.copy()

            # Delete the column of the variable X that is maximized out
            del self.new_cpt[X]

            # Get a list of all variables in the cpt
            self.variables_in_cpt = [value for value in list(self.new_cpt.columns) if value in self.all_variables]

            # Maximize-out
            self.new_cpt = self.new_cpt.groupby(self.variables_in_cpt)['p'].max().reset_index()

            # Extract instantiation of X from original cpt
            #self.new_cpt[X] = [None, None, None, None]

            return self.new_cpt

    def MultFactors(self, f, g):
        """
        Given two factors f and g, compute the multiplied factor h=fg.

        :param f: factor f
        :param g: factor g
        :return: h
        """
        return f*g

    def EliminationOrder(self, heuristic):
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X based on
        the min-degree heuristics or the min-fill heuristics.
        :param heurstic: min_degree or min_fill
        :return: order for the elimination of X as a list of variables
        """
        # Get the interaction graph of BN
        self.edges = list(self.G.edges)
        self.nodes = list(self.G.nodes)
        self.degree = self.G.degree

        # Draw the interaction graph of BN
        # nx.draw(self.G)
        # plt.show()

        # Create a list with the elimination order
        self.elimination_order = []

        if len(self.nodes) == 0:
            print('There are no nodes.')
            return

        elif len(self.nodes) == 1:
            print('There is only one node.')
            self.elimination_order.append(self.nodes[0])
            return self.elimination_order

        else:
            # Determine elimination order depending on the heuristic applied
            if heuristic == "MinFill":
                self.elimination_order = self.MinFill()
                print("The elimination order with 'MinFill' heuristic:")
                return self.elimination_order

            elif heuristic == "MinDegree":
                self.elimination_order = self.MinDegree()
                print("The elimination order with 'MinDegree' heuristic:")
                return self.elimination_order

            else:
                print("Heuristic has to be either 'MinFill' or 'MinDegree'.")
                return

    def MinFill(self):
        '''
        Elimination ordering by recursively favoring variables with fewer fill-ins required.
        :return: list of nodes in elimination order in the form ['var1, 'var2', ...]
        '''
        # Create a list with the elimination order
        self.elimination_order = []

        # Keep eliminating nodes as long as there are nodes left
        while len(self.G.nodes) > 0:
            self.fill_ins_per_node = {}

            # Get a list of the required fill ins per node if that node is eliminated
            for node in self.G.nodes:
                self.fill_ins_per_node[node] = []

                # Get a list of the neighbors per node
                self.neighbors_per_node = list(self.G.neighbors(node))

                # Get a list of all possible edges between those nodes
                self.possible_edges = list(combinations(self.neighbors_per_node, 2))

                # If a possible edge is not in the existing edges of the graph, a fill-in is required
                for possible_edge in self.possible_edges:
                    if possible_edge in self.G.edges:
                        continue
                    else:
                        self.fill_ins_per_node[node].append(possible_edge)

            # Eliminate the node with the minimum fill-ins required
            self.eliminated_node = min(self.fill_ins_per_node, key=self.fill_ins_per_node.get)
            self.elimination_order.append(self.eliminated_node)
            self.G.remove_node(self.eliminated_node)

            # Add the edges that are filled after elimination of that node
            for fill_in in self.fill_ins_per_node[self.eliminated_node]:
                self.G.add_edge(fill_in)

        return self.elimination_order

    def MinDegree(self):
        '''
        Elimination ordering by recursively favoring variables with fewer neighbors.
        :return: list of nodes in elimination order in the form ['var1, 'var2', ...]
        '''
        # Create a list with the elimination order
        self.elimination_order = []

        # Keep eliminating nodes as long as there are nodes left
        while len(self.G.nodes) > 0:
            self.degrees_per_node = {}

            # Count the number of degrees per node
            for node in self.G.nodes:
                self.degrees_per_node[node] = self.G.degree(node)

            # Eliminate the node with the minimum degree
            self.eliminated_node = min(self.degrees_per_node, key=self.degrees_per_node.get)
            self.elimination_order.append(self.eliminated_node)
            self.G.remove_node(self.eliminated_node)

        return self.elimination_order

    def VarElimination(self, X):
        """
        Sums out a set of variables X by using variable elimination

        :param X: set of variables that is summed out
        :return:
        """
        return X



def test_BN(filename, heuristic, variablename1, variablename2, Q, e):

    # Create a BN reasoner (outer class)
    BNReasoner_ = BNReasoner(filename)

    # Create a BN (inner class)
    org_BayesNet_ = BNReasoner_.bn

    # pruned_BayesNet_ = BNReasoner_.NetworkPruning(Q, e)
    # print(pruned_BayesNet_)

    # Get the conditional probability table of variable 1 in the BN
    cpt_variable1 = org_BayesNet_.get_cpt(variablename1)
    print(f'cpt_{variablename1}: \n {cpt_variable1}')

    # Sum-out a variable 2 of the cpt of variable 1 in the BN
    sum_out_variable2 = BNReasoner_.SumOutVar(variablename2, cpt_variable1)
    print(sum_out_variable2)

    # Maximize-out variable 2 of the cpt of variable 1 in the BN
    max_out_variable2 = BNReasoner_.MaxOutVar(variablename2, cpt_variable1)
    print(max_out_variable2)

    # # Check whether node set X and y are d-Separated by Z
    # X = 'dog-out'
    # Y = 'family-out'
    # Z = 'hear-bark'
    # d_separated = BNReasoner_.d_Seperation(X,Y,Z)
    # print(d_separated)

    # # Get the elimination order of the variables in the BN
    order = BNReasoner_.EliminationOrder(heuristic)

    print(order)

    return

#BN_dog = test_BN('testing/dog_problem.BIFXML', heuristic = 'MinFill', variablename1 = 'dog-out', variablename2 = 'family-out', Q = [], e = {})
BN_lecture_example = test_BN('testing/lecture_example.BIFXML', heuristic = 'MinFill', variablename1 = 'Rain?', variablename2 = 'Winter?', Q = ['Wet Grass?'], e = {"Rain?": False, "Winter?": True})

