from typing import Union
from BayesNet import BayesNet
import matplotlib.pyplot as plt
import networkx as nx

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

    def network_pruning(self, instantiation, cpt):
        """
        Given a set of query variables Q and evidence e, node- and edge-prunes the Bayesian network so that queries of
        the form P(Q|E) can still be correctly calculated.

        :param instantiation: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :param cpt: cpt to be filtered
        :return: node- and edge-pruned Bayesain network
        """

        # Creates and returns a new factor in which all probabilities which are incompatible with the instantiation
        # passed to the method to 0.
        self.new_cpt = self.bn.reduce_factor(instantiation, cpt)

        ## Get all the entries of a CPT which are compatible with the instantiation
        # compatible_instantiations_table = get_compatible_instantiations_table(instantiation, cpt)

        return self.new_cpt

    def d_Seperation(self, X, Y, Z):
        """
        Given three sets of variables X, Y, and Z, determines whether X is d-separated of Y given Z.

        :param X: set of variables X
        :param Y: set of variables Y
        :param Z: set of variables Z
        :return: True/False
        """
        return

    def Independence(self):
        """
        Given three sets of variables X, Y, and Z, determines whether X is independent of Y given Z.

        :param X: set of variables X
        :param Y: set of variables Y
        :param Z: set of variables Z
        :return: True/False
        """
        return

    def SumOutVar(self, X, cpt, all_variables):
        """
        Given a factor and a variable X, computes the CPT in which X is summed-out.

        :param X: set of variables X
        :param cpt: cpt to be filtered
        :return: True/False
        """
        self.new_cpt = cpt.copy()

        # Delete the column of the variable X that is summed out
        del self.new_cpt[X]

        # Get a list of all variables in the cpt
        self.variables_in_cpt = [value for value in list(self.new_cpt.columns) if value in all_variables]

        # Sum out
        self.new_cpt = self.new_cpt.groupby(self.variables_in_cpt)['p'].sum().reset_index()
        return self.new_cpt

    def MaxOutVar(self, X, cpt, all_variables):
        """
        Given a factor and a variable X, computes the CPT in which X is maximized-out. Also keeps track of which
        instantiation of X led to the maximized value

        :param X: set of variables X
        :param cpt: cpt to be filtered
        :return: True/False
        """
        self.new_cpt = cpt.copy()

        # Delete the column of the variable X that is maximized out
        del self.new_cpt[X]

        # Get a list of all variables in the cpt
        self.variables_in_cpt = [value for value in list(self.new_cpt.columns) if value in all_variables]

        # Maximize-out
        self.new_cpt = self.new_cpt.groupby(self.variables_in_cpt)['p'].max().reset_index()

        return self.new_cpt

    def MultFactors(self, f, g):
        """
        Given two factors f and g, compute the multiplied factor h=fg.

        :param f: factor f
        :param g: factor g
        :return: h
        """
        return f*g

    def MinFill(self, nodes, edges):
        '''
        Eliminates node first that lead to the fewest fill-in edges
        :param G: interaction graph
        :return: list with elimination order
        '''
        self.edges = list(edges)
        self.nodes = list(nodes)

        # Create a list with the elimination order
        self.elimination_order = []

        return self.elimination_order

    def MinDegree(self, nodes, edges):
        '''
        Eliminates nodes with the fewest neighbors first

        :param G: interaction graph
        :return: list with elimination order
        '''
        self.edges = list(edges)
        self.nodes = list(nodes)

        # Create a list with the elimination order
        self.elimination_order = []

        if len(nodes) == 0:
            print('There are no nodes.')

        elif len(nodes) == 1:
            self.elimination_order.append(node)

        else:
            # Keep counting the number of neighbours as long as there are edges left
            while len(self.edges) >= 1:

                if len(self.edges) == 1:
                    self.elimination_order.append(self.nodes[0])
                    self.elimination_order.append(self.nodes[1])
                    return self.elimination_order

                else:
                    # Count the number of neighbours of every variable
                    self.neighbour_count = {}
                    for node in self.nodes:
                        self.neighbour_count[node] = 0

                    for edge in self.edges:
                        for node in edge:
                            self.neighbour_count[node] += 1

                    # Find the node with the minimum amount of neighbours and delete all edges with that node
                    eliminated_node = min(self.neighbour_count, key=self.neighbour_count.get)
                    self.elimination_order.append(eliminated_node)

                    # Determine which nodes and edges are left
                    self.remaining_edges = []
                    self.remaining_nodes = []

                    for edge in self.edges:
                        if eliminated_node not in edge:
                            self.remaining_edges.append(edge)
                            for node in edge:
                                self.remaining_nodes.append(node)
                        else:
                            continue

                    self.edges = self.remaining_edges
                    self.nodes = self.remaining_nodes

        return self.elimination_order

    def EliminationOrder(self, heuristic):
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X based on
        the min-degree heuristics and the min-fill heuristics.

        :param X: set of variables X
        :param heurstic: min_degree or min_fill
        :return: order for the elimination of X as a list of variables
        """
        # Get the interaction graph of BN
        self.G = self.bn.get_interaction_graph()

        ## Get all nodes of the interaction graph of BN
        #print("Node set: ", self.G.nodes())

        ## Get all edges of the interaction graph of BN
        #print("Edge set: ", self.G.edges())

        ## Draw the intergraph of BN
        #nx.draw(self.G)
        #plt.show()

        # Determine elimination order depending on the heuristic applied
        if heuristic == "MinFill":
            elimination_order = self.MinFill(self.G.nodes, self.G.edges)
            return elimination_order

        elif heuristic == "MinDegree":
            elimination_order = self.MinDegree(self.G.nodes, self.G.edges)
            return elimination_order

        else:
            print("Heuristic has to be either 'MinFill' or 'MinDegree'.")
            return

def test_BN(filename):

    # Create a BN reasoner (outer class)
    BNReasoner_ = BNReasoner(filename)

    # Create a BN (inner class)
    org_BayesNet_ = BNReasoner_.bn

    # Get the structure of the BN
    structure = org_BayesNet_.structure
    print(f'structure: {structure}')

    # Get a list of all variables in the BN
    all_variables = org_BayesNet_.get_all_variables()
    print(f'all_variables: {all_variables}')

    # Get the interaction graph
    interaction_graph = org_BayesNet_.get_interaction_graph()
    print(f'interaction_graph: {interaction_graph}')

    ## Get a figure of the structure of the BN
    # drawed_structure = org_BayesNet_.draw_structure()
    # print(f'drawed_structure: {drawed_structure}')
    #
    ## Get the conditional probability tables of all the variables in the BN
    # all_cpts = org_BayesNet_.get_all_cpts()
    # print(f'all_cpts: {all_cpts}')
    #
    # # Get the conditional probability table of the variable 'dog-out' in the BN
    # cpt_dog_out = org_BayesNet_.get_cpt('dog-out')
    # print(f'cpt_dog_out: \n {cpt_dog_out}')
    #
    # # Sum-out the variable 'family-out' of the cdt of the variable 'dog-out' in the BN
    # sum_out_family_out = BNReasoner_.SumOutVar('family-out', cpt_dog_out, all_variables)
    # print(sum_out_family_out)
    #
    # # Maximize-out the variable 'max-out' of the cdt of the variable 'dog-out' in the BN
    # max_out_family_out = BNReasoner_.MaxOutVar('family-out', cpt_dog_out, all_variables)
    # print(max_out_family_out)

    # # Get the elimination order of the variables in the BN
    order = BNReasoner_.EliminationOrder(heuristic = "MinDegree")
    print(order)


    # pruned_BayesNet_ = BNReasoner_.network_pruning({"family_out": True, "bowel-problem": False}, cpt_dog_out)
    # print(pruned_BayesNet_)

    return

BN_dog = test_BN('testing/dog_problem.BIFXML')
