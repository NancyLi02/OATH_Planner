"""extending A* for disconnected graph"""

import matplotlib.pyplot as plt
import copy
import heapq

from ltl_automaton_planner.ltl_tools.buchi import check_label_for_buchi_edge
from ltl_automaton_planner.ltl_tools.discrete_plan import compute_path_from_pre
from ltl_automaton_planner.ltl_automaton_utilities import extract_numbers
from ltl_automaton_planner.ltl_tools.dstar import DStar

from networkx.classes.digraph import DiGraph
import time 


class Relaxation(DStar):
    def __init__(self, product, heuristic_type, deepcopy=False, start_node=None):
        super().__init__(product, heuristic_type, deepcopy, start_node) # Note that this product include imag_goal
        self.dist = {}
        for node in self.product.nodes():
            self.dist[node] = 0
        self.start_cluster = []
        self.initialize()
        
    def dijkstra(self, source):
        distances = {node: float('inf') for node in self.product}
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
 
            if current_distance > distances[current_node]:
                continue
            
            for succ in self.get_successors(current_node):
                distance = current_distance + self.cost(current_node, succ)
                if distance < distances[succ]:
                    distances[succ] = distance
                    heapq.heappush(priority_queue, (distance, succ))
                    
        return distances

    def dissect(self):
        # calculate the start cluster  
        distances = self.dijkstra(self.product, self.s_start)

        for node, distance in distances.items():
            if distance != float('inf'):
                self.start_cluster.append(node)
                
    def expand(self, t_prod_node):
        relaxed_edges_dict = {}
        t_ts_node = t_prod_node[0]
        t_buchi_node = t_prod_node[1]
        for f_buchi_node in self.product.graph['buchi'].predecessors(t_buchi_node): # is it correct?
            for f_ts_node in self.product.graph['ts'].predecessors(t_ts_node):
                # some ts predecessors are occluded using self.removed_list
                if (extract_numbers(f_ts_node), extract_numbers(t_ts_node)) not in self.removed_list:
                    f_prod_node = DiGraph.composition(f_ts_node, f_buchi_node)
                    if self.product.has_edge(f_prod_node, t_prod_node):
                        continue
                    label = self.product.graph['ts'].nodes[f_ts_node]['label']
                    cost = self.product.graph['ts'][f_ts_node][t_ts_node]['weight'] # action weight
                    action = self.product.graph['ts'][f_ts_node][t_ts_node]['action']
                    # Check if label is compatible with büchi (black magic for now, need to understand this better)
                    truth, dist = check_label_for_buchi_edge(self.graph['buchi'], label, f_buchi_node, t_buchi_node)
                    total_weight = cost + 10000*dist
                    if not truth:
                        relaxed_edges_dict[(f_prod_node, t_prod_node)] = total_weight
                        self.product.add_edge(f_prod_node, t_prod_node, transition_cost=cost, dist=dist, weight=total_weight, action=action)
        return relaxed_edges_dict
    

    def compute_path(self):
        while True:
            s, v = self.top_key()
            if v >= self.calculate_key(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            k_old = v
            self.U.pop(s)
            self.visited.add(s)

            if k_old < self.calculate_key(s):
                self.U[s] = self.calculate_key(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)
    
    def relax(self):     
        """Algorithm:
        starting from the existing node
        explore the one with smallest value and extend that node
        add the node to the waivor
        for the next node to check 
            if it already connected to the start set
            end 
        """
        self.dissect()
        
        
        priority_queue = []
        self.compute_path()
        for node in self.visited: 
            heapq.heappush(priority_queue, (self.U[node], node)) # explored node from goal
            
        while priority_queue:
            _, pop_node = heapq.heappop(priority_queue)
            
            if pop_node in self.start_cluster:
                break
            
            relaxed_edges = self.expand(pop_node) # these edges have already been added to product
            
            for edge, cost in relaxed_edges:
                if edge[0] in self.visited:
                    # need to do a comparison here
                    pass
                else:
                    heapq.heappush(priority_queue, (edge[0], self.calculate_key(edge[0])))
            
    def calculate_key(self, s):
        return [min(self.g[s], self.rhs[s]) + self.dist[s] + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s]) + self.dist[s]]

    
