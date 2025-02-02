
import os
import sys
import math
import matplotlib.pyplot as plt
import copy
import heapq

from ltl_automaton_planner.ltl_tools.run import ProdAut_Run
from ltl_automaton_planner.ltl_tools.discrete_plan import compute_path_from_pre
from ltl_automaton_planner.ltl_automaton_utilities import extract_numbers
from ltl_automaton_planner.ltl_tools.buchi import check_label_for_buchi_edge

from networkx import dijkstra_predecessor_and_distance
from networkx.classes.digraph import DiGraph
import time 

class DStar(object):
    def __init__(self, product, heuristic_type, deepcopy=True, add_imag=True, start_node=None, relaxation=False):
        self.relaxation = relaxation
        self.product = copy.deepcopy(product) if deepcopy else product
        self.heuristic_type = heuristic_type
        
        self.s_start = start_node if start_node is not None else list(self.product.graph['initial'])[0]
        if add_imag:
            self.augment_imaginary_goal_dstar()
        self.s_goal = 'imag_goal' 
        self.initialize()

    def initialize(self):
        self.g, self.rhs, self.U= {}, {}, {}
        self.km = 0

        for node in self.product.nodes():
            self.rhs[node] = float("inf")
            self.g[node] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.calculate_key(self.s_goal)
        self.s_last = self.s_start
        self.removed_list = set()
        self.relaxed_edges_dict = set()
        
    def augment_imaginary_goal(self):
        self.product.add_node('imag_goal')
        self.loop = {}
        # print("length of accepted state:", len(self.product.graph['accept']))
        for prod_target in self.product.graph['accept']:
            # print('prod_target', self.product.nodes[prod_target])
            # accepting state in self-loop
            if prod_target in self.product.predecessors(prod_target):
                self.loop[prod_target] = (self.product.edges[prod_target,prod_target]["weight"], [prod_target, prod_target])
                self.product.add_edge(prod_target, 'imag_goal', weight = self.loop[prod_target][0]) 
                continue
            else:
                cycle = {}
                loop_pre, loop_dist = dijkstra_predecessor_and_distance(self.product, prod_target)
                for target_pred in self.product.predecessors(prod_target):
                    if target_pred in loop_dist:
                        cycle[target_pred] = self.product.edges[target_pred,prod_target]["weight"] + loop_dist[target_pred]
                if cycle:
                    opti_pred = min(cycle, key = cycle.get)
                    suffix = compute_path_from_pre(loop_pre, opti_pred)
                    self.loop[prod_target] = (cycle[opti_pred], suffix)   
                    self.product.add_edge(prod_target, 'imag_goal', weight = self.loop[prod_target][0])          

        
    def augment_imaginary_goal_dstar(self):
        self.base_product = copy.deepcopy(self.product)
        self.product.add_node('imag_goal')
        self.loop = {}
        self.loop_graph = {}
        print("length of accepted state:", len(self.product.graph['accept']))
        for prod_target in self.product.graph['accept']:
            if len(self.product.pred[prod_target]) == 0:
                continue
            else:
                self.loop_graph[prod_target] = DStarSuffix(self.base_product, self.heuristic_type, prod_target, self.relaxation) 
        for prod_target, _ in self.loop_graph.items():
            self.loop_graph[prod_target].initialize()
            print(prod_target)
            suffix, sufcost, plantime = self.loop_graph[prod_target].dstar_plan()
            print(sufcost, plantime)
            #time.sleep(0.1)
            suffix = suffix[:-1]
            suffix.append(prod_target)
            self.loop[prod_target] = (sufcost, suffix)
            self.product.add_edge(prod_target, 'imag_goal', weight = sufcost) 
        print("************finish constructing imaginary goal****************")

    def reroute_prefix(self, modified_edges_dict, node_to_start): # full node edges and corresponding new cost
        start = time.time()
        self.s_curr = node_to_start
        path = [self.s_curr]
        path_cost = 0

        while self.s_curr != self.s_goal:
            s_list = {}
            for s in self.get_successors(self.s_curr):
                s_list[s] = self.g[s] + self.cost(self.s_curr, s)
                
            self.s_curr = min(s_list, key=s_list.get)
            path.append(self.s_curr) # here it goes one step further
            path_cost += self.product.edges[path[-2], path[-1]]["weight"]
            
            if (path[-2], path[-1]) in modified_edges_dict:
                path_cost -= self.product.edges[path[-2], path[-1]]["weight"]
                self.km += self.h(self.s_last, path[-2])
                self.s_last = path[-2]
                count = 0
                
                for edge, cost in modified_edges_dict.items():
                    count += 1
                    self.product.edges[edge]["weight"] = cost 
                    self.update_vertex(edge[0])
                
                path = path[:-1] # delete the last element
                self.s_curr = path[-1]
                
                self.compute_path()
                modified_edges_dict = {}
                
        return path, path_cost, time.time()-start 
    
    def reroute_suffix(self, modified_edges_dict, node_to_start):
        for edge, cost in modified_edges_dict.items():
            self.product.edges[edge]["weight"] = cost 
            self.update_vertex(edge[0])
        self.s_start = node_to_start # need to replan from scratch
        self.initialize()
        path, path_cost, plantime = self.dstar_plan()
        return path, path_cost, plantime
    
    def dstar_plan(self):
        start = time.time()
        self.compute_path()
        path , path_cost = self.extract_path()        
        return path, path_cost, time.time()-start
    
    def reroute_dstar(self, modified_edges_dict, update_info, segment="prefix", node_to_start=None):
        new_modified_edges_dict = copy.deepcopy(modified_edges_dict)
        relaxed_modified_edges_dict = self.relaxed_edges_change(update_info)
        new_modified_edges_dict.update(relaxed_modified_edges_dict)
        for d in update_info["deleted"]:
            self.removed_list.add((extract_numbers(d[0][0]), extract_numbers(d[1][0])))
        
        plantime_sum = 0
        for prod_target, dstar_graph in self.loop_graph.items():
            print(prod_target)
            suffix, sufcost, plantime = self.loop_graph[prod_target].reroute(modified_edges_dict, update_info)
            print(sufcost, plantime)
            #time.sleep(0.1)
            suffix = suffix[:-1]
            suffix.append(prod_target)
            plantime_sum += plantime
            if self.loop[prod_target][0] != sufcost:
                new_modified_edges_dict[(prod_target, self.s_goal)] = sufcost
            self.loop[prod_target] = (sufcost, suffix)
        if segment == "prefix":
            path, path_cost, plantime = self.reroute_prefix(new_modified_edges_dict, node_to_start)
        else:
            path, path_cost, plantime = self.reroute_suffix(new_modified_edges_dict, node_to_start)
        plantime_sum += plantime
        return path, path_cost, plantime_sum

    def compute_path(self):
        while True:
            s, v = self.top_key()
            if v >= self.calculate_key(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            k_old = v
            self.U.pop(s)
            
            if k_old < self.calculate_key(s):
                self.U[s] = self.calculate_key(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                if self.relaxation:
                    self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                if self.relaxation:
                    self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)

    # def compute_path_optimized(self):
    #     while True:
    #         s, v = self.top_key()
    #         if v >= self.calculate_key(self.s_start) and \
    #                 self.rhs[self.s_start] == self.g[self.s_start]:
    #             break
                    
    #         k_old = v
    #         if k_old < self.calculate_key(s):
    #             self.U[s] = self.calculate_key(s)
    #         elif self.g[s] > self.rhs[s]:
    #             self.g[s] = self.rhs[s]
    #             self.U.pop(s)
    #             for x in self.get_predecessors(s):
    #                 if x != self.s_goal:
    #                     self.rhs[x] = min(self.rhs[x], self.cost(x, s)+self.g[s])
    #                     self.update_vertex[x]
    #         else:
    #             temp = self.g[s]
    #             self.g[s] = float("inf")
    #             for x in self.get_predecessors(s):
    #                 if self.rhs[x] == self.cost(x, s)+temp:
    #                     if x != self.s_goal:
    #                         s_list = {}
    #                         for x_ in self.get_successors(x):
    #                             s_list[x_] = self.g[x_] + self.cost(x, x_)
                                
    #                         self.rhs[x] = min(s_list, key=s_list.get)
    #                 self.update_vertex[x]
                
    def basic_deletion(self):
        self.par[self.s_curr] = None
        s_list = {}
        for s in self.get_predecessors(self.s_last):
            s_list[s] = self.g[s] + self.cost(s, self.s_last)
        min_key, min_value = min(s_list.items(), key=s_list.get)
        self.rhs[self.s_last] = min_value
        if self.rhs[self.s_last] == float("inf"):
            self.par[self.s_last] = None
        else:
            self.par[self.s_last]= min_key
        self.update_vertex(self.s_last)
    
    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_successors(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.calculate_key(s)

    def calculate_key(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def top_key(self):
        """
        :return: return the min key and its value.
        """
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        if s_goal == 'imag_goal':
            return 10
        heuristic_type = self.heuristic_type  # heuristic type: Manhattan
        node_start_loc = extract_numbers(self.product.nodes[s_start]['ts'][0])
        node_goal_loc = extract_numbers(self.product.nodes[s_goal]['ts'][0])
        
        if heuristic_type == "manhattan":
            return abs(node_goal_loc[0] - node_start_loc[0]) + abs(node_goal_loc[1] - node_start_loc[1])
        else:
            return math.hypot(node_goal_loc[0] - node_start_loc[0], node_goal_loc[1] - node_start_loc[1])

    def cost(self, s_start, s_goal): # TODO
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        
        return self.product.edges[s_start, s_goal]["weight"]

    # def get_neighbor(self, s):
    #     nei_list = set()
    #     for u in self.U:
    #         s_next = tuple([s[i] + u[i] for i in range(2)])
    #         if s_next not in self.obs:
    #             nei_list.add(s_next)

    #     return nei_list
    
    def get_predecessors(self, s): 
        """ get the predecessor of s """
        return self.product.predecessors(s) 
    
    def get_successors(self, s):
        """ get the successor of s """
        return self.product.successors(s) 
    
    def get_suffix(self, s):
        if s in self.loop:
            return self.loop[s]
    
    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_start]
        path_cost = 0
        s = self.s_start

        while s != self.s_goal:
            g_list = {}
            for x in self.get_successors(s):
                g_list[x] = self.g[x] + self.cost(s, x)
            s = min(g_list, key=g_list.get)
            
            path.append(s)
            path_cost += self.product.edges[path[-2], path[-1]]["weight"]
            if s == self.s_goal:
                break

        return list(path), path_cost
    
    def expand(self, t_prod_node):
        if t_prod_node != "imag_goal":
            t_ts_node = t_prod_node[0]
            t_buchi_node = t_prod_node[1]
            for f_buchi_node in self.product.graph['buchi'].predecessors(t_buchi_node):
                for f_ts_node in self.product.graph['ts'].predecessors(t_ts_node):
                    # some ts predecessors are occluded if within self.removed_list
                    if (extract_numbers(f_ts_node[0]), extract_numbers(t_ts_node[0])) not in self.removed_list:
                        f_prod_node = self.product.composition(f_ts_node, f_buchi_node)
                        if self.product.has_edge(f_prod_node, t_prod_node):
                            continue
                        label = self.product.graph['ts'].nodes[f_ts_node]['label']
                        cost = self.product.graph['ts'][f_ts_node][t_ts_node]['weight'] # action weight
                        action = self.product.graph['ts'][f_ts_node][t_ts_node]['action']
                        truth, dist = check_label_for_buchi_edge(self.product.graph['buchi'], label, f_buchi_node, t_buchi_node)
                        total_weight = cost + 100000*dist
                        if not truth:
                            #self.relaxed_edges_dict[(f_prod_node, t_prod_node)] = total_weight
                            self.relaxed_edges_dict.add((f_prod_node, t_prod_node))
                            self.product.add_edge(f_prod_node, t_prod_node, transition_cost=cost, dist=dist, weight=total_weight, action=action)

    def relaxed_edges_change(self, update_info): # revise_local_pa
        # "remember the other way around!!!!"
        modified_pairs = update_info["modified"]
        deleted_pairs = update_info["deleted"]
        relabel_states = update_info["relabel"]
        modified_edges_relaxed = dict()
        
        to_delete=[]
        # add transition tuple (from, to, cost) 
        # already including bidirectional transition, no need to worry about the duality
        for re in self.relaxed_edges_dict:
            if re[1] == "imag_goal":
                continue
            f_relaxed_ts, f_relaxed_buchi = self.product.projection(re[0])
            t_relaxed_ts, t_relaxed_buchi = self.product.projection(re[1])
            for mod_pair in modified_pairs:
                if f_relaxed_ts == mod_pair[0] and t_relaxed_ts == mod_pair[1]:
                    label = self.product.graph['ts'].nodes[f_relaxed_ts]['label']
                    # guard = self.product.graph['buchi'].edges[f_relaxed_buchi, t_relaxed_buchi]['guard']
                    _, dist = check_label_for_buchi_edge(self.product.graph['buchi'], label, f_relaxed_buchi, t_relaxed_buchi)
                    violation_weight = 100000*dist
                    modified_edges_relaxed[re] = mod_pair[2]+violation_weight
                    
            for deleted_pair in deleted_pairs:
                if f_relaxed_ts == deleted_pair[0] and t_relaxed_ts == deleted_pair[1]:
                    modified_edges_relaxed[re] = float("inf")
                    # self.product.remove_edge(re[0], re[1])
                    to_delete.append(re)
                   
        for td in to_delete:
            self.relaxed_edges_dict.remove(td)
                    
        return modified_edges_relaxed



class DStarSuffix(object):
    def __init__(self, product, heuristic_type, s_start, relaxation=False):
        self.relaxation = relaxation
        self.product = product #copy.deepcopy(product)
        self.s_start = s_start
        self.imag_goal_name = 'imag_goal_' + str(self.s_start[0][0]) + '_' + str(self.s_start[0][1]) + '_' + str(self.s_start[1]) 
        self.s_goal = self.imag_goal_name
        # print(self.s_goal)
        self.deadend = False
        self.augment_imaginary_goal()
        
        # self.s_start, self.s_goal = s_start, s_goal
        # print(self.s_goal)
        self.heuristic_type = heuristic_type
        
    def initialize(self):
        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0

        for node in self.product.nodes():
            self.rhs[node] = float("inf")
            self.g[node] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.calculate_key(self.s_goal)
        self.visited = set()
        self.count = 0
        self.removed_list = set()
    
    def augment_imaginary_goal(self):
        
        self.product.add_node(self.imag_goal_name)
        # print("M start:", self.s_start)
        for target_pred in self.product.predecessors(self.s_start):
            # print("M: ", target_pred)
            if target_pred == self.s_start:
                self.product.add_edge(self.s_start, self.imag_goal_name, weight = self.product.edges[self.s_start, self.s_start]["weight"])
            else:
                self.product.add_edge(target_pred, self.imag_goal_name, weight = self.product.edges[target_pred, self.s_start]["weight"])
        # self.product.remove_edges_from([(targ_pred, self.s_start) for targ_pred in self.product.predecessors(self.s_start)])
        #print("predecessor of imaginary goal", [targ_pred for targ_pred in self.product.predecessors('imag_goal')])
        # self.product.remove_edge(target_pred, self.s_start)
                    
    def dstar_plan(self):
        start = time.time()
        found = self.compute_path()
        if not found:
            return [], float('inf'), time.time()-start
        init_path, path_cost = self.extract_path()
        return init_path, path_cost, time.time()-start

    def reroute(self, modified_edges_dict, update_info):
        for d in update_info["deleted"]:
            self.removed_list.add((extract_numbers(d[0][0]), extract_numbers(d[1][0])))

        start = time.time()
        self.s_curr = self.s_start
        self.s_last = self.s_start
        path = [self.s_start]
        path_cost = 0
        # need to change here
        # print(self.s_start)
        for edge, cost in modified_edges_dict.items():
            if edge[1] == self.s_start:
                self.product.edges[(edge[0], self.s_goal)]["weight"] = cost
            else:
                self.product.edges[edge]["weight"] = cost # float("inf")
            self.update_vertex(edge[0])

            found = self.compute_path()
            modified_edges_dict = {}
            if not found:
                return [], float("inf"), time.time()-start
        while self.s_curr != self.s_goal:
            s_list = {}
            for s in self.get_successors(self.s_curr):
                s_list[s] = self.g[s] + self.cost(self.s_curr, s)
                
            self.s_curr = min(s_list, key=s_list.get)
            path.append(self.s_curr) # here it goes one step further
            path_cost += self.product.edges[path[-2], path[-1]]["weight"]
        
        
        # while self.s_curr != self.s_goal:
        #     s_list = {}
        #     for s in self.get_successors(self.s_curr):
        #         s_list[s] = self.g[s] + self.cost(self.s_curr, s)
                
        #     self.s_curr = min(s_list, key=s_list.get)
        #     path.append(self.s_curr) # here it goes one step further
        #     path_cost += self.product.edges[path[-2], path[-1]]["weight"]
            
        #     if (path[-2], path[-1]) in modified_edges_dict or (path[-1] == self.s_goal and (path[-2], self.s_start) in modified_edges_dict):
        #         path_cost -= self.product.edges[path[-2], path[-1]]["weight"]
        #         self.km += self.h(self.s_last, path[-2])
        #         self.s_last = path[-2]

        #         # for edge in self.product.edges():
        #         #     if edge[0][0][0] == path[-2][0][0] and edge[1][0][0] == path[-1][0][0]:
        #         #         self.product.edges[edge]["weight"] = float("inf")
        #         #         self.update_vertex(edge[0])
        #         for edge, cost in modified_edges_dict:
        #             print(edge)
        #             self.product.edges[edge]["weight"] = cost # float("inf")
        #             self.update_vertex(edge[0])
        #         # self.update_vertex(path[-2]) # not only this is affected all 
        #         path = path[:-1] # delete the last element
        #         self.s_curr = path[-1]

        #         self.count += 1
        #         self.visited = set()
        #         found = self.compute_path()
        #         if not found:
        #             return [], float("inf"), time.time()-start
            
        # self.plot_visited(self.visited)
        # self.plot_path(path)
        return path, path_cost, time.time()-start # need to consider to remove the last node

    def compute_path(self):
        while True:
            if not self.U:
                return False
            s, v = self.top_key()
            # print(v)
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
                if self.relaxation:
                    self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)
            else:
                self.g[s] = float("inf")
                self.update_vertex(s)
                if self.relaxation:
                    self.expand(s)
                for x in self.get_predecessors(s):
                    self.update_vertex(x)
        return True
        
    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_successors(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.calculate_key(s)

    def calculate_key(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def top_key(self):
        """
        :return: return the min key and its value.
        """
        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal): # TODO: heuristic
        if  isinstance(s_goal, str) and s_goal.startswith("imag_goal"):
            return 0 #? what should be put here
        heuristic_type = self.heuristic_type  # heuristic type
        node_start_loc = extract_numbers(self.product.nodes[s_start]['ts'][0])
        node_goal_loc = extract_numbers(self.product.nodes[s_goal]['ts'][0])
        
        if heuristic_type == "manhattan":
            return abs(node_goal_loc[0] - node_start_loc[0]) + abs(node_goal_loc[1] - node_start_loc[1])
        else:
            return math.hypot(node_goal_loc[0] - node_start_loc[0], node_goal_loc[1] - node_start_loc[1])

    def cost(self, s_start, s_goal): # TODO
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        
        return self.product.edges[s_start, s_goal]["weight"]
    
    def get_predecessors(self, s): 
        """ get the predecessor of s """
        return self.product.predecessors(s) 
    
    def get_successors(self, s):
        """ get the successor of s """
        return self.product.successors(s) 

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        path = [self.s_start]
        path_cost = 0
        s = self.s_start

        while s != self.s_goal:
            g_list = {}
            for x in self.get_successors(s):
                g_list[x] = self.g[x] + self.cost(s, x)
            s = min(g_list, key=g_list.get)
            
            path.append(s)
            path_cost += self.product.edges[path[-2], path[-1]]["weight"]
            if s == self.s_goal:
                break

        return list(path), path_cost
    
    def expand(self, t_prod_node):
        if isinstance(t_prod_node, tuple):
            t_ts_node = t_prod_node[0]
            t_buchi_node = t_prod_node[1]
            # print(t_buchi_node)
            for f_buchi_node in self.product.graph['buchi'].predecessors(t_buchi_node):
                for f_ts_node in self.product.graph['ts'].predecessors(t_ts_node):
                    # some ts predecessors are occluded if within self.removed_list
                    if (extract_numbers(f_ts_node[0]), extract_numbers(t_ts_node[0])) not in self.removed_list:
                        f_prod_node = self.product.composition(f_ts_node, f_buchi_node)
                        if self.product.has_edge(f_prod_node, t_prod_node):
                            continue
                        label = self.product.graph['ts'].nodes[f_ts_node]['label']
                        cost = self.product.graph['ts'][f_ts_node][t_ts_node]['weight'] # action weight
                        action = self.product.graph['ts'][f_ts_node][t_ts_node]['action']
                        truth, dist = check_label_for_buchi_edge(self.product.graph['buchi'], label, f_buchi_node, t_buchi_node)
                        total_weight = cost + 100000*dist
                        if not truth:
                            self.product.add_edge(f_prod_node, t_prod_node, transition_cost=cost, dist=dist, weight=total_weight, action=action)
                        
        if t_prod_node == self.s_goal:
            t_ts_node = self.s_start[0]
            t_buchi_node = self.s_start[1]
            for f_buchi_node in self.product.graph['buchi'].predecessors(t_buchi_node):
                for f_ts_node in self.product.graph['ts'].predecessors(t_ts_node):
                    # some ts predecessors are occluded if within self.removed_list
                    if (extract_numbers(f_ts_node[0]), extract_numbers(t_ts_node[0])) not in self.removed_list:
                        f_prod_node = self.product.composition(f_ts_node, f_buchi_node)
                        if self.product.has_edge(f_prod_node, t_prod_node):
                            continue
                        label = self.product.graph['ts'].nodes[f_ts_node]['label']
                        cost = self.product.graph['ts'][f_ts_node][t_ts_node]['weight'] # action weight
                        action = self.product.graph['ts'][f_ts_node][t_ts_node]['action']
                        truth, dist = check_label_for_buchi_edge(self.product.graph['buchi'], label, f_buchi_node, t_buchi_node)
                        total_weight = cost + 100000*dist
                        if not truth:
                            self.product.add_edge(f_prod_node, self.s_goal, transition_cost=cost, dist=dist, weight=total_weight, action=action)