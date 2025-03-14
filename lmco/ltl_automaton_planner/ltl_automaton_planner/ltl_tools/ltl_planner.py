# -*- coding: utf-8 -*-
import yaml
# import rospy
from ltl_automaton_planner.ltl_tools.buchi import mission_to_buchi
from ltl_automaton_planner.ltl_tools.product import ProdAut
#from ts import distance, reach_waypoint
from ltl_automaton_planner.ltl_tools.discrete_plan import dijkstra_plan_networkX, dijkstra_plan_optimal, residual, dijkstra_targets
from ltl_automaton_planner.ltl_automaton_utilities import read_yaml_file, write_to_yaml, delete_file
from ltl_automaton_planner.ltl_tools.dijkstra import *
from ltl_automaton_planner.ltl_tools.dstar import *
import networkx as nx
import time

class LTLPlanner(object):
    def __init__(self, ts, hard_spec, soft_spec, beta=1000, gamma=10):
        self.hard_spec = hard_spec
        self.soft_spec = soft_spec
        self.ts = ts

        #Empty product
        self.product = None

        self.Time = 0
        self.curr_ts_state = None
        self.trace = [] # record the regions been visited
        self.traj = [] # record the full trajectory
        self.prefix_opt_log = [] 
        # record [(time, prefix, suffix, prefix_cost, suffix_cost, total_cost)]
        self.prefix_com_log = []
        # record [(time, no_messages)]
        self.suffix_opt_log = []
        self.suffix_opt_log = []
        self.beta = beta                    # importance of taking soft task into account
        self.gamma = gamma                  # cost ratio between prefix and suffix

    def write_to_log(self, data, segment="prefix"):
        file_name = self.algo+'_'+str(self.N)+'_'+segment+'.yaml'
        existing_data = read_yaml_file(file_name)
        print(existing_data)
        existing_data.append(data)
        write_to_yaml(existing_data, file_name)
    
    def optimal(self, algo='dstar', N=10):
        # rospy.loginfo("LTL Planner: --- Planning in progress ("+algo+") ---")
        # rospy.loginfo("LTL Planner: Hard task is: "+str(self.hard_spec))
        # rospy.loginfo("LTL Planner: Soft task is: "+str(self.soft_spec))
        self.N = N
        self.algo = algo
        delete_file(self.algo+'_'+str(self.N)+'_'+'prefix'+'.yaml')
        delete_file(self.algo+'_'+str(self.N)+'_'+'suffix'+'.yaml')    

        self.product = ProdAut(self.ts, mission_to_buchi(self.hard_spec, self.soft_spec), self.beta)
        self.product.graph['ts'].build_full()
        
        if algo == 'dstar' or algo == "dstar-relaxed":
            print("in dstar")
            start_time = time.time()
            self.product.build_full()
            elapsed_time = time.time() - start_time
            print(f"Product automaton constuction took {elapsed_time} seconds to run.")
            
            if algo == 'dstar':
                self.dstar = DStar(self.product, "manhattan", relaxation=False)
            elif algo == 'dstar-relaxed':
                self.dstar = DStar(self.product, "manhattan", relaxation=True)                 
            run, run_cost, plantime = self.dstar.dstar_plan()
            prefix = run[:-1]
            print(prefix)
            precost = run_cost
            sufcost, suffix = self.dstar.get_suffix(prefix[-1])
            print(suffix)
            precost -= sufcost
            self.run = ProdAut_Run(self.product, prefix, precost, suffix, sufcost, precost+self.gamma*sufcost)
            print(self.run)
            print("Dstar initial run compute time: ", plantime)
        elif algo == 'brute-force' or algo == 'local':
            start_time = time.time()
            self.product.build_full()
            elapsed_time = time.time() - start_time
            print(f"Product automaton constuction took {elapsed_time} seconds to run.")
            self.dijkstra = Dijkstra()
            self.run, plantime = self.dijkstra.dijkstra_plan_networkX(self.product, self.gamma)
            print("Dijkstra initial run compute time: ", plantime)
        elif algo == 'relaxed': 
            start_time = time.time()
            self.product.build_full_relaxed()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The function took {elapsed_time} seconds to run.")
            self.dijkstra = Dijkstra()
            self.run, plantime = self.dijkstra.dijkstra_plan_networkX(self.product, self.gamma)
            print("Dijkstra relaxed initial run compute time: ", plantime)
        elif algo == "relaxed-dstar":
            pass 

        if self.run == None:
            # rospy.logerr("LTL Planner: No valid plan has been found! Check you FTS or task")
            return False

        # rospy.loginfo("LTL Planner: --- Planning successful! ---")
        # rospy.logdebug("Prefix states: "+str([n for n in self.run.line]))
        # rospy.logdebug("Suffix states: "+str([n for n in self.run.loop]))

        # self.opt_log.append((self.Time, self.run.pre_plan, self.run.suf_plan, self.run.precost, self.run.sufcost, self.run.totalcost))
        self.last_time = self.Time
        self.acc_change = 0
        self.index = 0
        self.segment = 'line'
        
        # If prefix exists, init next move with prefix first action
        if self.run.pre_plan:
            self.next_move = self.run.pre_plan[self.index]
        # If prefix is empty, jump to suffix for init next move
        else:
            self.next_move = self.run.suf_plan[self.index]
            self.segment = 'loop'
        return True
            
    def dijkstra_rewire(self, exec_index): # baseline benchmark for bruteforce and relaxed
        self.old_run = self.run
        if exec_index <= len(self.run.line):
            print("Prefix")
            self.run, plantime = self.dijkstra.dijkstra_plan_with_initial(self.product, self.run.prefix[exec_index], segment="prefix")
            print(self.run.prefix)
            print(self.run.suffix)
            self.write_to_log([plantime, self.run.precost+self.gamma*self.run.sufcost], segment="prefix")
            print("Dijkstra replanning prefix compute time: ", plantime)
            self.run.prefix = self.old_run.prefix[:exec_index] + self.run.prefix
            # self.run.suffix = self.run.suffix
        else:
            print("Suffix")
            self.run, plantime = self.dijkstra.dijkstra_plan_with_initial(self.product, self.run.suffix[exec_index-len(self.run.line)+1], segment="suffix")
            print(self.run.prefix)
            print(self.run.suffix)
            print("Dijkstra replanning suffix compute time: ", plantime)
            self.write_to_log([plantime, self.run.precost+self.gamma*self.run.sufcost], segment="suffix")
            self.run.prefix = self.old_run.prefix + self.old_run.suffix[:exec_index-len(self.old_run.prefix)+1] + self.run.prefix
            print('\n')
            print(self.run.prefix)
            self.run.suffix = self.run.suffix
        if self.run == None:
            # rospy.logerr("LTL Planner: No valid plan has been found! Check you FTS or task")
            return False

        # self.run.prefix = self.old_run.prefix[:exec_index] + self.run.prefix
        # self.run.suffix = self.run.suffix
        self.run.prod_run_to_prod_edges()
        self.run.plan_output(self.product)
        self.run = self.run
        return True
    
    
    # loop starts with target and but ends with pred
    def local_rewire(self, exec_index): # baseline benchmark for bruteforce and relaxed
        self.old_run = self.run
        
        if exec_index <= len(self.run.line):
            pre_bridge, pre_bridge_cost, pre_plantime = dijkstra_targets(self.product, self.old_run.prefix[exec_index], self.old_run.prefix[exec_index+1:])
            res_cost, res_count = residual(self.old_run, pre_bridge[-1], "prefix") # res_count is minus
            
            self.run.prefix = self.old_run.prefix[: exec_index] + pre_bridge + self.old_run.prefix[res_count+1:] # from initial
            precost = pre_bridge_cost + res_cost # only starts from current node
            
            target_preds = []
            for targ_pred in self.product.predecessors(self.old_run.suffix[0]):
                target_preds.append(targ_pred)
            suf_bridge, suf_bridge_cost, suf_plantime = dijkstra_targets(self.product, self.old_run.suffix[0], target_preds)

            self.run.suffix = suf_bridge # not include target
            sufcost = suf_bridge_cost + self.product.edges[suf_bridge[-1],self.old_run.suffix[0]]["weight"]
            
            self.write_to_log([pre_plantime + suf_plantime, precost+self.gamma*sufcost], segment="prefix")
            
            self.run.prod_run_to_prod_edges()
            self.run.plan_output(self.product)
                
            print("Dijkstra replanning prefix compute time: ", pre_plantime+suf_plantime)
            return True
        else:
            # target_preds = []
            # for targ_pred in self.product.predecessors(self.old_run.suffix[0]):
            #     target_preds.append(targ_pred)
            suf_bridge, suf_bridge_cost, suf_plantime = dijkstra_targets(self.product, self.old_run.suffix[exec_index-len(self.run.line)+1], self.old_run.suffix[exec_index-len(self.run.line)+2:])
            res_cost, res_count = residual(self.old_run, suf_bridge[-1], "suffix") # res_count is minus
            
            self.run.prefix = self.old_run.prefix
            # self.run.prefix = suf_bridge + self.old_run.suffix[res_count:]
            self.run.suffix = self.old_run.suffix[:exec_index-len(self.old_run.line)+1] + suf_bridge + self.old_run.suffix[res_count+1:]
            
            self.run.prod_run_to_prod_edges()
            self.run.plan_output(self.product)
            precost = suf_bridge_cost + res_cost
            
            self.write_to_log([suf_plantime, precost+self.gamma*self.run.sufcost], segment="suffix")
                
            print("Dijkstra replanning suffix compute time: ", suf_plantime)
            return True
        return False
    
    
    def dstar_rewire(self, exec_index, modified_edges_dict, update_info, dstar=True): # baseline rewire for ours and local --- dstar: whether use dstar to do the rewire of suffix part as well
        if exec_index <= len(self.run.line):
            print("PREFIX")
            current_node = self.run.prefix[exec_index]  
            # print(exec_index)
            # print(len(self.run.prefix), " ", self.run.prefix)
            prefix, precost, plantime = self.dstar.reroute_dstar(modified_edges_dict, update_info, segment="prefix", node_to_start=current_node)
            print(prefix, precost)
            print("DStar replanning prefix compute time: ", plantime)
            prefix = prefix[:-1]
            prefix = self.run.prefix[:exec_index] + prefix
            sufcost, suffix = self.dstar.get_suffix(prefix[-1])
            precost -= sufcost
            # prefix, precost, suffix, sufcost = min(self.runs.values(), key = lambda p: p[1] + gamma*p[3])
            self.run = ProdAut_Run(self.product, prefix, precost, suffix, sufcost, precost+self.gamma*sufcost)
            self.write_to_log([plantime, precost+self.gamma*sufcost], segment="prefix")
            return True
        else: # at suffix phase
            print("SUFFIX")
            current_node = self.run.suffix[exec_index - len(self.run.prefix)+1]
            print(current_node)
            prefix, precost, plantime = self.dstar.reroute_dstar(modified_edges_dict, update_info, segment="suffix", node_to_start=current_node)
            prefix = prefix[:-1]
            print("DStar replanning suffix compute time: ", plantime)
            prefix = self.run.prefix + self.run.suffix[:exec_index-len(self.run.prefix)+1] + prefix
            # precost need to be revised
            sufcost, suffix = self.dstar.get_suffix(prefix[-1])
            print(prefix, precost)
            print(suffix, sufcost)
            precost -= sufcost
            self.run = ProdAut_Run(self.product, prefix, precost, suffix, sufcost, precost+self.gamma*sufcost)
            self.write_to_log([plantime, precost+self.gamma*sufcost], segment="suffix")
            return True
        return False
            
    def revise_product(self, update_info): # revise_local_pa
        # "remember the other way around!!!!"
        modified_pairs = update_info["modified"]
        deleted_pairs = update_info["deleted"]
        relabel_states = update_info["relabel"]
        modified_edges = {}
        
        # add transition tuple (from, to, cost) 
        # already including bidirectional transition, no need to worry about the duality
        for mod_pair in modified_pairs:
            for pa_node in self.product.nodes:
                ts_node, bu_node = self.product.projection(pa_node)
                if ts_node == mod_pair[0]:
                    label = self.product.graph['ts'].nodes[ts_node]['label'] # is it right to check the label of this node?
                    for bu_succ in self.product.graph['buchi'].successors(bu_node):
                        guard = self.product.graph['buchi'].edges[bu_node, bu_succ]['guard']
                        if guard.check(label):
                            edge = ((ts_node, bu_node), (mod_pair[1], bu_succ))
                            if self.product.has_edge(edge[0], edge[1]):
                                 self.product.edges[edge]["weight"] = mod_pair[2]
                            else:
                                self.product.add_edge((ts_node, bu_node), (mod_pair[1], bu_succ), weight=mod_pair[2])
                            self.product.graph['ts'][ts_node][mod_pair[1]]['weight'] = mod_pair[2]   # Adjust the weight
                            modified_edges[edge] = mod_pair[2]
        
        # remove transition (from, to)
        remove_list = list()
        for deleted_pair in deleted_pairs:
            print("deleted_pair:", deleted_pair)
            for pa_node in self.product.nodes:
                ts_node, bu_node = self.product.projection(pa_node)
                if ts_node == deleted_pair[0]:
                    label = self.product.graph['ts'].nodes[ts_node]['label']
                    for bu_succ in self.product.graph['buchi'].successors(bu_node):
                        guard = self.product.graph['buchi'].edges[bu_node, bu_succ]['guard']
                        if guard.check(label):
                            remove_list.append(((ts_node, bu_node), (deleted_pair[1], bu_succ)))
        self.product.remove_edges_from(remove_list)
        for edge in remove_list:
            modified_edges[edge] = float("inf")
            
        # relabel (b, pi_j)/(label, state)
        remove_list_relabel = list()
        for relabel_state in relabel_states:
            for ts_pre in self.product.graph['ts'].predecessors(relabel_state[1]):
                for pa_node in self.product.nodes:
                    ts_node, bu_node = self.product.projection(pa_node)
                    if ts_node == ts_pre:
                        for bu_succ in self.product.graph['buchi'].successors(bu_node):
                            guard = self.product.graph['buchi'].edges[bu_node, bu_succ]['guard']
                            if guard.check(relabel_state[0]):
                                self.product.add_edge((ts_node, bu_node), (relabel_state[1], bu_succ), transition_cost=0, action='added_transition', weight=0)
                                self.product.graph['ts'][ts_node][relabel_state[1]]['weight'] = 0   # Adjust the weight
                                self.product.graph['ts'][ts_node][relabel_state[1]]['action'] = 'relabeled_transition'
                for pa_node in self.product.nodes:
                    ts_node, bu_node = self.product.projection(pa_node)
                    if ts_node == ts_pre:
                        for bu_succ in self.product.graph['buchi'].successors(bu_node):
                            guard = self.product.graph['buchi'].edges[bu_node, bu_succ]['guard']
                            if not guard.check(relabel_state[0]):
                                remove_list_relabel.append(((ts_node, bu_node), (relabel_state[1], bu_succ)))
        self.product.remove_edges_from(remove_list_relabel)
        return modified_edges

