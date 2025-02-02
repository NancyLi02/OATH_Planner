# import rospy
from ltl_automaton_planner.ltl_tools.run import ProdAut_Run
from collections import defaultdict
from networkx import dijkstra_predecessor_and_distance
from ltl_automaton_planner.ltl_tools.discrete_plan import compute_path_from_pre#, dijkstra_predecessor_and_distance
import time 

class Dijkstra(object):
    def __init__(self) -> None:
        self.runs = {}
        self.loop = {}
    
    def dijkstra_plan_networkX(self, product, gamma=10):
        # requires a full construct of product automaton
        start = time.time()
        # minimal circles
        for prod_target in product.graph['accept']:
                    # print('prod_target', prod_target)
                    # accepting state in self-loop
                    if prod_target in product.predecessors(prod_target):
                            self.loop[prod_target] = (product.edges[prod_target,prod_target]["weight"], [prod_target, prod_target])
                            continue
                    else:
                            cycle = {}
                            if len(product.pred[prod_target]) != 0:
                                now = time.time()
                                loop_pre, loop_dist = dijkstra_predecessor_and_distance(product, prod_target)
                                # print(prod_target, time.time()-now)
                            for target_pred in product.predecessors(prod_target):
                                    if target_pred in loop_dist:
                                            cycle[target_pred] = product.edges[target_pred,prod_target]["weight"] + loop_dist[target_pred]
                            if cycle:
                                    opti_pred = min(cycle, key = cycle.get)
                                    #print("loop_pre", loop_pre)
                                    #print("opti_pred", opti_pred)
                                    suffix = compute_path_from_pre(loop_pre, opti_pred)
                                    self.loop[prod_target] = (cycle[opti_pred], suffix)
        # print("loop:", time.time()-start)
        new_start = time.time()
        # shortest line
        for prod_init in product.graph['initial']:
            # print("reach initial once")
            line = {}
            line_pre, line_dist = dijkstra_predecessor_and_distance(product, prod_init)
            line_pre.pop(prod_init)
            # print("target keys: ", self.loop.keys())
            for target in self.loop.keys():
                # print("checkpoint 1")
                # print("target")
                if target in line_dist:
                    line[target] = line_dist[target]+gamma*self.loop[target][0]
            # print("reach here")
            # print("line for each target", line)
            # print("product init: ", prod_init)
            if line:
                # print("checkpoint 2")
                opti_targ = min(line, key = line.get)
                # print("opti_targ!!!!: ", opti_targ)
                prefix = compute_path_from_pre(line_pre, opti_targ)
                precost = line_dist[opti_targ]
                self.runs[(prod_init, opti_targ)] = (prefix, precost, self.loop[opti_targ][1], self.loop[opti_targ][0])
        # best combination
        if self.runs:
            prefix, precost, suffix, sufcost = min(self.runs.values(), key = lambda p: p[1] + gamma*p[3])
            run = ProdAut_Run(product, prefix, precost, suffix, sufcost, precost+gamma*sufcost)
            # rospy.logdebug('Dijkstra_plan_networkX done within %.2fs: prefix cost %.2f, sufix cost %.2f' %(time.time()-start, precost, sufcost))
            # print("line:", time.time()-new_start)
            return run, time.time()-start
        
        # rospy.logerr('No accepting run found in optimal planning!')
        return None, None
    
    
    def dijkstra_plan_with_initial(self, product, prod_init, gamma=10, segment="prefix"):
        start = time.time()
        self.runs = {}
        self.loop = {}
        # minimal circles
        for prod_target in product.graph['accept']:
            #print('prod_target', prod_target)
            # accepting state in self-loop
            if prod_target in product.predecessors(prod_target):
                self.loop[prod_target] = (product.edges[prod_target,prod_target]["weight"], [prod_target, prod_target])
                continue
            else:
                cycle = {}
                if len(product.pred[prod_target]) != 0:
                    now = time.time()
                    loop_pre, loop_dist = dijkstra_predecessor_and_distance(product, prod_target)
                    # print(prod_target, time.time()-now)
                for target_pred in product.predecessors(prod_target):
                    if target_pred in loop_dist:
                        cycle[target_pred] = product.edges[target_pred,prod_target]["weight"] + loop_dist[target_pred]
                if cycle:
                    opti_pred = min(cycle, key = cycle.get)
                    suffix = compute_path_from_pre(loop_pre, opti_pred)
                    self.loop[prod_target] = (cycle[opti_pred], suffix)
        # shortest line
        # print("product_init", prod_init)
        line = {}
        line_pre, line_dist = dijkstra_predecessor_and_distance(product, prod_init)
        line_pre.pop(prod_init)
        for target in self.loop.keys():
            if target in line_dist:
                line[target] = line_dist[target]+gamma*self.loop[target][0]
        if line:
            opti_targ = min(line, key = line.get)
            prefix = compute_path_from_pre(line_pre, opti_targ)
            precost = line_dist[opti_targ]
            self.runs[(prod_init, opti_targ)] = (prefix, precost, self.loop[opti_targ][1], self.loop[opti_targ][0]) # (prefix, precost, suffix, sufcost)
        # best combination
        if self.runs:
            prefix, precost, suffix, sufcost = min(self.runs.values(), key = lambda p: p[1] + gamma*p[3])
            run = ProdAut_Run(product, prefix, precost, suffix, sufcost, precost+gamma*sufcost)
            # rospy.logdebug('Dijkstra_plan_networkX done within %.2fs: prefix cost %.2f, sufix cost %.2f' %(time.time()-start, precost, sufcost))
            return run, time.time()-start
    
        # rospy.logerr('No accepting run found in optimal planning!')
        return None, None