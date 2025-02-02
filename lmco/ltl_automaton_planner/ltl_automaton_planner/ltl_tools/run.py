# import rospy
from networkx.classes.digraph import DiGraph

class ProdAut_Run(object):
    # prefix, suffix in product run
    # prefix: init --> accept, suffix accept --> accept
    # line, loop in ts
    def __init__(self, product, prefix, precost, suffix, sufcost, totalcost):
        self.prefix = prefix
        self.precost = precost
        self.suffix = suffix
        self.sufcost = sufcost
        self.totalcost = totalcost
        #self.prod_run_to_prod_edges(product)
        self.plan_output(product)

    def prod_run_to_prod_edges(self):
        self.pre_prod_edges = zip(self.prefix[0:-1], self.prefix[1:])
        self.suf_prod_edges = zip(self.suffix[0:-1], self.suffix[1:])
        #########
        # line: a, b ,c , d, e, g 
        # pre_plan: act_a, act_b, act_c, act_d, act_e, act_g
        # loop: g, b, c, d, e, f, g
        # suf_plan: act_b, act_c, act_d.., act_g
        

    def plan_output(self, product):

        # Collect the nodes of the TS associated with the prefix plan
        self.line = [product.nodes[node]['ts'] for node in self.prefix]
        # Collect the nodes of the TS associated with the suffix plan
        self.loop = [product.nodes[node]['ts'] for node in self.suffix]
        # Append start of loop to the end to create a 'loop'
        self.loop.append(self.loop[0])


        # Collect prefix nodes in list of tuples e.g. [ (prefix_node_1, prefix_node_2), (prefix_node_2, prefix_node_3), ..., (prefix_node_n-1, prefix_node_n)]
        self.pre_ts_edges = zip(self.line[0:-1], self.line[1:])
        # Collect suffix nodes in list of tuples (see pre_ts_edges)
        self.suf_ts_edges = zip(self.loop[0:-1], self.loop[1:])

        # output plan --- for execution

        # Initialize prefix plan and cost
        self.pre_plan = list()
        # Initialize pre_plan cost
        self.pre_plan_cost = [0,]

        # Iterate over the nodes associated with the prefix (see pre_ts_edges)
        for ts_edge in self.pre_ts_edges:

            # Extract 'action' label between the two consecutive TS nodes of the prefix plan and add it to the pre_plan
            self.pre_plan.append(product.graph['ts'][ts_edge[0]][ts_edge[1]]['action'] )

            # Add the 'weight' label between the two consectuve TS nodes as the cost of the prefix plan
            self.pre_plan_cost.append(product.graph['ts'][ts_edge[0]][ts_edge[1]]['weight']) # action cost 

        # Initialize suffix plan and cost
        self.suf_plan = list()
        self.suf_plan_cost = [0,]

        # Iterate over the nodes associated with the suffix (see suf_ts_edges)
        for ts_edge in self.suf_ts_edges:
            
            # Extract 'action' label between the two consecutive TS nodes of the suffix plan and add it to the suf_plan
            self.suf_plan.append(product.graph['ts'][ts_edge[0]][ts_edge[1]]['action'] )

            # Add 'weight' label between the consecutive TS nodes of the suffix plan to the cost
            self.suf_plan_cost.append(product.graph['ts'][ts_edge[0]][ts_edge[1]]['weight']) # action cost

        # # rospy.loginfo('LTL Planner: Prefix plan: ' + str(self.pre_plan))
        # # rospy.loginfo('LTL Planner: Suffix plan: ' + str(self.suf_plan))