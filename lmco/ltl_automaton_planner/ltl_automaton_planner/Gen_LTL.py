def generate_ltl_formula(route, pickup_labels, delivery_labels):
    """
    Generate an LTL formula for any number of pickups and deliveries.
    
    The formula structure:
      1. <> (first event)
      2. [] (start -> X(condition U next && event))
      ...
    
    Segments between:
      - pickups: require loaded
      - first segment: insert loaded after start
      - segment into delivery: insert unloaded at end
      - segments between deliveries: require unloaded
    
    Forbidden nodes during each segment:
      all cluster pickups not yet visited plus deliveries not yet reached,
      excluding the next target node.
    """
    # identify cluster sets
    cluster_p = set(pickup_labels)
    cluster_d = set(delivery_labels)
    
    # extract sequence of visited pickups and deliveries in order
    visited_pickups = [n for n in route if n in cluster_p]
    visited_deliveries = [n for n in route if n in cluster_d]
    assert visited_pickups and visited_deliveries, "Route must include at least one pickup and one delivery"
    
    # helper to format forbidden conjunction
    fmt = lambda s: "&&".join(f"(!{n})" for n in sorted(s)) if s else "true"
    
    parts = []
    # 1. eventual first node
    parts.append(f"<>({route[0]})")
    
    # build segments
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i+1]
        
        # visited so far (start included)
        done_p = set(route[:i+1]) & cluster_p
        done_d = set(route[:i+1]) & cluster_d
        
        # nodes not yet visited
        not_p = cluster_p - done_p
        not_d = cluster_d - done_d
        
        # forbid everything not visited, except the next target
        forbid = (not_p | not_d) - {end}
        forb_str = f"({fmt(forbid)})"
        
        # determine prefix and suffix for LTL segment
        if i == 0:
            # first segment: insert loaded after start
            prefix = f"[]({start} -> X(loaded && {forb_str} U ({end} &&"
            suffix = " loaded))"
        else:
            # subsequent: choose loaded or unloaded based on start
            state = "loaded" if start in cluster_p else "unloaded"
            prefix = f"[]({start} && {state} -> X({forb_str} U ({end} &&"
            suffix = f" { 'loaded' if end in cluster_p else 'unloaded'})"
        
        parts.append(prefix + suffix + ")")
    
    # join with line-continuation
    return " \\\n".join(parts)

# Example usage:
route = ['dc', 'bc', 'ec', 'c']
pickup_labels = ['bc', 'cc', 'dc', 'ec']
delivery_labels = ['c']

print(generate_ltl_formula(route, pickup_labels, delivery_labels))
