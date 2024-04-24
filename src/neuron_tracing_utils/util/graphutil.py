from neuron_tracing_utils.util.java import snt


def prune_contiguous_dups(graph):
    """
    Removes contiguous nodes with duplicated x, y, z
    coordinate values while preserving the topology of the tree.
    Runs in-place.

    Args:
    graph (org.jgrapht.Graph): the graph

    Returns:
    Set: A set of duplicate indices removed from the graph
    """
    stack = [graph.getRoot()]
    dups = set()
    while stack:
        n = stack.pop()
        idx = (int(n.x), int(n.y), int(n.z))
        cedges = list(graph.outgoingEdgesOf(n))
        stack.extend([c.getTarget() for c in cedges])
        if graph.incomingEdgesOf(n).isEmpty():
            continue
        p = next(graph.incomingEdgesOf(n).iterator()).getSource()
        if idx == (int(p.x), int(p.y), int(p.z)):
            dups.add(idx)
            for c in cedges:
                graph.addEdge(p, c.getTarget())
            graph.removeVertex(n)  # removes all incident edges
    return dups


def prune_all_dups(graph):
    """
    Removes any subsequently encountered nodes during depth first search
    with the same x, y, z coordinate values as a visited node
    while preserving the topology of the tree. Runs in-place.

    Args:
    graph (org.jgrapht.Graph): the graph

    Returns:
    Set: A set of duplicate indices removed from the graph
    """
    stack = [graph.getRoot()]
    dups = set()
    visited = set()
    while stack:
        n = stack.pop()
        for c in graph.outgoingEdgesOf(n):
            stack.append(c.getTarget())
        idx = (int(n.x), int(n.y), int(n.z))
        if idx in visited:
            dups.add(idx)
            p = next(graph.incomingEdgesOf(n).iterator()).getSource()
            for c in graph.outgoingEdgesOf(n):
                graph.addEdge(p, c.getTarget())
            graph.removeVertex(n)  # removes all incident edges
        else:
            visited.add(idx)
    return dups


def get_components_iterative(graph):
    roots = [v for v in graph.vertexSet() if graph.inDegreeOf(v) == 0]
    components = []
    for root in roots:
        # create an empty graph
        comp = snt.DirectedWeightedGraph()
        # iterative depth-first search
        stack = [root]
        while stack:
            v = stack.pop()
            comp.addVertex(v)
            out_edges = graph.outgoingEdgesOf(v)
            for edge in out_edges:
                child = edge.getTarget()
                comp.addVertex(child)
                comp.addEdge(v, child)
                stack.append(child)
        components.append(comp)

    return components