def topological_sort(root):
    stack = []
    processing = []
    visited = []
    sorted = []

    def add_node(stack, node):
        stack.append(node)
        visited.append(node)
        processing.append(node)

    add_node(stack, root)
    while stack:
        node = stack.pop() 
        while processing[-1] is not node:
            sorted.append(processing.pop())

        for child in node._children:
            if not any(child is node for node in visited):
                add_node(stack, child)

    sorted.extend(processing[::-1])

    return sorted[::-1]

