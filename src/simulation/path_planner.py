import heapq

class Node:
    """
    A node class for A* Pathfinding.
    """
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"Node(pos={self.position}, f={self.f})"

def heuristic(pos1, pos2):
    """
    Manhattan distance heuristic.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def a_star_search(grid, start, end):
    """
    Returns a list of tuples as a path from the given start to the given end in the given grid.
    Grid values:
    0 = Empty space
    1 = Shelf obstacle
    2 = Charging station (traversable)
    """
    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize open and closed lists
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    
    # Store visited positions to avoid duplicate nodes in open list
    open_set = {start}
    closed_set = set()

    rows = len(grid)
    cols = len(grid[0])

    # Loop until open list is empty
    while open_list:
        # Get the current node
        _, current_node = heapq.heappop(open_list)
        current_pos = current_node.position
        open_set.discard(current_pos)
        closed_set.add(current_pos)

        # Found the goal
        if current_pos == end:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path (start to end)

        # Generate children (4-way connectivity: up, down, left, right)
        adjacent_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for new_position in adjacent_directions:
            # Get node position
            node_position = (current_pos[0] + new_position[0], current_pos[1] + new_position[1])

            # Make sure within range
            if not (0 <= node_position[0] < rows and 0 <= node_position[1] < cols):
                continue

            # Make sure walkability (0 and 2 are walkable, 1 is shelf)
            if grid[node_position[0]][node_position[1]] == 1:
                continue

            # Check if already visited
            if node_position in closed_set:
                continue

            # Create the child node
            child = Node(current_node, node_position)

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = heuristic(child.position, end_node.position)
            child.f = child.g + child.h

            # Check if this child is already in the open list with a lower g cost
            if node_position in open_set:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, (child.f, child))
            open_set.add(node_position)

    # Return empty list if no path found
    return []
