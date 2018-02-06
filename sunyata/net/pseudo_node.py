from collections import defaultdict


class NodeSequencer(object):
    """
    A cache that creates sequences of nodes connected via the > operator.

    It works by assigning and propagating 'colors' of nodes.
    """

    def __init__(self):
        self._next_color = 1
        self._node2color = {}
        self._color2nodes = defaultdict(list)
        self._prev_right = None

    def connect(self, left, right):
        """
        Evaluate one > comparison, returning a new Sequence.

        Note: the Sequence will be immediately thrown away unless this is the
        last > comparison of the "node > node > node ..." sequence.
        """
        # Save the new previous right node.
        self._prev_right = right

        # Either retrieve or invent the color of the node on the left.
        color = self._node2color.get(left)
        if color is None:
            color = self._next_color
            self._next_color += 1
            self._node2color[left] = color
            self._color2nodes[color].append(left)

        # Propagate the left node's color forward to the right node.
        self._node2color[right] = color
        self._color2nodes[color].append(right)

        # Return a Sequence of the nodes of that color.
        from .sequence import Sequence
        nodes = self._color2nodes[color]
        return Sequence(*nodes)


_SEQ = NodeSequencer()


class PseudoNode(object):
    def desugar(self):
        raise NotImplementedError

    def __gt__(self, right):
        return _SEQ.connect(self, right)

    def __mul__(self, mul):
        raise NotImplementedError

    def pseudo_node_to_pretty(self):
        """
        -> str
        """
        raise NotImplementedError
