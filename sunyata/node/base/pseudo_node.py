from collections import defaultdict
from copy import deepcopy


class Chainer(object):
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
        Evaluate one > comparison, returning a new Chain.

        Note: the Chain will be immediately thrown away unless this is the last
        > comparison of the "node > node > node ..." sequence.
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

        # Return a Chain of the nodes of that color.
        from ..network import Chain
        nodes = self._color2nodes[color]
        return Chain(*nodes)


_CHAIN_CACHE = Chainer()


class PseudoNode(object):
    def desugar(self):
        raise NotImplementedError

    def __gt__(self, right):
        return _CHAIN_CACHE.connect(self, right)

    def pseudo_node_to_pretty(self):
        """
        -> str
        """
        raise NotImplementedError
