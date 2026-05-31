"""Tests for motorch topological sort."""

import pytest
from motorch.autograd.topological_sort import topological_sort
from motorch.tensor import tensor


# ── helpers ───────────────────────────────────────────────────────────────────

def is_valid_topological_order(nodes):
    """
    Verify that every parent appears before all of its children in the sorted list.
    i.e. for every node, all of its _children appear later in the list.
    """
    index = {id(n): i for i, n in enumerate(nodes)}
    for node in nodes:
        for child in node._children:
            if id(child) not in index:
                return False, f"Child {child} not in sorted output"
            if index[id(node)] > index[id(child)]:
                return False, (
                    f"Parent appears after child: "
                    f"parent at {index[id(node)]}, child at {index[id(child)]}"
                )
    return True, ""


# ── single node ───────────────────────────────────────────────────────────────

class TestSingleNode:

    def test_single_node_returned(self):
        x = tensor(1.0, requires_grad=True)
        result = topological_sort(x)
        assert len(result) == 1
        assert result[0] is x

    def test_single_node_no_children(self):
        x = tensor(1.0, requires_grad=True)
        result = topological_sort(x)
        assert result[0]._children == []


# ── linear chains ─────────────────────────────────────────────────────────────

class TestLinearChain:

    def test_two_node_chain_length(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        result = topological_sort(z)
        assert len(result) == 3  # z, x, y

    def test_two_node_chain_root_first(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        result = topological_sort(z)
        assert result[0] is z

    def test_two_node_chain_order(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0)
        z = x + y
        result = topological_sort(z)
        valid, msg = is_valid_topological_order(result)
        assert valid, msg

    def test_three_node_chain_order(self):
        # w = (x + y) + v
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        v = tensor(3.0, requires_grad=True)
        z = x + y
        w = z + v
        result = topological_sort(w)
        valid, msg = is_valid_topological_order(result)
        assert valid, msg

    def test_three_node_chain_length(self):
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        v = tensor(3.0, requires_grad=True)
        z = x + y
        w = z + v
        result = topological_sort(w)
        assert len(result) == 5  # w, z, x, y, v

    def test_long_chain_order(self):
        x = tensor(1.0, requires_grad=True)
        node = x
        for _ in range(10):
            node = node + tensor(1.0, requires_grad=True)
        result = topological_sort(node)
        valid, msg = is_valid_topological_order(result)
        assert valid, msg


# ── diamond / fan-out-fan-in ──────────────────────────────────────────────────

class TestDiamondGraph:

    def test_diamond_order(self):
        # x feeds into both y and z, which both feed into w
        # w = (x + a) + (x + b)
        x = tensor(2.0, requires_grad=True)
        a = tensor(1.0, requires_grad=True)
        b = tensor(3.0, requires_grad=True)
        left  = x + a
        right = x + b
        w = left + right
        result = topological_sort(w)
        valid, msg = is_valid_topological_order(result)
        assert valid, msg

    def test_diamond_no_duplicates(self):
        x = tensor(2.0, requires_grad=True)
        a = tensor(1.0, requires_grad=True)
        b = tensor(3.0, requires_grad=True)
        left  = x + a
        right = x + b
        w = left + right
        result = topological_sort(w)
        ids = [id(n) for n in result]
        assert len(ids) == len(set(ids)), "Duplicate nodes in sorted output"

    def test_shared_leaf_visited_once(self):
        # x is a shared leaf — should appear exactly once
        x = tensor(2.0, requires_grad=True)
        y = x * x
        result = topological_sort(y)
        x_count = sum(1 for n in result if n is x)
        assert x_count == 1

    def test_diamond_root_first(self):
        x = tensor(2.0, requires_grad=True)
        a = tensor(1.0, requires_grad=True)
        b = tensor(3.0, requires_grad=True)
        left  = x + a
        right = x + b
        w = left + right
        result = topological_sort(w)
        assert result[0] is w


# ── all nodes present ─────────────────────────────────────────────────────────

class TestAllNodesPresent:

    def test_all_nodes_in_chain(self):
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        z = x + y
        result = topological_sort(z)
        result_ids = {id(n) for n in result}
        assert id(z) in result_ids
        assert id(x) in result_ids
        assert id(y) in result_ids

    def test_all_nodes_in_deep_graph(self):
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        v = tensor(3.0, requires_grad=True)
        w = tensor(4.0, requires_grad=True)
        z1 = x + y
        z2 = v + w
        out = z1 * z2
        result = topological_sort(out)
        result_ids = {id(n) for n in result}
        for node in [out, z1, z2, x, y, v, w]:
            assert id(node) in result_ids, f"{node} missing from sorted output"

    def test_no_extra_nodes(self):
        # only nodes reachable from root should appear
        x = tensor(1.0, requires_grad=True)
        y = tensor(2.0, requires_grad=True)
        z = x + y
        _unrelated = tensor(99.0, requires_grad=True)  # not in graph
        result = topological_sort(z)
        assert all(n is not _unrelated for n in result)


# ── no-grad leaves ────────────────────────────────────────────────────────────

class TestNoGradLeaves:

    def test_no_grad_leaf_included(self):
        # leaf tensors without requires_grad are still nodes in the graph
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=False)
        z = x + y
        result = topological_sort(z)
        result_ids = {id(n) for n in result}
        assert id(y) in result_ids

    def test_no_grad_leaf_after_parent(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=False)
        z = x + y
        result = topological_sort(z)
        valid, msg = is_valid_topological_order(result)
        assert valid, msg


# ── real backward integration ─────────────────────────────────────────────────

class TestBackwardIntegration:
    """Verify that the sort order produces correct gradients end-to-end."""

    def test_add_backward(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        z = x + y
        z.backward()
        assert float(x.grad.data) == pytest.approx(1.0)
        assert float(y.grad.data) == pytest.approx(1.0)

    def test_mul_backward(self):
        x = tensor(2.0, requires_grad=True)
        y = tensor(5.0, requires_grad=True)
        z = x * y
        z.backward()
        assert float(x.grad.data) == pytest.approx(5.0)
        assert float(y.grad.data) == pytest.approx(2.0)

    def test_chained_backward(self):
        # w = (x + y) * v; dw/dx = v = 4, dw/dy = v = 4, dw/dv = x+y = 5
        x = tensor(2.0, requires_grad=True)
        y = tensor(3.0, requires_grad=True)
        v = tensor(4.0, requires_grad=True)
        w = (x + y) * v
        w.backward()
        assert float(x.grad.data) == pytest.approx(4.0)
        assert float(y.grad.data) == pytest.approx(4.0)
        assert float(v.grad.data) == pytest.approx(5.0)

    def test_diamond_backward_accumulates(self):
        # z = x * x; dz/dx = 2x = 6
        x = tensor(3.0, requires_grad=True)
        z = x * x
        z.backward()
        assert float(x.grad.data) == pytest.approx(6.0)
