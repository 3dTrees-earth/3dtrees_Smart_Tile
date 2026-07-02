#!/usr/bin/env python3
"""Union-Find data structure with size-biased roots."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class UnionFind:
    """Disjoint-set structure that keeps the larger component as root."""

    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}

    def make_set(self, x, size: int = 0):
        """Create a new set containing only x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = size

    def find(self, x) -> int:
        """Find the root of the set containing x with path compression."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> int:
        """Merge two sets and return the size-biased root."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return root_x
        if self.size.get(root_x, 0) >= self.size.get(root_y, 0):
            self.parent[root_y] = root_x
            self.size[root_x] = self.size.get(root_x, 0) + self.size.get(root_y, 0)
            return root_x
        self.parent[root_x] = root_y
        self.size[root_y] = self.size.get(root_x, 0) + self.size.get(root_y, 0)
        return root_y

    def get_components(self) -> Dict[int, List[int]]:
        """Return connected components as {root: [members]}."""
        components = defaultdict(list)
        for x in self.parent:
            components[self.find(x)].append(x)
        return dict(components)
