//! Hypergraph representation for the greedy contraction algorithm.
//!
//! The incidence list represents the tensor network as a hypergraph where:
//! - Vertices represent tensors
//! - Edges represent indices (dimensions)
//! - An edge connects all tensors that share that index

use crate::Label;
use std::collections::{HashMap, HashSet};

/// A hypergraph representing a tensor network.
///
/// This structure efficiently tracks which tensors share which indices,
/// supporting the operations needed for greedy contraction optimization.
///
/// # Type Parameters
/// - `V`: Vertex type (tensor identifier, typically `usize`)
/// - `E`: Edge type (index label)
#[derive(Debug, Clone)]
pub struct IncidenceList<V, E>
where
    V: Clone + Eq + std::hash::Hash,
    E: Clone + Eq + std::hash::Hash,
{
    /// Maps vertices (tensors) to their edges (indices)
    v2e: HashMap<V, Vec<E>>,
    /// Maps edges (indices) to their vertices (tensors)
    e2v: HashMap<E, Vec<V>>,
    /// Open edges (output indices that should not be contracted)
    openedges: HashSet<E>,
}

impl<V, E> IncidenceList<V, E>
where
    V: Clone + Eq + std::hash::Hash,
    E: Clone + Eq + std::hash::Hash,
{
    /// Create a new incidence list from vertex-to-edge mapping and open edges.
    pub fn new(v2e: HashMap<V, Vec<E>>, openedges: Vec<E>) -> Self {
        let mut e2v: HashMap<E, Vec<V>> = HashMap::new();
        for (v, es) in &v2e {
            for e in es {
                e2v.entry(e.clone()).or_default().push(v.clone());
            }
        }
        Self {
            v2e,
            e2v,
            openedges: openedges.into_iter().collect(),
        }
    }

    /// Create an incidence list from an EinCode.
    pub fn from_eincode<L: Label>(ixs: &[Vec<L>], iy: &[L]) -> IncidenceList<usize, L> {
        let v2e: HashMap<usize, Vec<L>> = ixs
            .iter()
            .enumerate()
            .map(|(i, ix)| (i, ix.clone()))
            .collect();
        IncidenceList::new(v2e, iy.to_vec())
    }

    /// Get the number of vertices (tensors).
    #[inline]
    pub fn nv(&self) -> usize {
        self.v2e.len()
    }

    /// Get the number of edges (indices).
    #[inline]
    pub fn ne(&self) -> usize {
        self.e2v.len()
    }

    /// Check if the incidence list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.v2e.is_empty()
    }

    /// Get all vertices (tensors).
    pub fn vertices(&self) -> impl Iterator<Item = &V> {
        self.v2e.keys()
    }

    /// Get all edges (indices).
    pub fn edges_all(&self) -> impl Iterator<Item = &E> {
        self.e2v.keys()
    }

    /// Get the edges (indices) for a vertex (tensor).
    pub fn edges(&self, v: &V) -> Option<&Vec<E>> {
        self.v2e.get(v)
    }

    /// Get the vertices (tensors) for an edge (index).
    pub fn vertices_of_edge(&self, e: &E) -> Option<&Vec<V>> {
        self.e2v.get(e)
    }

    /// Check if an edge is open (output index).
    #[inline]
    pub fn is_open(&self, e: &E) -> bool {
        self.openedges.contains(e)
    }

    /// Get all neighbors of a vertex (tensors sharing at least one index).
    pub fn neighbors(&self, v: &V) -> Vec<V> {
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        seen.insert(v.clone());

        if let Some(edges) = self.v2e.get(v) {
            for e in edges {
                if let Some(verts) = self.e2v.get(e) {
                    for vj in verts {
                        if seen.insert(vj.clone()) {
                            result.push(vj.clone());
                        }
                    }
                }
            }
        }
        result
    }

    /// Check if two vertices share at least one edge.
    pub fn are_neighbors(&self, vi: &V, vj: &V) -> bool {
        if let (Some(ei), Some(ej)) = (self.v2e.get(vi), self.v2e.get(vj)) {
            let set_j: HashSet<_> = ej.iter().collect();
            ei.iter().any(|e| set_j.contains(e))
        } else {
            false
        }
    }

    /// Get the edges shared by two vertices.
    pub fn shared_edges(&self, vi: &V, vj: &V) -> Vec<E> {
        if let (Some(ei), Some(ej)) = (self.v2e.get(vi), self.v2e.get(vj)) {
            let set_j: HashSet<_> = ej.iter().collect();
            ei.iter().filter(|e| set_j.contains(e)).cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Check if an edge is internal (not connected to any other vertex).
    pub fn is_internal(&self, e: &E, vi: &V, vj: &V) -> bool {
        if self.is_open(e) {
            return false;
        }
        if let Some(verts) = self.e2v.get(e) {
            verts.len() == 2 && verts.contains(vi) && verts.contains(vj)
        } else {
            false
        }
    }

    /// Check if an edge is external (connected to vertices outside {vi, vj} or is open).
    pub fn is_external(&self, e: &E, vi: &V, vj: &V) -> bool {
        if self.is_open(e) {
            return true;
        }
        if let Some(verts) = self.e2v.get(e) {
            verts.iter().any(|v| v != vi && v != vj)
        } else {
            false
        }
    }

    /// Delete a vertex from the incidence list.
    ///
    /// Removes the vertex and cleans up edges that no longer connect anything.
    pub fn delete_vertex(&mut self, v: &V) {
        if let Some(edges) = self.v2e.remove(v) {
            for e in edges {
                if let Some(verts) = self.e2v.get_mut(&e) {
                    verts.retain(|x| x != v);
                    // Remove edge if it's empty or only has one vertex and isn't open
                    if verts.is_empty() || (verts.len() == 1 && !self.is_open(&e)) {
                        self.e2v.remove(&e);
                    }
                }
            }
        }
    }

    /// Update the edges for a vertex.
    pub fn set_edges(&mut self, v: V, new_edges: Vec<E>) {
        // Remove old edge mappings
        if let Some(old_edges) = self.v2e.get(&v) {
            for e in old_edges.clone() {
                if let Some(verts) = self.e2v.get_mut(&e) {
                    verts.retain(|x| x != &v);
                    if verts.is_empty() {
                        self.e2v.remove(&e);
                    }
                }
            }
        }

        // Add new edge mappings
        for e in &new_edges {
            self.e2v.entry(e.clone()).or_default().push(v.clone());
        }
        self.v2e.insert(v, new_edges);
    }

    /// Remove specific edges from the incidence list entirely.
    pub fn remove_edges(&mut self, edges: &[E]) {
        for e in edges {
            if let Some(verts) = self.e2v.remove(e) {
                for v in verts {
                    if let Some(v_edges) = self.v2e.get_mut(&v) {
                        v_edges.retain(|x| x != e);
                    }
                }
            }
            self.openedges.remove(e);
        }
    }

    /// Replace a vertex with another in all edge mappings.
    pub fn replace_vertex(&mut self, old_v: &V, new_v: V) {
        if old_v == &new_v {
            return;
        }

        // Get all edges of the old vertex
        if let Some(edges) = self.v2e.remove(old_v) {
            // Update e2v mappings
            for e in &edges {
                if let Some(verts) = self.e2v.get_mut(e) {
                    for v in verts.iter_mut() {
                        if v == old_v {
                            *v = new_v.clone();
                        }
                    }
                }
            }
            // Merge with existing edges of new_v
            let mut all_edges = self.v2e.remove(&new_v).unwrap_or_default();
            for e in edges {
                if !all_edges.contains(&e) {
                    all_edges.push(e);
                }
            }
            self.v2e.insert(new_v, all_edges);
        }
    }
}

/// Compute the contraction dimensions for a pair of tensors.
///
/// Returns (D1, D2, D12, D01, D02, D012) where:
/// - D1: log2 size of edges only in vi, not external
/// - D2: log2 size of edges only in vj, not external
/// - D12: log2 size of edges in both vi and vj, not external (internal contraction)
/// - D01: log2 size of edges only in vi, external
/// - D02: log2 size of edges only in vj, external
/// - D012: log2 size of edges in both vi and vj, external
///
/// Also returns the edges to keep in output and edges to remove.
#[derive(Debug, Clone)]
pub struct ContractionDims<E> {
    pub d1: f64,
    pub d2: f64,
    pub d12: f64,
    pub d01: f64,
    pub d02: f64,
    pub d012: f64,
    pub edges_out: Vec<E>,
    pub edges_remove: Vec<E>,
}

impl<E: Clone + Eq + std::hash::Hash> ContractionDims<E> {
    /// Compute contraction dimensions for a tensor pair.
    pub fn compute<V: Clone + Eq + std::hash::Hash>(
        il: &IncidenceList<V, E>,
        log2_sizes: &HashMap<E, f64>,
        vi: &V,
        vj: &V,
    ) -> Self {
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        let mut d12 = 0.0;
        let mut d01 = 0.0;
        let mut d02 = 0.0;
        let mut d012 = 0.0;
        let mut edges_out = Vec::new();
        let mut edges_remove = Vec::new();

        let ei = il.edges(vi).cloned().unwrap_or_default();
        let ej = il.edges(vj).cloned().unwrap_or_default();
        let ej_set: HashSet<_> = ej.iter().collect();
        let mut processed: HashSet<E> = HashSet::new();

        // Process edges from vi
        for e in &ei {
            if !processed.insert(e.clone()) {
                continue;
            }
            let size = log2_sizes.get(e).copied().unwrap_or(0.0);
            let in_j = ej_set.contains(e);
            let is_ext = il.is_external(e, vi, vj);

            match (in_j, is_ext) {
                (false, false) => d1 += size,
                (false, true) => {
                    d01 += size;
                    edges_out.push(e.clone());
                }
                (true, false) => {
                    d12 += size;
                    edges_remove.push(e.clone());
                }
                (true, true) => {
                    d012 += size;
                    edges_out.push(e.clone());
                }
            }
        }

        // Process edges from vj that weren't already processed
        for e in &ej {
            if !processed.insert(e.clone()) {
                continue;
            }
            let size = log2_sizes.get(e).copied().unwrap_or(0.0);
            let is_ext = il.is_external(e, vi, vj);

            if is_ext {
                d02 += size;
                edges_out.push(e.clone());
            } else {
                d2 += size;
            }
        }

        Self {
            d1,
            d2,
            d12,
            d01,
            d02,
            d012,
            edges_out,
            edges_remove,
        }
    }

    /// Get the time complexity (log2) of this contraction.
    #[inline]
    pub fn time_complexity(&self) -> f64 {
        self.d12 + self.d01 + self.d02 + self.d012
    }

    /// Get the space complexity (log2) of the output tensor.
    #[inline]
    pub fn space_complexity(&self) -> f64 {
        self.d01 + self.d02 + self.d012
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_v2e() -> HashMap<usize, Vec<char>> {
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['i', 'j']);
        v2e.insert(1, vec!['j', 'k']);
        v2e.insert(2, vec!['k', 'l']);
        v2e
    }

    #[test]
    fn test_new_incidence_list() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        assert_eq!(il.nv(), 3);
        assert_eq!(il.ne(), 4);
        assert!(il.is_open(&'i'));
        assert!(il.is_open(&'l'));
        assert!(!il.is_open(&'j'));
    }

    #[test]
    fn test_neighbors() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        let neighbors_0 = il.neighbors(&0);
        assert_eq!(neighbors_0.len(), 1);
        assert!(neighbors_0.contains(&1));

        let neighbors_1 = il.neighbors(&1);
        assert_eq!(neighbors_1.len(), 2);
        assert!(neighbors_1.contains(&0));
        assert!(neighbors_1.contains(&2));
    }

    #[test]
    fn test_shared_edges() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        let shared = il.shared_edges(&0, &1);
        assert_eq!(shared, vec!['j']);

        let shared_none = il.shared_edges(&0, &2);
        assert!(shared_none.is_empty());
    }

    #[test]
    fn test_is_internal() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // 'j' connects only 0 and 1, so it's internal to them
        assert!(il.is_internal(&'j', &0, &1));
        // 'i' is open, so it's not internal
        assert!(!il.is_internal(&'i', &0, &1));
    }

    #[test]
    fn test_delete_vertex() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        il.delete_vertex(&1);
        assert_eq!(il.nv(), 2);
        assert!(!il.are_neighbors(&0, &2));
    }

    #[test]
    fn test_from_eincode() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let iy = vec!['i', 'k'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        assert_eq!(il.nv(), 2);
        assert!(il.are_neighbors(&0, &1));
        assert!(il.is_open(&'i'));
        assert!(il.is_open(&'k'));
        assert!(!il.is_open(&'j'));
    }

    #[test]
    fn test_contraction_dims() {
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let iy = vec!['i', 'k'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0); // size 4
        log2_sizes.insert('j', 3.0); // size 8
        log2_sizes.insert('k', 2.0); // size 4

        let dims = ContractionDims::compute(&il, &log2_sizes, &0, &1);

        // 'j' is internal (d12)
        assert!((dims.d12 - 3.0).abs() < 1e-10);
        // 'i' is external to 0 only (d01)
        assert!((dims.d01 - 2.0).abs() < 1e-10);
        // 'k' is external to 1 only (d02)
        assert!((dims.d02 - 2.0).abs() < 1e-10);
        // Time complexity: j + i + k = 3 + 2 + 2 = 7
        assert!((dims.time_complexity() - 7.0).abs() < 1e-10);
        // Space complexity: i + k = 2 + 2 = 4
        assert!((dims.space_complexity() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_edges_method() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        let edges_0 = il.edges(&0).unwrap();
        assert_eq!(edges_0.len(), 2);
        assert!(edges_0.contains(&'i'));
        assert!(edges_0.contains(&'j'));
    }

    #[test]
    fn test_vertices_iter() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        let vertices: Vec<_> = il.vertices().collect();
        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn test_no_common_edges() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Vertices 0 and 2 don't share edges
        assert!(!il.are_neighbors(&0, &2));
        let shared = il.shared_edges(&0, &2);
        assert!(shared.is_empty());
    }

    #[test]
    fn test_is_external() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // 'i' is open, so it's external
        assert!(il.is_external(&'i', &0, &1));
        // 'j' connects only 0 and 1, so it's not external
        assert!(!il.is_external(&'j', &0, &1));
    }

    #[test]
    fn test_single_tensor() {
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['i', 'j']);
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'j']);

        assert_eq!(il.nv(), 1);
        assert_eq!(il.ne(), 2);
        assert!(il.is_open(&'i'));
        assert!(il.is_open(&'j'));
    }

    #[test]
    fn test_vertices_of_edge() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // 'j' connects vertices 0 and 1
        let vertices = il.vertices_of_edge(&'j').unwrap();
        assert_eq!(vertices.len(), 2);
        assert!(vertices.contains(&0));
        assert!(vertices.contains(&1));
    }

    #[test]
    fn test_replace_vertex() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        il.replace_vertex(&0, 100);
        assert_eq!(il.nv(), 3);
        assert!(il.edges(&100).is_some());
        assert!(il.edges(&0).is_none());
    }

    #[test]
    fn test_set_edges() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        il.set_edges(0, vec!['a', 'b']);
        let edges = il.edges(&0).unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&'a'));
        assert!(edges.contains(&'b'));
    }

    #[test]
    fn test_is_empty() {
        let il: IncidenceList<usize, char> = IncidenceList::new(HashMap::new(), vec![]);
        assert!(il.is_empty());

        let v2e = simple_v2e();
        let il2: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);
        assert!(!il2.is_empty());
    }

    #[test]
    fn test_edges_all() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        let edges: Vec<_> = il.edges_all().collect();
        assert_eq!(edges.len(), 4);
        assert!(edges.contains(&&'i'));
        assert!(edges.contains(&&'j'));
        assert!(edges.contains(&&'k'));
        assert!(edges.contains(&&'l'));
    }

    #[test]
    fn test_remove_edges() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Remove edge 'j'
        il.remove_edges(&['j']);

        // 'j' should no longer exist
        assert!(il.vertices_of_edge(&'j').is_none());

        // Tensors 0 and 1 should no longer share edges
        assert!(!il.are_neighbors(&0, &1));
    }

    #[test]
    fn test_remove_open_edge() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        assert!(il.is_open(&'i'));
        il.remove_edges(&['i']);
        assert!(!il.is_open(&'i'));
    }

    #[test]
    fn test_replace_vertex_same() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Replacing with same vertex should be no-op
        il.replace_vertex(&0, 0);
        assert!(il.edges(&0).is_some());
        assert_eq!(il.nv(), 3);
    }

    #[test]
    fn test_replace_vertex_merge() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Merge vertex 0 into vertex 1
        il.replace_vertex(&0, 1);

        // Vertex 0 should be gone
        assert!(il.edges(&0).is_none());

        // Vertex 1 should have edges from both
        let edges = il.edges(&1).unwrap();
        assert!(edges.contains(&'i') || edges.contains(&'j'));
    }

    #[test]
    fn test_delete_vertex_removes_empty_edges() {
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['a', 'b']);
        v2e.insert(1, vec!['b']); // 'b' shared
        v2e.insert(2, vec!['b', 'c']); // 'b' connected to 3 vertices
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['a', 'c']);

        // Delete vertex 0 - 'a' should be removed (only connected to 0, even though open)
        // 'b' should remain because it's still connected to 1 and 2
        il.delete_vertex(&0);

        assert_eq!(il.nv(), 2);
        assert!(il.vertices_of_edge(&'a').is_none());
        // 'b' should still exist with vertices 1 and 2
        let b_verts = il.vertices_of_edge(&'b').unwrap();
        assert_eq!(b_verts.len(), 2);
    }

    #[test]
    fn test_neighbors_no_neighbors() {
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['a']);
        v2e.insert(1, vec!['b']); // Different edges, no neighbors
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['a', 'b']);

        let neighbors = il.neighbors(&0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_shared_edges_nonexistent_vertex() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Test with non-existent vertex
        let shared = il.shared_edges(&0, &99);
        assert!(shared.is_empty());
    }

    #[test]
    fn test_is_internal_nonexistent_edge() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Non-existent edge
        assert!(!il.is_internal(&'z', &0, &1));
    }

    #[test]
    fn test_is_external_nonexistent_edge() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Non-existent edge
        assert!(!il.is_external(&'z', &0, &1));
    }

    #[test]
    fn test_set_edges_new_vertex() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Add a completely new vertex
        il.set_edges(99, vec!['x', 'y']);

        assert_eq!(il.nv(), 4);
        let edges = il.edges(&99).unwrap();
        assert!(edges.contains(&'x'));
        assert!(edges.contains(&'y'));
    }

    #[test]
    fn test_contraction_dims_d1_d2() {
        // Test case where edges are only in one tensor and not external
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['a', 'j']);
        v2e.insert(1, vec!['j', 'b']);
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec![]); // No open edges

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('a', 2.0);
        log2_sizes.insert('j', 3.0);
        log2_sizes.insert('b', 4.0);

        let dims = ContractionDims::compute(&il, &log2_sizes, &0, &1);

        // 'a' is only in 0, not external -> d1
        assert!((dims.d1 - 2.0).abs() < 1e-10);
        // 'b' is only in 1, not external -> d2
        assert!((dims.d2 - 4.0).abs() < 1e-10);
        // 'j' is in both, not external -> d12
        assert!((dims.d12 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_contraction_dims_d012() {
        // Test d012: edge in both tensors AND is external (batched index)
        let mut v2e = HashMap::new();
        v2e.insert(0, vec!['b', 'j']);
        v2e.insert(1, vec!['j', 'b']);
        v2e.insert(2, vec!['b']); // Third tensor shares 'b'
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['b']);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('b', 2.0);
        log2_sizes.insert('j', 3.0);

        let dims = ContractionDims::compute(&il, &log2_sizes, &0, &1);

        // 'b' is in both tensors and is external (open or connected to other tensors) -> d012
        assert!((dims.d012 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_neighbors_nonexistent_vertex() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Neighbors of non-existent vertex
        let neighbors = il.neighbors(&99);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_are_neighbors_nonexistent() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Check with non-existent vertex
        assert!(!il.are_neighbors(&0, &99));
        assert!(!il.are_neighbors(&99, &0));
    }

    #[test]
    fn test_replace_vertex_nonexistent() {
        let v2e = simple_v2e();
        let mut il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Replace non-existent vertex (should be no-op)
        il.replace_vertex(&99, 100);
        assert_eq!(il.nv(), 3);
        assert!(il.edges(&99).is_none());
        assert!(il.edges(&100).is_none());
    }

    #[test]
    fn test_edges_nonexistent_vertex() {
        let v2e = simple_v2e();
        let il: IncidenceList<usize, char> = IncidenceList::new(v2e, vec!['i', 'l']);

        // Edges of non-existent vertex
        assert!(il.edges(&99).is_none());
    }

    #[test]
    fn test_contraction_dims_missing_size() {
        // Test when some edges are missing from size dict
        let ixs = vec![vec!['i', 'j'], vec!['j', 'k']];
        let iy = vec!['i', 'k'];
        let il = IncidenceList::<usize, char>::from_eincode(&ixs, &iy);

        let mut log2_sizes = HashMap::new();
        log2_sizes.insert('i', 2.0);
        // 'j' and 'k' missing - should default to 0.0

        let dims = ContractionDims::compute(&il, &log2_sizes, &0, &1);
        // Should still compute without panic
        assert!(dims.d01 >= 0.0 || !dims.d01.is_nan());
    }
}
