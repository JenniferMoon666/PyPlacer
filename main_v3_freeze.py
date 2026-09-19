"""PyPlacer V3 Freeze: DPFPGA GP+LG + GPU-accelerated DP

V3 FREEZE STRATEGY:
- Small circuits (<50K movable): DPFPGA GP+LG + WindowDP with SA
  - Proven best results for small circuits (Ex1: 0.6053x)
- Medium/Large circuits (>=50K movable): DPFPGA GP+LG + GPUBatchDP (PyTorch scatter_reduce)
  - Proven best results for medium/large circuits (Ex2: 0.9347x, Ex3: 0.9841x, Ex4: 0.9552x)

V41 KEY IMPROVEMENTS over V40:
1. Replace Numba CUDA kernels with PyTorch scatter_reduce_ / scatter_add_ for batch delta HPWL
   - Numba CUDA kernels are slow on P102-100 Pascal GPU due to JIT compilation overhead
   - PyTorch's highly-optimized scatter operations leverage cuBLAS/cuDNN backends
2. GPUBatchDP uses PyTorch tensors on GPU for all evaluation
3. Generate ALL valid candidates (not just first per node), evaluate all on GPU, select best per node
4. CPU fallback still uses Numba prange kernels
5. Pipeline: Swap1 -> GPUBatchDP1 -> Swap2 -> GPUBatchDP2 -> Swap3 -> GPUBatchDP3

PYTORCH SCATTER_REDUCE STRATEGY:
- Upload positions to GPU as float32 tensors
- Compute per-net bounding boxes via scatter_reduce_ (amin/amax)
- Build flat (candidate, net) and (candidate, net, pin) index arrays on CPU with numpy
- Upload indices to GPU, gather pin positions, replace moved node positions with torch.where
- Compute new bounding boxes per (candidate, net) pair via scatter_reduce_
- Compute delta HPWL per pair, sum per candidate via scatter_add_
- Return results to CPU as float64
"""
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict

from numba import njit, prange

from main_v39 import (
    ISPD2016Benchmark, BASELINE, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT,
    GPUHPWLTracker, WindowDP, DSPRAMLegalizer, HybridLegalizer,
    save_placement, spread_positions, run_dreamplacefpga_lg,
    load_dpfpga_placement, run_dpfpga_gp_lg,
    _do_swap_refinement,
    HAS_DP_LG, HAS_MCF, _delta_hpwl_swap, _delta_hpwl_move
)


# ============================================================================
# CPU Fallback Kernels (Numba prange) - kept for CPU-only mode
# ============================================================================

@njit(parallel=True, cache=True)
def batch_delta_hpwl_move_cpu(pos_x, pos_y, node2net_start, node2net_flat,
                               net2node_start, net2node_flat,
                               move_nodes, move_new_x, move_new_y,
                               delta_hpwl, x_weight, y_weight, K):
    """Evaluate delta HPWL for K moves in parallel on CPU using prange."""
    for k in prange(K):
        node = move_nodes[k]
        new_x = move_new_x[k]
        new_y = move_new_y[k]

        total_delta = 0.0

        for ni in range(node2net_start[node], node2net_start[node + 1]):
            net = node2net_flat[ni]

            old_min_x = 1e18; old_max_x = -1e18
            old_min_y = 1e18; old_max_y = -1e18
            new_min_x = 1e18; new_max_x = -1e18
            new_min_y = 1e18; new_max_y = -1e18

            for pi in range(net2node_start[net], net2node_start[net + 1]):
                other = net2node_flat[pi]
                ox = pos_x[other]
                oy = pos_y[other]

                old_min_x = min(old_min_x, ox)
                old_max_x = max(old_max_x, ox)
                old_min_y = min(old_min_y, oy)
                old_max_y = max(old_max_y, oy)

                if other == node:
                    nx_val = new_x; ny_val = new_y
                else:
                    nx_val = ox; ny_val = oy

                new_min_x = min(new_min_x, nx_val)
                new_max_x = max(new_max_x, nx_val)
                new_min_y = min(new_min_y, ny_val)
                new_max_y = max(new_max_y, ny_val)

            old_hpwl = x_weight * (old_max_x - old_min_x) + y_weight * (old_max_y - old_min_y)
            new_hpwl = x_weight * (new_max_x - new_min_x) + y_weight * (new_max_y - new_min_y)
            total_delta += new_hpwl - old_hpwl

        delta_hpwl[k] = total_delta


@njit(parallel=True, cache=True)
def batch_delta_hpwl_swap_cpu(pos_x, pos_y, node2net_start, node2net_flat,
                               net2node_start, net2node_flat,
                               swap_node1, swap_node2,
                               delta_hpwl, x_weight, y_weight, K):
    """Evaluate delta HPWL for K swaps in parallel on CPU using prange."""
    for k in prange(K):
        n1 = swap_node1[k]
        n2 = swap_node2[k]
        old_x1 = pos_x[n1]; old_y1 = pos_y[n1]
        old_x2 = pos_x[n2]; old_y2 = pos_y[n2]

        total_delta = 0.0

        # Process nets of n1 (including shared nets)
        for ni in range(node2net_start[n1], node2net_start[n1 + 1]):
            net = node2net_flat[ni]

            old_min_x = 1e18; old_max_x = -1e18
            old_min_y = 1e18; old_max_y = -1e18
            new_min_x = 1e18; new_max_x = -1e18
            new_min_y = 1e18; new_max_y = -1e18

            for pi in range(net2node_start[net], net2node_start[net + 1]):
                other = net2node_flat[pi]
                ox = pos_x[other]
                oy = pos_y[other]

                old_min_x = min(old_min_x, ox)
                old_max_x = max(old_max_x, ox)
                old_min_y = min(old_min_y, oy)
                old_max_y = max(old_max_y, oy)

                if other == n1:
                    nx_val = old_x2; ny_val = old_y2
                elif other == n2:
                    nx_val = old_x1; ny_val = old_y1
                else:
                    nx_val = ox; ny_val = oy

                new_min_x = min(new_min_x, nx_val)
                new_max_x = max(new_max_x, nx_val)
                new_min_y = min(new_min_y, ny_val)
                new_max_y = max(new_max_y, ny_val)

            old_hpwl = x_weight * (old_max_x - old_min_x) + y_weight * (old_max_y - old_min_y)
            new_hpwl = x_weight * (new_max_x - new_min_x) + y_weight * (new_max_y - new_min_y)
            total_delta += new_hpwl - old_hpwl

        # Process nets of n2 that are NOT shared with n1
        for ni in range(node2net_start[n2], node2net_start[n2 + 1]):
            net = node2net_flat[ni]

            shared = False
            for ni2 in range(node2net_start[n1], node2net_start[n1 + 1]):
                if node2net_flat[ni2] == net:
                    shared = True
                    break

            if shared:
                continue

            old_min_x = 1e18; old_max_x = -1e18
            old_min_y = 1e18; old_max_y = -1e18
            new_min_x = 1e18; new_max_x = -1e18
            new_min_y = 1e18; new_max_y = -1e18

            for pi in range(net2node_start[net], net2node_start[net + 1]):
                other = net2node_flat[pi]
                ox = pos_x[other]
                oy = pos_y[other]

                old_min_x = min(old_min_x, ox)
                old_max_x = max(old_max_x, ox)
                old_min_y = min(old_min_y, oy)
                old_max_y = max(old_max_y, oy)

                if other == n1:
                    nx_val = old_x2; ny_val = old_y2
                elif other == n2:
                    nx_val = old_x1; ny_val = old_y1
                else:
                    nx_val = ox; ny_val = oy

                new_min_x = min(new_min_x, nx_val)
                new_max_x = max(new_max_x, nx_val)
                new_min_y = min(new_min_y, ny_val)
                new_max_y = max(new_max_y, ny_val)

            old_hpwl = x_weight * (old_max_x - old_min_x) + y_weight * (old_max_y - old_min_y)
            new_hpwl = x_weight * (new_max_x - new_min_x) + y_weight * (new_max_y - new_min_y)
            total_delta += new_hpwl - old_hpwl

        delta_hpwl[k] = total_delta


# ============================================================================
# GPUBatchDP Class (V41: PyTorch scatter_reduce)
# ============================================================================

class GPUBatchDP:
    """GPU Batch Detailed Placement using PyTorch scatter_reduce for batch delta HPWL.

    V41 Strategy:
    1. Generate ALL valid candidate moves for all movable nodes (10 directions)
    2. Evaluate ALL candidates' delta HPWL on GPU using PyTorch scatter_reduce_
    3. Select best (most negative delta) per node using _select_best_per_node
    4. Apply best moves/swaps sequentially with conflict resolution
    5. Falls back to Numba CPU parallel (prange) if CUDA unavailable
    """

    def __init__(self, benchmark, device='cuda', seed=42, gpu_tracker=None):
        self.bm = benchmark
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.gpu_tracker = gpu_tracker

        bm = benchmark

        # Pre-upload static data to GPU as PyTorch tensors
        if self.device == 'cuda':
            try:
                self.node2net_start_t = torch.tensor(bm.node2net_start, dtype=torch.long, device=device)
                self.node2net_flat_t = torch.tensor(bm.node2net_flat, dtype=torch.long, device=device)
                self.net2node_start_t = torch.tensor(bm.net2node_start, dtype=torch.long, device=device)
                self.net2node_flat_t = torch.tensor(bm.net2node_flat, dtype=torch.long, device=device)
                net_sizes = bm.net2node_start[1:] - bm.net2node_start[:-1]
                self.net_ids_exp_t = torch.tensor(
                    np.repeat(np.arange(bm.num_nets), net_sizes), dtype=torch.long, device=device)
                self.num_nets = bm.num_nets
                print(f"  [GPUBatchDP] Static data uploaded to GPU (PyTorch scatter_reduce mode)")
            except Exception as e:
                print(f"  [GPUBatchDP] GPU upload failed: {e}, falling back to CPU")
                self.device = 'cpu'
        else:
            print(f"  [GPUBatchDP] Using CPU fallback (Numba prange)")

        # Build movable indices
        movable_mask = (~bm.is_fixed)
        if isinstance(movable_mask, torch.Tensor):
            movable_mask = movable_mask.cpu().numpy()
        self.movable_indices = np.where(movable_mask)[0]
        self.n_movable = len(self.movable_indices)

    def _generate_move_candidates(self, pos_x, pos_y, centroids_x, centroids_y, site_count, max_per_node=3):
        """Generate ALL valid candidate moves for all movable nodes.

        10 candidate directions per node:
        [pref1, pref2, pref3, sec1, sec2, sec3, left, right, up, down]

        V41: Returns ALL valid candidates (multiple per node possible).
        GPU will evaluate all, then best per node is selected via _select_best_per_node.

        Args:
            max_per_node: limit candidates per node (keep first = most promising). 0 = no limit.
        """
        bm = self.bm
        size_x = int(bm.size_x)
        size_y = int(bm.size_y)
        movable = self.movable_indices
        n = len(movable)

        # Vectorized: compute current integer positions
        cur_x = np.round(pos_x[movable]).astype(np.int64)
        cur_y = np.round(pos_y[movable]).astype(np.int64)

        # Compute direction to centroid
        dx_c = centroids_x[movable] - pos_x[movable]
        dy_c = centroids_y[movable] - pos_y[movable]

        # Preferred direction - primary: larger |delta|, secondary: smaller
        abs_dx = np.abs(dx_c); abs_dy = np.abs(dy_c)
        pref_dx1 = np.where(abs_dx >= abs_dy, np.sign(dx_c).astype(np.int64), np.int64(0))
        pref_dy1 = np.where(abs_dx < abs_dy, np.sign(dy_c).astype(np.int64), np.int64(0))
        sec_dx1 = np.where(abs_dx < abs_dy, np.sign(dx_c).astype(np.int64), np.int64(0))
        sec_dy1 = np.where(abs_dx >= abs_dy, np.sign(dy_c).astype(np.int64), np.int64(0))

        # Zero out tiny deltas
        pref_dx1[abs_dx < 0.5] = 0; pref_dy1[abs_dy < 0.5] = 0
        sec_dx1[abs_dx < 0.5] = 0; sec_dy1[abs_dy < 0.5] = 0

        # Multi-step towards centroid
        pref_dx2 = 2 * pref_dx1; pref_dy2 = 2 * pref_dy1
        pref_dx3 = 3 * pref_dx1; pref_dy3 = 3 * pref_dy1
        sec_dx2 = 2 * sec_dx1; sec_dy2 = 2 * sec_dy1
        sec_dx3 = 3 * sec_dx1; sec_dy3 = 3 * sec_dy1

        # Build candidate list: 10 candidates per node
        # [pref1, pref2, pref3, sec1, sec2, sec3, left, right, up, down]
        cand_dx = np.stack([pref_dx1, pref_dx2, pref_dx3, sec_dx1, sec_dx2, sec_dx3,
                            -np.ones(n, dtype=np.int64), np.ones(n, dtype=np.int64),
                            np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)], axis=1)
        cand_dy = np.stack([pref_dy1, pref_dy2, pref_dy3, sec_dy1, sec_dy2, sec_dy3,
                            np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
                            -np.ones(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=1)

        # Compute new positions for all candidates
        new_x = cur_x[:, None] + cand_dx  # [n, 10]
        new_y = cur_y[:, None] + cand_dy  # [n, 10]

        # Validity checks: within bounds
        valid = (new_x >= 0) & (new_x < size_x) & (new_y >= 0) & (new_y < size_y)

        # Skip candidates that don't move (dx=dy=0)
        no_move = (cand_dx == 0) & (cand_dy == 0)
        valid &= ~no_move

        # Vectorized site group and capacity check
        node_sg = bm.node_site_group[movable]
        safe_x = np.clip(new_x, 0, size_x - 1)
        safe_y = np.clip(new_y, 0, size_y - 1)

        for c in range(10):
            sg_match = bm.site_grid[safe_x[:, c], safe_y[:, c]] == node_sg
            valid[:, c] &= sg_match
            cap_ok = site_count[safe_x[:, c], safe_y[:, c]] < bm.site_cap_grid[safe_x[:, c], safe_y[:, c]]
            valid[:, c] &= cap_ok

        # V41: Return ALL valid candidates (not just first per node)
        valid_rows, valid_cols = np.where(valid)

        # Limit to max_per_node candidates per node (keep first = most promising)
        if max_per_node > 0 and len(valid_rows) > 0:
            # Compute rank within each node group
            # valid_rows are sorted, so same-node candidates are consecutive
            changes = np.concatenate([[True], valid_rows[:-1] != valid_rows[1:]])
            change_positions = np.where(changes)[0]
            group_id = np.cumsum(changes) - 1
            rank_in_group = np.arange(len(valid_rows)) - change_positions[group_id]
            keep = rank_in_group < max_per_node
            valid_rows = valid_rows[keep]
            valid_cols = valid_cols[keep]

        move_nodes = movable[valid_rows]
        move_new_x = new_x[valid_rows, valid_cols].astype(np.float64)
        move_new_y = new_y[valid_rows, valid_cols].astype(np.float64)

        return move_nodes, move_new_x, move_new_y

    def _generate_swap_candidates(self, pos_x, pos_y, centroids_x, centroids_y, site_inst_map, max_per_node=3):
        """Generate ALL valid candidate swaps for all movable nodes (FULLY VECTORIZED).

        Uses site_inst_map[x, y, sg] lookup table for O(1) partner lookup.
        Tries 5 directions: centroid primary + 4 cardinal.

        V41: Returns ALL valid (node, partner) pairs from all 5 directions.

        Args:
            max_per_node: limit candidates per node (keep first = most promising). 0 = no limit.
        """
        bm = self.bm
        size_x = int(bm.size_x)
        size_y = int(bm.size_y)
        movable = self.movable_indices
        n = len(movable)

        # Vectorized: compute current positions and centroid direction
        cur_x = np.round(pos_x[movable]).astype(np.int64)
        cur_y = np.round(pos_y[movable]).astype(np.int64)
        dx_c = centroids_x[movable] - pos_x[movable]
        dy_c = centroids_y[movable] - pos_y[movable]

        # Primary direction
        abs_dx = np.abs(dx_c); abs_dy = np.abs(dy_c)
        pref_dx = np.where(abs_dx >= abs_dy, np.sign(dx_c).astype(np.int64), np.int64(0))
        pref_dy = np.where(abs_dx < abs_dy, np.sign(dy_c).astype(np.int64), np.int64(0))
        pref_dx[abs_dx < 0.5] = 0; pref_dy[abs_dy < 0.5] = 0

        # Build target positions for all 5 directions: [centroid, left, right, up, down]
        all_target_x = np.stack([
            cur_x + pref_dx,
            cur_x - 1, cur_x + 1,
            cur_x, cur_x
        ], axis=1)  # [n, 5]
        all_target_y = np.stack([
            cur_y + pref_dy,
            cur_y, cur_y,
            cur_y - 1, cur_y + 1
        ], axis=1)  # [n, 5]

        # Check bounds and not-same-position
        in_bounds = (all_target_x >= 0) & (all_target_x < size_x) & \
                    (all_target_y >= 0) & (all_target_y < size_y)
        not_same = (all_target_x != cur_x[:, None]) | (all_target_y != cur_y[:, None])
        valid_dir = in_bounds & not_same

        # Clip for safe indexing
        safe_tx = np.clip(all_target_x, 0, size_x - 1)
        safe_ty = np.clip(all_target_y, 0, size_y - 1)

        # Look up swap partners from site_inst_map (vectorized)
        node_sg = bm.node_site_group[movable].astype(np.int64)
        # For each direction, look up partner
        partners = np.full((n, 5), -1, dtype=np.int64)
        for d in range(5):
            p = site_inst_map[safe_tx[:, d], safe_ty[:, d], node_sg]
            valid = valid_dir[:, d] & (p >= 0) & (p != movable)
            partners[:, d] = np.where(valid, p, -1)

        # V41: Return ALL valid (node, partner) pairs from all directions
        has_partner = partners >= 0  # [n, 5]
        valid_rows, valid_cols = np.where(has_partner)

        # Limit to max_per_node candidates per node (keep first = most promising)
        if max_per_node > 0 and len(valid_rows) > 0:
            # Compute rank within each node group
            # valid_rows are sorted, so same-node candidates are consecutive
            changes = np.concatenate([[True], valid_rows[:-1] != valid_rows[1:]])
            change_positions = np.where(changes)[0]
            group_id = np.cumsum(changes) - 1
            rank_in_group = np.arange(len(valid_rows)) - change_positions[group_id]
            keep = rank_in_group < max_per_node
            valid_rows = valid_rows[keep]
            valid_cols = valid_cols[keep]

        swap_n1 = movable[valid_rows]
        swap_n2 = partners[valid_rows, valid_cols]

        # Filter out invalid partners (shouldn't happen but just in case)
        valid = swap_n2 >= 0
        return swap_n1[valid], swap_n2[valid]

    def _select_best_per_node(self, nodes, new_x, new_y, delta):
        """Select the best (most negative delta) candidate per node.

        Args:
            nodes: array of node indices (may have duplicates)
            new_x: array of new x positions
            new_y: array of new y positions
            delta: array of delta HPWL values

        Returns:
            (best_nodes, best_new_x, best_new_y, best_delta) with one entry per unique node
        """
        if len(nodes) == 0:
            return nodes, new_x, new_y, delta

        # Sort by (node, delta) so that for each node, the best delta comes first
        # lexsort sorts by last key first, so we put delta last
        order = np.lexsort((delta, nodes))
        sorted_nodes = nodes[order]
        # Find first occurrence of each unique node (which will be the best delta
        # since we want the most negative, and lexsort with delta sorts ascending)
        # Actually, we want the most negative delta, so we need to sort by delta descending
        # within each node group. Let's use a different approach:
        # Sort by (node, delta) where delta is negated so most negative comes first
        order = np.lexsort((-delta, nodes))
        sorted_nodes = nodes[order]
        _, first_idx = np.unique(sorted_nodes, return_index=True)
        best_idx = order[first_idx]

        return nodes[best_idx], new_x[best_idx], new_y[best_idx], delta[best_idx]

    def _select_best_swap_per_node(self, swap_n1, swap_n2, delta):
        """Select the best (most negative delta) swap candidate per n1 node.

        Args:
            swap_n1: array of node1 indices (may have duplicates)
            swap_n2: array of node2 indices
            delta: array of delta HPWL values

        Returns:
            (best_n1, best_n2, best_delta) with one entry per unique n1
        """
        if len(swap_n1) == 0:
            return swap_n1, swap_n2, delta

        # Sort by (n1, -delta) so that for each n1, the most negative delta comes first
        order = np.lexsort((-delta, swap_n1))
        sorted_n1 = swap_n1[order]
        _, first_idx = np.unique(sorted_n1, return_index=True)
        best_idx = order[first_idx]

        return swap_n1[best_idx], swap_n2[best_idx], delta[best_idx]

    def _eval_batch_move_pytorch(self, pos_x, pos_y, move_nodes, move_new_x, move_new_y):
        """Evaluate batch delta HPWL for moves using PyTorch scatter_reduce on GPU.

        Algorithm:
        1. Upload pos_x, pos_y to GPU as float32 tensors
        2. Compute per-net bounding boxes using scatter_reduce_
        3. Build flat (candidate, net) pair indices on CPU using numpy
        4. Build flat (candidate, net, pin) triplet indices on CPU
        5. Upload index arrays to GPU
        6. Gather pin positions, replace moved node positions with torch.where
        7. Compute new bounding boxes per pair using scatter_reduce_
        8. Compute delta HPWL per pair, sum per candidate via scatter_add_
        9. Return results to CPU as float64, or None if batch too large
        """
        K = len(move_nodes)
        if K == 0:
            return np.array([], dtype=np.float64)

        bm = self.bm
        device = self.device

        # Step 1: Upload positions to GPU as float32
        x_gpu = torch.tensor(pos_x, dtype=torch.float32, device=device)
        y_gpu = torch.tensor(pos_y, dtype=torch.float32, device=device)

        # Step 2: Compute per-net bounding boxes using scatter_reduce_
        pin_xs = x_gpu[self.net2node_flat_t]
        pin_ys = y_gpu[self.net2node_flat_t]

        net_min_x = torch.full((self.num_nets,), 1e18, dtype=torch.float32, device=device)
        net_max_x = torch.full((self.num_nets,), -1e18, dtype=torch.float32, device=device)
        net_min_y = torch.full((self.num_nets,), 1e18, dtype=torch.float32, device=device)
        net_max_y = torch.full((self.num_nets,), -1e18, dtype=torch.float32, device=device)

        net_min_x.scatter_reduce_(0, self.net_ids_exp_t, pin_xs, reduce='amin', include_self=True)
        net_max_x.scatter_reduce_(0, self.net_ids_exp_t, pin_xs, reduce='amax', include_self=True)
        net_min_y.scatter_reduce_(0, self.net_ids_exp_t, pin_ys, reduce='amin', include_self=True)
        net_max_y.scatter_reduce_(0, self.net_ids_exp_t, pin_ys, reduce='amax', include_self=True)

        # Step 3: Build flat (candidate, net) pair indices on CPU
        node2net_start = bm.node2net_start
        node2net_flat = bm.node2net_flat
        net2node_start = bm.net2node_start
        net2node_flat = bm.net2node_flat

        move_nodes_int = move_nodes.astype(np.int64)

        # Bounds check: move_nodes must be valid
        assert np.all(move_nodes_int >= 0), f"move_nodes has negative values: min={move_nodes_int.min()}"
        assert np.all(move_nodes_int < len(node2net_start) - 1), \
            f"move_nodes out of bounds: max={move_nodes_int.max()}, len(node2net_start)={len(node2net_start)}"

        net_counts = node2net_start[move_nodes_int + 1] - node2net_start[move_nodes_int]
        total_pairs = int(net_counts.sum())

        # Build pair_cand and pair_net using vectorized numpy (no Python loop)
        pair_cand = np.repeat(np.arange(K, dtype=np.int64), net_counts)
        net_starts = node2net_start[move_nodes_int[pair_cand]]
        net_offsets = np.arange(total_pairs, dtype=np.int64) - np.repeat(
            np.cumsum(net_counts) - net_counts, net_counts)
        idx_into_flat = (net_starts + net_offsets).astype(np.int64)

        # Bounds check: flat indices into node2net_flat
        assert np.all(idx_into_flat >= 0), f"flat indices negative: min={idx_into_flat.min()}"
        assert np.all(idx_into_flat < len(node2net_flat)), \
            f"flat indices out of bounds: max={idx_into_flat.max()}, len(node2net_flat)={len(node2net_flat)}"

        pair_net = node2net_flat[idx_into_flat]

        # Bounds check: pair_net must be valid net indices
        assert np.all(pair_net >= 0), f"pair_net negative: min={pair_net.min()}"
        assert np.all(pair_net < len(net2node_start) - 1), \
            f"pair_net out of bounds: max={pair_net.max()}, len(net2node_start)={len(net2node_start)}"

        # Filter out high-degree nets (their delta HPWL is negligible and they explode triplet count)
        MAX_NET_DEGREE = 100
        net_degrees = net2node_start[pair_net + 1] - net2node_start[pair_net]
        keep_pair = net_degrees <= MAX_NET_DEGREE
        if not np.all(keep_pair):
            n_filtered = int((~keep_pair).sum())
            pair_cand = pair_cand[keep_pair]
            pair_net = pair_net[keep_pair]
            net_degrees = net_degrees[keep_pair]
            total_pairs = len(pair_cand)
            if total_pairs == 0:
                return None

        # Step 4: Build flat (candidate, net, pin) triplet indices on CPU
        pin_counts = net2node_start[pair_net + 1] - net2node_start[pair_net]
        total_triplets = int(pin_counts.sum())

        # Diagnostic prints before triplet expansion
        if total_triplets > 50_000_000:
            print(f"  [V41] Move eval: K={K}, pairs={total_pairs}, triplets={total_triplets/1e6:.1f}M")

        # MAX_TRIPLETS: skip batch if too many triplets to avoid OOM
        MAX_TRIPLETS = 100_000_000
        if total_triplets > MAX_TRIPLETS:
            print(f"  [V41] Too many triplets ({total_triplets/1e6:.1f}M), skipping batch")
            return None

        triplet_pair = np.repeat(np.arange(total_pairs, dtype=np.int64), pin_counts)

        pin_offsets = net2node_start[pair_net]
        pin_enumeration = np.arange(total_triplets, dtype=np.int64) - np.repeat(
            np.cumsum(pin_counts) - pin_counts, pin_counts)
        triplet_pin_node = net2node_flat[(pin_offsets[triplet_pair] + pin_enumeration).astype(np.int64)]

        # Step 5: Upload index arrays to GPU
        pair_cand_t = torch.tensor(pair_cand, dtype=torch.long, device=device)
        pair_net_t = torch.tensor(pair_net, dtype=torch.long, device=device)
        triplet_pair_t = torch.tensor(triplet_pair, dtype=torch.long, device=device)
        triplet_pin_node_t = torch.tensor(triplet_pin_node, dtype=torch.long, device=device)
        move_nodes_t = torch.tensor(move_nodes_int, dtype=torch.long, device=device)
        move_new_x_t = torch.tensor(move_new_x.astype(np.float32), dtype=torch.float32, device=device)
        move_new_y_t = torch.tensor(move_new_y.astype(np.float32), dtype=torch.float32, device=device)

        # Step 6: Gather pin positions
        pin_x = x_gpu[triplet_pin_node_t]
        pin_y = y_gpu[triplet_pin_node_t]

        # Step 7: Replace moved node positions
        moved_node_per_triplet = move_nodes_t[pair_cand_t[triplet_pair_t]]
        is_moved = (triplet_pin_node_t == moved_node_per_triplet)
        pin_x = torch.where(is_moved, move_new_x_t[pair_cand_t[triplet_pair_t]], pin_x)
        pin_y = torch.where(is_moved, move_new_y_t[pair_cand_t[triplet_pair_t]], pin_y)

        # Step 8: Compute new bounding boxes per pair using scatter_reduce_
        new_min_x = torch.full((total_pairs,), 1e18, dtype=torch.float32, device=device)
        new_max_x = torch.full((total_pairs,), -1e18, dtype=torch.float32, device=device)
        new_min_y = torch.full((total_pairs,), 1e18, dtype=torch.float32, device=device)
        new_max_y = torch.full((total_pairs,), -1e18, dtype=torch.float32, device=device)

        new_min_x.scatter_reduce_(0, triplet_pair_t, pin_x, reduce='amin', include_self=True)
        new_max_x.scatter_reduce_(0, triplet_pair_t, pin_x, reduce='amax', include_self=True)
        new_min_y.scatter_reduce_(0, triplet_pair_t, pin_y, reduce='amin', include_self=True)
        new_max_y.scatter_reduce_(0, triplet_pair_t, pin_y, reduce='amax', include_self=True)

        # Step 9: Get old bounding boxes
        old_min_x = net_min_x[pair_net_t]
        old_max_x = net_max_x[pair_net_t]
        old_min_y = net_min_y[pair_net_t]
        old_max_y = net_max_y[pair_net_t]

        # Step 10: Compute delta HPWL per pair
        x_w = np.float32(DPFPGA_X_WEIGHT)
        y_w = np.float32(DPFPGA_Y_WEIGHT)
        old_hpwl = x_w * (old_max_x - old_min_x) + y_w * (old_max_y - old_min_y)
        new_hpwl = x_w * (new_max_x - new_min_x) + y_w * (new_max_y - new_min_y)
        pair_delta = new_hpwl - old_hpwl

        # Step 11: Sum per candidate
        cand_delta = torch.zeros(K, dtype=torch.float32, device=device)
        cand_delta.scatter_add_(0, pair_cand_t, pair_delta)

        # Step 12: Return to CPU as float64
        result = cand_delta.cpu().numpy().astype(np.float64)

        # Step 13: Free GPU memory
        del x_gpu, y_gpu, pin_xs, pin_ys
        del net_min_x, net_max_x, net_min_y, net_max_y
        del pair_cand_t, pair_net_t, triplet_pair_t, triplet_pin_node_t
        del move_nodes_t, move_new_x_t, move_new_y_t
        del pin_x, pin_y, moved_node_per_triplet, is_moved
        del new_min_x, new_max_x, new_min_y, new_max_y
        del old_min_x, old_max_x, old_min_y, old_max_y
        del old_hpwl, new_hpwl, pair_delta, cand_delta
        torch.cuda.empty_cache()

        return result

    def _eval_batch_swap_pytorch(self, pos_x, pos_y, swap_node1, swap_node2):
        """Evaluate batch delta HPWL for swaps using PyTorch scatter_reduce on GPU.

        Same as move but:
        - Create pairs for BOTH n1 and n2 nets (concatenate)
        - Position replacement: if pin is n1 -> use n2's pos, if pin is n2 -> use n1's pos
        - NO shared net check (accept slight double-counting error for huge speedup)
        """
        K = len(swap_node1)
        if K == 0:
            return np.array([], dtype=np.float64)

        bm = self.bm
        device = self.device

        # Upload positions to GPU as float32
        x_gpu = torch.tensor(pos_x, dtype=torch.float32, device=device)
        y_gpu = torch.tensor(pos_y, dtype=torch.float32, device=device)

        # Compute per-net bounding boxes using scatter_reduce_
        pin_xs = x_gpu[self.net2node_flat_t]
        pin_ys = y_gpu[self.net2node_flat_t]

        net_min_x = torch.full((self.num_nets,), 1e18, dtype=torch.float32, device=device)
        net_max_x = torch.full((self.num_nets,), -1e18, dtype=torch.float32, device=device)
        net_min_y = torch.full((self.num_nets,), 1e18, dtype=torch.float32, device=device)
        net_max_y = torch.full((self.num_nets,), -1e18, dtype=torch.float32, device=device)

        net_min_x.scatter_reduce_(0, self.net_ids_exp_t, pin_xs, reduce='amin', include_self=True)
        net_max_x.scatter_reduce_(0, self.net_ids_exp_t, pin_xs, reduce='amax', include_self=True)
        net_min_y.scatter_reduce_(0, self.net_ids_exp_t, pin_ys, reduce='amin', include_self=True)
        net_max_y.scatter_reduce_(0, self.net_ids_exp_t, pin_ys, reduce='amax', include_self=True)

        # Build flat (candidate, net) pair indices on CPU
        # For swaps, we include nets of BOTH n1 and n2 (no shared net check)
        node2net_start = bm.node2net_start
        node2net_flat = bm.node2net_flat
        net2node_start = bm.net2node_start
        net2node_flat = bm.net2node_flat

        swap_n1_int = swap_node1.astype(np.int64)
        swap_n2_int = swap_node2.astype(np.int64)

        # Bounds check: swap nodes must be valid
        assert np.all(swap_n1_int >= 0), f"swap_n1 has negative values: min={swap_n1_int.min()}"
        assert np.all(swap_n1_int < len(node2net_start) - 1), \
            f"swap_n1 out of bounds: max={swap_n1_int.max()}, len(node2net_start)={len(node2net_start)}"
        assert np.all(swap_n2_int >= 0), f"swap_n2 has negative values: min={swap_n2_int.min()}"
        assert np.all(swap_n2_int < len(node2net_start) - 1), \
            f"swap_n2 out of bounds: max={swap_n2_int.max()}, len(node2net_start)={len(node2net_start)}"

        # Build pairs for n1 nets (vectorized, no Python loop)
        net_counts_n1 = node2net_start[swap_n1_int + 1] - node2net_start[swap_n1_int]
        total_pairs_n1 = int(net_counts_n1.sum())
        pair_cand_n1 = np.repeat(np.arange(K, dtype=np.int64), net_counts_n1)
        net_starts_n1 = node2net_start[swap_n1_int[pair_cand_n1]]
        net_offsets_n1 = np.arange(total_pairs_n1, dtype=np.int64) - np.repeat(
            np.cumsum(net_counts_n1) - net_counts_n1, net_counts_n1)
        idx_into_flat_n1 = (net_starts_n1 + net_offsets_n1).astype(np.int64)

        # Bounds check: n1 flat indices into node2net_flat
        assert np.all(idx_into_flat_n1 >= 0), f"n1 flat indices negative: min={idx_into_flat_n1.min()}"
        assert np.all(idx_into_flat_n1 < len(node2net_flat)), \
            f"n1 flat indices out of bounds: max={idx_into_flat_n1.max()}, len(node2net_flat)={len(node2net_flat)}"

        pair_net_n1 = node2net_flat[idx_into_flat_n1]

        # Bounds check: n1 pair_net must be valid net indices
        assert np.all(pair_net_n1 >= 0), f"pair_net_n1 negative: min={pair_net_n1.min()}"
        assert np.all(pair_net_n1 < len(net2node_start) - 1), \
            f"pair_net_n1 out of bounds: max={pair_net_n1.max()}, len(net2node_start)={len(net2node_start)}"

        # Build pairs for n2 nets (vectorized, no Python loop)
        net_counts_n2 = node2net_start[swap_n2_int + 1] - node2net_start[swap_n2_int]
        total_pairs_n2 = int(net_counts_n2.sum())
        pair_cand_n2 = np.repeat(np.arange(K, dtype=np.int64), net_counts_n2)
        net_starts_n2 = node2net_start[swap_n2_int[pair_cand_n2]]
        net_offsets_n2 = np.arange(total_pairs_n2, dtype=np.int64) - np.repeat(
            np.cumsum(net_counts_n2) - net_counts_n2, net_counts_n2)
        idx_into_flat_n2 = (net_starts_n2 + net_offsets_n2).astype(np.int64)

        # Bounds check: n2 flat indices into node2net_flat
        assert np.all(idx_into_flat_n2 >= 0), f"n2 flat indices negative: min={idx_into_flat_n2.min()}"
        assert np.all(idx_into_flat_n2 < len(node2net_flat)), \
            f"n2 flat indices out of bounds: max={idx_into_flat_n2.max()}, len(node2net_flat)={len(node2net_flat)}"

        pair_net_n2 = node2net_flat[idx_into_flat_n2]

        # Bounds check: n2 pair_net must be valid net indices
        assert np.all(pair_net_n2 >= 0), f"pair_net_n2 negative: min={pair_net_n2.min()}"
        assert np.all(pair_net_n2 < len(net2node_start) - 1), \
            f"pair_net_n2 out of bounds: max={pair_net_n2.max()}, len(net2node_start)={len(net2node_start)}"

        # Concatenate n1 and n2 pairs
        pair_cand = np.concatenate([pair_cand_n1, pair_cand_n2])
        pair_net = np.concatenate([pair_net_n1, pair_net_n2])
        total_pairs = len(pair_cand)

        # Filter out high-degree nets (their delta HPWL is negligible and they explode triplet count)
        MAX_NET_DEGREE = 100
        net_degrees = net2node_start[pair_net + 1] - net2node_start[pair_net]
        keep_pair = net_degrees <= MAX_NET_DEGREE
        if not np.all(keep_pair):
            pair_cand = pair_cand[keep_pair]
            pair_net = pair_net[keep_pair]
            total_pairs = len(pair_cand)
            if total_pairs == 0:
                return None

        # Build flat (candidate, net, pin) triplet indices on CPU
        pin_counts = net2node_start[pair_net + 1] - net2node_start[pair_net]
        total_triplets = int(pin_counts.sum())

        # Diagnostic prints before triplet expansion
        if total_triplets > 50_000_000:
            print(f"  [V41] Swap eval: K={K}, pairs={total_pairs}, triplets={total_triplets/1e6:.1f}M")

        # MAX_TRIPLETS: skip batch if too many triplets to avoid OOM
        MAX_TRIPLETS = 100_000_000
        if total_triplets > MAX_TRIPLETS:
            print(f"  [V41] Too many triplets ({total_triplets/1e6:.1f}M), skipping batch")
            return None

        triplet_pair = np.repeat(np.arange(total_pairs, dtype=np.int64), pin_counts)

        pin_offsets = net2node_start[pair_net]
        pin_enumeration = np.arange(total_triplets, dtype=np.int64) - np.repeat(
            np.cumsum(pin_counts) - pin_counts, pin_counts)
        triplet_pin_node = net2node_flat[(pin_offsets[triplet_pair] + pin_enumeration).astype(np.int64)]

        # Upload index arrays to GPU
        pair_cand_t = torch.tensor(pair_cand, dtype=torch.long, device=device)
        pair_net_t = torch.tensor(pair_net, dtype=torch.long, device=device)
        triplet_pair_t = torch.tensor(triplet_pair, dtype=torch.long, device=device)
        triplet_pin_node_t = torch.tensor(triplet_pin_node, dtype=torch.long, device=device)
        swap_n1_t = torch.tensor(swap_n1_int, dtype=torch.long, device=device)
        swap_n2_t = torch.tensor(swap_n2_int, dtype=torch.long, device=device)

        # Gather pin positions
        pin_x = x_gpu[triplet_pin_node_t]
        pin_y = y_gpu[triplet_pin_node_t]

        # Position replacement for swap:
        # if pin is n1 -> use n2's pos, if pin is n2 -> use n1's pos
        n1_per_triplet = swap_n1_t[pair_cand_t[triplet_pair_t]]
        n2_per_triplet = swap_n2_t[pair_cand_t[triplet_pair_t]]

        is_n1 = (triplet_pin_node_t == n1_per_triplet)
        is_n2 = (triplet_pin_node_t == n2_per_triplet)

        old_x1_per_triplet = x_gpu[n1_per_triplet]
        old_y1_per_triplet = y_gpu[n1_per_triplet]
        old_x2_per_triplet = x_gpu[n2_per_triplet]
        old_y2_per_triplet = y_gpu[n2_per_triplet]

        pin_x = torch.where(is_n1, old_x2_per_triplet, torch.where(is_n2, old_x1_per_triplet, pin_x))
        pin_y = torch.where(is_n1, old_y2_per_triplet, torch.where(is_n2, old_y1_per_triplet, pin_y))

        # Compute new bounding boxes per pair using scatter_reduce_
        new_min_x = torch.full((total_pairs,), 1e18, dtype=torch.float32, device=device)
        new_max_x = torch.full((total_pairs,), -1e18, dtype=torch.float32, device=device)
        new_min_y = torch.full((total_pairs,), 1e18, dtype=torch.float32, device=device)
        new_max_y = torch.full((total_pairs,), -1e18, dtype=torch.float32, device=device)

        new_min_x.scatter_reduce_(0, triplet_pair_t, pin_x, reduce='amin', include_self=True)
        new_max_x.scatter_reduce_(0, triplet_pair_t, pin_x, reduce='amax', include_self=True)
        new_min_y.scatter_reduce_(0, triplet_pair_t, pin_y, reduce='amin', include_self=True)
        new_max_y.scatter_reduce_(0, triplet_pair_t, pin_y, reduce='amax', include_self=True)

        # Get old bounding boxes
        old_min_x = net_min_x[pair_net_t]
        old_max_x = net_max_x[pair_net_t]
        old_min_y = net_min_y[pair_net_t]
        old_max_y = net_max_y[pair_net_t]

        # Compute delta HPWL per pair
        x_w = np.float32(DPFPGA_X_WEIGHT)
        y_w = np.float32(DPFPGA_Y_WEIGHT)
        old_hpwl = x_w * (old_max_x - old_min_x) + y_w * (old_max_y - old_min_y)
        new_hpwl = x_w * (new_max_x - new_min_x) + y_w * (new_max_y - new_min_y)
        pair_delta = new_hpwl - old_hpwl

        # Sum per candidate
        cand_delta = torch.zeros(K, dtype=torch.float32, device=device)
        cand_delta.scatter_add_(0, pair_cand_t, pair_delta)

        # Return to CPU as float64
        result = cand_delta.cpu().numpy().astype(np.float64)

        # Free GPU memory
        del x_gpu, y_gpu, pin_xs, pin_ys
        del net_min_x, net_max_x, net_min_y, net_max_y
        del pair_cand_t, pair_net_t, triplet_pair_t, triplet_pin_node_t
        del swap_n1_t, swap_n2_t
        del pin_x, pin_y, n1_per_triplet, n2_per_triplet
        del is_n1, is_n2
        del old_x1_per_triplet, old_y1_per_triplet, old_x2_per_triplet, old_y2_per_triplet
        del new_min_x, new_max_x, new_min_y, new_max_y
        del old_min_x, old_max_x, old_min_y, old_max_y
        del old_hpwl, new_hpwl, pair_delta, cand_delta
        torch.cuda.empty_cache()

        return result

    def _eval_batch_move(self, pos_x, pos_y, move_nodes, move_new_x, move_new_y):
        """Evaluate batch delta HPWL for moves. Uses PyTorch GPU if available, else CPU."""
        K = len(move_nodes)
        if K == 0:
            return np.array([], dtype=np.float64)

        if self.device == 'cuda':
            return self._eval_batch_move_pytorch(pos_x, pos_y, move_nodes, move_new_x, move_new_y)
        else:
            # CPU fallback using Numba prange
            delta_hpwl = np.zeros(K, dtype=np.float64)
            bm = self.bm
            batch_delta_hpwl_move_cpu(
                pos_x, pos_y,
                bm.node2net_start, bm.node2net_flat,
                bm.net2node_start, bm.net2node_flat,
                move_nodes, move_new_x, move_new_y,
                delta_hpwl, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT, K)
            return delta_hpwl

    def _eval_batch_swap(self, pos_x, pos_y, swap_node1, swap_node2):
        """Evaluate batch delta HPWL for swaps. Uses PyTorch GPU if available, else CPU."""
        K = len(swap_node1)
        if K == 0:
            return np.array([], dtype=np.float64)

        if self.device == 'cuda':
            return self._eval_batch_swap_pytorch(pos_x, pos_y, swap_node1, swap_node2)
        else:
            # CPU fallback using Numba prange
            delta_hpwl = np.zeros(K, dtype=np.float64)
            bm = self.bm
            batch_delta_hpwl_swap_cpu(
                pos_x, pos_y,
                bm.node2net_start, bm.node2net_flat,
                bm.net2node_start, bm.net2node_flat,
                swap_node1, swap_node2,
                delta_hpwl, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT, K)
            return delta_hpwl

    def _apply_best_moves(self, dp_x, dp_y, move_nodes, move_new_x, move_new_y, delta, site_count):
        """Apply best improving moves with conflict resolution.

        Since each node has only one candidate (after _select_best_per_node),
        we just sort by delta and apply sequentially with target site conflict resolution.

        Returns: number of moves applied
        """
        improving = delta < -0.01
        if not improving.any():
            return 0

        imp_idx = np.where(improving)[0]
        imp_delta = delta[imp_idx]
        imp_nodes = move_nodes[imp_idx]
        imp_new_x = move_new_x[imp_idx]
        imp_new_y = move_new_y[imp_idx]

        # Sort by delta (best first)
        order = np.argsort(imp_delta)
        imp_nodes = imp_nodes[order]
        imp_delta = imp_delta[order]
        imp_new_x = imp_new_x[order]
        imp_new_y = imp_new_y[order]

        # Apply sequentially with target site conflict resolution
        bm = self.bm
        applied = 0
        target_sites = set()

        for i in range(len(imp_nodes)):
            if imp_delta[i] >= -0.01:
                break
            node = int(imp_nodes[i])
            target_x = int(round(imp_new_x[i]))
            target_y = int(round(imp_new_y[i]))
            target = (target_x, target_y)
            if target in target_sites:
                continue

            # Check capacity at target site
            if site_count[target_x, target_y] >= bm.site_cap_grid[target_x, target_y]:
                continue

            # Update site_count: decrement old site, increment new site
            old_x = int(round(dp_x[node]))
            old_y = int(round(dp_y[node]))
            site_count[old_x, old_y] -= 1
            site_count[target_x, target_y] += 1

            # Apply move
            dp_x[node] = imp_new_x[i]
            dp_y[node] = imp_new_y[i]
            target_sites.add(target)
            applied += 1

        return applied

    def _apply_best_swaps(self, dp_x, dp_y, swap_node1, swap_node2, delta):
        """Apply best improving swaps with conflict resolution.

        1. Filter to improving swaps (delta < -0.01)
        2. Sort by delta (best first)
        3. Apply sequentially, skipping node conflicts (no node in two swaps)

        Returns: number of swaps applied
        """
        improving = delta < -0.01
        if not improving.any():
            return 0

        imp_idx = np.where(improving)[0]
        imp_n1 = swap_node1[imp_idx]
        imp_n2 = swap_node2[imp_idx]
        imp_delta = delta[imp_idx]

        # Sort by delta (best first)
        order = np.argsort(imp_delta)
        imp_n1 = imp_n1[order]
        imp_n2 = imp_n2[order]
        imp_delta = imp_delta[order]

        # Apply sequentially with node conflict resolution
        applied = 0
        used_nodes = set()

        for i in range(len(imp_n1)):
            if imp_delta[i] >= -0.01:
                break
            n1 = int(imp_n1[i])
            n2 = int(imp_n2[i])
            if n1 in used_nodes or n2 in used_nodes:
                continue

            dp_x[n1], dp_x[n2] = dp_x[n2], dp_x[n1]
            dp_y[n1], dp_y[n2] = dp_y[n2], dp_y[n1]
            used_nodes.add(n1)
            used_nodes.add(n2)
            applied += 1

        return applied

    def run(self, pos_x, pos_y, time_budget=60):
        """Main batch DP loop. Alternates between move batches and swap batches.

        V41: Evaluate ALL candidates on GPU via PyTorch scatter_reduce, pick best per node,
        apply sequentially with conflict resolution. All hot paths vectorized.

        Args:
            pos_x: torch tensor of x positions (on device)
            pos_y: torch tensor of y positions (on device)
            time_budget: time limit in seconds

        Returns:
            (final_pos_x, final_pos_y): torch tensors on device
        """
        bm = self.bm
        start_time = time.time()

        # Work with numpy arrays on CPU
        dp_x = pos_x.cpu().numpy().copy().astype(np.float64)
        dp_y = pos_y.cpu().numpy().copy().astype(np.float64)

        if self.n_movable == 0:
            return pos_x.clone(), pos_y.clone()

        size_x = int(bm.size_x)
        size_y = int(bm.size_y)
        movable = self.movable_indices
        n_movable = len(movable)

        # Vectorized site count grid builder
        def build_site_count():
            sc = np.zeros((size_x, size_y), dtype=np.int64)
            ix = np.clip(np.round(dp_x[movable]).astype(np.int64), 0, size_x - 1)
            iy = np.clip(np.round(dp_y[movable]).astype(np.int64), 0, size_y - 1)
            np.add.at(sc, (ix, iy), 1)
            return sc

        # Vectorized site instance map builder: site_inst_map[x, y, sg] = last instance at (x,y) with sg
        max_sg = int(bm.node_site_group.max()) + 1
        def build_site_inst_map():
            sim = np.full((size_x, size_y, max_sg), -1, dtype=np.int64)
            ix = np.clip(np.round(dp_x[movable]).astype(np.int64), 0, size_x - 1)
            iy = np.clip(np.round(dp_y[movable]).astype(np.int64), 0, size_y - 1)
            sg = bm.node_site_group[movable].astype(np.int64)
            sim[ix, iy, sg] = movable
            return sim

        # Initial HPWL
        if self.gpu_tracker is not None:
            best_hpwl = self.gpu_tracker.compute_weighted_hpwl(dp_x, dp_y)
        else:
            best_hpwl = float('inf')
        best_x = dp_x.copy()
        best_y = dp_y.copy()

        no_improve_count = 0
        batch_num = 0
        total_moves_applied = 0
        total_swaps_applied = 0
        total_candidates_eval = 0

        # Compute initial centroids
        if self.gpu_tracker is not None:
            centroids_x, centroids_y = self.gpu_tracker.compute_centroids(dp_x, dp_y)
        else:
            centroids_x = dp_x.copy()
            centroids_y = dp_y.copy()

        while time.time() - start_time < time_budget:
            # Alternate between move and swap phases
            for phase in ['move', 'swap']:
                if time.time() - start_time >= time_budget:
                    break

                batch_num += 1

                if phase == 'move':
                    t0 = time.time()
                    site_count = build_site_count()
                    t_sc = time.time() - t0
                    move_nodes, move_new_x, move_new_y = self._generate_move_candidates(
                        dp_x, dp_y, centroids_x, centroids_y, site_count, max_per_node=3)
                    t_gen = time.time() - t0 - t_sc

                    n_cands = len(move_nodes)
                    if n_cands == 0:
                        continue

                    # Subsample if too many candidates
                    MAX_CANDIDATES = 800_000
                    if n_cands > MAX_CANDIDATES:
                        idx = self.rng.choice(n_cands, MAX_CANDIDATES, replace=False)
                        move_nodes = move_nodes[idx]
                        move_new_x = move_new_x[idx]
                        move_new_y = move_new_y[idx]
                        n_cands = MAX_CANDIDATES

                    # Evaluate candidates on GPU
                    t2 = time.time()
                    delta = self._eval_batch_move(dp_x, dp_y, move_nodes, move_new_x, move_new_y)
                    t_eval = time.time() - t2

                    if delta is None or len(delta) != n_cands:
                        # Batch was skipped (too many triplets)
                        continue

                    # Select best per node
                    best_nodes, best_new_x, best_new_y, best_delta = self._select_best_per_node(
                        move_nodes, move_new_x, move_new_y, delta)

                    # Apply best improving moves with conflict resolution
                    n_applied = self._apply_best_moves(
                        dp_x, dp_y, best_nodes, best_new_x, best_new_y, best_delta, site_count)
                    total_moves_applied += n_applied
                    total_candidates_eval += n_cands
                    t1 = time.time()
                    if batch_num <= 3:
                        print(f"  [V41] Move batch {batch_num}: n_cands={n_cands}, best_per_node={len(best_nodes)}, "
                              f"applied={n_applied}, sc={t_sc:.2f}s, gen={t_gen:.2f}s, "
                              f"eval={t_eval:.2f}s, apply={t1-t2-t_eval:.2f}s")

                else:  # swap phase
                    t0 = time.time()
                    site_inst_map = build_site_inst_map()
                    t_sim = time.time() - t0
                    swap_n1, swap_n2 = self._generate_swap_candidates(
                        dp_x, dp_y, centroids_x, centroids_y, site_inst_map, max_per_node=3)
                    t_gen = time.time() - t0 - t_sim

                    n_cands = len(swap_n1)
                    if n_cands == 0:
                        continue

                    # Subsample if too many candidates
                    MAX_CANDIDATES = 800_000
                    if n_cands > MAX_CANDIDATES:
                        idx = self.rng.choice(n_cands, MAX_CANDIDATES, replace=False)
                        swap_n1 = swap_n1[idx]
                        swap_n2 = swap_n2[idx]
                        n_cands = MAX_CANDIDATES

                    # Evaluate candidates on GPU
                    t2 = time.time()
                    delta = self._eval_batch_swap(dp_x, dp_y, swap_n1, swap_n2)
                    t_eval = time.time() - t2

                    if delta is None or len(delta) != n_cands:
                        # Batch was skipped (too many triplets)
                        continue

                    # Select best per node (for n1)
                    best_n1, best_n2, best_delta = self._select_best_swap_per_node(
                        swap_n1, swap_n2, delta)

                    # Apply best improving swaps with conflict resolution
                    n_applied = self._apply_best_swaps(dp_x, dp_y, best_n1, best_n2, best_delta)
                    total_swaps_applied += n_applied
                    total_candidates_eval += n_cands
                    t1 = time.time()
                    if batch_num <= 3:
                        print(f"  [V41] Swap batch {batch_num}: n_cands={n_cands}, best_per_node={len(best_n1)}, "
                              f"applied={n_applied}, sim={t_sim:.2f}s, gen={t_gen:.2f}s, "
                              f"eval={t_eval:.2f}s, apply={t1-t2-t_eval:.2f}s")

                # Periodic HPWL check and progress report
                if batch_num % 5 == 0:
                    elapsed = time.time() - start_time
                    if self.gpu_tracker is not None:
                        hpwl = self.gpu_tracker.compute_weighted_hpwl(dp_x, dp_y)
                    else:
                        hpwl = 0.0

                    if hpwl < best_hpwl:
                        best_hpwl = hpwl
                        best_x = dp_x.copy()
                        best_y = dp_y.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    print(f"  [V41] Batch {batch_num}: cands={total_candidates_eval}, "
                          f"moves={total_moves_applied}, swaps={total_swaps_applied}, "
                          f"HPWL={hpwl:.0f}, best={best_hpwl:.0f}, "
                          f"applied={n_applied}, batch_t={t1-t0:.2f}s, time={elapsed:.1f}s")

                    # Update centroids periodically
                    if self.gpu_tracker is not None and batch_num % 20 == 0:
                        centroids_x, centroids_y = self.gpu_tracker.compute_centroids(dp_x, dp_y)

                    # Early stop if no improvement for 100 consecutive HPWL checks
                    if no_improve_count >= 20:
                        print(f"  [V41] No improvement for 100 checks, stopping early")
                        break

            if no_improve_count >= 20:
                break

        # Revert to best if needed
        if best_hpwl < float('inf'):
            dp_x = best_x
            dp_y = best_y

        elapsed = time.time() - start_time
        print(f"  [V41] Done: batches={batch_num}, moves={total_moves_applied}, "
              f"swaps={total_swaps_applied}, best_HPWL={best_hpwl:.0f}, time={elapsed:.1f}s")

        return (torch.from_numpy(dp_x.astype(np.float32)).to(self.device),
                torch.from_numpy(dp_y.astype(np.float32)).to(self.device))


# ============================================================================
# Modified run_placement for V41
# ============================================================================

def _run_gp_lg(bm, device, is_small, num_movable, bench_start, time_limit):
    """Run PyPlacer GP + LG stages. Returns (legal_x_cpu, legal_y_cpu, gp_hpwl, lg_hpwl)."""
    from main_v39 import _run_gp_lg as v39_run_gp_lg
    return v39_run_gp_lg(bm, device, is_small, num_movable, bench_start, time_limit)


def run_placement(benchmark_dir, output_dir, device='cuda', time_limit=None,
                  use_dpfpga=False, load_pl_file=None):
    bm = ISPD2016Benchmark(benchmark_dir, device=device)
    num_movable = bm.num_movable
    is_small = num_movable < 50000
    is_medium = num_movable < 500000

    # ===== V3 Freeze: All circuits use V41 pipeline with DPFPGA GP+LG =====
    # Small circuits: DPFPGA GP+LG + WindowDP with SA
    # Medium/Large circuits: DPFPGA GP+LG + GPUBatchDP (PyTorch scatter_reduce)
    print(f"\n{'='*60}")
    if is_small:
        print(f"  PyPlacer V3 Freeze: Small circuit -> DPFPGA GP+LG + WindowDP(SA)")
    else:
        mode = "DPFPGA GP+LG + Swap+GPUBatchDP" if (use_dpfpga or load_pl_file) else "PyPlacer GP + LG + Swap+GPUBatchDP"
        print(f"  PyPlacer V3 Freeze: Medium/Large circuit -> {mode}")
    print(f"  {bm.name}: {num_movable} movable, {bm.num_nets} nets, FPGA {bm.size_x}x{bm.size_y}")
    print(f"  DREAMPlaceFPGA LG available: {HAS_DP_LG}")
    print(f"  MCF available: {HAS_MCF}")
    if not is_small:
        print(f"  GPU Batch DP: {'PyTorch scatter_reduce' if device == 'cuda' else 'CPU (prange)'}")
        print(f"  GPU HPWL Tracker: {'YES' if device == 'cuda' else 'NO'}")
    print(f"{'='*60}")

    bench_start = time.time()

    # ===== Option: Load placement from DREAMPlaceFPGA's GP+LG output =====
    if use_dpfpga or load_pl_file:
        if load_pl_file:
            pl_file = load_pl_file
        else:
            pl_file = run_dpfpga_gp_lg(bm.name, benchmark_dir)
            if pl_file is None:
                print(f"  [ERROR] DREAMPlaceFPGA GP+LG failed, falling back to PyPlacer GP")
                use_dpfpga = False

        if use_dpfpga or load_pl_file:
            legal_x, legal_y = load_dpfpga_placement(bm, pl_file)
            legal_x_dev = legal_x.to(device)
            legal_y_dev = legal_y.to(device)
            lg_hpwl_raw = bm.compute_hpwl_fast(legal_x_dev, legal_y_dev)
            lg_hpwl = bm.compute_hpwl_fast(legal_x_dev, legal_y_dev, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
            del legal_x_dev, legal_y_dev
            baseline = BASELINE.get(bm.name, 0)
            print(f"  [DPFPGA] LG HPWL (raw):      {lg_hpwl_raw:.0f}")
            print(f"  [DPFPGA] LG HPWL (weighted): {lg_hpwl:.0f} (baseline: {baseline}, ratio: {lg_hpwl/baseline:.2f}x)")
            gp_hpwl = lg_hpwl
            legal_x_cpu = legal_x.cpu()
            legal_y_cpu = legal_y.cpu()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()
            print(f"  Skipping GP and LG stages (using DREAMPlaceFPGA output)")
            print(f"  Time so far: {time.time()-bench_start:.1f}s")
        else:
            legal_x_cpu, legal_y_cpu, gp_hpwl, lg_hpwl = _run_gp_lg(
                bm, device, is_small, num_movable, bench_start, time_limit)
    else:
        legal_x_cpu, legal_y_cpu, gp_hpwl, lg_hpwl = _run_gp_lg(
            bm, device, is_small, num_movable, bench_start, time_limit)

    if legal_x_cpu is None:
        return None

    # Create GPU HPWL tracker for the rest of the pipeline
    gpu_tracker = None
    if device == 'cuda':
        gpu_tracker = GPUHPWLTracker(bm, device=device)
        print(f"  [GPU-HPWL] Tracker initialized (scatter_reduce_ acceleration)")

    # V41: No GPU Directed Move Phase (Stage 3 removed - it always reverts, wastes time)

    # DSP/BRAM/IO swap refinement (SLICE swap skipped for medium/large)
    pos_x_np = legal_x_cpu.numpy().copy().astype(np.float64)
    pos_y_np = legal_y_cpu.numpy().copy().astype(np.float64)

    pos_x_np, pos_y_np = _do_swap_refinement(bm, pos_x_np, pos_y_np, is_small, is_medium, "Swap1")

    refine_x = torch.from_numpy(pos_x_np.astype(np.float32)).to(device)
    refine_y = torch.from_numpy(pos_y_np.astype(np.float32)).to(device)

    for fid in bm.fixed_indices:
        refine_x[fid] = float(bm.fixed_pos[fid, 0])
        refine_y[fid] = float(bm.fixed_pos[fid, 1])

    refine_hpwl = bm.compute_hpwl_fast(refine_x, refine_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    baseline = BASELINE.get(bm.name, 0)
    print(f"  [After Swap1] Weighted HPWL: {refine_hpwl:.0f} (ratio: {refine_hpwl/baseline:.4f}x)")

    if time_limit and (time.time() - bench_start) > time_limit: return None

    # ===== Stage 4: Detailed Placement (Round 1) =====
    # V41: Use GPUBatchDP for medium/large circuits, WindowDP for small
    dp1_time = 1200 if is_small else (5400 if is_medium else 5400)

    if is_small:
        print(f"\n[Stage 4] Detailed Placement Round 1 (WindowDP+GPU+Centroid+SA, budget={dp1_time}s)")
        dp = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,
                      sa_enabled=True, use_centroid_guidance=True)
        dp_pos_x, dp_pos_y = dp.run(refine_x, refine_y, num_iterations=100000000, time_budget=dp1_time)
    else:
        print(f"\n[Stage 4] Detailed Placement Round 1 (GPUBatchDP, budget={dp1_time}s)")
        dp = GPUBatchDP(bm, device=device, seed=42, gpu_tracker=gpu_tracker)
        dp_pos_x, dp_pos_y = dp.run(refine_x, refine_y, time_budget=dp1_time)

    dp_hpwl = bm.compute_hpwl_fast(dp_pos_x, dp_pos_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    print(f"  [After DP1] Weighted HPWL: {dp_hpwl:.0f} (ratio: {dp_hpwl/baseline:.4f}x)")

    if time_limit and (time.time() - bench_start) > time_limit: return None

    # ===== Stage 5: Second round Swap + DP =====
    pos_x_np = dp_pos_x.cpu().numpy().copy().astype(np.float64)
    pos_y_np = dp_pos_y.cpu().numpy().copy().astype(np.float64)

    pos_x_np, pos_y_np = _do_swap_refinement(bm, pos_x_np, pos_y_np, is_small, is_medium, "Swap2")

    round2_x = torch.from_numpy(pos_x_np.astype(np.float32)).to(device)
    round2_y = torch.from_numpy(pos_y_np.astype(np.float32)).to(device)
    for fid in bm.fixed_indices:
        round2_x[fid] = float(bm.fixed_pos[fid, 0])
        round2_y[fid] = float(bm.fixed_pos[fid, 1])

    round2_hpwl = bm.compute_hpwl_fast(round2_x, round2_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    print(f"  [After Swap2] Weighted HPWL: {round2_hpwl:.0f} (ratio: {round2_hpwl/baseline:.4f}x)")

    dp2_time = 600 if is_small else (1800 if is_medium else 1800)

    if is_small:
        print(f"\n[Stage 5b] Second round DP (WindowDP+GPU+Centroid+SA, budget={dp2_time}s)")
        dp2 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,
                       sa_enabled=True, use_centroid_guidance=True)
        dp2_pos_x, dp2_pos_y = dp2.run(round2_x, round2_y, num_iterations=100000000, time_budget=dp2_time)
    else:
        print(f"\n[Stage 5b] Second round DP (GPUBatchDP, budget={dp2_time}s)")
        dp2 = GPUBatchDP(bm, device=device, seed=43, gpu_tracker=gpu_tracker)
        dp2_pos_x, dp2_pos_y = dp2.run(round2_x, round2_y, time_budget=dp2_time)

    dp2_hpwl = bm.compute_hpwl_fast(dp2_pos_x, dp2_pos_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    print(f"  [After DP2] Weighted HPWL: {dp2_hpwl:.0f} (ratio: {dp2_hpwl/baseline:.4f}x)")

    if time_limit and (time.time() - bench_start) > time_limit: return None

    # ===== Stage 6: Third round Swap + DP (fine-tuning) =====
    pos_x_np = dp2_pos_x.cpu().numpy().copy().astype(np.float64)
    pos_y_np = dp2_pos_y.cpu().numpy().copy().astype(np.float64)

    pos_x_np, pos_y_np = _do_swap_refinement(bm, pos_x_np, pos_y_np, is_small, is_medium, "Swap3")

    round3_x = torch.from_numpy(pos_x_np.astype(np.float32)).to(device)
    round3_y = torch.from_numpy(pos_y_np.astype(np.float32)).to(device)
    for fid in bm.fixed_indices:
        round3_x[fid] = float(bm.fixed_pos[fid, 0])
        round3_y[fid] = float(bm.fixed_pos[fid, 1])

    round3_hpwl = bm.compute_hpwl_fast(round3_x, round3_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    print(f"  [After Swap3] Weighted HPWL: {round3_hpwl:.0f} (ratio: {round3_hpwl/baseline:.4f}x)")

    dp3_time = 300 if is_small else (900 if is_medium else 900)

    if is_small:
        print(f"\n[Stage 6b] Third round DP (WindowDP+GPU+Centroid+SA, budget={dp3_time}s)")
        dp3 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,
                       sa_enabled=True, use_centroid_guidance=True)
        final_pos_x, final_pos_y = dp3.run(round3_x, round3_y, num_iterations=100000000, time_budget=dp3_time)
    else:
        print(f"\n[Stage 6b] Third round DP (GPUBatchDP, budget={dp3_time}s)")
        dp3 = GPUBatchDP(bm, device=device, seed=44, gpu_tracker=gpu_tracker)
        final_pos_x, final_pos_y = dp3.run(round3_x, round3_y, time_budget=dp3_time)

    final_hpwl_raw = bm.compute_hpwl_fast(final_pos_x, final_pos_y)
    final_hpwl = bm.compute_hpwl_fast(final_pos_x, final_pos_y, DPFPGA_X_WEIGHT, DPFPGA_Y_WEIGHT)
    ratio = final_hpwl / baseline if baseline > 0 else 0
    print(f"\n  Final HPWL (raw):      {final_hpwl_raw:.0f}")
    print(f"  Final HPWL (weighted): {final_hpwl:.0f} (x_w={DPFPGA_X_WEIGHT}, y_w={DPFPGA_Y_WEIGHT})")
    print(f"  Baseline:              {baseline}")
    print(f"  Ratio:                 {ratio:.4f}x")
    print(f"  Total time: {time.time()-bench_start:.1f}s")

    save_placement(bm, final_pos_x, final_pos_y, output_dir)
    return final_hpwl, ratio


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    benchmark_base = r"D:\Codes\VLSI\DREAMPlaceFPGA-main\benchmarks\sample_ispd2016_benchmarks"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    use_dpfpga = '--dpfpga' in sys.argv
    load_pl_file = None
    examples = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--dpfpga':
            i += 1
        elif args[i] == '--load-pl':
            if i + 1 < len(args):
                load_pl_file = args[i + 1]
                i += 2
            else:
                print("ERROR: --load-pl requires a file path")
                return
        else:
            examples.append(args[i])
            i += 1

    if not examples:
        examples = ['FPGA-example1', 'FPGA-example2', 'FPGA-example3', 'FPGA-example4']

    results = {}
    for ex in examples:
        ex_dir = os.path.join(benchmark_base, ex)
        if not os.path.exists(ex_dir):
            print(f"Benchmark not found: {ex_dir}")
            continue
        try:
            hpwl, ratio = run_placement(ex_dir, output_dir, device=device,
                                        use_dpfpga=use_dpfpga, load_pl_file=load_pl_file)
            if hpwl is not None:
                results[ex] = (hpwl, ratio)
            else:
                print(f"\n[TIMEOUT] {ex}")
        except Exception as e:
            print(f"\n[ERROR] {ex} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"{'Example':<20} {'HPWL':>15} {'Baseline':>15} {'Ratio':>10}")
    print(f"{'-'*60}")
    for ex in examples:
        if ex in results:
            hpwl, ratio = results[ex]
            baseline = BASELINE.get(ex, 0)
            print(f"{ex:<20} {hpwl:>15.0f} {baseline:>15} {ratio:>10.4f}x")
        else:
            print(f"{ex:<20} {'FAILED/TIMEOUT':>15}")


if __name__ == '__main__':
    main()
