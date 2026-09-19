import torch
import numpy as np
from collections import defaultdict
import time
from numba import njit


@njit(cache=True)
def _legalize_bfs(inst_x, inst_y, cap_grid, site_type, size_y, size_x):
    """BFS-based legalization for any site type using 2D grid capacity map."""
    num_inst = inst_x.shape[0]
    assignment = np.full(num_inst, -1, np.int64)
    queue_y = np.zeros(size_x * size_y, dtype=np.int32)
    queue_x = np.zeros(size_x * size_y, dtype=np.int32)
    visited_tag = np.zeros((size_y, size_x), dtype=np.int32)
    DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)

    for i in range(num_inst):
        rx = int(round(inst_x[i]))
        ry = int(round(inst_y[i]))
        rx = min(max(rx, 0), size_x - 1)
        ry = min(max(ry, 0), size_y - 1)

        if cap_grid[ry, rx] > 0:
            cap_grid[ry, rx] -= 1
            assignment[i] = ry * size_x + rx
            continue

        visited_tag[ry, rx] = i + 1
        head = 0
        tail = 1
        queue_y[0] = ry
        queue_x[0] = rx
        found = False
        while head < tail:
            cy = queue_y[head]
            cx = queue_x[head]
            head += 1
            for d in range(4):
                ny = cy + DIRS[d, 0]
                nx = cx + DIRS[d, 1]
                if 0 <= ny < size_y and 0 <= nx < size_x and visited_tag[ny, nx] != i + 1:
                    if cap_grid[ny, nx] > 0:
                        cap_grid[ny, nx] -= 1
                        assignment[i] = ny * size_x + nx
                        found = True
                        break
                    visited_tag[ny, nx] = i + 1
                    queue_y[tail] = ny
                    queue_x[tail] = nx
                    tail += 1
            if found:
                break
    return assignment


@njit(cache=True)
def _legalize_bfs_weighted(inst_x, inst_y, target_x, target_y, cap_grid, size_y, size_x):
    """BFS-based legalization with net-center-aware target positions."""
    num_inst = inst_x.shape[0]
    assignment = np.full(num_inst, -1, np.int64)
    queue_y = np.zeros(size_x * size_y, dtype=np.int32)
    queue_x = np.zeros(size_x * size_y, dtype=np.int32)
    visited_tag = np.zeros((size_y, size_x), dtype=np.int32)
    DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)

    for i in range(num_inst):
        rx = int(round(target_x[i]))
        ry = int(round(target_y[i]))
        rx = min(max(rx, 0), size_x - 1)
        ry = min(max(ry, 0), size_y - 1)

        if cap_grid[ry, rx] > 0:
            cap_grid[ry, rx] -= 1
            assignment[i] = ry * size_x + rx
            continue

        visited_tag[ry, rx] = i + 1
        head = 0
        tail = 1
        queue_y[0] = ry
        queue_x[0] = rx
        found = False
        while head < tail:
            cy = queue_y[head]
            cx = queue_x[head]
            head += 1
            for d in range(4):
                ny = cy + DIRS[d, 0]
                nx = cx + DIRS[d, 1]
                if 0 <= ny < size_y and 0 <= nx < size_x and visited_tag[ny, nx] != i + 1:
                    if cap_grid[ny, nx] > 0:
                        cap_grid[ny, nx] -= 1
                        assignment[i] = ny * size_x + nx
                        found = True
                        break
                    visited_tag[ny, nx] = i + 1
                    queue_y[tail] = ny
                    queue_x[tail] = nx
                    tail += 1
            if found:
                break
    return assignment


@njit(cache=True)
def _legalize_bfs_weighted_v2(inst_x, inst_y, target_x, target_y, cap_grid, size_y, size_x):
    """BFS with weighted target + GP fallback for better wirelength.

    Try target position first, then GP position, then BFS from target.
    This ensures instances with good net-center targets get placed optimally.
    """
    num_inst = inst_x.shape[0]
    assignment = np.full(num_inst, -1, np.int64)
    queue_y = np.zeros(size_x * size_y, dtype=np.int32)
    queue_x = np.zeros(size_x * size_y, dtype=np.int32)
    visited_tag = np.zeros((size_y, size_x), dtype=np.int32)
    DIRS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int32)

    for i in range(num_inst):
        # Try target position first (net center)
        rx = int(round(target_x[i]))
        ry = int(round(target_y[i]))
        rx = min(max(rx, 0), size_x - 1)
        ry = min(max(ry, 0), size_y - 1)

        if cap_grid[ry, rx] > 0:
            cap_grid[ry, rx] -= 1
            assignment[i] = ry * size_x + rx
            continue

        # Try GP position as fallback
        gx = int(round(inst_x[i]))
        gy = int(round(inst_y[i]))
        gx = min(max(gx, 0), size_x - 1)
        gy = min(max(gy, 0), size_y - 1)

        if (gx != rx or gy != ry) and cap_grid[gy, gx] > 0:
            cap_grid[gy, gx] -= 1
            assignment[i] = gy * size_x + gx
            continue

        # BFS from target position
        visited_tag[ry, rx] = i + 1
        # Also mark GP position as visited if different
        if gx != rx or gy != ry:
            visited_tag[gy, gx] = i + 1
            queue_y[0] = ry
            queue_x[0] = rx
            queue_y[1] = gy
            queue_x[1] = gx
            head = 0
            tail = 2
        else:
            head = 0
            tail = 1
            queue_y[0] = ry
            queue_x[0] = rx

        found = False
        while head < tail:
            cy = queue_y[head]
            cx = queue_x[head]
            head += 1
            for d in range(4):
                ny = cy + DIRS[d, 0]
                nx = cx + DIRS[d, 1]
                if 0 <= ny < size_y and 0 <= nx < size_x and visited_tag[ny, nx] != i + 1:
                    if cap_grid[ny, nx] > 0:
                        cap_grid[ny, nx] -= 1
                        assignment[i] = ny * size_x + nx
                        found = True
                        break
                    visited_tag[ny, nx] = i + 1
                    queue_y[tail] = ny
                    queue_x[tail] = nx
                    tail += 1
            if found:
                break
    return assignment


class Legalizer:
    def __init__(self, benchmark, device='cuda'):
        self.bm = benchmark
        self.device = device
        self.size_x = benchmark.size_x
        self.size_y = benchmark.size_y

    def _build_cap_grid(self, sg):
        bm = self.bm
        cap = np.zeros((bm.size_y, bm.size_x), dtype=np.int64)
        for x in range(bm.size_x):
            for y in range(bm.size_y):
                if bm.site_grid[x, y] == sg:
                    cap[y, x] = int(bm.site_cap_grid[x, y])
        return cap

    def _compute_net_centers_vectorized(self, inst_indices, pos_x, pos_y):
        """Vectorized net-center computation using numpy operations.

        Much faster than per-instance Python loop for large SLICE groups.
        """
        bm = self.bm
        num_inst = len(inst_indices)
        pos_x_np = pos_x.cpu().numpy()
        pos_y_np = pos_y.cpu().numpy()

        target_x = pos_x[inst_indices].cpu().numpy().copy()
        target_y = pos_y[inst_indices].cpu().numpy().copy()

        if not hasattr(bm, 'node2net_list'):
            return target_x, target_y

        # For each instance, compute weighted average of net centers
        # Net center = mean position of other nodes in the net
        for i in range(num_inst):
            inst_id = inst_indices[i]
            n_start = bm.node2net_start[inst_id]
            n_end = bm.node2net_start[inst_id + 1]
            if n_start == n_end:
                continue

            cx, cy = 0.0, 0.0
            cnt = 0
            for ni in range(n_start, n_end):
                net_id = bm.node2net_flat[ni]
                s = bm.net2node_start[net_id]
                e = bm.net2node_start[net_id + 1]
                nn = e - s
                if nn < 2:
                    continue
                ncx, ncy = 0.0, 0.0
                net_nn = 0
                for j in range(s, e):
                    nd = bm.net2node_flat[j]
                    if nd != inst_id:
                        ncx += pos_x_np[nd]
                        ncy += pos_y_np[nd]
                        net_nn += 1
                if net_nn > 0:
                    cx += ncx / net_nn
                    cy += ncy / net_nn
                    cnt += 1
            if cnt > 0:
                # Blend: more weight to net center for high-connectivity instances
                alpha = min(0.8, 0.3 + 0.5 * min(cnt / 10.0, 1.0))
                target_x[i] = (1 - alpha) * target_x[i] + alpha * (cx / cnt)
                target_y[i] = (1 - alpha) * target_y[i] + alpha * (cy / cnt)

        return target_x, target_y

    def _compute_connectivity(self, inst_indices):
        """Compute connectivity (number of connected nets) for each instance."""
        bm = self.bm
        connectivity = np.zeros(len(inst_indices), dtype=np.int32)
        for i, inst_id in enumerate(inst_indices):
            if inst_id < len(bm.node2net_list):
                connectivity[i] = len(bm.node2net_list[inst_id])
        return connectivity

    def legalize(self, pos_x, pos_y, wl_aware=True):
        bm = self.bm
        start_time = time.time()

        legal_x = pos_x.clone()
        legal_y = pos_y.clone()

        for sg in range(4):
            site_mask = (bm.site_grid == sg)
            site_positions = np.argwhere(site_mask)
            if len(site_positions) == 0:
                continue

            inst_mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            inst_indices = np.where(inst_mask)[0]
            num_inst = len(inst_indices)

            if num_inst == 0:
                continue

            inst_x = pos_x[inst_indices].cpu().numpy().astype(np.float32)
            inst_y = pos_y[inst_indices].cpu().numpy().astype(np.float32)

            cap = self._build_cap_grid(sg)

            sg_start = time.time()

            if sg == 0:
                # SLICE: Use BFS with distance-based sorting
                # Sort instances by distance to nearest available site (ascending)
                # so instances closest to available sites are placed first
                dist_map = np.full((int(bm.size_y), int(bm.size_x)), 999999, dtype=np.int32)
                bfs_q = []
                bfs_head = 0
                for yy in range(int(bm.size_y)):
                    for xx in range(int(bm.size_x)):
                        if cap[yy, xx] > 0:
                            dist_map[yy, xx] = 0
                            bfs_q.append((yy, xx))
                while bfs_head < len(bfs_q):
                    cy, cx = bfs_q[bfs_head]
                    bfs_head += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < int(bm.size_y) and 0 <= nx < int(bm.size_x) and dist_map[ny, nx] > dist_map[cy, cx] + 1:
                            dist_map[ny, nx] = dist_map[cy, cx] + 1
                            bfs_q.append((ny, nx))
                inst_dist = np.array([dist_map[int(round(inst_y[i])), int(round(inst_x[i]))] for i in range(num_inst)])
                sort_order = np.argsort(inst_dist)
                inst_x_s = inst_x[sort_order]
                inst_y_s = inst_y[sort_order]
                assignment_bfs = _legalize_bfs(
                    inst_x_s, inst_y_s, cap, sg,
                    int(bm.size_y), int(bm.size_x))
                assignment = np.full(num_inst, -1, np.int64)
                assignment[sort_order] = assignment_bfs
                assignment_bfs = assignment
            else:
                # DSP/BRAM/IO: use net-center-aware BFS
                if wl_aware and num_inst <= 50000:
                    # Sort by connectivity for non-SLICE too
                    connectivity = self._compute_connectivity(inst_indices)
                    sort_order = np.argsort(-connectivity)
                    inst_indices_sorted = inst_indices[sort_order]
                    inst_x_sorted = inst_x[sort_order]
                    inst_y_sorted = inst_y[sort_order]

                    target_x, target_y = self._compute_net_centers_vectorized(
                        inst_indices_sorted, pos_x, pos_y)
                    assignment_bfs = _legalize_bfs_weighted_v2(
                        inst_x_sorted, inst_y_sorted,
                        target_x.astype(np.float32),
                        target_y.astype(np.float32),
                        cap, int(bm.size_y), int(bm.size_x))

                    assignment = np.full(num_inst, -1, np.int64)
                    assignment[sort_order] = assignment_bfs
                    assignment_bfs = assignment
                else:
                    assignment_bfs = _legalize_bfs(
                        inst_x, inst_y, cap, sg,
                        int(bm.size_y), int(bm.size_x))

            for i in range(num_inst):
                if assignment_bfs[i] >= 0:
                    inst_id = inst_indices[i]
                    sy = assignment_bfs[i] // int(bm.size_x)
                    sx = assignment_bfs[i] % int(bm.size_x)
                    legal_x[inst_id] = float(sx)
                    legal_y[inst_id] = float(sy)

            assigned = int((assignment_bfs >= 0).sum())
            sg_time = time.time() - sg_start
            print(f"  [LG] Site group {sg}: assigned {assigned}/{num_inst} instances "
                  f"(BFS-WL, {sg_time:.1f}s)")

        for fid in bm.fixed_indices:
            legal_x[fid] = float(bm.fixed_pos[fid, 0])
            legal_y[fid] = float(bm.fixed_pos[fid, 1])

        return legal_x, legal_y


@njit(cache=True)
def _delta_hpwl_swap(pos_x, pos_y, inst1, inst2, x1_old, y1_old, x2_old, y2_old,
                     node2net_start, node2net_flat, net2node_start, net2node_flat):
    delta = 0.0
    n1_start = node2net_start[inst1]
    n1_end = node2net_start[inst1 + 1]
    n2_start = node2net_start[inst2]
    n2_end = node2net_start[inst2 + 1]

    seen = np.zeros(10000, dtype=np.int64)
    seen_cnt = 0

    for ni in range(n1_start, n1_end):
        net_id = node2net_flat[ni]
        seen[seen_cnt] = net_id
        seen_cnt += 1

    for ni in range(n2_start, n2_end):
        net_id = node2net_flat[ni]
        dup = False
        for si in range(seen_cnt):
            if seen[si] == net_id:
                dup = True
                break
        if not dup:
            seen[seen_cnt] = net_id
            seen_cnt += 1

    for si in range(seen_cnt):
        net_id = seen[si]
        ns = net2node_start[net_id]
        ne = net2node_start[net_id + 1]
        nn = ne - ns
        if nn < 2:
            continue

        x_max_old = -1e18
        x_min_old = 1e18
        y_max_old = -1e18
        y_min_old = 1e18
        has1 = False
        has2 = False

        for j in range(ns, ne):
            nd = net2node_flat[j]
            xv = pos_x[nd]
            yv = pos_y[nd]
            if xv > x_max_old:
                x_max_old = xv
            if xv < x_min_old:
                x_min_old = xv
            if yv > y_max_old:
                y_max_old = yv
            if yv < y_min_old:
                y_min_old = yv
            if nd == inst1:
                has1 = True
            elif nd == inst2:
                has2 = True

        if not has1 and not has2:
            continue

        old_hpwl = (x_max_old - x_min_old) + (y_max_old - y_min_old)

        x1_new = x2_old if has1 else x1_old
        y1_new = y2_old if has1 else y1_old
        x2_new = x1_old if has2 else x2_old
        y2_new = y1_old if has2 else y2_old

        x_max_new = -1e18
        x_min_new = 1e18
        y_max_new = -1e18
        y_min_new = 1e18

        for j in range(ns, ne):
            nd = net2node_flat[j]
            if nd == inst1:
                xv = x1_new
                yv = y1_new
            elif nd == inst2:
                xv = x2_new
                yv = y2_new
            else:
                xv = pos_x[nd]
                yv = pos_y[nd]
            if xv > x_max_new:
                x_max_new = xv
            if xv < x_min_new:
                x_min_new = xv
            if yv > y_max_new:
                y_max_new = yv
            if yv < y_min_new:
                y_min_new = yv

        new_hpwl = (x_max_new - x_min_new) + (y_max_new - y_min_new)
        delta += (new_hpwl - old_hpwl)

    return delta


@njit(cache=True)
def _delta_hpwl_move(pos_x, pos_y, inst_id, new_x, new_y,
                     node2net_start, node2net_flat, net2node_start, net2node_flat):
    """Compute delta HPWL for moving a single instance to a new position."""
    delta = 0.0
    old_x = pos_x[inst_id]
    old_y = pos_y[inst_id]

    n_start = node2net_start[inst_id]
    n_end = node2net_start[inst_id + 1]

    for ni in range(n_start, n_end):
        net_id = node2net_flat[ni]
        ns = net2node_start[net_id]
        ne = net2node_start[net_id + 1]
        nn = ne - ns
        if nn < 2:
            continue

        x_max_old = -1e18
        x_min_old = 1e18
        y_max_old = -1e18
        y_min_old = 1e18
        has_inst = False

        for j in range(ns, ne):
            nd = net2node_flat[j]
            if nd == inst_id:
                xv = old_x
                yv = old_y
                has_inst = True
            else:
                xv = pos_x[nd]
                yv = pos_y[nd]
            if xv > x_max_old:
                x_max_old = xv
            if xv < x_min_old:
                x_min_old = xv
            if yv > y_max_old:
                y_max_old = yv
            if yv < y_min_old:
                y_min_old = yv

        if not has_inst:
            continue

        old_hpwl = (x_max_old - x_min_old) + (y_max_old - y_min_old)

        # Compute new bounding box with new position
        x_max_new = -1e18
        x_min_new = 1e18
        y_max_new = -1e18
        y_min_new = 1e18

        for j in range(ns, ne):
            nd = net2node_flat[j]
            if nd == inst_id:
                xv = new_x
                yv = new_y
            else:
                xv = pos_x[nd]
                yv = pos_y[nd]
            if xv > x_max_new:
                x_max_new = xv
            if xv < x_min_new:
                x_min_new = xv
            if yv > y_max_new:
                y_max_new = yv
            if yv < y_min_new:
                y_min_new = yv

        new_hpwl = (x_max_new - x_min_new) + (y_max_new - y_min_new)
        delta += (new_hpwl - old_hpwl)

    return delta


class DetailedPlacer:
    def __init__(self, benchmark, device='cuda', seed=42):
        self.bm = benchmark
        self.device = device
        self.rng = np.random.RandomState(seed)

    def run(self, pos_x, pos_y, num_iterations=50000000, time_budget=60):
        bm = self.bm
        start_time = time.time()

        dp_x = pos_x.cpu().numpy().copy()
        dp_y = pos_y.cpu().numpy().copy()

        node2net_start = bm.node2net_start
        node2net_flat = bm.node2net_flat
        net2node_start = bm.net2node_start
        net2node_flat = bm.net2node_flat

        movable_mask = (~bm.is_fixed)
        if isinstance(movable_mask, torch.Tensor):
            movable_mask = movable_mask.cpu().numpy()
        movable_indices = np.where(movable_mask)[0]
        n_movable = len(movable_indices)

        if n_movable == 0:
            return pos_x.clone(), pos_y.clone()

        node_site_group = bm.node_site_group
        site_grid = bm.site_grid

        # Group movable indices by site group for efficient same-group swaps
        sg_groups = {}
        for idx in range(n_movable):
            inst = movable_indices[idx]
            sg = node_site_group[inst]
            if sg not in sg_groups:
                sg_groups[sg] = []
            sg_groups[sg].append(idx)

        # Convert to arrays for fast random access
        sg_group_arrays = {sg: np.array(indices, dtype=np.int64) for sg, indices in sg_groups.items()}
        sg_group_sizes = {sg: len(arr) for sg, arr in sg_group_arrays.items()}

        # Build position occupancy grid for move operations
        pos_owner = np.full((bm.size_x, bm.size_y), -1, dtype=np.int64)
        for inst in movable_indices:
            ix = int(round(dp_x[inst]))
            iy = int(round(dp_y[inst]))
            if 0 <= ix < bm.size_x and 0 <= iy < bm.size_y:
                pos_owner[ix, iy] = inst

        # Precompute site positions per site group for move operations
        sg_site_positions = {}
        for sg in sg_group_sizes.keys():
            sg_site_positions[sg] = np.argwhere(site_grid == sg)

        # Precompute weighted sampling for site groups
        sg_keys = list(sg_group_sizes.keys())
        sg_weights = np.array([sg_group_sizes[sg] for sg in sg_keys], dtype=np.float64)
        sg_weights /= sg_weights.sum()

        accepted = 0
        temperature = 1.0

        for iteration in range(num_iterations):
            if time.time() - start_time >= time_budget:
                break

            # Pick a site group, then pick two instances from that group
            sg = self.rng.choice(sg_keys, p=sg_weights)
            sg_size = sg_group_sizes[sg]
            if sg_size < 2:
                continue

            idx1_pos = self.rng.randint(0, sg_size)
            idx2_pos = self.rng.randint(0, sg_size)
            if idx1_pos == idx2_pos:
                continue

            idx1 = sg_group_arrays[sg][idx1_pos]
            idx2 = sg_group_arrays[sg][idx2_pos]

            inst1 = movable_indices[idx1]
            inst2 = movable_indices[idx2]

            x1_old = dp_x[inst1]
            y1_old = dp_y[inst1]
            x2_old = dp_x[inst2]
            y2_old = dp_y[inst2]

            # Same site group: swap is always legal (both positions have the correct site type)
            # No need for legality check

            delta = _delta_hpwl_swap(
                dp_x, dp_y, inst1, inst2, x1_old, y1_old, x2_old, y2_old,
                node2net_start, node2net_flat, net2node_start, net2node_flat
            )

            if delta < 0 or self.rng.rand() < np.exp(-delta / (temperature + 1e-10)):
                dp_x[inst1] = x2_old
                dp_y[inst1] = y2_old
                dp_x[inst2] = x1_old
                dp_y[inst2] = y1_old
                pos_owner[int(round(x1_old)), int(round(y1_old))] = inst2
                pos_owner[int(round(x2_old)), int(round(y2_old))] = inst1
                accepted += 1

            # Move attempt: try moving a random instance to an empty site
            sg_m = self.rng.choice(sg_keys, p=sg_weights)
            sg_m_size = sg_group_sizes[sg_m]
            if sg_m_size >= 1:
                idx_m_pos = self.rng.randint(0, sg_m_size)
                idx_m = sg_group_arrays[sg_m][idx_m_pos]
                inst_m = movable_indices[idx_m]
                old_xm = dp_x[inst_m]
                old_ym = dp_y[inst_m]
                site_arr = sg_site_positions[sg_m]
                site_idx = self.rng.randint(0, len(site_arr))
                new_xm = float(site_arr[site_idx, 0])
                new_ym = float(site_arr[site_idx, 1])
                nxi = int(round(new_xm))
                nyi = int(round(new_ym))
                oxi = int(round(old_xm))
                oyi = int(round(old_ym))
                if (nxi != oxi or nyi != oyi) and 0 <= nxi < bm.size_x and 0 <= nyi < bm.size_y and pos_owner[nxi, nyi] == -1:
                    delta_m = _delta_hpwl_move(
                        dp_x, dp_y, inst_m, new_xm, new_ym,
                        node2net_start, node2net_flat, net2node_start, net2node_flat
                    )
                    if delta_m < 0 or self.rng.rand() < np.exp(-delta_m / (temperature + 1e-10)):
                        pos_owner[oxi, oyi] = -1
                        pos_owner[nxi, nyi] = inst_m
                        dp_x[inst_m] = new_xm
                        dp_y[inst_m] = new_ym
                        accepted += 1

            if iteration % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"  [DP] accepted={accepted}/{iteration+1}, T={temperature:.4f}, time={elapsed:.1f}s")

            temperature = max(0.0001, temperature * 0.99999)

        print(f"  [DP] Done: accepted={accepted}/{iteration+1}, time={time.time()-start_time:.1f}s")
        return torch.from_numpy(dp_x).float().to(self.device), torch.from_numpy(dp_y).float().to(self.device)
