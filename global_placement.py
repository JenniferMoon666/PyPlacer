import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict


class EPlaceGlobalPlacer:
    def __init__(self, benchmark, device='cuda', num_bins=None):
        self.bm = benchmark
        self.device = device
        self.size_x = benchmark.size_x
        self.size_y = benchmark.size_y

        # Use coarser bins for better density gradient signal
        # Fine bins (512x512) cause overflow to be near-zero when instances are sparse
        self.movable_idx = benchmark.movable_indices
        self.fixed_idx = benchmark.fixed_indices
        self.num_movable = benchmark.num_movable
        self.num_fixed = benchmark.num_fixed

        # Use adaptive bin sizes based on instance count
        # More instances → finer bins for better density gradient
        if num_bins is not None:
            self.num_bins_x = num_bins
            self.num_bins_y = num_bins
        else:
            n_movable = len(self.movable_idx)
            nb = max(32, min(128, int(np.sqrt(n_movable / 50))))
            self.num_bins_x = max(32, min(128, nb))
            self.num_bins_y = max(32, min(128, int(nb * self.size_y / self.size_x)))
        self.bin_size_x = self.size_x / self.num_bins_x
        self.bin_size_y = self.size_y / self.num_bins_y

        self.pos_x = benchmark.pos_x.clone()
        self.pos_y = benchmark.pos_y.clone()

        self._random_init_movable()
        self._precompute_density_params()
        self._precompute_fft_params()
        self._precompute_net_data_vectorized()

    def _random_init_movable(self):
        bm = self.bm
        rng = np.random.RandomState(42)

        for sg in range(4):
            mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            inst_indices = np.where(mask)[0]
            if len(inst_indices) == 0:
                continue

            positions = bm.site_positions.get(sg, np.empty((0, 2)))
            if len(positions) == 0:
                continue

            chosen = rng.choice(len(positions), size=len(inst_indices), replace=True)
            selected = positions[chosen]
            self.pos_x[inst_indices] = torch.from_numpy(selected[:, 0]).float().to(self.device)
            self.pos_y[inst_indices] = torch.from_numpy(selected[:, 1]).float().to(self.device)

        noise_x = (torch.rand_like(self.pos_x) - 0.5) * 2.0
        noise_y = (torch.rand_like(self.pos_y) - 0.5) * 2.0
        noise_x[bm.is_fixed_t] = 0
        noise_y[bm.is_fixed_t] = 0
        self.pos_x += noise_x
        self.pos_y += noise_y
        self.pos_x.clamp_(0, self.size_x - 1)
        self.pos_y.clamp_(0, self.size_y - 1)

    def _precompute_density_params(self):
        bm = self.bm
        self.site_type_capacity = torch.zeros((4, self.num_bins_x, self.num_bins_y),
                                               dtype=torch.float32, device=self.device)
        self.site_type_bin_mask = torch.zeros((4, self.num_bins_x, self.num_bins_y),
                                               dtype=torch.bool, device=self.device)
        for sg in range(4):
            positions = bm.site_positions.get(sg, np.empty((0, 2)))
            for pos in positions:
                x, y = int(pos[0]), int(pos[1])
                bx = min(int(x / self.bin_size_x), self.num_bins_x - 1)
                by = min(int(y / self.bin_size_y), self.num_bins_y - 1)
                self.site_type_capacity[sg, bx, by] += bm.site_cap_grid[x, y]
                self.site_type_bin_mask[sg, bx, by] = True

        self.target_density = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.target_capacity = torch.zeros((4, self.num_bins_x, self.num_bins_y),
                                            dtype=torch.float32, device=self.device)
        for sg in range(4):
            mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            num_inst = mask.sum().item()
            total_cap = self.site_type_capacity[sg].sum().item()
            site_bins = self.site_type_bin_mask[sg]
            num_site_bins = site_bins.sum().item()
            if total_cap > 0 and num_site_bins > 0:
                # Target: each bin should hold instances proportional to its site capacity
                # This means bins with more site capacity should hold more instances
                # Scale so total target = num_instances (not total site capacity)
                bin_cap = self.site_type_capacity[sg]
                total_bin_cap = bin_cap.sum()
                if total_bin_cap > 0:
                    self.target_capacity[sg] = bin_cap * (num_inst / total_bin_cap)
                else:
                    self.target_capacity[sg] = torch.where(
                        site_bins,
                        torch.full_like(self.target_capacity[sg], num_inst / num_site_bins),
                        torch.zeros_like(self.target_capacity[sg])
                    )
                self.target_density[sg] = num_inst / total_cap
            else:
                self.target_density[sg] = 0.0

    def _precompute_fft_params(self):
        M = self.num_bins_x
        N = self.num_bins_y

        # DCT-II frequency vectors: wu_k = pi*k/M, wv_k = pi*k/N
        wu = torch.arange(M, dtype=torch.float32, device=self.device).mul(np.pi / M).view(M, 1)
        wv = torch.arange(N, dtype=torch.float32, device=self.device).mul(np.pi / N).view(1, N)

        wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
        wu2_plus_wv2[0, 0] = 1.0

        self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
        self.inv_wu2_plus_wv2[0, 0] = 0.0
        self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(0.5)
        self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(0.5)

    def _precompute_site_attraction(self):
        bm = self.bm
        self.nearest_site_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        self.nearest_site_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)

        for sg in range(4):
            mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            inst_indices = np.where(mask)[0]
            if len(inst_indices) == 0:
                continue

            inst_x = self.pos_x[inst_indices]
            inst_y = self.pos_y[inst_indices]

            ix = inst_x.round().long().clamp(0, self.size_x - 1)
            iy = inst_y.round().long().clamp(0, self.size_y - 1)

            site_mask = torch.tensor(bm.site_grid == sg, dtype=torch.bool, device=self.device)
            site_xs = torch.where(site_mask)[0].float()
            site_ys = torch.where(site_mask)[1].float()

            if len(site_xs) == 0:
                continue

            chunk = 500
            for c in range(0, len(inst_indices), chunk):
                ce = min(c + chunk, len(inst_indices))
                idx = inst_indices[c:ce]
                cx = ix[c:ce].unsqueeze(1).float()
                cy = iy[c:ce].unsqueeze(1).float()
                sx = site_xs.unsqueeze(0)
                sy = site_ys.unsqueeze(0)
                dist = (cx - sx).pow(2) + (cy - sy).pow(2)
                nearest = dist.argmin(dim=1)
                self.nearest_site_x[idx] = site_xs[nearest]
                self.nearest_site_y[idx] = site_ys[nearest]

        for fid in bm.fixed_indices:
            self.nearest_site_x[fid] = bm.fixed_pos[fid, 0]
            self.nearest_site_y[fid] = bm.fixed_pos[fid, 1]

    def _precompute_net_data_vectorized(self):
        bm = self.bm
        valid_net_ids = []
        valid_pin_offsets = []
        for net_id in range(bm.num_nets):
            s = bm.net2node_start[net_id]
            e = bm.net2node_start[net_id + 1]
            if e - s >= 2:
                valid_net_ids.append(net_id)
                valid_pin_offsets.append((s, e))

        self.num_valid_nets = len(valid_net_ids)

        all_pin_node_ids = []
        all_pin_net_ids = []
        for new_net_id, net_id in enumerate(valid_net_ids):
            s = bm.net2node_start[net_id]
            e = bm.net2node_start[net_id + 1]
            for pin_offset in range(s, e):
                all_pin_node_ids.append(bm.net2node_flat[pin_offset])
                all_pin_net_ids.append(new_net_id)

        self.pin_node_ids = torch.tensor(all_pin_node_ids, dtype=torch.int64, device=self.device)
        self.pin_net_ids = torch.tensor(all_pin_net_ids, dtype=torch.int64, device=self.device)
        self.num_total_pins = len(all_pin_node_ids)

        net_sizes = []
        for net_id in valid_net_ids:
            s = bm.net2node_start[net_id]
            e = bm.net2node_start[net_id + 1]
            net_sizes.append(e - s)
        self.net_sizes = torch.tensor(net_sizes, dtype=torch.float32, device=self.device)

    def _compute_wl_gradient_wawl(self, gamma=1.0):
        bm = self.bm
        inv_gamma = 1.0 / max(gamma, 1e-6)

        pin_x = self.pos_x[self.pin_node_ids]
        pin_y = self.pos_y[self.pin_node_ids]

        net_max_x = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
        net_min_x = torch.zeros(self.num_valid_nets, device=self.device).fill_(1e18)
        net_max_x.scatter_reduce_(0, self.pin_net_ids, pin_x, reduce='amax', include_self=True)
        net_min_x.scatter_reduce_(0, self.pin_net_ids, pin_x, reduce='amin', include_self=True)
        net_max_y = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
        net_min_y = torch.zeros(self.num_valid_nets, device=self.device).fill_(1e18)
        net_max_y.scatter_reduce_(0, self.pin_net_ids, pin_y, reduce='amax', include_self=True)
        net_min_y.scatter_reduce_(0, self.pin_net_ids, pin_y, reduce='amin', include_self=True)

        pin_net_max_x = net_max_x[self.pin_net_ids]
        pin_net_min_x = net_min_x[self.pin_net_ids]
        pin_net_max_y = net_max_y[self.pin_net_ids]
        pin_net_min_y = net_min_y[self.pin_net_ids]

        x_range = pin_net_max_x - pin_net_min_x
        y_range = pin_net_max_y - pin_net_min_y
        x_active = x_range > 1e-6
        y_active = y_range > 1e-6

        grad_pin_x = torch.zeros(self.num_total_pins, dtype=torch.float32, device=self.device)
        grad_pin_y = torch.zeros(self.num_total_pins, dtype=torch.float32, device=self.device)

        if x_active.any():
            scaled_xp = (pin_x - pin_net_max_x.detach()) * inv_gamma
            lse_xp = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
            lse_xp.scatter_reduce_(0, self.pin_net_ids, scaled_xp, reduce='amax', include_self=True)
            pin_lse_xp = lse_xp[self.pin_net_ids]
            exp_xp = (scaled_xp - pin_lse_xp).exp()
            xexp_xp = pin_x * exp_xp

            sum_exp_xp = torch.zeros(self.num_valid_nets, device=self.device)
            sum_xexp_xp = torch.zeros(self.num_valid_nets, device=self.device)
            sum_exp_xp.scatter_add_(0, self.pin_net_ids, exp_xp)
            sum_xexp_xp.scatter_add_(0, self.pin_net_ids, xexp_xp)
            pin_sum_exp_xp = sum_exp_xp[self.pin_net_ids].clamp(min=1e-10)
            pin_sum_xexp_xp = sum_xexp_xp[self.pin_net_ids]

            wavg_xp = pin_sum_xexp_xp / pin_sum_exp_xp

            scaled_xn = (pin_net_min_x.detach() - pin_x) * inv_gamma
            lse_xn = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
            lse_xn.scatter_reduce_(0, self.pin_net_ids, scaled_xn, reduce='amax', include_self=True)
            pin_lse_xn = lse_xn[self.pin_net_ids]
            exp_xn = (scaled_xn - pin_lse_xn).exp()
            xexp_xn = pin_x * exp_xn

            sum_exp_xn = torch.zeros(self.num_valid_nets, device=self.device)
            sum_xexp_xn = torch.zeros(self.num_valid_nets, device=self.device)
            sum_exp_xn.scatter_add_(0, self.pin_net_ids, exp_xn)
            sum_xexp_xn.scatter_add_(0, self.pin_net_ids, xexp_xn)
            pin_sum_exp_xn = sum_exp_xn[self.pin_net_ids].clamp(min=1e-10)
            pin_sum_xexp_xn = sum_xexp_xn[self.pin_net_ids]

            wavg_xn = pin_sum_xexp_xn / pin_sum_exp_xn

            grad_pin_x = torch.where(
                x_active,
                (1.0 + (pin_x - wavg_xp) * inv_gamma) * exp_xp / pin_sum_exp_xp
                - (1.0 + (pin_x - wavg_xn) * inv_gamma) * exp_xn / pin_sum_exp_xn,
                torch.zeros_like(grad_pin_x)
            )

        if y_active.any():
            scaled_yp = (pin_y - pin_net_max_y.detach()) * inv_gamma
            lse_yp = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
            lse_yp.scatter_reduce_(0, self.pin_net_ids, scaled_yp, reduce='amax', include_self=True)
            pin_lse_yp = lse_yp[self.pin_net_ids]
            exp_yp = (scaled_yp - pin_lse_yp).exp()
            yexp_yp = pin_y * exp_yp

            sum_exp_yp = torch.zeros(self.num_valid_nets, device=self.device)
            sum_yexp_yp = torch.zeros(self.num_valid_nets, device=self.device)
            sum_exp_yp.scatter_add_(0, self.pin_net_ids, exp_yp)
            sum_yexp_yp.scatter_add_(0, self.pin_net_ids, yexp_yp)
            pin_sum_exp_yp = sum_exp_yp[self.pin_net_ids].clamp(min=1e-10)
            pin_sum_yexp_yp = sum_yexp_yp[self.pin_net_ids]

            wavg_yp = pin_sum_yexp_yp / pin_sum_exp_yp

            scaled_yn = (pin_net_min_y.detach() - pin_y) * inv_gamma
            lse_yn = torch.zeros(self.num_valid_nets, device=self.device).fill_(-1e18)
            lse_yn.scatter_reduce_(0, self.pin_net_ids, scaled_yn, reduce='amax', include_self=True)
            pin_lse_yn = lse_yn[self.pin_net_ids]
            exp_yn = (scaled_yn - pin_lse_yn).exp()
            yexp_yn = pin_y * exp_yn

            sum_exp_yn = torch.zeros(self.num_valid_nets, device=self.device)
            sum_yexp_yn = torch.zeros(self.num_valid_nets, device=self.device)
            sum_exp_yn.scatter_add_(0, self.pin_net_ids, exp_yn)
            sum_yexp_yn.scatter_add_(0, self.pin_net_ids, yexp_yn)
            pin_sum_exp_yn = sum_exp_yn[self.pin_net_ids].clamp(min=1e-10)
            pin_sum_yexp_yn = sum_yexp_yn[self.pin_net_ids]

            wavg_yn = pin_sum_yexp_yn / pin_sum_exp_yn

            grad_pin_y = torch.where(
                y_active,
                (1.0 + (pin_y - wavg_yp) * inv_gamma) * exp_yp / pin_sum_exp_yp
                - (1.0 + (pin_y - wavg_yn) * inv_gamma) * exp_yn / pin_sum_exp_yn,
                torch.zeros_like(grad_pin_y)
            )

        grad_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        grad_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        grad_x.scatter_add_(0, self.pin_node_ids, grad_pin_x)
        grad_y.scatter_add_(0, self.pin_node_ids, grad_pin_y)

        grad_x[bm.is_fixed_t] = 0
        grad_y[bm.is_fixed_t] = 0

        return grad_x, grad_y

    def _compute_wl_gradient_vectorized(self, gamma=1.0):
        bm = self.bm

        pin_x = self.pos_x[self.pin_node_ids]
        pin_y = self.pos_y[self.pin_node_ids]

        net_max_x = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('-inf'), device=self.device),
            0, self.pin_net_ids, pin_x, reduce='amax', include_self=True
        )
        net_min_x = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('inf'), device=self.device),
            0, self.pin_net_ids, pin_x, reduce='amin', include_self=True
        )
        net_max_y = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('-inf'), device=self.device),
            0, self.pin_net_ids, pin_y, reduce='amax', include_self=True
        )
        net_min_y = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('inf'), device=self.device),
            0, self.pin_net_ids, pin_y, reduce='amin', include_self=True
        )

        pin_net_max_x = net_max_x[self.pin_net_ids]
        pin_net_min_x = net_min_x[self.pin_net_ids]
        pin_net_max_y = net_max_y[self.pin_net_ids]
        pin_net_min_y = net_min_y[self.pin_net_ids]

        x_range = pin_net_max_x - pin_net_min_x
        y_range = pin_net_max_y - pin_net_min_y

        x_active = x_range > 1e-6
        y_active = y_range > 1e-6

        grad_pin_x = torch.zeros(self.num_total_pins, dtype=torch.float32, device=self.device)
        grad_pin_y = torch.zeros(self.num_total_pins, dtype=torch.float32, device=self.device)

        if x_active.any():
            scaled_x = (pin_x - pin_net_max_x) / gamma
            lse_x = torch.zeros(self.num_valid_nets, device=self.device)
            lse_x.scatter_reduce_(0, self.pin_net_ids, scaled_x, reduce='amax', include_self=True)
            pin_lse_x = lse_x[self.pin_net_ids]
            wx_max = (scaled_x - pin_lse_x).exp()

            sum_wx_max = torch.zeros(self.num_valid_nets, device=self.device)
            sum_wx_max.scatter_add_(0, self.pin_net_ids, wx_max)
            pin_sum_wx_max = sum_wx_max[self.pin_net_ids]
            wx_max_norm = wx_max / pin_sum_wx_max.clamp(min=1e-10)

            scaled_x_neg = -(pin_x - pin_net_min_x) / gamma
            lse_x_neg = torch.zeros(self.num_valid_nets, device=self.device)
            lse_x_neg.scatter_reduce_(0, self.pin_net_ids, scaled_x_neg, reduce='amax', include_self=True)
            pin_lse_x_neg = lse_x_neg[self.pin_net_ids]
            wx_min = (scaled_x_neg - pin_lse_x_neg).exp()

            sum_wx_min = torch.zeros(self.num_valid_nets, device=self.device)
            sum_wx_min.scatter_add_(0, self.pin_net_ids, wx_min)
            pin_sum_wx_min = sum_wx_min[self.pin_net_ids]
            wx_min_norm = wx_min / pin_sum_wx_min.clamp(min=1e-10)

            grad_pin_x = torch.where(x_active, wx_max_norm - wx_min_norm, torch.zeros_like(grad_pin_x))

        if y_active.any():
            scaled_y = (pin_y - pin_net_max_y) / gamma
            lse_y = torch.zeros(self.num_valid_nets, device=self.device)
            lse_y.scatter_reduce_(0, self.pin_net_ids, scaled_y, reduce='amax', include_self=True)
            pin_lse_y = lse_y[self.pin_net_ids]
            wy_max = (scaled_y - pin_lse_y).exp()

            sum_wy_max = torch.zeros(self.num_valid_nets, device=self.device)
            sum_wy_max.scatter_add_(0, self.pin_net_ids, wy_max)
            pin_sum_wy_max = sum_wy_max[self.pin_net_ids]
            wy_max_norm = wy_max / pin_sum_wy_max.clamp(min=1e-10)

            scaled_y_neg = -(pin_y - pin_net_min_y) / gamma
            lse_y_neg = torch.zeros(self.num_valid_nets, device=self.device)
            lse_y_neg.scatter_reduce_(0, self.pin_net_ids, scaled_y_neg, reduce='amax', include_self=True)
            pin_lse_y_neg = lse_y_neg[self.pin_net_ids]
            wy_min = (scaled_y_neg - pin_lse_y_neg).exp()

            sum_wy_min = torch.zeros(self.num_valid_nets, device=self.device)
            sum_wy_min.scatter_add_(0, self.pin_net_ids, wy_min)
            pin_sum_wy_min = sum_wy_min[self.pin_net_ids]
            wy_min_norm = wy_min / pin_sum_wy_min.clamp(min=1e-10)

            grad_pin_y = torch.where(y_active, wy_max_norm - wy_min_norm, torch.zeros_like(grad_pin_y))

        grad_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        grad_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        grad_x.scatter_add_(0, self.pin_node_ids, grad_pin_x)
        grad_y.scatter_add_(0, self.pin_node_ids, grad_pin_y)

        grad_x[bm.is_fixed_t] = 0
        grad_y[bm.is_fixed_t] = 0

        return grad_x, grad_y

    def _compute_density_map(self, site_group_id):
        """Compute density map - instance count per bin."""
        bm = self.bm
        density_map = torch.zeros(self.num_bins_x, self.num_bins_y,
                                   dtype=torch.float32, device=self.device)

        mask = (bm.node_site_group_t == site_group_id) & (~bm.is_fixed_t)
        idx = torch.where(mask)[0]

        if len(idx) == 0:
            return density_map

        # Instance center in bin coordinates
        center_x = (self.pos_x[idx] + 0.5) / self.bin_size_x
        center_y = (self.pos_y[idx] + 0.5) / self.bin_size_y

        x_coords = center_x.clamp(0, self.num_bins_x - 1).long()
        y_coords = center_y.clamp(0, self.num_bins_y - 1).long()

        density_map.index_put_(
            (x_coords, y_coords),
            torch.ones(len(idx), dtype=torch.float32, device=self.device),
            accumulate=True
        )

        return density_map

    def _compute_density_map_soft(self, site_group_id, half_size_x=None, half_size_y=None):
        """Soft spreading density map like DREAMPlaceFPGA.
        Each instance spreads its density over a region of size 2*halfSize x 2*halfSize.
        """
        bm = self.bm
        density_map = torch.zeros(self.num_bins_x, self.num_bins_y,
                                   dtype=torch.float32, device=self.device)

        mask = (bm.node_site_group_t == site_group_id) & (~bm.is_fixed_t)
        idx = torch.where(mask)[0]

        if len(idx) == 0:
            return density_map

        if half_size_x is None:
            half_size_x = self.bin_size_x * 0.5
        if half_size_y is None:
            half_size_y = self.bin_size_y * 0.5

        # Instance positions in bin coordinates
        pos_x_bin = self.pos_x[idx] / self.bin_size_x
        pos_y_bin = self.pos_y[idx] / self.bin_size_y

        # Spread range in bin coordinates
        hs_x = half_size_x / self.bin_size_x
        hs_y = half_size_y / self.bin_size_y

        # For each instance, spread to affected bins
        bin_lo_x = (pos_x_bin - hs_x).clamp(0, self.num_bins_x - 1).long()
        bin_hi_x = (pos_x_bin + hs_x).clamp(0, self.num_bins_x - 1).long() + 1
        bin_lo_y = (pos_y_bin - hs_y).clamp(0, self.num_bins_y - 1).long()
        bin_hi_y = (pos_y_bin + hs_y).clamp(0, self.num_bins_y - 1).long() + 1

        # Use simple spreading: add 1/(num_affected_bins) to each affected bin
        for i in range(len(idx)):
            bx_lo = bin_lo_x[i].item()
            bx_hi = min(bin_hi_x[i].item(), self.num_bins_x)
            by_lo = bin_lo_y[i].item()
            by_hi = min(bin_hi_y[i].item(), self.num_bins_y)

            num_bins = (bx_hi - bx_lo) * (by_hi - by_lo)
            if num_bins > 0:
                density_map[bx_lo:bx_hi, by_lo:by_hi] += 1.0 / num_bins

        return density_map

    def _compute_density_map_smooth(self, site_group_id, sigma=1.5):
        density_map = self._compute_density_map(site_group_id)

        # Convert count to area density (area per bin)
        # Each instance has area=1, so density_map is already area sum per bin
        # No need to divide by bin_area here - overflow calculation handles it

        if sigma > 0:
            # Use larger sigma for finer bins to simulate area spreading
            # Instance size=1, bin_size≈0.33, so instance spans ~3 bins
            # sigma should be ~1.5 bins to cover this
            effective_sigma = sigma
            k = int(effective_sigma * 3) * 2 + 1
            k = min(k, 15)
            x = torch.arange(k, dtype=torch.float32, device=self.device) - k // 2
            gauss = torch.exp(-x.pow(2) / (2 * effective_sigma ** 2))
            gauss = gauss / gauss.sum()
            kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
            kernel = kernel.view(1, 1, k, k)

            density_map = F.conv2d(
                density_map.view(1, 1, self.num_bins_x, self.num_bins_y),
                kernel,
                padding=k // 2
            ).view(self.num_bins_x, self.num_bins_y)

        return density_map

    def _dct2(self, x):
        """2D DCT-II using FFT (like scipy.fft.dctn(x, type=2))."""
        M, N = x.shape
        # DCT-II along dim 0
        v = torch.cat([x[::2], x[1::2].flip(0)], dim=0)
        V = torch.fft.fft(v, dim=0)
        k = torch.arange(M, dtype=x.dtype, device=x.device)
        w0 = torch.exp(-1j * torch.pi * k / (2 * M))
        result = (V * w0.unsqueeze(1)).real
        # DCT-II along dim 1
        v2 = torch.cat([result[:, ::2], result[:, 1::2].flip(1)], dim=1)
        V2 = torch.fft.fft(v2, dim=1)
        k2 = torch.arange(N, dtype=x.dtype, device=x.device)
        w1 = torch.exp(-1j * torch.pi * k2 / (2 * N))
        result2 = (V2 * w1.unsqueeze(0)).real
        return result2

    def _idct2(self, X):
        """2D iDCT-II (inverse of DCT-II) using FFT."""
        M, N = X.shape
        # iDCT along dim 1
        k2 = torch.arange(N, dtype=X.dtype, device=X.device)
        w1 = torch.exp(1j * torch.pi * k2 / (2 * N))
        V2 = X * w1.unsqueeze(0)
        v2 = torch.fft.ifft(V2, dim=1).real
        result = torch.zeros_like(X)
        result[:, ::2] = v2[:, :N - N // 2]
        result[:, 1::2] = v2[:, N - N // 2:].flip(1)
        # iDCT along dim 0
        k = torch.arange(M, dtype=X.dtype, device=X.device)
        w0 = torch.exp(1j * torch.pi * k / (2 * M))
        V = result * w0.unsqueeze(1)
        v = torch.fft.ifft(V, dim=0).real
        result2 = torch.zeros_like(X)
        result2[::2] = v[:M - M // 2]
        result2[1::2] = v[M - M // 2:].flip(0)
        return result2

    def _idxst_idct(self, X):
        """iDST-iDCT: first iDCT along dim 1, then iDST along dim 0.
        This computes the inverse of DST along dim 0 and DCT along dim 1."""
        M, N = X.shape
        # iDCT along dim 1
        k2 = torch.arange(N, dtype=X.dtype, device=X.device)
        w1 = torch.exp(1j * torch.pi * k2 / (2 * N))
        V2 = X * w1.unsqueeze(0)
        v2 = torch.fft.ifft(V2, dim=1).real
        result = torch.zeros_like(X)
        result[:, ::2] = v2[:, :N - N // 2]
        result[:, 1::2] = v2[:, N - N // 2:].flip(1)
        # iDST along dim 0: iDST(X) = flip(iDCT(flip(X, dims=[0])), dims=[0])
        result_flipped = result.flip(0)
        k = torch.arange(M, dtype=X.dtype, device=X.device)
        w0 = torch.exp(1j * torch.pi * k / (2 * M))
        V = result_flipped * w0.unsqueeze(1)
        v = torch.fft.ifft(V, dim=0).real
        idct_result = torch.zeros_like(X)
        idct_result[::2] = v[:M - M // 2]
        idct_result[1::2] = v[M - M // 2:].flip(0)
        return idct_result.flip(0)

    def _idct_idxst(self, X):
        """iDCT-iDST: first iDST along dim 1, then iDCT along dim 0."""
        M, N = X.shape
        # iDST along dim 1: iDST(X) = flip(iDCT(flip(X, dims=[1])), dims=[1])
        X_flipped = X.flip(1)
        k2 = torch.arange(N, dtype=X.dtype, device=X.device)
        w1 = torch.exp(1j * torch.pi * k2 / (2 * N))
        V2 = X_flipped * w1.unsqueeze(0)
        v2 = torch.fft.ifft(V2, dim=1).real
        idct_result = torch.zeros_like(X)
        idct_result[:, ::2] = v2[:, :N - N // 2]
        idct_result[:, 1::2] = v2[:, N - N // 2:].flip(1)
        result = idct_result.flip(1)
        # iDCT along dim 0
        k = torch.arange(M, dtype=X.dtype, device=X.device)
        w0 = torch.exp(1j * torch.pi * k / (2 * M))
        V = result * w0.unsqueeze(1)
        v = torch.fft.ifft(V, dim=0).real
        result2 = torch.zeros_like(X)
        result2[::2] = v[:M - M // 2]
        result2[1::2] = v[M - M // 2:].flip(0)
        return result2

    def _solve_poisson_fft(self, density_map, target_density_val, site_group_id=None):
        # Use per-bin target capacity instead of scalar target_density * bin_area
        # This correctly accounts for heterogeneous site distribution
        if site_group_id is not None:
            target = self.target_capacity[site_group_id]
        else:
            bin_area = self.bin_size_x * self.bin_size_y
            target = target_density_val * bin_area
        overflow = density_map - target

        density_for_fft = overflow / (self.bin_size_x * self.bin_size_y)

        # Use DCT instead of FFT (like DREAMPlaceFPGA)
        # DCT assumes Neumann boundary conditions (zero gradient at boundary)
        auv = self._dct2(density_for_fft)

        # Compute potential using iDCT
        auv_filtered = auv * self.inv_wu2_plus_wv2
        potential_map = self._idct2(auv_filtered)

        # Compute field using iDST-iDCT and iDCT-iDST
        auv_wu = auv * self.wu_by_wu2_plus_wv2_half * 2
        auv_wv = auv * self.wv_by_wu2_plus_wv2_half * 2

        # iDST-iDCT for field_x
        field_map_x = self._idxst_idct(auv_wu)
        # iDCT-iDST for field_y
        field_map_y = self._idct_idxst(auv_wv)

        return potential_map, field_map_x, field_map_y, overflow

    def _compute_obj_and_grad(self, gamma, density_weight, wirelength_model='wawl'):
        bm = self.bm

        wl_fn = self._compute_wl_gradient_vectorized if wirelength_model == 'lse' else self._compute_wl_gradient_wawl
        wl_grad_x, wl_grad_y = wl_fn(gamma)

        density_grad_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        density_grad_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        density_cost = 0.0

        for sg in range(4):
            density_map = self._compute_density_map_smooth(sg, sigma=1.0)
            potential_map, field_x, field_y, overflow = self._solve_poisson_fft(
                density_map, self.target_density[sg], site_group_id=sg)

            mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
            idx = torch.where(mask)[0]
            if len(idx) == 0:
                continue

            x_coords = (self.pos_x[idx] / self.bin_size_x).clamp(0, self.num_bins_x - 1).long()
            y_coords = (self.pos_y[idx] / self.bin_size_y).clamp(0, self.num_bins_y - 1).long()

            density_grad_x[idx] = field_x[x_coords, y_coords]
            density_grad_y[idx] = field_y[x_coords, y_coords]

            density_cost += potential_map.sum().item()

        total_grad_x = wl_grad_x + density_weight * density_grad_x
        total_grad_y = wl_grad_y + density_weight * density_grad_y

        total_grad_x[bm.is_fixed_t] = 0
        total_grad_y[bm.is_fixed_t] = 0

        wl_cost = self._compute_hpwl_vectorized()
        obj = wl_cost + density_weight * density_cost

        return obj, total_grad_x, total_grad_y

    def run(self, num_iterations=2000, lr=0.01, density_weight=8e-5,
            gamma_start=1.0, gamma_end=4.0, verbose=True, wirelength_model='wawl',
            optimizer='nesterov', use_best=True, alpha_min=0.0, density_weight_start=None,
            fixed_dw_schedule=False, min_dw_ratio=0.01):
        bm = self.bm
        start_time = time.time()

        if optimizer == 'nesterov':
            return self._run_nesterov(num_iterations, lr, density_weight,
                                       gamma_start, gamma_end, verbose, wirelength_model,
                                       use_best=use_best, alpha_min=alpha_min,
                                       density_weight_start=density_weight_start,
                                       fixed_dw_schedule=fixed_dw_schedule)

        return self._run_multistage(num_iterations, lr, density_weight,
                                     gamma_start, gamma_end, verbose, wirelength_model,
                                     use_best=use_best, fixed_dw_schedule=fixed_dw_schedule,
                                     density_weight_start=density_weight_start,
                                     min_dw_ratio=min_dw_ratio)

    def _run_multistage(self, num_iterations, lr, density_weight,
                         gamma_start, gamma_end, verbose, wirelength_model,
                         use_best=True, fixed_dw_schedule=False,
                         density_weight_start=None, min_dw_ratio=0.01):
        bm = self.bm
        start_time = time.time()
        wl_fn = self._compute_wl_gradient_vectorized if wirelength_model == 'lse' else self._compute_wl_gradient_wawl

        best_hpwl = float('inf')
        best_pos_x = self.pos_x.clone()
        best_pos_y = self.pos_y.clone()

        vel_x = torch.zeros_like(self.pos_x)
        vel_y = torch.zeros_like(self.pos_y)

        base_gamma = gamma_start
        adaptive_density_weight = density_weight
        min_density_weight = density_weight * min_dw_ratio
        prev_overflow = 1.0

        for iteration in range(num_iterations):
            wl_grad_x, wl_grad_y = wl_fn(base_gamma)

            density_grad_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
            density_grad_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)

            max_overflow = 0.0
            for sg in range(4):
                density_map = self._compute_density_map_smooth(sg, sigma=0.3)
                _, field_x, field_y, overflow = self._solve_poisson_fft(
                    density_map, self.target_density[sg], site_group_id=sg)

                mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
                idx = torch.where(mask)[0]
                if len(idx) == 0:
                    continue

                x_coords = (self.pos_x[idx] / self.bin_size_x).clamp(0, self.num_bins_x - 1).long()
                y_coords = (self.pos_y[idx] / self.bin_size_y).clamp(0, self.num_bins_y - 1).long()

                density_grad_x[idx] = field_x[x_coords, y_coords]
                density_grad_y[idx] = field_y[x_coords, y_coords]

                # Only count overflow in bins that have sites
                site_bins = self.site_type_bin_mask[sg]
                if site_bins.sum() > 0:
                    sg_overflow = (overflow[site_bins] > 0).float().mean().item()
                else:
                    sg_overflow = 0.0
                max_overflow = max(max_overflow, sg_overflow)

            if iteration == 0:
                wl_norm = (wl_grad_x ** 2 + wl_grad_y ** 2).sum().sqrt().item()
                dens_norm = (density_grad_x ** 2 + density_grad_y ** 2).sum().sqrt().item()
                if fixed_dw_schedule:
                    # Use scheduled density weight (linear ramp from start to target)
                    dw_start = density_weight_start if density_weight_start is not None else density_weight
                    adaptive_density_weight = dw_start
                elif dens_norm > 1e-10:
                    adaptive_density_weight = density_weight * wl_norm / dens_norm
                    adaptive_density_weight = max(adaptive_density_weight, min_density_weight)
                    # Cap initial auto_dw to prevent over-spreading when overflow is already low
                    adaptive_density_weight = min(adaptive_density_weight, density_weight)
                else:
                    adaptive_density_weight = density_weight
                if verbose:
                    print(f"  [Momentum] iter0: wl_norm={wl_norm:.2e}, dens_norm={dens_norm:.2e}, auto_dw={adaptive_density_weight:.6f}")

            base_gamma = gamma_start * (10.0 ** (0.5 * max_overflow + 0.0))
            base_gamma = max(gamma_start, min(base_gamma, gamma_end))

            if iteration > 0 and iteration % 10 == 0:
                if fixed_dw_schedule:
                    # Linear ramp from dw_start to density_weight
                    dw_start = density_weight_start if density_weight_start is not None else density_weight
                    progress = min(iteration / num_iterations, 1.0)
                    adaptive_density_weight = dw_start + progress * (density_weight - dw_start)
                else:
                    wl_norm = (wl_grad_x ** 2 + wl_grad_y ** 2).sum().sqrt().item()
                    dens_norm = (density_grad_x ** 2 + density_grad_y ** 2).sum().sqrt().item()
                    if dens_norm > 1e-10:
                        target_dw = density_weight * wl_norm / dens_norm
                        target_dw = max(target_dw, min_density_weight)
                        adaptive_density_weight = 0.8 * adaptive_density_weight + 0.2 * target_dw

                    # Gradually increase density weight when overflow is high
                    if max_overflow > 0.3:
                        adaptive_density_weight *= 1.1
                    elif max_overflow > 0.1:
                        adaptive_density_weight *= 1.05

                    # V12: Spread-aware dw boost - prevent instance collapse
                    if iteration > 0 and iteration % 100 == 0:
                        sg0_mask = (bm.node_site_group_t == 0) & (~bm.is_fixed_t)
                        sg0_idx = torch.where(sg0_mask)[0]
                        if len(sg0_idx) > 100:
                            x_range = self.pos_x[sg0_idx].max() - self.pos_x[sg0_idx].min()
                            coverage = x_range.item() / self.size_x
                            if coverage < 0.5:
                                adaptive_density_weight *= 2.0
                            elif coverage < 0.65:
                                adaptive_density_weight *= 1.3

                    # Cap density weight to prevent explosion
                    adaptive_density_weight = max(adaptive_density_weight, min_density_weight)
                    adaptive_density_weight = min(adaptive_density_weight, density_weight * 10)

            total_grad_x = wl_grad_x + adaptive_density_weight * density_grad_x
            total_grad_y = wl_grad_y + adaptive_density_weight * density_grad_y

            # Global gradient normalization: scale by RMS norm
            # This allows instances with larger gradients to move more
            movable_mask = ~bm.is_fixed_t
            movable_idx = torch.where(movable_mask)[0]
            lr_cur = lr * (0.9995 ** iteration)
            grad_sq = total_grad_x[movable_idx] ** 2 + total_grad_y[movable_idx] ** 2
            rms_norm = (grad_sq.mean() + 1e-10).sqrt()
            total_grad_x[movable_idx] = lr_cur * total_grad_x[movable_idx] / rms_norm
            total_grad_y[movable_idx] = lr_cur * total_grad_y[movable_idx] / rms_norm

            with torch.no_grad():
                vel_x = 0.9 * vel_x - total_grad_x
                vel_y = 0.9 * vel_y - total_grad_y

                self.pos_x += vel_x
                self.pos_y += vel_y

                self.pos_x.clamp_(0, self.size_x - 1)
                self.pos_y.clamp_(0, self.size_y - 1)

                self.pos_x[bm.is_fixed_t] = torch.tensor(
                    bm.fixed_pos[bm.fixed_indices, 0], dtype=torch.float32, device=self.device)
                self.pos_y[bm.is_fixed_t] = torch.tensor(
                    bm.fixed_pos[bm.fixed_indices, 1], dtype=torch.float32, device=self.device)

            if (iteration + 1) % 50 == 0 or iteration == 0:
                hpwl = self._compute_hpwl_vectorized()
                elapsed = time.time() - start_time
                if hpwl < best_hpwl:
                    best_hpwl = hpwl
                    best_pos_x = self.pos_x.clone()
                    best_pos_y = self.pos_y.clone()
                if verbose:
                    print(f"  GP iter {iteration+1}/{num_iterations}: "
                          f"HPWL={hpwl:.0f}, gamma={base_gamma:.4f}, "
                          f"dw={adaptive_density_weight:.6f}, overflow={max_overflow:.4f}, "
                          f"lr={lr_cur:.6f}, time={elapsed:.1f}s")

            prev_overflow = max_overflow

        self.pos_x = best_pos_x if use_best else self.pos_x
        self.pos_y = best_pos_y if use_best else self.pos_y

        # Post-GP: report spread metrics
        sg = 0
        inst_mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
        idx = torch.where(inst_mask)[0]
        if len(idx) > 0:
            x_range = self.pos_x[idx].max() - self.pos_x[idx].min()
            y_range = self.pos_y[idx].max() - self.pos_y[idx].min()
            print(f"  [GP-End] Spread: x_range={x_range:.1f}/{self.size_x:.0f}, y_range={y_range:.1f}/{self.size_y:.0f}")

        return self.pos_x, self.pos_y

    def _spread_instances(self, num_iters=200):
        """Spread instances to fill the entire FPGA region using density gradient.
        This is critical for LG quality - instances must be spread across all available sites.
        Uses gradually increasing density weight while preserving wirelength quality."""
        bm = self.bm
        
        # Check initial spread
        sg = 0
        inst_mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
        idx = torch.where(inst_mask)[0]
        if len(idx) == 0:
            print(f"  [Spread] No movable instances, skipping")
            return
        init_x_range = self.pos_x[idx].max() - self.pos_x[idx].min()
        init_y_range = self.pos_y[idx].max() - self.pos_y[idx].min()
        print(f"  [Spread] Starting: x_range={init_x_range:.1f}/{self.size_x:.0f}, y_range={init_y_range:.1f}/{self.size_y:.0f}, num_iters={num_iters}", flush=True)
        
        # Compute initial wirelength gradient
        wl_grad_x, wl_grad_y = self._compute_wl_gradient_wawl(1.0)
        wl_norm = (wl_grad_x ** 2 + wl_grad_y ** 2).sum().sqrt().item()
        
        # Start with small density weight, gradually increase
        density_weight = 0.01
        
        for it in range(num_iters):
            total_grad_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
            total_grad_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
            
            for sg in range(4):
                density_map = self._compute_density_map_smooth(sg, sigma=2.0)
                _, field_x, field_y, overflow = self._solve_poisson_fft(density_map, self.target_density[sg])
                
                mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
                idx = torch.where(mask)[0]
                if len(idx) == 0:
                    continue
                
                x_coords = (self.pos_x[idx] / self.bin_size_x).clamp(0, self.num_bins_x - 1).long()
                y_coords = (self.pos_y[idx] / self.bin_size_y).clamp(0, self.num_bins_y - 1).long()
                
                total_grad_x[idx] = field_x[x_coords, y_coords]
                total_grad_y[idx] = field_y[x_coords, y_coords]
            
            # Check spread: compute x and y range of instances
            sg = 0
            inst_mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
            idx = torch.where(inst_mask)[0]
            if len(idx) == 0:
                continue
            inst_x = self.pos_x[idx]
            inst_y = self.pos_y[idx]
            x_range = inst_x.max() - inst_x.min()
            y_range = inst_y.max() - inst_y.min()
            
            # Stop when instances cover most of the FPGA
            if x_range > 0.8 * self.size_x and y_range > 0.8 * self.size_y:
                print(f"  [Spread] Done at iter {it}: x_range={x_range:.1f}, y_range={y_range:.1f}", flush=True)
                break
            
            # Increase density weight over iterations to spread instances
            density_weight *= 1.02
            
            # Combined gradient: wirelength + density
            # Recompute wirelength gradient periodically
            if it % 20 == 0:
                wl_grad_x, wl_grad_y = self._compute_wl_gradient_wawl(1.0)
            
            combined_x = wl_grad_x + density_weight * total_grad_x
            combined_y = wl_grad_y + density_weight * total_grad_y
            
            # Normalize gradient per-instance, then scale step
            movable = ~bm.is_fixed_t
            # Per-instance gradient magnitude
            per_inst_norm = (combined_x[movable] ** 2 + combined_y[movable] ** 2).sqrt()
            per_inst_norm = per_inst_norm.clamp(min=1e-10)
            
            # Step size: move each instance by at most max_step units
            max_step = 2.0
            # Normalize direction, then scale
            step_x = max_step * combined_x[movable] / per_inst_norm
            step_y = max_step * combined_y[movable] / per_inst_norm
            
            self.pos_x[movable] -= step_x
            self.pos_y[movable] -= step_y
            self.pos_x.clamp_(0, self.size_x - 1)
            self.pos_y.clamp_(0, self.size_y - 1)
            
            if it % 50 == 0:
                hpwl = self._compute_hpwl_vectorized()
                print(f"  [Spread] iter {it}: HPWL={hpwl:.0f}, dw={density_weight:.4f}, "
                      f"x_range={x_range:.1f}/{self.size_x:.0f}, y_range={y_range:.1f}/{self.size_y:.0f}")

    def _run_nesterov(self, num_iterations, lr, density_weight,
                       gamma_start, gamma_end, verbose, wirelength_model,
                       use_best=True, alpha_min=0.0, density_weight_start=None,
                       fixed_dw_schedule=False):
        bm = self.bm

        best_hpwl = float('inf')
        best_pos_x = self.pos_x.clone()
        best_pos_y = self.pos_y.clone()

        v_x = self.pos_x.clone()
        v_y = self.pos_y.clone()

        wl_grad_init_x, wl_grad_init_y = self._compute_wl_gradient_wawl(gamma_start)
        wl_norm = (wl_grad_init_x ** 2 + wl_grad_init_y ** 2).sum().sqrt().item()
        dens_grad_init_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        dens_grad_init_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=self.device)
        for sg in range(4):
            density_map = self._compute_density_map_smooth(sg, sigma=1.0)
            _, field_x, field_y, overflow = self._solve_poisson_fft(density_map, self.target_density[sg], site_group_id=sg)
            mask = (bm.node_site_group_t == sg) & (~bm.is_fixed_t)
            idx = torch.where(mask)[0]
            if len(idx) == 0:
                continue
            x_coords = (self.pos_x[idx] / self.bin_size_x).clamp(0, self.num_bins_x - 1).long()
            y_coords = (self.pos_y[idx] / self.bin_size_y).clamp(0, self.num_bins_y - 1).long()
            dens_grad_init_x[idx] = field_x[x_coords, y_coords]
            dens_grad_init_y[idx] = field_y[x_coords, y_coords]
        dens_norm = (dens_grad_init_x ** 2 + dens_grad_init_y ** 2).sum().sqrt().item()
        if dens_norm > 1e-10 and not fixed_dw_schedule:
            # Match deepseek version: auto-scale initial dw by wl/density gradient ratio
            auto_dw = density_weight * wl_norm / dens_norm
            if density_weight_start is not None:
                adaptive_density_weight = max(auto_dw, density_weight_start)
            else:
                adaptive_density_weight = auto_dw
            print(f"  [DEBUG] GP2 init: wl_norm={wl_norm:.2e}, dens_norm={dens_norm:.2e}, auto_dw={auto_dw:.6f}, dw_start={density_weight_start}, final_dw={adaptive_density_weight:.6f}")
        else:
            adaptive_density_weight = density_weight_start if density_weight_start is not None else density_weight

        obj_k, g_k_x, g_k_y = self._compute_obj_and_grad(gamma_start, adaptive_density_weight, wirelength_model)

        v_k_1_x = v_x - lr * g_k_x
        v_k_1_y = v_y - lr * g_k_y
        v_k_1_x.clamp_(0, self.size_x - 1)
        v_k_1_y.clamp_(0, self.size_y - 1)
        v_k_1_x[bm.is_fixed_t] = torch.tensor(bm.fixed_pos[bm.fixed_indices, 0], dtype=torch.float32, device=self.device)
        v_k_1_y[bm.is_fixed_t] = torch.tensor(bm.fixed_pos[bm.fixed_indices, 1], dtype=torch.float32, device=self.device)

        self.pos_x.copy_(v_k_1_x)
        self.pos_y.copy_(v_k_1_y)
        obj_k_1, g_k_1_x, g_k_1_y = self._compute_obj_and_grad(gamma_start, adaptive_density_weight, wirelength_model)

        alpha_k = torch.tensor(lr, dtype=torch.float32, device=self.device)
        a_k = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        u_x = v_x.clone()
        u_y = v_y.clone()

        base_gamma = gamma_start

        start_time = time.time()

        for iteration in range(num_iterations):
            max_overflow = 0.0
            for sg in range(4):
                density_map = self._compute_density_map_smooth(sg, sigma=1.0)
                _, _, _, overflow = self._solve_poisson_fft(density_map, self.target_density[sg], site_group_id=sg)
                # Only count overflow in bins that have sites
                site_bins = self.site_type_bin_mask[sg]
                if site_bins.sum() > 0:
                    sg_overflow = (overflow[site_bins] > 0).float().mean().item()
                else:
                    sg_overflow = 0.0
                max_overflow = max(max_overflow, sg_overflow)

            base_gamma = gamma_start * (10.0 ** (0.5 * max_overflow))
            base_gamma = max(gamma_start, min(base_gamma, gamma_end))

            if iteration > 0 and iteration % 10 == 0:
                if fixed_dw_schedule:
                    # Linear ramp from density_weight_start to density_weight
                    dw_start = density_weight_start if density_weight_start is not None else density_weight
                    progress = min(iteration / num_iterations, 1.0)
                    adaptive_density_weight = dw_start + progress * (density_weight - dw_start)
                else:
                    wl_norm = (g_k_x ** 2 + g_k_y ** 2).sum().sqrt().item()
                    dens_part_x = g_k_x - self._compute_wl_gradient_vectorized(base_gamma)[0]
                    dens_norm = (dens_part_x ** 2).sum().sqrt().item()
                    if dens_norm > 1e-10:
                        target_dw = density_weight * wl_norm / dens_norm
                        adaptive_density_weight = 0.8 * adaptive_density_weight + 0.2 * target_dw

                    # Gradually adjust density weight based on overflow
                    if max_overflow > 0.3:
                        adaptive_density_weight *= 1.05
                    elif max_overflow < 0.03:
                        adaptive_density_weight *= 0.98
                    # Clamp to reasonable range
                    adaptive_density_weight = max(adaptive_density_weight, density_weight * 0.1)
                    adaptive_density_weight = min(adaptive_density_weight, density_weight * 5)
                # Don't decrease density weight even when overflow is low

            a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
            coef = (a_k - 1) / a_kp1

            max_backtrack = 2
            backtrack_cnt = 0

            while True:
                u_kp1_x = v_x - alpha_k * g_k_x
                u_kp1_y = v_y - alpha_k * g_k_y

                v_kp1_x = u_kp1_x + coef * (u_kp1_x - u_x)
                v_kp1_y = u_kp1_y + coef * (u_kp1_y - u_y)

                v_kp1_x.clamp_(0, self.size_x - 1)
                v_kp1_y.clamp_(0, self.size_y - 1)
                v_kp1_x[bm.is_fixed_t] = torch.tensor(bm.fixed_pos[bm.fixed_indices, 0], dtype=torch.float32, device=self.device)
                v_kp1_y[bm.is_fixed_t] = torch.tensor(bm.fixed_pos[bm.fixed_indices, 1], dtype=torch.float32, device=self.device)

                self.pos_x.copy_(v_kp1_x)
                self.pos_y.copy_(v_kp1_y)
                obj_kp1, g_kp1_x, g_kp1_y = self._compute_obj_and_grad(base_gamma, adaptive_density_weight, wirelength_model)

                alpha_kp1 = torch.sqrt(
                    ((v_kp1_x - v_x) ** 2 + (v_kp1_y - v_y) ** 2).sum() /
                    ((g_kp1_x - g_k_x) ** 2 + (g_kp1_y - g_k_y) ** 2).sum().clamp(min=1e-20)
                )

                backtrack_cnt += 1
                if alpha_kp1 > 0.95 * alpha_k or backtrack_cnt >= max_backtrack:
                    alpha_k.copy_(max(alpha_kp1.item(), alpha_min))
                    break
                else:
                    alpha_k.copy_(max(alpha_kp1.item(), alpha_min))

            u_x.copy_(u_kp1_x)
            u_y.copy_(u_kp1_y)
            v_x.copy_(v_kp1_x)
            v_y.copy_(v_kp1_y)
            g_k_x.copy_(g_kp1_x)
            g_k_y.copy_(g_kp1_y)
            obj_k = obj_kp1
            a_k.copy_(a_kp1)

            self.pos_x.copy_(v_x)
            self.pos_y.copy_(v_y)

            if (iteration + 1) % 50 == 0 or iteration == 0:
                hpwl = self._compute_hpwl_vectorized()
                elapsed = time.time() - start_time
                if hpwl < best_hpwl:
                    best_hpwl = hpwl
                    best_pos_x = self.pos_x.clone()
                    best_pos_y = self.pos_y.clone()

                if verbose:
                    print(f"  GP[Nesterov] iter {iteration+1}/{num_iterations}: "
                          f"HPWL={hpwl:.0f}, gamma={base_gamma:.4f}, "
                          f"dw={adaptive_density_weight:.6f}, overflow={max_overflow:.4f}, "
                          f"alpha={alpha_k.item():.6f}, time={elapsed:.1f}s")

        self.pos_x = best_pos_x if use_best else self.pos_x
        self.pos_y = best_pos_y if use_best else self.pos_y
        # Instead of a separate spread phase, we rely on the GP optimizer
        # with proper density weight to spread instances during optimization.
        # self._spread_instances(num_iters=200)

        return self.pos_x, self.pos_y

    def _compute_hpwl_vectorized(self):
        pin_x = self.pos_x[self.pin_node_ids]
        pin_y = self.pos_y[self.pin_node_ids]

        net_max_x = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('-inf'), device=self.device),
            0, self.pin_net_ids, pin_x, reduce='amax', include_self=True
        )
        net_min_x = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('inf'), device=self.device),
            0, self.pin_net_ids, pin_x, reduce='amin', include_self=True
        )
        net_max_y = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('-inf'), device=self.device),
            0, self.pin_net_ids, pin_y, reduce='amax', include_self=True
        )
        net_min_y = torch.scatter_reduce(
            torch.full((self.num_valid_nets,), float('inf'), device=self.device),
            0, self.pin_net_ids, pin_y, reduce='amin', include_self=True
        )

        hpwl_per_net = (net_max_x - net_min_x) + (net_max_y - net_min_y)
        return hpwl_per_net.sum().item()
