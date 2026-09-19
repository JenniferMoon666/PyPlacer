import os
import re
import numpy as np
import torch
from collections import defaultdict


class ISPD2016Benchmark:
    def __init__(self, benchmark_dir, device='cuda'):
        self.device = device
        self.benchmark_dir = benchmark_dir
        self.name = os.path.basename(benchmark_dir.rstrip('/\\'))

        aux_file = os.path.join(benchmark_dir, 'design.aux')
        with open(aux_file, 'r') as f:
            for line in f:
                if line.startswith('design'):
                    parts = line.strip().split()
                    self.node_file = os.path.join(benchmark_dir, parts[2])
                    self.net_file = os.path.join(benchmark_dir, parts[3])
                    self.wts_file = os.path.join(benchmark_dir, parts[4])
                    self.pl_file = os.path.join(benchmark_dir, parts[5])
                    self.scl_file = os.path.join(benchmark_dir, parts[6])
                    self.lib_file = os.path.join(benchmark_dir, parts[7])
                    break

        self._parse_scl()
        self._parse_nodes()
        self._parse_nets()
        self._parse_pl()
        self._build_tensors()
        print(f"[Benchmark] {self.name}: {self.num_nodes} nodes, {self.num_nets} nets, "
              f"FPGA {self.size_x}x{self.size_y}, movable={self.num_movable}")

    def _parse_scl(self):
        self.site_types = {}
        self.site_capacity = {}
        self.size_x = 0
        self.size_y = 0

        with open(self.scl_file, 'r') as f:
            content = f.read()

        site_blocks = re.findall(r'SITE\s+(\w+)\s*\n(.*?)END SITE', content, re.DOTALL)
        for site_name, block in site_blocks:
            resources = {}
            for line in block.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        resources[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
            self.site_types[site_name] = resources
            self.site_capacity[site_name] = sum(resources.values())

        sitemap_match = re.search(r'SITEMAP\s+(\d+)\s+(\d+)', content)
        if sitemap_match:
            self.size_x = int(sitemap_match.group(1))
            self.size_y = int(sitemap_match.group(2))

        self.site_type_names = ['SLICE', 'DSP', 'BRAM', 'IO']
        self.site_type_to_id = {name: i for i, name in enumerate(self.site_type_names)}

        self.site_grid = np.full((self.size_x, self.size_y), -1, dtype=np.int32)
        self.site_cap_grid = np.zeros((self.size_x, self.size_y), dtype=np.int32)

        sitemap_start = content.index('SITEMAP')
        sitemap_section = content[sitemap_start:]
        matches = re.findall(r'^(\d+)\s+(\d+)\s+(\w+)', sitemap_section, re.MULTILINE)

        for x_str, y_str, stype in matches:
            x, y = int(x_str), int(y_str)
            if 0 <= x < self.size_x and 0 <= y < self.size_y:
                if stype in self.site_type_to_id:
                    self.site_grid[x, y] = self.site_type_to_id[stype]
                if stype in self.site_capacity:
                    self.site_cap_grid[x, y] = self.site_capacity[stype]

        self.site_positions = {}
        for sg_id, sg_name in enumerate(self.site_type_names):
            positions = np.argwhere(self.site_grid == sg_id)
            caps = self.site_cap_grid[self.site_grid == sg_id]
            self.site_positions[sg_id] = positions
            total_cap = int(caps.sum()) if len(caps) > 0 else 0
            print(f"  Site {sg_name}: {len(positions)} sites, capacity={total_cap}")

    def _parse_nodes(self):
        self.node_names = []
        self.node_types = []
        self.inst_type_to_site = {
            'LUT1': 0, 'LUT2': 0, 'LUT3': 0, 'LUT4': 0, 'LUT5': 0, 'LUT6': 0,
            'FDRE': 0, 'CARRY8': 0,
            'DSP48E2': 1,
            'RAMB36E2': 2,
            'IBUF': 3, 'OBUF': 3, 'BUFGCE': 3
        }

        with open(self.node_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.node_names.append(parts[0])
                    self.node_types.append(parts[1])

        self.num_nodes = len(self.node_names)
        self.node_site_group = np.zeros(self.num_nodes, dtype=np.int32)
        for i, ntype in enumerate(self.node_types):
            self.node_site_group[i] = self.inst_type_to_site.get(ntype, 0)

        self.node_name_to_id = {name: i for i, name in enumerate(self.node_names)}

        sg_counts = [0, 0, 0, 0]
        for sg in self.node_site_group:
            sg_counts[sg] += 1
        for sg_id, sg_name in enumerate(self.site_type_names):
            print(f"  Inst {sg_name}: {sg_counts[sg_id]}")

    def _parse_nets(self):
        self.net_names = []
        self.net2pin_node = []

        with open(self.net_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('net'):
                parts = line.split()
                net_name = parts[1]
                pins = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('endnet'):
                    pin_parts = lines[i].strip().split()
                    if len(pin_parts) >= 2:
                        pins.append(pin_parts[0])
                    i += 1
                self.net_names.append(net_name)
                self.net2pin_node.append(pins)
            i += 1

        self.num_nets = len(self.net_names)

    def _parse_pl(self):
        self.fixed_pos = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        self.is_fixed = np.zeros(self.num_nodes, dtype=bool)

        with open(self.pl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    name = parts[0]
                    if name in self.node_name_to_id:
                        nid = self.node_name_to_id[name]
                        x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                        self.fixed_pos[nid] = [x, y, z]
                        if len(parts) >= 5 and parts[4] == 'FIXED':
                            self.is_fixed[nid] = True

        self.movable_mask = ~self.is_fixed
        self.movable_indices = np.where(self.movable_mask)[0]
        self.fixed_indices = np.where(self.is_fixed)[0]
        self.num_movable = len(self.movable_indices)
        self.num_fixed = len(self.fixed_indices)

    def _build_tensors(self):
        node2net = defaultdict(list)
        net2node = defaultdict(list)

        for net_id, pins in enumerate(self.net2pin_node):
            for pin_node_name in pins:
                if pin_node_name in self.node_name_to_id:
                    node_id = self.node_name_to_id[pin_node_name]
                    node2net[node_id].append(net_id)
                    net2node[net_id].append(node_id)

        self.node2net_list = [node2net.get(i, []) for i in range(self.num_nodes)]

        self.node2net_flat = []
        self.node2net_start = [0]
        for node_id in range(self.num_nodes):
            nets = node2net.get(node_id, [])
            self.node2net_flat.extend(nets)
            self.node2net_start.append(len(self.node2net_flat))
        self.node2net_flat = np.array(self.node2net_flat, dtype=np.int64)
        self.node2net_start = np.array(self.node2net_start, dtype=np.int64)

        self.net2node_flat = []
        self.net2node_start = [0]
        for net_id in range(self.num_nets):
            nodes = net2node.get(net_id, [])
            self.net2node_flat.extend(nodes)
            self.net2node_start.append(len(self.net2node_flat))
        self.net2node_flat = np.array(self.net2node_flat, dtype=np.int64)
        self.net2node_start = np.array(self.net2node_start, dtype=np.int64)

        pos_x = np.zeros(self.num_nodes, dtype=np.float32)
        pos_y = np.zeros(self.num_nodes, dtype=np.float32)
        for i in range(self.num_nodes):
            if self.is_fixed[i]:
                pos_x[i] = self.fixed_pos[i, 0]
                pos_y[i] = self.fixed_pos[i, 1]
            else:
                if self.fixed_pos[i, 0] >= 0:
                    pos_x[i] = self.fixed_pos[i, 0]
                    pos_y[i] = self.fixed_pos[i, 1]

        self.pos_x = torch.tensor(pos_x, dtype=torch.float32, device=self.device)
        self.pos_y = torch.tensor(pos_y, dtype=torch.float32, device=self.device)
        self.is_fixed_t = torch.tensor(self.is_fixed, dtype=torch.bool, device=self.device)
        self.node_site_group_t = torch.tensor(self.node_site_group, dtype=torch.int32, device=self.device)
        self.site_grid_t = torch.tensor(self.site_grid, dtype=torch.int32, device=self.device)
        self.site_cap_grid_t = torch.tensor(self.site_cap_grid, dtype=torch.int32, device=self.device)
        self.net2node_flat_t = torch.tensor(self.net2node_flat, dtype=torch.int64, device=self.device)
        self.net2node_start_t = torch.tensor(self.net2node_start, dtype=torch.int64, device=self.device)

    def compute_hpwl(self, pos_x=None, pos_y=None):
        if pos_x is None:
            pos_x = self.pos_x
        if pos_y is None:
            pos_y = self.pos_y

        total_hpwl = 0.0
        px = pos_x.cpu().numpy()
        py = pos_y.cpu().numpy()

        for net_id in range(self.num_nets):
            s = self.net2node_start[net_id]
            e = self.net2node_start[net_id + 1]
            if e - s < 2:
                continue
            nodes = self.net2node_flat[s:e]
            xs = px[nodes]
            ys = py[nodes]
            hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            total_hpwl += hpwl

        return total_hpwl

    def compute_hpwl_gpu(self, pos_x=None, pos_y=None):
        if pos_x is None:
            pos_x = self.pos_x
        if pos_y is None:
            pos_y = self.pos_y

        total_hpwl = torch.tensor(0.0, device=self.device)

        for net_id in range(self.num_nets):
            s = self.net2node_start[net_id]
            e = self.net2node_start[net_id + 1]
            if e - s < 2:
                continue
            nodes = self.net2node_flat_t[s:e]
            xs = pos_x[nodes]
            ys = pos_y[nodes]
            hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
            total_hpwl = total_hpwl + hpwl

        return total_hpwl.item()

    def compute_hpwl_fast(self, pos_x=None, pos_y=None, x_weight=1.0, y_weight=1.0):
        """Compute HPWL with optional direction weights (matching DREAMPlaceFPGA's 0.7/1.2)."""
        if pos_x is None:
            pos_x = self.pos_x
        if pos_y is None:
            pos_y = self.pos_y

        pin_x = pos_x[self.net2node_flat_t]
        pin_y = pos_y[self.net2node_flat_t]

        net_ids = torch.arange(self.num_nets, device=self.device)
        net_sizes = self.net2node_start_t[1:] - self.net2node_start_t[:-1]
        net_ids_expanded = net_ids.repeat_interleave(net_sizes)

        x_max = torch.zeros(self.num_nets, dtype=torch.float32, device=self.device)
        x_min = torch.full((self.num_nets,), float('inf'), dtype=torch.float32, device=self.device)
        y_max = torch.zeros(self.num_nets, dtype=torch.float32, device=self.device)
        y_min = torch.full((self.num_nets,), float('inf'), dtype=torch.float32, device=self.device)

        x_max.scatter_reduce_(0, net_ids_expanded, pin_x, reduce='amax', include_self=True)
        x_min.scatter_reduce_(0, net_ids_expanded, pin_x, reduce='amin', include_self=True)
        y_max.scatter_reduce_(0, net_ids_expanded, pin_y, reduce='amax', include_self=True)
        y_min.scatter_reduce_(0, net_ids_expanded, pin_y, reduce='amin', include_self=True)

        multi_pin = net_sizes > 1
        hpwl_x = (x_max[multi_pin] - x_min[multi_pin]).sum()
        hpwl_y = (y_max[multi_pin] - y_min[multi_pin]).sum()
        return (x_weight * hpwl_x + y_weight * hpwl_y).item()
