"""PyPlacer V39: GP + DREAMPlaceFPGA C++ LUT/FF Legalization + MCF DSP/RAM
+ GPU Directed Move + Multi-round Swap Refinement + GPU-accelerated WindowDP

V3 KEY IMPROVEMENTS over V2 (V38):
1. GPU-accelerated HPWL tracking in WindowDP (replaces slow np.maximum.at)
2. GPU centroid computation for directed move guidance
3. GPU Directed Move phase (coarse optimization before WindowDP)
4. Multi-round optimization: DirectedMove -> Swap -> DP -> Swap -> DP -> Swap -> DP
5. Larger DP time budgets (1.5x) with GPU HPWL tracking + centroid guidance
6. Weighted HPWL (0.7/1.2) used throughout for DREAMPlaceFPGA comparison
7. Net-weighted instance selection (GPU computes per-net HPWL contributions)
8. SA enabled for medium circuits (50K-500K movable), not just small
9. SLICE swap skipped for medium/large circuits (too slow, WindowDP handles it)
10. Centroid-guided moves in WindowDP (prefer moves towards ideal positions)

GPU UTILIZATION STRATEGY:
- GPU scatter_reduce_ for fast batch HPWL evaluation
- GPU scatter_add for centroid computation (ideal positions)
- GPU for net HPWL contribution analysis (instance prioritization)
- GPU Directed Move phase uses GPU for both centroids and HPWL evaluation
- Positions stay on CPU numpy arrays, copied to GPU only for batch operations
"""
import os
import sys
import time
import gc
import torch
import numpy as np
from collections import defaultdict
from numba import njit

# ===== Setup DREAMPlaceFPGA import paths =====
_dp_root = r'D:\Codes\VLSI\DeepSeek2\DREAMPlaceFPGA-local'
_dp_pkg = os.path.join(_dp_root, 'dreamplacefpga')
if _dp_root not in sys.path:
    sys.path.insert(0, _dp_root)
if _dp_pkg not in sys.path:
    sys.path.insert(0, _dp_pkg)

# Add build directories for compiled C++ modules
for _build_dir in ['build_cpu', 'build_v142_cuda', 'build_v142_fix7', 'build_v142_fix8']:
    _bd = os.path.join(_dp_root, _build_dir)
    if os.path.exists(_bd) and _bd not in sys.path:
        sys.path.insert(0, _bd)

_torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(_torch_lib):
    os.add_dll_directory(_torch_lib)

# Import DREAMPlaceFPGA modules
try:
    import dreamplacefpga.ops.lut_ff_legalization.lut_ff_legalization as lut_ff_legalization
    import dreamplacefpga.ops.precondWL.precondWL as precondWL
    import dreamplacefpga.ops.sortNode2Pin.sortNode2Pin as sortNode2Pin
    import dreamplacefpga.ops.hpwl.hpwl as hpwl
    import dreamplacefpga.ops.pin_pos.pin_pos as pin_pos
    import dreamplacefpga.ops.move_boundary.move_boundary as move_boundary
    import dreamplacefpga.ops.demandMap.demandMap as demandMap
    import dreamplacefpga.ops.electric_potential.electric_overflow as electric_overflow
    import dreamplacefpga.ops.draw_place.draw_place as draw_place
    from dreamplacefpga.PlaceDB import PlaceDBFPGA
    from dreamplacefpga.Params import ParamsFPGA
    from dreamplacefpga.BasicPlace import PlaceDataCollectionFPGA, PlaceOpCollectionFPGA
    HAS_DP_LG = True
    print("[V39] DREAMPlaceFPGA modules loaded successfully")
except ImportError as e:
    HAS_DP_LG = False
    print(f"[V39] WARNING: DREAMPlaceFPGA modules not available: {e}")
    print("[V39] Falling back to BFS legalization")

# Import DREAMPlaceFPGA C++ MCF module (from the old path)
_dp_main = r'D:\Codes\VLSI\DREAMPlaceFPGA-main'
if _dp_main not in sys.path:
    sys.path.insert(0, _dp_main)

try:
    import dreamplacefpga.ops.dsp_ram_legalization.legalize_cpp as _legalize_cpp
    HAS_MCF = True
    print("[V39] DREAMPlaceFPGA legalize_cpp module loaded successfully")
except ImportError as e:
    HAS_MCF = False
    print(f"[V39] WARNING: legalize_cpp not available: {e}")

from benchmark import ISPD2016Benchmark
from global_placement import EPlaceGlobalPlacer
from legalization import Legalizer, _delta_hpwl_swap, _delta_hpwl_move, _legalize_bfs, _legalize_bfs_weighted_v2


BASELINE = {
    'FPGA-example1': 13562,
    'FPGA-example2': 2914068,
    'FPGA-example3': 7781857,
    'FPGA-example4': 8221614,
}

# DREAMPlaceFPGA direction weights for HPWL (from PlaceDB.py)
DPFPGA_X_WEIGHT = 0.7
DPFPGA_Y_WEIGHT = 1.2


def save_placement(bm, pos_x, pos_y, output_dir):
    pos_x_np = pos_x.cpu().numpy(); pos_y_np = pos_y.cpu().numpy()
    output_file = os.path.join(output_dir, f"{bm.name}_placement.txt")
    with open(output_file, 'w') as f:
        for i in range(bm.num_nodes):
            name = bm.node_names[i] if i < len(bm.node_names) else f"inst_{i}"
            x = int(round(pos_x_np[i])); y = int(round(pos_y_np[i])); z = 0
            f.write(f"{name} {x} {y} {z}\n")
    print(f"  Placement saved to: {output_file}")
    return output_file


def spread_positions(pos_x, pos_y, bm, target_coverage=0.85):
    movable_mask = ~bm.is_fixed
    for sg in range(4):
        mask = (bm.node_site_group == sg) & movable_mask
        if isinstance(mask, torch.Tensor): mask_np = mask.cpu().numpy()
        else: mask_np = mask
        indices = np.where(mask_np)[0]
        if len(indices) == 0: continue
        site_mask = (bm.site_grid == sg)
        if isinstance(site_mask, torch.Tensor): site_mask_np = site_mask.cpu().numpy()
        else: site_mask_np = site_mask
        site_positions = np.argwhere(site_mask_np)
        if len(site_positions) == 0: continue
        site_x_min = float(site_positions[:, 0].min()); site_x_max = float(site_positions[:, 0].max())
        site_y_min = float(site_positions[:, 1].min()); site_y_max = float(site_positions[:, 1].max())
        inst_x = pos_x[indices]; inst_y = pos_y[indices]
        x_min, x_max = inst_x.min().item(), inst_x.max().item()
        y_min, y_max = inst_y.min().item(), inst_y.max().item()
        x_range = x_max - x_min; y_range = y_max - y_min
        target_x_range = (site_x_max - site_x_min) * target_coverage
        target_y_range = (site_y_max - site_y_min) * target_coverage
        if x_range < 1e-6 or y_range < 1e-6: continue
        scale_x = max(1.0, target_x_range / x_range)
        scale_y = max(1.0, target_y_range / y_range)
        scale = min(scale_x, scale_y)
        center_x = (x_min + x_max) / 2; center_y = (y_min + y_max) / 2
        target_center_x = (site_x_min + site_x_max) / 2; target_center_y = (site_y_min + site_y_max) / 2
        with torch.no_grad():
            pos_x[indices] = (inst_x - center_x) * scale + target_center_x
            pos_y[indices] = (inst_y - center_y) * scale + target_center_y
            pos_x.clamp_(0, bm.size_x - 1); pos_y.clamp_(0, bm.size_y - 1)
    return pos_x, pos_y


def run_dreamplacefpga_lg(bm, gp_pos_x, gp_pos_y, benchmark_dir, device='cpu'):
    """Run DREAMPlaceFPGA's C++ LUT/FF legalization on PyPlacer's GP output."""
    lg_device = torch.device('cpu')

    print("  [DP-LG] Reading benchmark with DREAMPlaceFPGA PlaceDB...")
    tt = time.time()

    orig_cwd = os.getcwd()
    os.chdir(benchmark_dir)

    try:
        params = ParamsFPGA()
        aux_file = None
        for f in os.listdir(benchmark_dir):
            if f.endswith('.aux') or f.endswith('.AUX'):
                aux_file = os.path.join(benchmark_dir, f)
                break

        if aux_file is not None:
            params.aux_input = aux_file
        else:
            for f in os.listdir(benchmark_dir):
                if f.endswith('.device'):
                    params.interchange_device = os.path.join(benchmark_dir, f)
                    break

        params.dtype = 'float32'
        params.gpu = 0
        params.num_threads = 8
        params.enable_fillers = 0
        params.enableTimingPreclustering = 0
        params.lg_alpha = 0.0
        params.lg_beta = 0.0
        params.ffPinWeight = 3.0
        params.unit_pin_capacity = 50
        params.ignore_net_degree = 100
        params.deterministic_flag = 1
        params.routability_opt_flag = 0
        params.timing_driven_flag = 0

        placedb = PlaceDBFPGA()
        placedb(params)
    finally:
        os.chdir(orig_cwd)

    print(f"  [DP-LG] PlaceDB loaded: {placedb.num_physical_nodes} nodes, "
          f"{placedb.num_movable_nodes} movable, {placedb.num_nets} nets "
          f"({time.time()-tt:.1f}s)")

    print("  [DP-LG] Mapping positions from PyPlacer to DREAMPlaceFPGA ordering...")
    tt = time.time()

    py_name2id = {}
    for i in range(bm.num_nodes):
        if i < len(bm.node_names):
            py_name2id[bm.node_names[i]] = i

    dp_name2id = placedb.node_name2id_map

    dp2py = np.full(placedb.num_physical_nodes, -1, dtype=np.int64)
    py2dp = np.full(bm.num_nodes, -1, dtype=np.int64)

    matched = 0
    for name, dp_id in dp_name2id.items():
        if name in py_name2id:
            py_id = py_name2id[name]
            if dp_id < placedb.num_physical_nodes and py_id < bm.num_nodes:
                dp2py[dp_id] = py_id
                py2dp[py_id] = dp_id
                matched += 1

    print(f"  [DP-LG] Matched {matched}/{placedb.num_physical_nodes} nodes "
          f"({time.time()-tt:.1f}s)")

    if matched < placedb.num_physical_nodes * 0.9:
        print(f"  [DP-LG] WARNING: Only {matched}/{placedb.num_physical_nodes} nodes matched!")
        return None, None

    num_nodes = placedb.num_nodes
    num_physical = placedb.num_physical_nodes

    pos = torch.zeros(num_nodes * 2, dtype=torch.float32, device=lg_device)

    gp_x_np = gp_pos_x.cpu().numpy()
    gp_y_np = gp_pos_y.cpu().numpy()

    for dp_id in range(num_physical):
        py_id = dp2py[dp_id]
        if py_id >= 0:
            pos[dp_id] = float(gp_x_np[py_id])
            pos[num_nodes + dp_id] = float(gp_y_np[py_id])
        else:
            pos[dp_id] = float(placedb.node_x[dp_id])
            pos[num_nodes + dp_id] = float(placedb.node_y[dp_id])

    for dp_id in range(placedb.num_movable_nodes, num_physical):
        pos[dp_id] = float(placedb.node_x[dp_id])
        pos[num_nodes + dp_id] = float(placedb.node_y[dp_id])

    print("  [DP-LG] Building DataCollections...")
    tt = time.time()

    class SimpleParams:
        pass
    p = SimpleParams()
    p.num_threads = params.num_threads
    p.dtype = params.dtype
    p.routability_opt_flag = 0
    p.ffPinWeight = params.ffPinWeight
    p.unit_pin_capacity = params.unit_pin_capacity
    p.deterministic_flag = 1
    p.ignore_net_degree = params.ignore_net_degree

    pos_param = torch.nn.ParameterList([torch.nn.Parameter(pos)])

    data_collections = PlaceDataCollectionFPGA(pos_param, p, placedb, lg_device)

    print(f"  [DP-LG] DataCollections built ({time.time()-tt:.1f}s)")

    print("  [DP-LG] Building ops...")
    tt = time.time()

    precondwl_op = precondWL.PrecondWL(
        flat_node2pin_start=data_collections.flat_node2pin_start_map,
        flat_node2pin=data_collections.flat_node2pin_map,
        pin2net_map=data_collections.pin2net_map,
        flat_net2pin=data_collections.flat_net2pin_start_map,
        net_weights=data_collections.net_weights,
        num_nodes=placedb.num_nodes,
        num_movable_nodes=placedb.num_physical_nodes,
        device=lg_device,
        num_threads=params.num_threads)

    sort_node2pin_op = sortNode2Pin.SortNode2Pin(
        flat_node2pin_start=data_collections.flat_node2pin_start_map,
        flat_node2pin=data_collections.flat_node2pin_map,
        num_nodes=placedb.num_physical_nodes,
        device=lg_device,
        num_threads=params.num_threads)

    avgLUTArea = data_collections.node_areas[:num_physical][data_collections.node2fence_region_map == 0].sum()
    avgLUTArea /= placedb.node_count[0]
    avgFFArea = data_collections.node_areas[:num_physical][data_collections.node2fence_region_map == 1].sum()
    avgFFArea /= placedb.node_count[1]

    inst_areas = data_collections.node_areas[:num_physical].detach().clone()
    inst_areas[data_collections.node2fence_region_map > 1] = 0.0
    inst_areas[data_collections.node2fence_region_map == 0] /= avgLUTArea
    inst_areas[data_collections.node2fence_region_map == 1] /= avgFFArea

    site_types = data_collections.site_type_map.detach().clone()
    site_types[site_types > 1] = 0

    if len(data_collections.net_weights):
        net_wts = data_collections.net_weights
    else:
        net_wts = torch.ones(placedb.num_nets, dtype=torch.float32, device=lg_device)

    lg_op = lut_ff_legalization.LegalizeCLB(
        lutFlopIndices=data_collections.flop_lut_indices,
        nodeNames=placedb.node_names,
        flop2ctrlSet=data_collections.flop2ctrlSetId_map,
        flop_ctrlSet=data_collections.flop_ctrlSets,
        pin2node=data_collections.pin2node_map,
        pin2net=data_collections.pin2net_map,
        snkpin2tnet=data_collections.snkpin2tnet_map,
        net2tnet_start=data_collections.net2tnet_start_map,
        flat_tnet2pin_map=data_collections.flat_tnet2pin_map,
        flat_net2pin=data_collections.flat_net2pin_map,
        flat_net2pin_start=data_collections.flat_net2pin_start_map,
        flat_node2pin=data_collections.flat_node2pin_map,
        flat_node2pin_start=data_collections.flat_node2pin_start_map,
        node2fence=data_collections.node2fence_region_map,
        pin_types=data_collections.pin_typeIds,
        lut_type=data_collections.lut_type,
        net_wts=net_wts,
        tnet_wts=data_collections.tnet_weights,
        avg_lut_area=avgLUTArea,
        avg_ff_area=avgFFArea,
        inst_areas=inst_areas,
        pin_offset_x=data_collections.lg_pin_offset_x,
        pin_offset_y=data_collections.lg_pin_offset_y,
        site_types=site_types,
        site_xy=data_collections.lg_siteXYs,
        node_size_x=data_collections.node_size_x[:num_physical],
        node_size_y=data_collections.node_size_y[:num_physical],
        node2outpin=data_collections.node2outpinIdx_map[:num_physical],
        net2pincount=data_collections.net2pincount_map,
        node2pincount=data_collections.node2pincount_map,
        spiral_accessor=data_collections.spiral_accessor,
        num_nets=placedb.num_nets,
        num_movable_nodes=placedb.num_movable_nodes,
        num_nodes=num_physical,
        num_sites_x=placedb.num_sites_x,
        num_sites_y=placedb.num_sites_y,
        xWirelenWt=placedb.xWirelenWt,
        yWirelenWt=placedb.yWirelenWt,
        lg_alpha=params.lg_alpha,
        lg_beta=params.lg_beta,
        enableTimingPreclustering=params.enableTimingPreclustering,
        nbrDistEnd=placedb.nbrDistEnd,
        num_threads=params.num_threads,
        device=lg_device)

    print(f"  [DP-LG] Ops built ({time.time()-tt:.1f}s)")

    print("  [DP-LG] Running C++ LUT/FF legalization...")
    lg_start = time.time()

    precond_wl = precondwl_op()

    _, sortedNetIdx = torch.sort(data_collections.net2pincount_map)
    sortedNetIdx = sortedNetIdx.to(torch.int32)
    _, sortedNetMap = torch.sort(sortedNetIdx)
    sortedNetMap = sortedNetMap.to(torch.int32)

    _, sortedPinIdx = torch.sort(sortedNetMap[data_collections.pin2net_map.to(torch.long)])
    sortedPinIdx = sortedPinIdx.to(torch.int32)
    _, sortedPinMap = torch.sort(sortedPinIdx)
    sortedPinMap = sortedPinMap.to(torch.int32)

    node2pinId0 = sort_node2pin_op(sortedPinMap)

    _, sortedNodeIdx = torch.sort(node2pinId0)
    sortedNodeIdx = sortedNodeIdx.to(torch.int32)
    _, sortedNodeMap = torch.sort(sortedNodeIdx)
    sortedNodeMap = sortedNodeMap.to(torch.int32)

    lg_op.initialize(pos_param[0], precond_wl[:num_physical],
                     sortedNodeMap, sortedNodeIdx, sortedNetMap, sortedNetIdx, sortedPinMap)

    activeStatus = torch.zeros(placedb.num_sites_x * placedb.num_sites_y,
                               dtype=torch.int, device=lg_device)
    illegalStatus = torch.zeros(num_physical, dtype=torch.int, device=lg_device)

    DLStatus = 1
    dlIter = 0
    iter_stable = 0
    prevAct = 0

    print("  [DP-LG] Starting DL iterations...")
    while DLStatus == 1:
        lg_op.runDLIter(pos_param[0], precond_wl[:num_physical],
                        sortedNodeMap, sortedNodeIdx, sortedNetMap, sortedNetIdx, sortedPinMap,
                        activeStatus, illegalStatus, dlIter)

        active_cnt = activeStatus.sum().item()
        illegal_cnt = illegalStatus.sum().item()

        if dlIter == 0 or dlIter % 10 == 0:
            print(f"  [DP-LG] DL iter {dlIter}: active={active_cnt}, illegal={illegal_cnt}, stable={iter_stable}")

        if prevAct == illegal_cnt + active_cnt:
            iter_stable += 1
        else:
            iter_stable = 0

        dlIter += 1
        if active_cnt > 0:
            DLStatus = 1
        elif illegal_cnt > 0:
            DLStatus = -1
        else:
            DLStatus = 0

        prevAct = illegal_cnt + active_cnt
        if dlIter > 100 or iter_stable > 5:
            DLStatus = 0

    print(f"  [DP-LG] DL iterations done: status={DLStatus}, iter={dlIter}, "
          f"active={activeStatus.sum().item()}, illegal={illegalStatus.sum().item()}")

    node_z = data_collections.node_z[:placedb.num_movable_nodes].detach().clone()
    pos_param[0].data.copy_(lg_op.ripUP_Greedy_slotAssign(
        pos_param[0], precond_wl[:num_physical], node_z,
        sortedNodeMap, sortedNodeIdx, sortedNetMap, sortedNetIdx, sortedPinMap))

    lg_time = time.time() - lg_start
    print(f"  [DP-LG] C++ LUT/FF legalization completed ({lg_time:.1f}s)")

    print("  [DP-LG] Mapping positions back to PyPlacer ordering...")
    lg_pos_x = pos_param[0][:num_physical].detach().cpu().numpy()
    lg_pos_y = pos_param[0][num_nodes:num_nodes + num_physical].detach().cpu().numpy()

    legal_x = gp_pos_x.cpu().clone()
    legal_y = gp_pos_y.cpu().clone()

    for dp_id in range(num_physical):
        py_id = dp2py[dp_id]
        if py_id >= 0:
            legal_x[py_id] = float(lg_pos_x[dp_id])
            legal_y[py_id] = float(lg_pos_y[dp_id])

    del data_collections, lg_op, precondwl_op, sort_node2pin_op
    del pos_param, precond_wl
    del activeStatus, illegalStatus, node_z
    del sortedNetIdx, sortedNetMap, sortedPinIdx, sortedPinMap
    del sortedNodeIdx, sortedNodeMap, node2pinId0
    del inst_areas, site_types, net_wts
    gc.collect()

    return legal_x, legal_y


class DSPRAMLegalizer:
    """MCF legalization for DSP/RAM/IO instances."""

    def __init__(self, benchmark, device='cuda'):
        self.bm = benchmark
        self.device = device
        self.size_x = benchmark.size_x
        self.size_y = benchmark.size_y

    def _build_site_data(self, sg):
        bm = self.bm
        site_mask = (bm.site_grid == sg)
        site_pos = np.argwhere(site_mask)
        site_caps = bm.site_cap_grid[site_mask].astype(np.int64)
        return site_pos, site_caps

    def _compute_precond(self, inst_indices):
        bm = self.bm
        num_inst = len(inst_indices)
        precond = np.ones(num_inst, dtype=np.float64)
        for i, inst_id in enumerate(inst_indices):
            if inst_id < len(bm.node2net_start) - 1:
                n_nets = bm.node2net_start[inst_id + 1] - bm.node2net_start[inst_id]
                precond[i] = 1.0 + n_nets * 0.5
        return precond

    def _mcf_legalize(self, inst_x, inst_y, site_pos, site_caps, precond,
                       lg_max_dist_init=20.0, lg_max_dist_incr=10.0,
                       lg_flow_cost_scale=100.0):
        num_inst = len(inst_x)
        if num_inst == 0:
            return np.full(0, -1, dtype=np.int64)

        virtual_sites = []
        virtual_to_physical = []
        for s in range(len(site_pos)):
            for _ in range(int(site_caps[s])):
                virtual_sites.append(site_pos[s])
                virtual_to_physical.append(s)
        virtual_sites = np.array(virtual_sites, dtype=np.float64)
        virtual_to_physical = np.array(virtual_to_physical, dtype=np.int64)

        if len(virtual_sites) == 0:
            return np.full(num_inst, -1, dtype=np.int64)

        num_sites = len(virtual_sites)
        sites_flat = virtual_sites.flatten()

        locX = inst_x.astype(np.float64)
        locY = inst_y.astype(np.float64)
        precondArr = precond.astype(np.float64)
        movVal = [0.0, 0.0]
        outLoc = [0.0] * (2 * num_inst)

        try:
            _legalize_cpp.legalize(
                locX, locY, num_inst, num_sites,
                sites_flat, precondArr,
                lg_max_dist_init, lg_max_dist_incr, lg_flow_cost_scale,
                movVal, outLoc)
        except Exception as e:
            print(f"  [MCF] legalize_cpp failed: {e}")
            return np.full(num_inst, -1, dtype=np.int64)

        outLoc = np.array(outLoc)
        out_x = outLoc[:num_inst]
        out_y = outLoc[num_inst:]

        assignment = np.full(num_inst, -1, dtype=np.int64)
        site_used = np.zeros(len(site_pos), dtype=np.int64)

        for i in range(num_inst):
            if out_x[i] == 0.0 and out_y[i] == 0.0:
                continue
            best_dist = float('inf')
            best_site = -1
            for s in range(len(site_pos)):
                if site_used[s] >= site_caps[s]:
                    continue
                dist = abs(out_x[i] - site_pos[s, 0]) + abs(out_y[i] - site_pos[s, 1])
                if dist < best_dist:
                    best_dist = dist
                    best_site = s
            if best_site >= 0 and best_dist < 2.0:
                assignment[i] = best_site
                site_used[best_site] += 1

        return assignment

    def legalize(self, pos_x, pos_y):
        """Legalize DSP/RAM/IO instances only (SLICE already done by DREAMPlaceFPGA)."""
        bm = self.bm
        start_time = time.time()
        legal_x = pos_x.clone()
        legal_y = pos_y.clone()

        for sg in range(1, 4):  # Only DSP, BRAM, IO (skip SLICE=0)
            sg_start = time.time()
            sg_name = ['SLICE', 'DSP', 'BRAM', 'IO'][sg]

            site_pos, site_caps = self._build_site_data(sg)
            if len(site_pos) == 0:
                continue

            inst_mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            inst_indices = np.where(inst_mask)[0]
            num_inst = len(inst_indices)
            if num_inst == 0:
                continue

            inst_x = pos_x[inst_indices].cpu().numpy().astype(np.float64)
            inst_y = pos_y[inst_indices].cpu().numpy().astype(np.float64)

            if HAS_MCF and num_inst < 50000:
                precond = self._compute_precond(inst_indices)
                assignment = self._mcf_legalize(
                    inst_x, inst_y, site_pos, site_caps, precond,
                    lg_max_dist_init=20.0, lg_max_dist_incr=10.0,
                    lg_flow_cost_scale=100.0)
                mcf_assigned = int((assignment >= 0).sum())
                print(f"  [LG-{sg_name}] MCF assigned {mcf_assigned}/{num_inst}")

                for i in range(num_inst):
                    if assignment[i] >= 0:
                        inst_id = inst_indices[i]
                        s_idx = assignment[i]
                        legal_x[inst_id] = float(site_pos[s_idx, 0])
                        legal_y[inst_id] = float(site_pos[s_idx, 1])
            else:
                cap = np.zeros((int(bm.size_y), int(bm.size_x)), dtype=np.int64)
                for x in range(int(bm.size_x)):
                    for y in range(int(bm.size_y)):
                        if bm.site_grid[x, y] == sg:
                            cap[y, x] = int(bm.site_cap_grid[x, y])
                assignment_bfs = _legalize_bfs(
                    inst_x.astype(np.float32), inst_y.astype(np.float32),
                    cap, sg, int(bm.size_y), int(bm.size_x))
                size_x_int = int(bm.size_x)
                assigned_count = 0
                for i in range(num_inst):
                    if assignment_bfs[i] >= 0:
                        inst_id = inst_indices[i]
                        linear_idx = assignment_bfs[i]
                        ax = linear_idx % size_x_int
                        ay = linear_idx // size_x_int
                        legal_x[inst_id] = float(ax)
                        legal_y[inst_id] = float(ay)
                        assigned_count += 1
                print(f"  [LG-{sg_name}] BFS assigned {assigned_count}/{num_inst}")

            sg_time = time.time() - sg_start
            print(f"  [LG-{sg_name}] completed ({sg_time:.1f}s)")

        for fid in bm.fixed_indices:
            legal_x[fid] = float(bm.fixed_pos[fid, 0])
            legal_y[fid] = float(bm.fixed_pos[fid, 1])

        return legal_x, legal_y


@njit(cache=True)
def _greedy_swap_refinement(pos_x, pos_y, inst_ids, size_y, size_x,
                             node2net_start, node2net_flat, net2node_start, net2node_flat,
                             max_passes, radius):
    """Greedy swap-based HPWL refinement after LG."""
    num_inst = inst_ids.shape[0]
    max_per_site = 40

    site_count = np.zeros((size_x, size_y), dtype=np.int32)
    site_insts = np.full((size_x, size_y, max_per_site), -1, dtype=np.int64)

    for i in range(num_inst):
        inst_id = inst_ids[i]
        ix = int(round(pos_x[inst_id]))
        iy = int(round(pos_y[inst_id]))
        ix = min(max(ix, 0), size_x - 1)
        iy = min(max(iy, 0), size_y - 1)
        slot = site_count[ix, iy]
        if slot < max_per_site:
            site_insts[ix, iy, slot] = inst_id
            site_count[ix, iy] = slot + 1

    total_improvement = 0.0
    total_swaps = 0

    for pass_num in range(max_passes):
        pass_improvement = 0.0
        num_swaps = 0

        for i in range(num_inst):
            inst_id = inst_ids[i]
            ix = int(round(pos_x[inst_id]))
            iy = int(round(pos_y[inst_id]))
            ix = min(max(ix, 0), size_x - 1)
            iy = min(max(iy, 0), size_y - 1)

            best_delta = 0.0
            best_swap_id = -1

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = ix + dx
                    ny = iy + dy
                    if nx < 0 or nx >= size_x or ny < 0 or ny >= size_y:
                        continue

                    for slot in range(site_count[nx, ny]):
                        other_id = site_insts[nx, ny, slot]
                        if other_id < 0 or other_id == inst_id:
                            continue

                        delta = _delta_hpwl_swap(
                            pos_x, pos_y, inst_id, other_id,
                            pos_x[inst_id], pos_y[inst_id],
                            pos_x[other_id], pos_y[other_id],
                            node2net_start, node2net_flat,
                            net2node_start, net2node_flat)

                        if delta < best_delta:
                            best_delta = delta
                            best_swap_id = other_id

            if best_swap_id >= 0 and best_delta < -0.01:
                old_x1 = pos_x[inst_id]; old_y1 = pos_y[inst_id]
                old_x2 = pos_x[best_swap_id]; old_y2 = pos_y[best_swap_id]

                ix1 = int(round(old_x1)); iy1 = int(round(old_y1))
                ix2 = int(round(old_x2)); iy2 = int(round(old_y2))

                for slot in range(site_count[ix1, iy1]):
                    if site_insts[ix1, iy1, slot] == inst_id:
                        site_insts[ix1, iy1, slot] = best_swap_id
                        break
                for slot in range(site_count[ix2, iy2]):
                    if site_insts[ix2, iy2, slot] == best_swap_id:
                        site_insts[ix2, iy2, slot] = inst_id
                        break

                pos_x[inst_id] = old_x2; pos_y[inst_id] = old_y2
                pos_x[best_swap_id] = old_x1; pos_y[best_swap_id] = old_y1

                pass_improvement += best_delta
                num_swaps += 1

        total_improvement += pass_improvement
        total_swaps += num_swaps
        if num_swaps == 0:
            break

    return total_improvement, total_swaps


class GPUHPWLTracker:
    """GPU-accelerated weighted HPWL computation and centroid guidance.

    V3 improvements:
    - GPU scatter_reduce_ for fast HPWL evaluation
    - GPU centroid computation for directed move guidance
    - Per-net HPWL contribution analysis for instance prioritization

    Key insight: positions stay on CPU (numpy), only copied to GPU
    for periodic HPWL evaluation and centroid computation.
    """

    def __init__(self, benchmark, device='cuda'):
        self.bm = benchmark
        self.device = device
        bm = benchmark

        self.net2node_flat_gpu = torch.tensor(bm.net2node_flat, dtype=torch.long, device=device)
        net_sizes = bm.net2node_start[1:] - bm.net2node_start[:-1]
        self.net_ids_exp = torch.tensor(
            np.repeat(np.arange(bm.num_nets), net_sizes), dtype=torch.long, device=device)
        self.multi_pin_mask = torch.tensor(net_sizes > 1, dtype=torch.bool, device=device)
        self.num_nets = bm.num_nets

        # Pin-level data for centroid computation
        # pin2node: for each pin (node-net connection), which node it belongs to
        # pin2net: for each pin, which net it connects to
        pin2node = np.repeat(np.arange(bm.num_nodes), np.diff(bm.node2net_start))
        self.pin2node_gpu = torch.tensor(pin2node, dtype=torch.long, device=device)
        self.pin2net_gpu = torch.tensor(bm.node2net_flat, dtype=torch.long, device=device)
        self.net_sizes_gpu = torch.tensor(net_sizes, dtype=torch.float32, device=device)

    def compute_weighted_hpwl(self, pos_x, pos_y):
        """Compute weighted HPWL using GPU scatter_reduce_.

        Args:
            pos_x: numpy array or CPU tensor of x positions
            pos_y: numpy array or CPU tensor of y positions
        Returns:
            float: weighted HPWL value
        """
        if isinstance(pos_x, np.ndarray):
            x_gpu = torch.tensor(pos_x, dtype=torch.float32, device=self.device)
            y_gpu = torch.tensor(pos_y, dtype=torch.float32, device=self.device)
        else:
            x_gpu = pos_x.to(self.device)
            y_gpu = pos_y.to(self.device)

        pin_xs = x_gpu[self.net2node_flat_gpu]
        pin_ys = y_gpu[self.net2node_flat_gpu]

        x_max = torch.full((self.num_nets,), -1e18, device=self.device)
        x_min = torch.full((self.num_nets,), 1e18, device=self.device)
        y_max = torch.full((self.num_nets,), -1e18, device=self.device)
        y_min = torch.full((self.num_nets,), 1e18, device=self.device)

        x_max.scatter_reduce_(0, self.net_ids_exp, pin_xs, reduce='amax', include_self=True)
        x_min.scatter_reduce_(0, self.net_ids_exp, pin_xs, reduce='amin', include_self=True)
        y_max.scatter_reduce_(0, self.net_ids_exp, pin_ys, reduce='amax', include_self=True)
        y_min.scatter_reduce_(0, self.net_ids_exp, pin_ys, reduce='amin', include_self=True)

        hpwl_x = (x_max[self.multi_pin_mask] - x_min[self.multi_pin_mask]).sum()
        hpwl_y = (y_max[self.multi_pin_mask] - y_min[self.multi_pin_mask]).sum()

        return (DPFPGA_X_WEIGHT * hpwl_x + DPFPGA_Y_WEIGHT * hpwl_y).item()

    def compute_centroids(self, pos_x, pos_y):
        """Compute weighted centroid for each node using GPU.

        For each node, the centroid is the average position of all OTHER nodes
        on the nets connected to that node. This gives the "ideal" position
        for minimizing wirelength.

        Returns:
            (centroid_x, centroid_y): numpy arrays of shape [num_nodes]
        """
        bm = self.bm
        device = self.device

        if isinstance(pos_x, np.ndarray):
            x_gpu = torch.tensor(pos_x, dtype=torch.float32, device=device)
            y_gpu = torch.tensor(pos_y, dtype=torch.float32, device=device)
        else:
            x_gpu = pos_x.to(device)
            y_gpu = pos_y.to(device)

        # Step 1: Per-net sum and count of node positions
        node_xs = x_gpu[self.net2node_flat_gpu]
        node_ys = y_gpu[self.net2node_flat_gpu]

        net_sum_x = torch.zeros(self.num_nets, dtype=torch.float32, device=device)
        net_sum_y = torch.zeros(self.num_nets, dtype=torch.float32, device=device)
        net_sum_x.scatter_add_(0, self.net_ids_exp, node_xs)
        net_sum_y.scatter_add_(0, self.net_ids_exp, node_ys)

        # Step 2: For each pin, compute centroid of OTHER nodes on the same net
        # centroid = (sum - self_pos) / (count - 1)
        pin_net_sum_x = net_sum_x[self.pin2net_gpu]
        pin_net_sum_y = net_sum_y[self.pin2net_gpu]
        pin_net_count = self.net_sizes_gpu[self.pin2net_gpu]
        pin_node_x = x_gpu[self.pin2node_gpu]
        pin_node_y = y_gpu[self.pin2node_gpu]

        safe_count = torch.clamp(pin_net_count - 1, min=1)
        pin_centroid_x = (pin_net_sum_x - pin_node_x) / safe_count
        pin_centroid_y = (pin_net_sum_y - pin_node_y) / safe_count

        # Single-pin nets: centroid = self position (no move)
        single_pin = (pin_net_count <= 1)
        pin_centroid_x[single_pin] = pin_node_x[single_pin]
        pin_centroid_y[single_pin] = pin_node_y[single_pin]

        # Step 3: For each node, average the centroids of its pins
        num_pins = len(pin_centroid_x)
        node_centroid_x = torch.zeros(bm.num_nodes, dtype=torch.float32, device=device)
        node_centroid_y = torch.zeros(bm.num_nodes, dtype=torch.float32, device=device)
        node_pin_count = torch.zeros(bm.num_nodes, dtype=torch.float32, device=device)

        node_centroid_x.scatter_add_(0, self.pin2node_gpu, pin_centroid_x)
        node_centroid_y.scatter_add_(0, self.pin2node_gpu, pin_centroid_y)
        node_pin_count.scatter_add_(0, self.pin2node_gpu,
                                     torch.ones(num_pins, dtype=torch.float32, device=device))

        safe_pin_count = torch.clamp(node_pin_count, min=1)
        node_centroid_x /= safe_pin_count
        node_centroid_y /= safe_pin_count

        return node_centroid_x.cpu().numpy(), node_centroid_y.cpu().numpy()

    def compute_net_hpwl_contributions(self, pos_x, pos_y):
        """Compute per-net weighted HPWL contribution using GPU.

        Returns numpy array of shape [num_nets] with weighted HPWL per net.
        Single-pin nets have 0 contribution.
        """
        if isinstance(pos_x, np.ndarray):
            x_gpu = torch.tensor(pos_x, dtype=torch.float32, device=self.device)
            y_gpu = torch.tensor(pos_y, dtype=torch.float32, device=self.device)
        else:
            x_gpu = pos_x.to(self.device)
            y_gpu = pos_y.to(self.device)

        pin_xs = x_gpu[self.net2node_flat_gpu]
        pin_ys = y_gpu[self.net2node_flat_gpu]

        x_max = torch.full((self.num_nets,), -1e18, device=self.device)
        x_min = torch.full((self.num_nets,), 1e18, device=self.device)
        y_max = torch.full((self.num_nets,), -1e18, device=self.device)
        y_min = torch.full((self.num_nets,), 1e18, device=self.device)

        x_max.scatter_reduce_(0, self.net_ids_exp, pin_xs, reduce='amax', include_self=True)
        x_min.scatter_reduce_(0, self.net_ids_exp, pin_xs, reduce='amin', include_self=True)
        y_max.scatter_reduce_(0, self.net_ids_exp, pin_ys, reduce='amax', include_self=True)
        y_min.scatter_reduce_(0, self.net_ids_exp, pin_ys, reduce='amin', include_self=True)

        net_hpwl = DPFPGA_X_WEIGHT * (x_max - x_min) + DPFPGA_Y_WEIGHT * (y_max - y_min)
        net_hpwl[~self.multi_pin_mask] = 0.0

        return net_hpwl.cpu().numpy()


class WindowDP:
    """Window-based detailed placement with GPU-accelerated HPWL tracking.

    V3 improvements over V2:
    - Uses GPUHPWLTracker for fast weighted HPWL evaluation
    - Net-weighted instance selection (focus on high-HPWL nets)
    - SA enabled for medium circuits (not just small)
    - GPU centroid-guided moves (directed towards ideal positions)
    - Better temperature schedule with higher initial temperature
    """

    def __init__(self, benchmark, device='cuda', seed=42, greedy_only=False,
                 gpu_tracker=None, sa_enabled=True, focus_ratio=0.5,
                 use_centroid_guidance=True):
        self.bm = benchmark; self.device = device; self.rng = np.random.RandomState(seed)
        self.greedy_only = greedy_only
        self.gpu_tracker = gpu_tracker
        self.sa_enabled = sa_enabled and not greedy_only
        self.focus_ratio = focus_ratio  # fraction of iterations using net-weighted selection
        self.use_centroid_guidance = use_centroid_guidance and (gpu_tracker is not None)

    def _compute_inst_probs(self, dp_x, dp_y, movable_indices, node2net_start, node2net_flat):
        """Compute instance selection probabilities based on net HPWL contributions."""
        if self.gpu_tracker is None:
            return None
        net_hpwl = self.gpu_tracker.compute_net_hpwl_contributions(dp_x, dp_y)
        # Vectorized: sum net HPWL contributions for each node
        pin_net_hpwl = net_hpwl[node2net_flat]  # [total_pins]
        node_hpwl = np.add.reduceat(pin_net_hpwl, node2net_start[:-1])  # [num_nodes]
        inst_priority = node_hpwl[movable_indices]
        # Add small epsilon to avoid zero probabilities
        inst_priority = np.maximum(inst_priority, 1e-6)
        return inst_priority / inst_priority.sum()

    def run(self, pos_x, pos_y, num_iterations=100000000, time_budget=60):
        bm = self.bm; start_time = time.time()
        dp_x = pos_x.cpu().numpy().copy(); dp_y = pos_y.cpu().numpy().copy()
        node2net_start = bm.node2net_start; node2net_flat = bm.node2net_flat
        net2node_start = bm.net2node_start; net2node_flat = bm.net2node_flat
        movable_mask = (~bm.is_fixed)
        if isinstance(movable_mask, torch.Tensor): movable_mask = movable_mask.cpu().numpy()
        movable_indices = np.where(movable_mask)[0]; n_movable = len(movable_indices)
        if n_movable == 0: return pos_x.clone(), pos_y.clone()
        node_site_group = bm.node_site_group; site_grid = bm.site_grid
        pos_to_insts = defaultdict(list)
        for inst in movable_indices:
            ix = int(round(dp_x[inst])); iy = int(round(dp_y[inst]))
            pos_to_insts[(ix, iy)].append(inst)
        inst_pos = {}
        for inst in movable_indices:
            ix = int(round(dp_x[inst])); iy = int(round(dp_y[inst]))
            inst_pos[inst] = (ix, iy)
        adj_offsets_1 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        adj_offsets_2 = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        adj_offsets_3 = [(-3, 0), (3, 0), (0, -3), (0, 3), (-2, -1), (-2, 1), (2, -1), (2, 1),
                         (-1, -2), (-1, 2), (1, -2), (1, 2)]
        accepted = 0; swap_accepted = 0; move_accepted = 0
        # Temperature schedule: higher T0 for SA-enabled, with adaptive cooling
        if self.sa_enabled:
            temperature = 5.0 if n_movable >= 50000 else 2.0
            # Cooling rate: ensure T drops to ~0.001 by end of budget
            est_iters = time_budget * 800  # rough estimate of iterations per second
            cool_rate = max(0.999990, (0.001 / temperature) ** (1.0 / max(est_iters, 1)))
        else:
            temperature = 1.0
            cool_rate = 0.999995
        best_hpwl = float('inf')
        best_x = dp_x.copy(); best_y = dp_y.copy()
        size_x = int(bm.size_x); size_y = int(bm.size_y)

        # GPU HPWL tracking interval
        hpwl_check_interval = 50000 if n_movable > 100000 else 20000
        # Net-weighted instance selection probabilities (updated periodically)
        inst_probs = None
        probs_update_interval = hpwl_check_interval * 2  # update every 2 HPWL checks
        # Centroid guidance (updated with HPWL checks)
        centroids_x = None; centroids_y = None

        for iteration in range(num_iterations):
            if time.time() - start_time >= time_budget: break

            # Instance selection: net-weighted or random
            if inst_probs is not None and self.rng.rand() < self.focus_ratio:
                idx = self.rng.choice(n_movable, p=inst_probs)
            else:
                idx = self.rng.randint(0, n_movable)

            inst1 = movable_indices[idx]; sg1 = node_site_group[inst1]
            ix1, iy1 = inst_pos[inst1]
            neighbor_found = False; neighbors = []
            # Search radius 1
            for dx, dy in adj_offsets_1:
                nx, ny = ix1 + dx, iy1 + dy
                if 0 <= nx < size_x and 0 <= ny < size_y:
                    for inst2 in pos_to_insts.get((nx, ny), []):
                        if inst2 != inst1 and node_site_group[inst2] == sg1: neighbors.append(inst2)
            # Search radius 2
            if not neighbors:
                for dx, dy in adj_offsets_2:
                    nx, ny = ix1 + dx, iy1 + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y:
                        for inst2 in pos_to_insts.get((nx, ny), []):
                            if inst2 != inst1 and node_site_group[inst2] == sg1: neighbors.append(inst2)
            # Search radius 3 (for small circuits or SA with high temperature)
            if not neighbors and (n_movable < 50000 or temperature > 1.0):
                for dx, dy in adj_offsets_3:
                    nx, ny = ix1 + dx, iy1 + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y:
                        for inst2 in pos_to_insts.get((nx, ny), []):
                            if inst2 != inst1 and node_site_group[inst2] == sg1: neighbors.append(inst2)
            if neighbors:
                inst2 = neighbors[self.rng.randint(0, len(neighbors))]
                ix2, iy2 = inst_pos[inst2]
                x1_old = dp_x[inst1]; y1_old = dp_y[inst1]; x2_old = dp_x[inst2]; y2_old = dp_y[inst2]
                delta = _delta_hpwl_swap(dp_x, dp_y, inst1, inst2, x1_old, y1_old, x2_old, y2_old,
                    node2net_start, node2net_flat, net2node_start, net2node_flat)
                accept_swap = delta < 0 or (self.sa_enabled and self.rng.rand() < np.exp(-delta / (temperature + 1e-10)))
                if accept_swap:
                    dp_x[inst1] = x2_old; dp_y[inst1] = y2_old; dp_x[inst2] = x1_old; dp_y[inst2] = y1_old
                    old_pos1 = inst_pos[inst1]; old_pos2 = inst_pos[inst2]
                    pos_to_insts[old_pos1].remove(inst1); pos_to_insts[old_pos2].remove(inst2)
                    pos_to_insts[old_pos2].append(inst1); pos_to_insts[old_pos1].append(inst2)
                    inst_pos[inst1] = old_pos2; inst_pos[inst2] = old_pos1
                    accepted += 1; swap_accepted += 1; neighbor_found = True
            if not neighbor_found or self.rng.rand() < 0.3:
                # Move: prefer centroid-directed moves when available
                if centroids_x is not None and self.use_centroid_guidance:
                    cx = centroids_x[inst1]; cy = centroids_y[inst1]
                    dx_c = cx - dp_x[inst1]; dy_c = cy - dp_y[inst1]
                    # Build directed move offsets: prefer centroid direction first
                    preferred = []
                    if abs(dx_c) >= abs(dy_c):
                        if abs(dx_c) > 0.5: preferred.append((int(np.sign(dx_c)), 0))
                        if abs(dy_c) > 0.5: preferred.append((0, int(np.sign(dy_c))))
                    else:
                        if abs(dy_c) > 0.5: preferred.append((0, int(np.sign(dy_c))))
                        if abs(dx_c) > 0.5: preferred.append((int(np.sign(dx_c)), 0))
                    if abs(dx_c) > 0.5 and abs(dy_c) > 0.5:
                        preferred.append((int(np.sign(dx_c)), int(np.sign(dy_c))))
                    # Add remaining radius-1 offsets as fallback
                    other = [o for o in adj_offsets_1 if o not in preferred]
                    move_offsets = preferred + other
                elif temperature > 1.0 and n_movable < 50000:
                    move_offsets = adj_offsets_1 + adj_offsets_2 + adj_offsets_3
                    self.rng.shuffle(move_offsets)
                elif self.rng.rand() < 0.7:
                    move_offsets = list(adj_offsets_1)
                else:
                    move_offsets = list(adj_offsets_2)
                    self.rng.shuffle(move_offsets)
                for dx, dy in move_offsets:
                    nx, ny = ix1 + dx, iy1 + dy
                    if 0 <= nx < size_x and 0 <= ny < size_y:
                        if site_grid[nx, ny] == sg1:
                            cap = bm.site_cap_grid[nx, ny]
                            cur_count = len(pos_to_insts.get((nx, ny), []))
                            if cur_count < cap:
                                new_x = float(nx); new_y = float(ny)
                                delta_m = _delta_hpwl_move(dp_x, dp_y, inst1, new_x, new_y,
                                    node2net_start, node2net_flat, net2node_start, net2node_flat)
                                accept_move = delta_m < 0 or (self.sa_enabled and self.rng.rand() < np.exp(-delta_m / (temperature + 1e-10)))
                                if accept_move:
                                    old_pos = inst_pos[inst1]; pos_to_insts[old_pos].remove(inst1)
                                    dp_x[inst1] = new_x; dp_y[inst1] = new_y
                                    pos_to_insts[(nx, ny)].append(inst1); inst_pos[inst1] = (nx, ny)
                                    accepted += 1; move_accepted += 1
                                break
            temperature = max(0.0001, temperature * cool_rate)
            if (iteration + 1) % hpwl_check_interval == 0:
                elapsed = time.time() - start_time
                if self.gpu_tracker is not None:
                    hpwl = self.gpu_tracker.compute_weighted_hpwl(dp_x, dp_y)
                else:
                    pin_x = dp_x[bm.net2node_flat]; pin_y = dp_y[bm.net2node_flat]
                    net_sizes = bm.net2node_start[1:] - bm.net2node_start[:-1]
                    net_ids = np.arange(bm.num_nets); net_ids_exp = np.repeat(net_ids, net_sizes)
                    x_max = np.full(bm.num_nets, -1e18); x_min = np.full(bm.num_nets, 1e18)
                    y_max = np.full(bm.num_nets, -1e18); y_min = np.full(bm.num_nets, 1e18)
                    np.maximum.at(x_max, net_ids_exp, pin_x); np.minimum.at(x_min, net_ids_exp, pin_x)
                    np.maximum.at(y_max, net_ids_exp, pin_y); np.minimum.at(y_min, net_ids_exp, pin_y)
                    multi_pin = net_sizes > 1
                    hpwl = np.sum(DPFPGA_X_WEIGHT * (x_max[multi_pin] - x_min[multi_pin]) +
                                  DPFPGA_Y_WEIGHT * (y_max[multi_pin] - y_min[multi_pin]))
                if hpwl < best_hpwl: best_hpwl = hpwl; best_x = dp_x.copy(); best_y = dp_y.copy()
                sa_str = f", SA-T={temperature:.4f}" if self.sa_enabled else ""
                print(f"  [DP] accepted={accepted}/{iteration+1} (swap={swap_accepted},move={move_accepted}), "
                      f"HPWL={hpwl:.0f}, best={best_hpwl:.0f}{sa_str}, time={elapsed:.1f}s")
                # Update net-weighted instance selection probabilities
                if (iteration + 1) % probs_update_interval == 0:
                    inst_probs = self._compute_inst_probs(dp_x, dp_y, movable_indices,
                                                           node2net_start, node2net_flat)
                # Update centroid guidance
                if self.use_centroid_guidance and (iteration + 1) % probs_update_interval == 0:
                    centroids_x, centroids_y = self.gpu_tracker.compute_centroids(dp_x, dp_y)
        if best_hpwl < float('inf'): dp_x = best_x; dp_y = best_y
        elapsed = time.time() - start_time
        print(f"  [DP] Done: accepted={accepted}/{iteration+1} (swap={swap_accepted},move={move_accepted}), "
              f"best_HPWL={best_hpwl:.0f}, time={elapsed:.1f}s")
        return torch.from_numpy(dp_x).float().to(self.device), torch.from_numpy(dp_y).float().to(self.device)


def load_dpfpga_placement(bm, pl_file):
    """Load DREAMPlaceFPGA's .final.pl or .gp.pl placement file."""
    pos_x = torch.zeros(bm.num_nodes, dtype=torch.float32)
    pos_y = torch.zeros(bm.num_nodes, dtype=torch.float32)

    if hasattr(bm, 'pos_x') and bm.pos_x is not None:
        pos_x.copy_(bm.pos_x.cpu())
        pos_y.copy_(bm.pos_y.cpu())

    name_to_id = bm.node_name_to_id if hasattr(bm, 'node_name_to_id') else {}
    loaded = 0
    skipped = 0

    with open(pl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('UCLA'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue

            if name in name_to_id:
                nid = name_to_id[name]
                pos_x[nid] = x
                pos_y[nid] = y
                loaded += 1
            else:
                skipped += 1

    print(f"  [Load-PL] Loaded {loaded} nodes, skipped {skipped} from {pl_file}")
    return pos_x, pos_y


def run_dpfpga_gp_lg(benchmark_name, benchmark_dir):
    """Run DREAMPlaceFPGA's GP+LG flow and return the output .final.pl file path."""
    dp_root = r'D:\Codes\VLSI\DeepSeek2\DREAMPlaceFPGA-local'

    # Check for existing results first (avoid re-running)
    result_dir = os.path.join(dp_root, 'results', benchmark_name)
    final_pl = os.path.join(result_dir, f'{benchmark_name}.final.pl')
    if os.path.exists(final_pl):
        print(f"  [DPFPGA] Using existing result: {final_pl}")
        return final_pl

    final_pl_root = os.path.join(dp_root, 'results', f'{benchmark_name}.final.pl')
    if os.path.exists(final_pl_root):
        print(f"  [DPFPGA] Using existing result: {final_pl_root}")
        return final_pl_root

    for root, dirs, files in os.walk(os.path.join(dp_root, 'results')):
        for f in files:
            if f == f'{benchmark_name}.final.pl':
                path = os.path.join(root, f)
                print(f"  [DPFPGA] Using existing result: {path}")
                return path

    # No existing result found, run DREAMPlaceFPGA
    test_dir = os.path.join(dp_root, 'test')
    config_file = os.path.join(test_dir, f'{benchmark_name}.json')

    if not os.path.exists(config_file):
        print(f"  [DPFPGA] Config not found: {config_file}")
        return None

    import json
    with open(config_file, 'r') as f:
        config = json.load(f)

    if not torch.cuda.is_available():
        config['gpu'] = 0
        config['routability_opt_flag'] = 0

    temp_config = os.path.join(dp_root, f'temp_{benchmark_name}.json')
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  [DPFPGA] Running DREAMPlaceFPGA GP+LG for {benchmark_name}...")
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(dp_root, 'dreamplacefpga', 'Placer.py'), temp_config],
        cwd=dp_root,
        capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  [DPFPGA] DREAMPlaceFPGA failed with return code {result.returncode}")
        print(f"  [DPFPGA] stderr: {result.stderr[:2000]}")
        return None

    # Check for output again after running
    if os.path.exists(final_pl):
        print(f"  [DPFPGA] Output: {final_pl}")
        return final_pl
    if os.path.exists(final_pl_root):
        print(f"  [DPFPGA] Output: {final_pl_root}")
        return final_pl_root
    for root, dirs, files in os.walk(os.path.join(dp_root, 'results')):
        for f in files:
            if f == f'{benchmark_name}.final.pl':
                path = os.path.join(root, f)
                print(f"  [DPFPGA] Found output: {path}")
                return path

    print(f"  [DPFPGA] No .final.pl output found")
    return None


def _run_gp_lg(bm, device, is_small, num_movable, bench_start, time_limit):
    """Run PyPlacer GP + LG stages. Returns (legal_x_cpu, legal_y_cpu, gp_hpwl, lg_hpwl)."""
    gp1_iters = 2000; gp1_lr = 0.01
    gp1_dw = 0.01

    if is_small:
        print(f"\n[Stage 1a] GP Phase 1 - Spread (Nesterov, {gp1_iters} iters, dw={gp1_dw})")
        gp = EPlaceGlobalPlacer(bm, device=device)
        gp_pos_x, gp_pos_y = gp.run(
            num_iterations=gp1_iters, lr=gp1_lr, density_weight=gp1_dw,
            gamma_start=1.0, gamma_end=4.0, verbose=True,
            wirelength_model='wawl', optimizer='nesterov')
    else:
        gp1_dw_start = 0.5; gp1_dw_end = 0.01
        print(f"\n[Stage 1a] GP Phase 1 - Spread (Nesterov, {gp1_iters} iters, "
              f"dw={gp1_dw_start}->{gp1_dw_end}, fixed schedule)")
        gp = EPlaceGlobalPlacer(bm, device=device)
        gp_pos_x, gp_pos_y = gp.run(
            num_iterations=gp1_iters, lr=gp1_lr, density_weight=gp1_dw_end,
            gamma_start=1.0, gamma_end=4.0, verbose=True,
            wirelength_model='wawl', optimizer='nesterov',
            density_weight_start=gp1_dw_start, fixed_dw_schedule=True)
    gp1_hpwl = bm.compute_hpwl_fast(gp_pos_x, gp_pos_y)
    print(f"  GP1 HPWL: {gp1_hpwl:.0f} ({time.time()-bench_start:.1f}s)")

    if time_limit and (time.time() - bench_start) > time_limit: return None, None, 0, 0

    if is_small:
        movable_mask = ~bm.is_fixed
        movable_x = gp_pos_x[movable_mask]; movable_y = gp_pos_y[movable_mask]
        centroid_x = movable_x.mean(); centroid_y = movable_y.mean()
        std_x = movable_x.std() + 1e-10; std_y = movable_y.std() + 1e-10
        target_std_x = bm.size_x * 0.20; target_std_y = bm.size_y * 0.20
        scale_x = (target_std_x / std_x).clamp(1.0, 2.0); scale_y = (target_std_y / std_y).clamp(1.0, 2.0)
        scale = min(scale_x, scale_y)
        if scale > 1.1:
            print(f"\n[Spreading] std=({std_x:.1f},{std_y:.1f}), scale={scale:.2f}")
            with torch.no_grad():
                gp_pos_x[movable_mask] = (movable_x - centroid_x) * scale + centroid_x
                gp_pos_y[movable_mask] = (movable_y - centroid_y) * scale + centroid_y
                gp_pos_x.clamp_(0, bm.size_x - 1); gp_pos_y.clamp_(0, bm.size_y - 1)
            gp1_hpwl = bm.compute_hpwl_fast(gp_pos_x, gp_pos_y)
            print(f"  After spreading HPWL: {gp1_hpwl:.0f}")

    if time_limit and (time.time() - bench_start) > time_limit: return None, None, 0, 0

    if not is_small:
        gp2_iters = 3000; gp2_lr = 0.01
        gp2_bins = max(32, min(128, int(np.sqrt(num_movable / 50))))
        gp2_dw_start = 0.1; gp2_dw_end = 0.005
        print(f"\n[Stage 1b] GP Phase 2 - Refine (Momentum, {gp2_iters} iters, "
              f"bins={gp2_bins}, dw={gp2_dw_start}->{gp2_dw_end})")
        gp2 = EPlaceGlobalPlacer(bm, device=device, num_bins=gp2_bins)
        gp2.pos_x = gp_pos_x.clone(); gp2.pos_y = gp_pos_y.clone()
        gp2._precompute_net_data_vectorized()
        gp_pos_x, gp_pos_y = gp2.run(
            num_iterations=gp2_iters, lr=gp2_lr, density_weight=gp2_dw_end,
            gamma_start=1.0, gamma_end=4.0, verbose=True,
            wirelength_model='wawl', optimizer='momentum',
            use_best=True, density_weight_start=gp2_dw_start,
            fixed_dw_schedule=True)
        del gp2
    else:
        print(f"\n[Stage 1b] Skipping GP2 for small circuit")

    gp_hpwl = bm.compute_hpwl_fast(gp_pos_x, gp_pos_y)
    if not is_small:
        movable_mask = ~bm.is_fixed
        if isinstance(movable_mask, torch.Tensor): movable_mask_np = movable_mask.cpu().numpy()
        else: movable_mask_np = movable_mask
        movable_x = gp_pos_x[movable_mask]; movable_y = gp_pos_y[movable_mask]
        x_range = movable_x.max() - movable_x.min()
        y_range = movable_y.max() - movable_y.min()
        x_coverage = x_range / bm.size_x
        y_coverage = y_range / bm.size_y
        print(f"  [GP-End] Spread: x_range={x_range:.1f}/{bm.size_x}, y_range={y_range:.1f}/{bm.size_y}, "
              f"coverage=({x_coverage:.2f}, {y_coverage:.2f})")
        if x_coverage < 0.5 or y_coverage < 0.5:
            print(f"  [Spreading] Low coverage after GP2, spreading instances...")
            gp_pos_x, gp_pos_y = spread_positions(gp_pos_x, gp_pos_y, bm, target_coverage=0.85)
            gp_hpwl = bm.compute_hpwl_fast(gp_pos_x, gp_pos_y)
            print(f"  After spreading HPWL: {gp_hpwl:.0f}")

    print(f"  Final GP HPWL: {gp_hpwl:.0f} ({time.time()-bench_start:.1f}s)")

    if time_limit and (time.time() - bench_start) > time_limit: return None, None, 0, 0

    gp_pos_x_cpu = gp_pos_x.cpu()
    gp_pos_y_cpu = gp_pos_y.cpu()
    del gp
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    # ===== Stage 2: Legalization =====
    lg_start = time.time()
    use_cpp_lg = HAS_DP_LG and num_movable < 50000

    if use_cpp_lg:
        print(f"\n[Stage 2] C++ DL LG (SLICE) + BFS DSP/RAM Legalization")
        try:
            slice_legal_x, slice_legal_y = run_dreamplacefpga_lg(
                bm, gp_pos_x_cpu, gp_pos_y_cpu, bm.benchmark_dir, device='cpu')
            if slice_legal_x is not None:
                legal_x = slice_legal_x.clone()
                legal_y = slice_legal_y.clone()
                dsp_lg = DSPRAMLegalizer(bm, device=device)
                legal_x, legal_y = dsp_lg.legalize(legal_x, legal_y)
                print(f"  [LG] C++ DL LG succeeded")
            else:
                print(f"  [LG] C++ DL LG failed, falling back to BFS")
                lg = HybridLegalizer(bm, device=device)
                legal_x, legal_y = lg.legalize(gp_pos_x_cpu, gp_pos_y_cpu, wl_aware=False)
        except Exception as e:
            print(f"  [LG] C++ DL LG exception: {e}")
            print(f"  [LG] Falling back to BFS")
            lg = HybridLegalizer(bm, device=device)
            legal_x, legal_y = lg.legalize(gp_pos_x_cpu, gp_pos_y_cpu, wl_aware=False)
    else:
        print(f"\n[Stage 2] BFS SLICE + BFS DSP/RAM Legalization")
        lg = HybridLegalizer(bm, device=device)
        legal_x, legal_y = lg.legalize(gp_pos_x_cpu, gp_pos_y_cpu, wl_aware=False)

    legal_x_dev = legal_x.to(device)
    legal_y_dev = legal_y.to(device)
    lg_hpwl = bm.compute_hpwl_fast(legal_x_dev, legal_y_dev)
    del legal_x_dev, legal_y_dev
    lg_loss = lg_hpwl / gp_hpwl if gp_hpwl > 0 else 0
    print(f"  [LG] After LG: HPWL={lg_hpwl:.0f} (loss: {lg_loss:.2f}x, {time.time()-lg_start:.1f}s)")

    if time_limit and (time.time() - bench_start) > time_limit: return None, None, 0, 0

    legal_x_cpu = legal_x.cpu()
    legal_y_cpu = legal_y.cpu()
    return legal_x_cpu, legal_y_cpu, gp_hpwl, lg_hpwl


def _do_swap_refinement(bm, pos_x_np, pos_y_np, is_small, is_medium, label="Swap"):
    """Run swap refinement for all site groups. Returns (pos_x_np, pos_y_np)."""
    print(f"\n[{label}] Swap Refinement")
    swap_start = time.time()
    for sg in range(4):
        inst_mask = (bm.node_site_group == sg) & (~bm.is_fixed)
        inst_indices = np.where(inst_mask)[0]
        num_inst = len(inst_indices)
        if num_inst < 2: continue
        sg_name = ['SLICE', 'DSP', 'BRAM', 'IO'][sg]
        if sg == 0:  # SLICE
            if is_small:
                radius, max_passes = 15, 10
            else:
                # Medium and large circuits: SLICE swap too slow, rely on WindowDP instead
                print(f"  [{label}-{sg_name}] {num_inst} instances, SKIPPED (use WindowDP)")
                continue
        else:  # DSP/BRAM/IO
            radius = 5 if is_small else 3
            max_passes = 3 if is_small else 2
        print(f"  [{label}-{sg_name}] {num_inst} instances, radius={radius}, max_passes={max_passes}")
        improvement, num_swaps = _greedy_swap_refinement(
            pos_x_np, pos_y_np, inst_indices.astype(np.int64),
            int(bm.size_y), int(bm.size_x),
            bm.node2net_start, bm.node2net_flat,
            bm.net2node_start, bm.net2node_flat,
            max_passes, radius)
        print(f"  [{label}-{sg_name}] improvement={improvement:.0f}, swaps={num_swaps}, "
              f"{time.time()-swap_start:.1f}s)")
    return pos_x_np, pos_y_np


def run_placement(benchmark_dir, output_dir, device='cuda', time_limit=None,
                  use_dpfpga=False, load_pl_file=None):
    bm = ISPD2016Benchmark(benchmark_dir, device=device)
    num_movable = bm.num_movable
    is_small = num_movable < 50000
    is_medium = num_movable < 500000

    print(f"\n{'='*60}")
    mode = "DPFPGA GP+LG + Swap+DP" if (use_dpfpga or load_pl_file) else "PyPlacer GP + LG + Swap+DP"
    print(f"  PyPlacer V39 (GPU-HPWL): {mode}")
    print(f"  {bm.name}: {num_movable} movable, {bm.num_nets} nets, FPGA {bm.size_x}x{bm.size_y}")
    print(f"  DREAMPlaceFPGA LG available: {HAS_DP_LG}")
    print(f"  MCF available: {HAS_MCF}")
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

    # ===== Stage 3: GPU Directed Move Phase (coarse optimization) =====
    pos_x_np = legal_x_cpu.numpy().copy().astype(np.float64)
    pos_y_np = legal_y_cpu.numpy().copy().astype(np.float64)

    if gpu_tracker is not None and not is_small:
        print(f"\n[Stage 3] GPU Directed Move Phase (centroid-guided)")
        dm_start = time.time()
        dm_iterations = 5 if is_medium else 3
        dm_hpwl = gpu_tracker.compute_weighted_hpwl(pos_x_np, pos_y_np)
        print(f"  [DirectedMove] Starting HPWL: {dm_hpwl:.0f}")
        for dm_iter in range(dm_iterations):
            centroids_x, centroids_y = gpu_tracker.compute_centroids(pos_x_np, pos_y_np)

            movable_mask = (~bm.is_fixed)
            if isinstance(movable_mask, torch.Tensor): movable_mask = movable_mask.cpu().numpy()
            movable_indices = np.where(movable_mask)[0]

            # Compute distance to centroid for priority (farthest first)
            dx = centroids_x[movable_indices] - pos_x_np[movable_indices]
            dy = centroids_y[movable_indices] - pos_y_np[movable_indices]
            dist = np.abs(dx) + np.abs(dy)
            order = np.argsort(-dist)
            # Only process top 5000 most "unhappy" instances
            top_k = min(5000, len(order))
            order = order[:top_k]

            # Save state for revert
            saved_x = pos_x_np.copy(); saved_y = pos_y_np.copy()

            # Position occupancy tracking
            pos_to_count = defaultdict(int)
            for inst in movable_indices:
                ix = int(round(pos_x_np[inst])); iy = int(round(pos_y_np[inst]))
                pos_to_count[(ix, iy)] += 1

            num_moved = 0
            for idx in order:
                inst_id = movable_indices[idx]
                sg = bm.node_site_group[inst_id]
                ix = int(round(pos_x_np[inst_id])); iy = int(round(pos_y_np[inst_id]))
                cx = centroids_x[inst_id]; cy = centroids_y[inst_id]
                dx_c = cx - pos_x_np[inst_id]; dy_c = cy - pos_y_np[inst_id]

                # Build candidate moves towards centroid
                candidates = []
                if abs(dx_c) >= abs(dy_c):
                    if abs(dx_c) > 0.5: candidates.append((ix + int(np.sign(dx_c)), iy))
                    if abs(dy_c) > 0.5: candidates.append((ix, iy + int(np.sign(dy_c))))
                else:
                    if abs(dy_c) > 0.5: candidates.append((ix, iy + int(np.sign(dy_c))))
                    if abs(dx_c) > 0.5: candidates.append((ix + int(np.sign(dx_c)), iy))

                for nx, ny in candidates:
                    if 0 <= nx < bm.size_x and 0 <= ny < bm.size_y:
                        if bm.site_grid[nx, ny] == sg:
                            cap = bm.site_cap_grid[nx, ny]
                            cur_count = pos_to_count.get((nx, ny), 0)
                            if cur_count < cap:
                                new_x = float(nx); new_y = float(ny)
                                pos_to_count[(ix, iy)] -= 1
                                pos_to_count[(nx, ny)] += 1
                                pos_x_np[inst_id] = new_x; pos_y_np[inst_id] = new_y
                                num_moved += 1
                                break

            # Check HPWL after batch - revert if worse
            new_hpwl = gpu_tracker.compute_weighted_hpwl(pos_x_np, pos_y_np)
            if new_hpwl < dm_hpwl:
                dm_hpwl = new_hpwl
                print(f"  [DirectedMove] Iter {dm_iter}: moved={num_moved}/{top_k}, "
                      f"HPWL={new_hpwl:.0f} (IMPROVED), time={time.time()-dm_start:.1f}s")
            else:
                # Revert
                pos_x_np[:] = saved_x; pos_y_np[:] = saved_y
                print(f"  [DirectedMove] Iter {dm_iter}: moved={num_moved}/{top_k}, "
                      f"HPWL={new_hpwl:.0f} (REVERTED), time={time.time()-dm_start:.1f}s")
                break  # Stop if no improvement
        print(f"  [DirectedMove] Done ({time.time()-dm_start:.1f}s)")

    # DSP/BRAM/IO swap refinement (SLICE swap skipped for medium/large)
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
    # V3: Larger time budgets + GPU HPWL tracking + centroid guidance
    # Medium/large circuits get more DP time since SLICE swap is skipped
    # Use greedy-only for all circuits when starting from DPFPGA LG (already near-optimal)
    dp1_time = 1200 if is_small else (5400 if is_medium else 5400)
    dp1_greedy = True  # Greedy-only: SA hurts when starting from good DPFPGA LG output
    dp1_sa = False

    print(f"\n[Stage 4] Detailed Placement Round 1 (WindowDP+GPU+Centroid, budget={dp1_time}s, "
          f"greedy)")
    dp = WindowDP(bm, device=device, greedy_only=dp1_greedy, gpu_tracker=gpu_tracker, sa_enabled=dp1_sa,
                  use_centroid_guidance=True)
    dp_pos_x, dp_pos_y = dp.run(refine_x, refine_y, num_iterations=100000000, time_budget=dp1_time)

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
    dp2_greedy = True
    dp2_sa = False
    print(f"\n[Stage 5b] Second round DP (WindowDP+GPU+Centroid, budget={dp2_time}s, "
          f"greedy)")
    dp2 = WindowDP(bm, device=device, greedy_only=dp2_greedy, gpu_tracker=gpu_tracker, sa_enabled=dp2_sa,
                   use_centroid_guidance=True)
    dp2_pos_x, dp2_pos_y = dp2.run(round2_x, round2_y, num_iterations=100000000, time_budget=dp2_time)

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
    # DP3: greedy-only for all sizes (fine-tuning, no SA needed)
    print(f"\n[Stage 6b] Third round DP (WindowDP+GPU+Centroid, budget={dp3_time}s, greedy)")
    dp3 = WindowDP(bm, device=device, greedy_only=True, gpu_tracker=gpu_tracker, sa_enabled=False,
                   use_centroid_guidance=True)
    final_pos_x, final_pos_y = dp3.run(round3_x, round3_y, num_iterations=100000000, time_budget=dp3_time)

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


class HybridLegalizer:
    """Fallback hybrid legalization: MCF for DSP/RAM/IO, BFS for SLICE."""

    def __init__(self, benchmark, device='cuda'):
        self.bm = benchmark
        self.device = device
        self.size_x = benchmark.size_x
        self.size_y = benchmark.size_y

    def _build_site_data(self, sg):
        bm = self.bm
        site_mask = (bm.site_grid == sg)
        site_pos = np.argwhere(site_mask)
        site_caps = bm.site_cap_grid[site_mask].astype(np.int64)
        return site_pos, site_caps

    def _compute_precond(self, inst_indices):
        bm = self.bm
        num_inst = len(inst_indices)
        precond = np.ones(num_inst, dtype=np.float64)
        for i, inst_id in enumerate(inst_indices):
            if inst_id < len(bm.node2net_start) - 1:
                n_nets = bm.node2net_start[inst_id + 1] - bm.node2net_start[inst_id]
                precond[i] = 1.0 + n_nets * 0.5
        return precond

    def _mcf_legalize(self, inst_x, inst_y, site_pos, site_caps, precond,
                       lg_max_dist_init=20.0, lg_max_dist_incr=10.0,
                       lg_flow_cost_scale=100.0):
        num_inst = len(inst_x)
        if num_inst == 0:
            return np.full(0, -1, dtype=np.int64)

        virtual_sites = []
        virtual_to_physical = []
        for s in range(len(site_pos)):
            for _ in range(int(site_caps[s])):
                virtual_sites.append(site_pos[s])
                virtual_to_physical.append(s)
        virtual_sites = np.array(virtual_sites, dtype=np.float64)
        virtual_to_physical = np.array(virtual_to_physical, dtype=np.int64)

        if len(virtual_sites) == 0:
            return np.full(num_inst, -1, dtype=np.int64)

        num_sites = len(virtual_sites)
        sites_flat = virtual_sites.flatten()

        locX = inst_x.astype(np.float64)
        locY = inst_y.astype(np.float64)
        precondArr = precond.astype(np.float64)
        movVal = [0.0, 0.0]
        outLoc = [0.0] * (2 * num_inst)

        try:
            _legalize_cpp.legalize(
                locX, locY, num_inst, num_sites,
                sites_flat, precondArr,
                lg_max_dist_init, lg_max_dist_incr, lg_flow_cost_scale,
                movVal, outLoc)
        except Exception as e:
            print(f"  [MCF] legalize_cpp failed: {e}")
            return np.full(num_inst, -1, dtype=np.int64)

        outLoc = np.array(outLoc)
        out_x = outLoc[:num_inst]
        out_y = outLoc[num_inst:]

        assignment = np.full(num_inst, -1, dtype=np.int64)
        site_used = np.zeros(len(site_pos), dtype=np.int64)

        for i in range(num_inst):
            if out_x[i] == 0.0 and out_y[i] == 0.0:
                continue
            best_dist = float('inf')
            best_site = -1
            for s in range(len(site_pos)):
                if site_used[s] >= site_caps[s]:
                    continue
                dist = abs(out_x[i] - site_pos[s, 0]) + abs(out_y[i] - site_pos[s, 1])
                if dist < best_dist:
                    best_dist = dist
                    best_site = s
            if best_site >= 0 and best_dist < 2.0:
                assignment[i] = best_site
                site_used[best_site] += 1

        return assignment

    def legalize(self, pos_x, pos_y, wl_aware=False):
        bm = self.bm
        start_time = time.time()
        legal_x = pos_x.clone()
        legal_y = pos_y.clone()

        for sg in range(4):
            sg_start = time.time()
            sg_name = ['SLICE', 'DSP', 'BRAM', 'IO'][sg]

            site_pos, site_caps = self._build_site_data(sg)
            if len(site_pos) == 0:
                continue

            inst_mask = (bm.node_site_group == sg) & (~bm.is_fixed)
            inst_indices = np.where(inst_mask)[0]
            num_inst = len(inst_indices)
            if num_inst == 0:
                continue

            inst_x = pos_x[inst_indices].cpu().numpy().astype(np.float64)
            inst_y = pos_y[inst_indices].cpu().numpy().astype(np.float64)

            if sg == 0:
                cap = np.zeros((int(bm.size_y), int(bm.size_x)), dtype=np.int64)
                for x in range(int(bm.size_x)):
                    for y in range(int(bm.size_y)):
                        if bm.site_grid[x, y] == sg:
                            cap[y, x] = int(bm.site_cap_grid[x, y])
                assignment = _legalize_bfs(
                    inst_x.astype(np.float32), inst_y.astype(np.float32),
                    cap, sg, int(bm.size_y), int(bm.size_x))
                assignment_format = 'linear_grid'
            else:
                if HAS_MCF and num_inst < 50000:
                    precond = self._compute_precond(inst_indices)
                    assignment = self._mcf_legalize(
                        inst_x, inst_y, site_pos, site_caps, precond,
                        lg_max_dist_init=20.0, lg_max_dist_incr=10.0,
                        lg_flow_cost_scale=100.0)
                    mcf_assigned = int((assignment >= 0).sum())
                    print(f"  [LG-{sg_name}] MCF assigned {mcf_assigned}/{num_inst}")
                    assignment_format = 'site_pos'
                else:
                    cap = np.zeros((int(bm.size_y), int(bm.size_x)), dtype=np.int64)
                    for x in range(int(bm.size_x)):
                        for y in range(int(bm.size_y)):
                            if bm.site_grid[x, y] == sg:
                                cap[y, x] = int(bm.site_cap_grid[x, y])
                    assignment = _legalize_bfs(
                        inst_x.astype(np.float32), inst_y.astype(np.float32),
                        cap, sg, int(bm.size_y), int(bm.size_x))
                    assignment_format = 'linear_grid'

            assigned_count = 0
            size_x_int = int(bm.size_x)
            if sg == 0 or assignment_format == 'linear_grid':
                for i in range(num_inst):
                    if assignment[i] >= 0:
                        inst_id = inst_indices[i]
                        linear_idx = assignment[i]
                        ax = linear_idx % size_x_int
                        ay = linear_idx // size_x_int
                        legal_x[inst_id] = float(ax)
                        legal_y[inst_id] = float(ay)
                        assigned_count += 1
            else:
                for i in range(num_inst):
                    if assignment[i] >= 0:
                        inst_id = inst_indices[i]
                        s_idx = assignment[i]
                        legal_x[inst_id] = float(site_pos[s_idx, 0])
                        legal_y[inst_id] = float(site_pos[s_idx, 1])
                        assigned_count += 1

            sg_time = time.time() - sg_start
            print(f"  [LG-{sg_name}] assigned {assigned_count}/{num_inst} instances "
                  f"({sg_time:.1f}s)")

        for fid in bm.fixed_indices:
            legal_x[fid] = float(bm.fixed_pos[fid, 0])
            legal_y[fid] = float(bm.fixed_pos[fid, 1])

        return legal_x, legal_y


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
