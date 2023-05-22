from tb_plugin.torch_tb_profiler.profiler.module_op import ModuleStats, Stats
from tb_plugin.torch_tb_profiler.profiler.data import (DistributedRunProfileData,
                                             RunProfileData)
from tb_plugin.torch_tb_profiler.profiler.node import (DurationEvent, EventTypes,
                                                KernelEvent, ModuleEvent,
                                                OperatorEvent,
                                                PLProfileEvent)
from tb_plugin.torch_tb_profiler.profiler.module_op import (
    _build_module_hierarchy, aggegate_module_view)

from tb_plugin.torch_tb_profiler.profiler.node import OperatorNode
from tb_plugin.torch_tb_profiler.profiler.op_agg import ModuleAggregator, OperatorAgg
# from tb_plugin.torch_tb_profiler.profiler.module_op import OperatorAgg

from typing import List, Dict
import json
from prettytable import PrettyTable
from tb_plugin.torch_tb_profiler.profiler.event_parser import CommLibTypes, EventParser, ProfileRole, traverse_tid2tree
from tb_plugin.torch_tb_profiler.profiler import trace


SCHEMA_VERSION = 1
WORKER_NAME = 'worker0'

class PrintLogger:
    """
    PrintLogger is a pseudo logger that adapts to the general logger interface.
    """

    def info(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)

    def error(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)

    def warning(self, *args, **kwargs):
        print(*args, **kwargs, flush=True)

logger = PrintLogger()

def print_tid2tree(node: OperatorNode, tid2tree: Dict[int, OperatorNode], level=0):
    print("   "*level + f"v:{node}")
    for cnode in node.children:
        print_tid2tree(cnode, tid2tree, level+1)


def dump_stats_new(level: int, stats: List[Stats], module_tree=None):
    """testing purpose"""
    for stat in stats:
        print(
            f"{'    ' * level}{stat.name.replace('nn.Module: ', '')}_{stat.id}: {stat.host_duration} , {stat.device_duration}"
        )
        name = '    ' * level + stat.name.replace('nn.Module: ', '') + str(stat.id)
        if module_tree:
            module_tree.add_row([name, stat.host_duration, stat.device_duration, stat.self_host_duration, stat.self_device_duration])
        dump_stats_new(level + 1, stat.children, module_tree)

    
if __name__ == "__main__":
    # mini_traces 下面有几个小trace文件的例子
    with open(
            "test_traces/mini_traces/mini3.json",
            'r') as load_f:
        content = json.load(load_f)
    print(content.keys())

    # profile = RunProfileData.from_json(WORKER_NAME, 1, content)
    # stats = aggegate_module_view(profile.tid2tree, profile.events)
    # module_tree = PrettyTable(float_format="1.1", left_padding_width=0, right_padding_width=0)
    # module_tree.field_names = ["Op name", "host_duration", "device_duration", "self_host_duration","self_device_duration"]
    # # dump_stats_new(0, stats, module_tree)
    # print(module_tree)

    tb_forward = PrettyTable(float_format="1.1")
    tb_forward.field_names = ["Op name", "host_duration", "device_duration", "self_host_duration","self_device_duration"]

    # <<<<<<<<<<<<<< new add >>>>>>>>>>>>>>>>>>>>
    # 开始解析trace文件中的各项 event ，生成op树
    parser = EventParser()
    events = []
    trace_body = content['traceEvents']
    fwd_bwd_events = []
    profiler_start_ts = float('inf')
    for data in trace_body:
        if data.get('cat') == 'forward_backward':
            fwd_bwd_events.append(data)
        else:
            event = trace.create_event(data, is_pytorch_lightning=False)
            if event is not None:
                profiler_start_ts = min(profiler_start_ts, event.ts)
                events.append(event)

    events.sort(key=lambda e: e.ts)
    forward_backward_events = trace.create_association_events(fwd_bwd_events)
    tid2tree, pl_tid2tree = parser.parse(events, forward_backward_events)
    # tid2tree 是一个以 tid 为 key 的字典,包含了所有操作的父子关系，一般来说我们只会有一个 tid
    # print_tid2tree(tid2tree[80245], tid2tree)# mini1
    # print_tid2tree(tid2tree[240003], tid2tree)# mini2
    # <<<<<<<<<<<<<< new add >>>>>>>>>>>>>>>>>>>>
    

    # from tb_plugin.torch_tb_profiler.profiler.overall_parser import OverallParser
    # overall_parser = OverallParser()
    # overall_parser.aggregate(parser.steps, parser.role_ranges)

    module_aggregator = ModuleAggregator()
    # module_aggregator.aggregate(profile.tid2tree)
    module_aggregator.aggregate(tid2tree)
    for agg in module_aggregator.op_list_groupby_name:
        agg : OperatorAgg
        tb_forward.add_row([agg.name, agg.host_duration, agg.device_duration, agg.self_host_duration, agg.self_device_duration])
    print(tb_forward)

    stats = aggegate_module_view(tid2tree, events)
    module_tree = PrettyTable(float_format="1.1", left_padding_width=0, right_padding_width=0)
    module_tree.field_names = ["Op name", "host_duration", "device_duration", "self_host_duration","self_device_duration"]
    dump_stats_new(0, stats, module_tree)
    print(module_tree)

