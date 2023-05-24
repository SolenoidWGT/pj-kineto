# kinto代码结构(待更新)

- EventParser.parse
  - parse_nodes：根据时间的包含和间隔关系建立event的父子调用关系
    - _parse_node：这个函数遍历所有的 evnet ，根据 event type 创建不同的节点，比如 kernel 执行的 DeviceNode 节点，或者 cudalaunchkernel 的 RuntimeNode 节点
      - update_communication_node
      - externalid_to_runtime.pop + op.runtimes.extend(runtime_nodes) 
        			原始代码是在这里建立了 runtime node（cudalaunchkernel）和 op 的关联(associate CUDA Runtimes with CPU events)
        	build_tree
      - _build_tree：Construct the BackwardNode and replace the original backward nodes 这函数会建立op树,build_tree 函数会被调用两次，分别构建fw和bw阶段的op树
        - build_tree_internal
        - build_tree_relationship：首先按照所有 op 的 start_time和 end_time对 oplist1排序，获得一个排完序后的oplist实际上是一个前序遍历optree的结果遍历排序后的 oplist 然后根据 strat time 和 end time 确定各个op间的调用关系（父子关系）
        - fill_stats：这个函数会自底向上填充 op 树节点的数据，这时候会累加各个 op 的 host/device time （注意不是moudle time）
        - _get_modules：如果存在fwd过程，会执行到这里，其会Get the ModuleNodes and backward root nodes If there are any ModuleNodes, the backward roots will be removed from the tree。so that later a new BackwardNode will be replaced.
    - parse_steps：为了生成 role_ranges 
    - update_device_steps
      - _find_device_steps：这里会计算出来各个 step （也就是我们调用 prof.step的阶段）下，上一个step的结束时间，step的device侧的最小开始和最大结束时间，以及是否启动了kernel
      - （并不是）这里会把 cudaLaunchKernel 节点(RuntimeNode)和 kernel （deviceNode）关联起来 _update_steps_duration 这个函数会根据上一个函数的结果更新每个 step 的起止时间
    - generate_communication_nodes	

## 1. EventParser.parse

注意下 RuntimeNode 节点，注释里面写的是需要使用 external_id 作为 RuntimeNode 与 OperatorNode或者 ProfilerStepNode 标记父子关系的记号，但是实际上我们抓出来的trace中很多 RuntimeNode 找不到包含对应 external_id 的 OperatorNode（很可能是pytorch生成trace时就没有打这个记号），因此导致了module view中没有device时间的bug，这个bug可以通过贪心的方法搜索关联性获得 RuntimeNode 和 OperatorNode 的父子关系解决，见[修复pr](https://github.com/SolenoidWGT/pj-kineto/pull/1)

```
    # For OperatorNode and ProfilerStepNode:
    #   Use time interval containing relationship to build father-child correlation,
    #   which is consistent with autograd profiler.
    # For RuntimeNode:
    #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
    #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
    #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
    # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
```

另外的bug就是 kinto 和 pytorch 中的 trace 文件的 eventname可能因为版本问题现在没有对齐，比如 kinto的关键字 `Runtime` 应该修改为 `cuda_runtime`

节点（Node）的类型：

| Node类型          | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| BaseNode          | 所有node类型的基类，一个node最基本的要素就是包含起止时间，名字，type，以及tid |
| CommunicationNode | 描述通信算子的node，需要额外记录input数据的大小，从而方便计算带宽 |
| HostNode          | 描述一般的cpu函数调用                                        |
| ProfilerStepNode  | 描述profier自身的一些执行情况，似乎只用于占位                |
| ModuleNode        |                                                              |
| BackwardNode      |                                                              |
| DataLoaderNode    |                                                              |
| OptimizerNode     |                                                              |
| RuntimeNode       | 专门为 cudaLaunchKernel 函数创建的 node，其会和一个或若干个 kernel event 关联在一起 |
| DeviceNode        | 描述 cuda kernel，MEMCPY，MEMSET 的 node                     |

时间（Event）类型

event类型是直接对应 trace 文件中每一个json里面的内容，可以直接从 trace 文件中搜索。

| event类型       | 描述                       |
| --------------- | -------------------------- |
| KERNEL          | cuda 算子                  |
| MEMORY          | 显存profiling的event       |
| PYTHON          | 应该是描述的是python线程？ |
| PYTHON_FUNCTION | python函数                 |
|                 |                            |



### 1.1 parse_nodes 执行流程

1. _parse_node创建node

这个过程并不会建树，而是遍历trace文件把每一个event变成各种node，devlce/runtime node会被存到 device_node_list， runtime_node_list 等列表中。python_function等op node会被存到 tid2list 这个字典里。

2. 更新每种 nccl 通信算子的总时间

```python
 if CommLibTypes.Nccl in self.comm_lib:
        for event in events:
            if event.type == EventTypes.KERNEL:
                self._update_communication_node(event)
```

3. kinto是在这里建立了 runtime node（cudalaunchkernel）和 op 的关联(associate CUDA Runtimes with CPU events)，我们发现有的runtime的external_id无法对应到一个op上去，应该是kinto/pytorch的bug

```python
for op_list in tid2list.values():
    for op in op_list:
        # print(f"Pop op {op.external_id} list from externalid_to_runtime")
        # runtime_nodes = externalid_to_runtime.pop(op.external_id, []) # 这个是原来的方法，我们发现有的runtime的external_id无法对应到一个op上去，应该是kinto/pytorch的bug
        runtime_nodes,_ = find_op_correlation_runtime_node(op, externalid_to_runtime)
        if runtime_nodes:
            print(f"op:{op.name} with external_id:{op.external_id} add runtime nodes: {runtime_nodes}")
            op.runtimes.extend(runtime_nodes)
        else:
            print(f"op:{op.name} with external_id:{op.external_id} has no relative tunime_nodes")
```

### 1.2 build_tree执行流程

该阶段会循环遍历 tid2list 字典，将其转化为  tid2tree 字典，根据"最早开始且最晚结束"的op作为root节点：

```python
# Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
op_list.sort(key=lambda x: (x.start_time, -x.end_time))
```

然后递归执行build_tree，实际建立父子关系的函数为 _build_tree_internal， 最终返回是根节点的 tid2tree 字典

然后调用  _build_tree_internal ，建立op树。下面这段是是确定节点间父子关系的逻辑，在函数 build_tree_relationship 中

```python
node_stack.append(root_node)	# 先把整棵树的 root 节点入栈
for node in host_node_list:	# host_node_list 就是刚才排序后的 op_list，按照 "最早开始且最晚结束" 节点顺序排序
    while True:  # break loop when the node is inserted.
        tail_node = node_stack[-1]	# 把当前子树root pop出来，即tail_node
        if node.start_time < tail_node.end_time:
            if node.end_time <= tail_node.end_time: # 如果这个节点在 tail_node 启动后执行，且早于 tail_node 结束，说明是 tail_node 的儿子节点
                tail_node.children.append(node)
                # node.parent_node = weakref.ref(tail_node)
                node_stack.append(node)	
            else:
                logger.error('Error in input data: ranges on the same thread should not intersect!'
                             'Father:({},{},{}) Child:({},{},{})'
                             .format(tail_node.name, tail_node.start_time, tail_node.end_time,
                                     node.name, node.start_time, node.end_time))
            break	# 尝试遍历新加入的 node 是否还有孩子节点
        else:	 # 如果这个节点在 tail_node 启动后执行，且晚于 tail_node 结束，说明这个节点是和 tail_node 并列关系的节点，tail_node 的孩子已经找完了，把 tail_node pop 掉
            node_stack.pop() 
```

还有一个 remove_dup_nodes 操作，这个是为了把相同函数的循环合并成一个 node，这样显示起来比较友好，且省内存，因为有时候循环次数会非常大，直接解析trace顶不住的。

build_tree_relationship 完成后，我们就获得了 op 树，由于刚辞提到的bug，我们要在这里手动关联下 runtime node 和 op node 的关系：

```python
externalid_to_runtime = self.externalid_to_runtime
print(f"externalid_to_runtime:{externalid_to_runtime}")
print("\n")
print("<<<<<<<<<<< second traverse_tid2tree >>>>>>>>>>>>>>>>>>>> ")
traverse_tid2tree(root_node, None, externalid_to_runtime, 0)

root_node.fill_stats()
```

我们手动关联  runtime node 和 op node  必须要在 `root_node.fill_stats()` 操作之前，这样才能正确累加出按树结构的 module  粒度的 device 时间。

在build完成树后，会根据是否存在 fwd_bwd_map 来觉得是否关联 fwd 和 bwd 阶段，这就引入了第二个bug，目前这个 fwd_bwd_map 是不存在的

### 1.3 parse_steps 执行过程

待续

### 1.4 update_device_steps 

# Kineto
--------------------------------------------------------------------------------------------
Kineto is part of the PyTorch Profiler.

The Kineto project was started to help enable

- **performance observability and diagnostics** across common ML bottleneck components
- **actionable recommendations** for common issues
- integration of external system-level profiling tools
- integration with popular visualization platforms and analysis pipelines

A central component is libkineto, a profiling library with special focus on low-overhead GPU timeline tracing.

The PyTorch Profiler TensorBoard plugin provides powerful and intuitive visualizations of profiling results, as well as actionable recommendations, and is the best way to experience the new PyTorch Profiler.

## Libkineto

Libkineto is an in-process profiling library integrated with the PyTorch Profiler. Please refer to the [README](libkineto/README.md) file in the `libkineto` folder as well as documentation on the [new PyTorch Profiler API](https://pytorch.org/docs/master/profiler.html).

## PyTorch TensorBoard Profiler

The goal of the PyTorch TensorBoard Profiler is to provide a seamless and intuitive end-to-end profiling experience, including straightforward collection from PyTorch and insightful visualizations and recommendations in the TensorBoard UI.
Please refer to the [README](tb_plugin/README.md) file in the `tb_plugin` folder.

## Future Development Direction:

Some areas we're currently working on:

- Support for tracing distributed workloads
- Trace processing, analysis and recommendation engine
- System-level activities, multiple tracing sources
- Profiling and monitoring daemon for larger scale deployments

## Releases and Contributing

We will follow the PyTorch release schedule which roughly happens on a 3 month basis.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the infrastructure in a different direction than you might be aware of. We expect the architecture to keep evolving.

## License

Kineto has a BSD-style license, as found in the [LICENSE](LICENSE) file.
