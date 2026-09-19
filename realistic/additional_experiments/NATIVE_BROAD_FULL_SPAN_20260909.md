# Native Broad：完整生成记录span对齐

本变体只对齐Native Broad的完整记录范围，其他设置保留。版本`native_broad_full_span_20260909_v1`；kth、category × Qwen3-8B、Gemma4-E4B四组分别重新discovery选头、排序、冻结并执行完整K扫描。

## 唯一方法修正

选头key从每条已注册生成记录的末尾token改为该记录完整的原始token span。直接调用主实验`parse_trace_record`、`align_trace_sites`和`_visible_item_spans`，以每题原有最终答案query作为可见前缀边界；每条记录的literal_token_start/end须精确对应原始生成IDs及文本。无法精确表示的完整span按主实验规则排除，不近似取边界、不重写自然trace。若仍有完整span，按实际J计算；若一个也没有，记录输入不可用。旧版不可用的最终回答query不新增。

每头M、H、J由这些完整span的attention mass计算，直接调用主实验`_broad_span_metrics`（含epsilon=1e-12）。先输入内、seed内、再seed等权得到全局排名。category保留自然轨迹中全部已注册记录，不额外按所问类别过滤；已完成的Non-thinking目标类别变体不变。

## 保留设置

每模型×任务300个原输入，discovery seeds1234–1253共200题，confirmation seeds1254–1263共100题。原始prompt、gold、自然输出、最终回答query和模型修订不变。Qwen K=1/2/4/8/16/32/64/128，Gemma K=1/2/4/6/8。逐层匹配三个随机对照，seeds7000–7002，优先不重叠、容量不足时只允许最少重叠。每次只在一次prefill的最终回答query置零pre-O切片，最多继续64tokens，以最终任务答案正确性评分。Clean、Selected、Random全部新生成。

K仍按confirmation主分析的最大Random−Selected展示，属于事后探索性峰值。两个任务、模型及26个K比较在本轮构成一个Holm族，Clean-correct另成一族；95%区间使用seed配对bootstrap20000次（seed20260907），逐点区间未校正选K。相对末尾token版本的Δ变化只在相同样本、相同K上配对比较；Clean正确性须与旧版一致。自然准确率、Non-thinking和Targeted保持已完成结果。

## 执行和审计

先只加载固定tokenizer，核验1200条Native plans的完整span及原始输入；冻结几何后才采集discovery attention。每模型每任务写出全部200条discovery状态（不可用原因保留），仅有效且所有头共享的discovery样本进入排名；每组20个discovery seed和10个confirmation seed均须有支持。所有头的各span原始mass保存，供重新计算分数和完整排名。

小规模检查每任务×模型2题、最小/最大K，共16个case×K、80条条件输出；category包含city和flower问法，kth覆盖首个可用seed的最小/最大有效level。双模型通过后执行全量，最多2247个case×K（11235条条件输出记录）。最终点数由只读几何阶段确定，不依据消融结果筛选。准确率和新旧配对变化只对新设置有效的同一端点集合计算，覆盖率单列。

归档、冻结文件、源自然输入、几何、完整排名、所有K和对照、原始输出评分与分析表必须核验通过才更新HTML。局部浮点复核预先允许最多8个binary64可表示数间隔，零值须精确一致，重算的完整头顺序须与冻结排序完全一致；实际数值差写入审计。GPU端使用exact验证。冻结包及大输出不提交git，旧版本不覆盖。

该修正完成后只确认Native Broad的记录span定义与主实验一致。现有K展示与随机对照方案已冻结，仍保留它们与主实验配置之间的适配说明。



Anonymous release note: This technical protocol is retained for package construction. Deployment/account records are excluded; it is not the byte-identical historical protocol artifact.
