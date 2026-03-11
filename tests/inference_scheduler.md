## 调度逻辑梳理

### 1. 推理性能指标基本概念

####  1.1 请求级别
- **TTFT(Time to First Token)**
    1. 含义: 首Token延迟, 从发送请求到推理引擎生成第一个输出 token 的时间。衡量推理引擎对单个请求的响应速度, TTFT 越低, 用户体验越好
    2. 公式: **TTFT = T_queue + T_prefill + T_net**
        - **T_queue:** 表示每个prompt请求在调度队列中的等待时间
        - **T_prefill:** 表示每个prompt请求执行prefill推理阶段的耗时
        - **T_net:** 表示网络延迟

- **TPOT(Time Per Output Token)**
    1. 含义: 单Token生成时间, 系统生成每个输出 token 所需的时间。
    2. 公式: **TPOT: = （总生成时间 - TTFT） / （输出的token数 - 1）**


#### 1.2 系统级别
- **InputToken Per Sec:** 系统每秒能够处理的输入 token 数量
- **OutputToken Per Sec:** 系统每秒能够生成的输出 token 数量
- **Concurrency:** 系统在同一时间正在处理的请求数量



### 2. 推理调度逻辑比较

- **原始调度逻辑:** 
    1. **waiting queue** 存储待**prefill**的请求,**running queue**存储**deocde**阶段的请求
    2. 优先**waiting queue**中的请求(**prefill**), 直到**waiting queue**为空时停止
    3. 如果**waiting queue**为空, 则处理**running queue**中的请求(**decode**)

    **特点**: 
    1. 如果此时请求受限max_num_sequences,此时刚做完prefill并处于**decode**阶段的请求的**TPOT**会变长. 也就是说该调度是优先 **TTFT** 的


- **新调度逻辑(chunked_prefill):**
    1. **running queue** 可能会同时存在**prefill**阶段和**decode**阶段的请求
    2. 对于需要**prefill** 的请求要按照限制的 **long_prefill_token_threshold** 进行切分.

    **特点**: 
    1. 对于很长的输入 prompt，可以把它分成几块，在多个步骤中处理完 Prefill 阶段，避免单步计算量过大阻塞其他请求
    2. 对于prefix caching. cache hit命中率会提升.


- bechmark






 





