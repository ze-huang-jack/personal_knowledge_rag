# 2 个月冲刺：AI Agent 实习（5月→7月）

> 仅剩 2 个月，策略必须聚焦：**砍掉所有"学了有用但不是立刻能用"的东西，只做能写进简历的事。**

---

## 总原则

- **不系统学理论** — 用到什么查什么
- **每天写代码** — 最少 4h，周末 8h+
- **跳过 Transformer 数学推导、不刷 LeetCode** — 没时间
- **所有学习围绕项目进行** — 项目里遇到什么学什么
- **第 8 周开始投简历** — 边投边完善

---

## 8 周时间分配

| 周次      | 阶段              | 核心产出                              |
| ------- | --------------- | --------------------------------- |
| 第 1 周   | LLM 基础速通        | 跑通 API、理解 Prompt/Function Calling |
| 第 2-3 周 | 项目 1：RAG 知识库    | 完整可用的文档问答系统                       |
| 第 4-5 周 | 项目 2：多 Agent 协作 | 多 Agent 自动协作系统（简历核心）              |
| 第 6-7 周 | 项目打磨 + 面试准备     | GitHub README、Demo 视频、八股文         |
| 第 8 周起  | 投递 + 面试         | 持续迭代简历和项目                         |

---

## 第 1 周：LLM 基础速通

### 目标
能熟练调用 API，理解 Agent 的核心概念，为项目做准备。

### 每日安排

**Day 1-2：API 上手**
- 注册 OpenAI / Anthropic / DeepSeek 的 API，各拿到 key
- 用 Python 分别调用三个 API，完成同一个任务：**"把一段中文新闻总结成 3 个要点"**
- 对比三个模型的输出差异
- 理解这几个参数：`temperature`、`top_p`、`max_tokens`、`system` vs `user` message

**Day 3-4：Prompt Engineering**
- 写出以下 5 种 prompt，全部跑通：
  1. Few-shot：给 3 个示例，让模型对新的输入做分类
  2. Chain-of-Thought：让模型分步推理一个数学应用题
  3. ReAct：让模型一边思考一边调用"工具"（先模拟，不用真的调 API）
  4. JSON 结构化输出：让模型稳定输出指定 JSON 格式，用 pydantic 校验
  5. System prompt：设计一个角色的 system prompt（如"毒舌代码审查员"）
- **产出物**：一个 `prompt_lab.py` 脚本，包含上述 5 个 demo

**Day 5-6：Function Calling**
- 阅读 OpenAI Function Calling 文档
- 定义一个工具：`get_weather(city: str) -> dict`
- 让 LLM 在对话中自动判断何时调用这个工具
- 实现完整的 loop：用户提问 → LLM 决定调工具 → 执行工具 → 把结果传回 LLM → LLM 生成最终回复
- **关键代码模式**（你会反复用到）：

```python
# Agent 核心循环 —— 后面所有项目都基于这个模式
messages = [{"role": "user", "content": user_input}]
while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)  # assistant message with tool_calls
        for tool_call in msg.tool_calls:
            result = execute_tool(tool_call.function.name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    else:
        # 最终回复
        return msg.content
```

**Day 7：RAG 快速概念**
- 理解 RAG 流程：文档 → 切分 → Embedding → 向量库 → 检索 → 生成
- 跑通一个最简 RAG demo（LangChain 的 `create_retrieval_chain`，30 行代码）
- 理解 Chunk size、overlap、相似度检索 三个概念

---

## 第 2-3 周：项目 1——个人知识库 RAG 系统

### 项目概述

做一个**可用、有 UI** 的文档问答系统。面试官可以打开浏览器直接试。

### 功能清单

**核心功能：**
- [ ] 上传 PDF / Markdown / TXT 文件
- [ ] 自动切片 + Embedding 存入向量数据库
- [ ] 输入问题，返回答案 + 引用原文来源（哪份文件、第几段）
- [ ] 支持多轮对话（记住之前的上下文）
- [ ] Web UI 界面

**加分功能：**
- [ ] 支持上传多个文件，跨文件检索
- [ ] 显示检索到的相关片段（让用户看到检索过程）
- [ ] 支持切换不同的 LLM（如 GPT-4o vs DeepSeek）

### 技术栈

```
后端：FastAPI
RAG 框架：LangChain
向量数据库：Chroma（轻量，嵌入式不需要单独部署）
Embedding：OpenAI text-embedding-3-small 或 BGE（本地）
LLM：OpenAI GPT-4o / DeepSeek-V4
前端：Streamlit 或 Gradio（不用写 HTML/CSS）
```

### 项目结构

```
rag-knowledge-base/
├── app.py              # FastAPI 主入口 + API 路由
├── rag_engine.py       # RAG 核心逻辑：加载、切片、检索、生成
├── streamlit_ui.py     # Streamlit 前端
├── requirements.txt
├── data/               # 用户上传的文档存这里
├── chroma_db/          # Chroma 持久化目录
└── README.md
```

### API 设计

```
POST /upload          # 上传文件，返回文件 ID
POST /chat            # 发送消息，返回答案 + 引用来源
GET  /documents       # 列出已上传的文档
DELETE /documents/{id} # 删除文档
```

### 核心代码片段（rag_engine.py）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载文档
def load_and_split(file_path: str):
    loader = TextLoader(file_path)  # 或 PyPDFLoader
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)

# 2. 存入向量库
def index_documents(splits, collection_name="default"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name,
    )

# 3. RAG 问答
def ask(question: str, collection_name="default"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = (
        "你是一个知识库助手。根据以下检索到的上下文回答问题。"
        "如果上下文中没有答案，就说不知道，不要编造。"
        "回答时注明信息来源。\n\n"
        "上下文：\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, chain)

    result = rag_chain.invoke({"input": question})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["context"]],
    }
```

### 这个项目展示的能力

- 熟练使用 LangChain 生态
- 理解 RAG 全流程
- 能写 FastAPI 后端 + 简单前端
- 理解向量检索原理
- 关注用户体验（引用来源、多轮对话）

---

## 第 4-5 周：项目 2——多 Agent 协作系统

### 选一个方向做

我建议做 **「代码分析 Agent 团队」**，因为面试官大概率也是写代码的，容易共鸣。

### 场景描述

用户上传一个 GitHub 仓库链接（或本地路径），三个 Agent 自动协作：

```
用户：分析这个项目的代码质量
         │
    ┌────▼────┐
    │ Planner │  拆解任务："1. 先找到核心模块 2. 检查代码规范 3. 看测试覆盖率"
    └────┬────┘
         │ 分发子任务
    ┌────▼─────────────────────┐
    │                          │
┌───▼───┐  ┌───▼───┐  ┌────▼───┐
│Coder  │  │Reviewer│  │Tester  │
│Agent  │  │Agent   │  │Agent   │
│分析架构│  │查代码  │  │查测试  │
│识别模式│  │质量/安全│  │覆盖情况│
└───┬───┘  └───┬───┘  └────┬───┘
    │          │           │
    └──────────▼───────────┘
         │ 汇总
    ┌────▼─────┐
    │Reporter  │  整合报告 → 输出 Markdown 分析报告
    │Agent     │
    └──────────┘
```

### 技术选型

**用 LangGraph**（不用 AutoGen）。理由：LangGraph 的 StateGraph 对面试官来说认知成本更低，而且 LangGraph 在简历上更常见。

### 项目结构

```
agent-code-review/
├── main.py             # 入口：启动整个 Agent 流程
├── agents/
│   ├── planner.py      # Planner Agent：任务分解
│   ├── coder.py        # Coder Agent：分析代码架构
│   ├── reviewer.py     # Reviewer Agent：检查代码质量
│   ├── tester.py       # Tester Agent：检查测试覆盖
│   └── reporter.py     # Reporter Agent：汇总报告
├── tools/
│   ├── file_tools.py   # 读文件、列目录、搜索代码
│   ├── git_tools.py    # git log、git diff
│   └── shell_tools.py  # 运行 pytest、ruff 等命令
├── graph.py            # LangGraph 状态图定义
├── state.py            # Agent 共享状态定义
├── ui.py               # Streamlit 前端
└── requirements.txt
```

### LangGraph 核心状态图设计

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

# 1. 定义共享状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    repo_path: str
    plan: dict                # Planner 产出的任务计划
    coder_report: str         # Coder 的分析结果
    review_report: str        # Reviewer 的审查结果
    test_report: str          # Tester 的检查结果
    final_report: str         # 最终汇总报告
    next_step: str            # 路由字段

# 2. 定义各 Agent 节点
def planner_node(state: AgentState) -> AgentState:
    """分析项目结构，拆解成子任务"""
    # 1. 先用 file_tools 扫描项目结构
    # 2. LLM 分析并生成任务计划
    # 3. 返回更新后的 state，设置 next_step
    pass

def coder_node(state: AgentState) -> AgentState:
    """执行代码分析子任务"""
    pass

def reviewer_node(state: AgentState) -> AgentState:
    """执行代码审查子任务"""
    pass

def tester_node(state: AgentState) -> AgentState:
    """执行测试检查子任务"""
    pass

def reporter_node(state: AgentState) -> AgentState:
    """汇总三个 Agent 的结果，生成最终报告"""
    pass

# 3. 路由函数
def router(state: AgentState) -> str:
    return state["next_step"]

# 4. 构建图
graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_node("reviewer", reviewer_node)
graph.add_node("tester", tester_node)
graph.add_node("reporter", reporter_node)

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", router, {
    "coder": "coder",
    "reviewer": "reviewer",
    "tester": "tester",
    "reporter": "reporter",
    "end": END,
})
# 各 worker 完成后都回到 planner，由 planner 决定下一步
graph.add_edge("coder", "planner")
graph.add_edge("reviewer", "planner")
graph.add_edge("tester", "planner")
graph.add_conditional_edges("reporter", router, {"end": END})

app = graph.compile(checkpointer=MemorySaver())
```

### 每个 Agent 的具体实现思路

**Planner Agent：**
- 工具：`list_files(path)`、`read_file(path)`、`search_code(pattern)`
- System prompt：*"你是技术负责人。分析项目结构，制定代码审查计划。每次只分配一个子任务给对应的 Agent。完成后根据结果决定下一步。"*
- 输出：JSON 格式的子任务，指定分配给哪个 Agent

**Coder Agent：**
- 工具：`read_file(path)`、`search_code(pattern)`、`get_git_history(file)`
- System prompt：*"你是资深工程师。分析代码架构：模块划分是否合理？依赖关系是否清晰？核心逻辑是否有明显 bug？"*

**Reviewer Agent：**
- 工具：`read_file(path)`、`run_linter(path)`
- System prompt：*"你是代码审查专家。检查命名规范、代码风格、安全漏洞（SQL 注入、XSS 等）、错误处理是否完善。"*

**Tester Agent：**
- 工具：`list_files(path)`、`read_file(path)`、`run_pytest(path)`
- System prompt：*"你是测试专家。检查测试覆盖情况：有没有测试？测试是否覆盖了边界条件？有没有关键的未测试模块？"*

**Reporter Agent：**
- 无额外工具，纯生成
- System prompt：*"你是技术写作专家。根据三个 Agent 的分析结果，生成一份结构化的 Markdown 代码质量报告。"*

### 工具函数实现细节

```python
# tools/file_tools.py
import os
import ast
from pathlib import Path

def list_files(repo_path: str, pattern: str = "*", max_depth: int = 3) -> str:
    """列出目录结构，限制深度防止 token 爆炸"""
    files = []
    base = Path(repo_path)
    for path in base.rglob(pattern):
        relative = path.relative_to(base)
        if len(relative.parts) <= max_depth:
            files.append(str(relative))
    # 限制返回数量
    return "\n".join(files[:200])

def read_file(repo_path: str, file_path: str) -> str:
    """读取文件内容，限制长度"""
    full_path = Path(repo_path) / file_path
    # 安全检查：防止路径穿越
    full_path.resolve().relative_to(Path(repo_path).resolve())
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return content[:12000]  # 限制 token

def search_code(repo_path: str, keyword: str) -> str:
    """在项目中搜索关键词"""
    results = []
    for path in Path(repo_path).rglob("*.py"):
        with open(path, "r", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if keyword in line:
                    results.append(f"{path}:{i}: {line.strip()}")
        if len(results) >= 50:
            break
    return "\n".join(results)
```

### 前端展示

用 Streamlit 做一个简洁的界面：
- 输入框：填入仓库路径或 GitHub URL
- 进度展示：实时显示 Agent 执行过程（哪个 Agent 在执行什么任务）
- 最终报告：渲染 Markdown 格式的代码质量报告
- 对话区：用户可以追问"展开说说安全漏洞那个部分"

### 这个项目展示的能力

- **LangGraph 状态机** — 当前最火的 Agent 编排方式
- **多 Agent 协作** — Planner-Worker-Critic 分工模式
- **工具设计** — 为 LLM 设计合理、安全的工具
- **工程化** — 状态管理、错误处理、日志追踪
- **有 UI** — 面试官能直接体验
- **贴近真实场景** — 代码审查是每个团队的刚需

---

## 第 6-7 周：项目打磨 + 面试准备

### 项目打磨

**GitHub 仓库要求：**
- [ ] README.md：项目介绍 + 架构图（Mermaid 或截图）+ 本地运行步骤 + Demo GIF/截图
- [ ] `requirements.txt` 或 `pyproject.toml` — 一键安装依赖
- [ ] 环境变量模板 `.env.example`
- [ ] 代码注释清晰，函数有 type hints
- [ ] 使用 `python-dotenv` 管理 API Key，不要 hardcode

**Demo 视频/GIF：**
- [ ] 用 OBS 或 ScreenToGif 录一个 1-2 分钟的演示
- [ ] 展示核心功能的使用流程
- [ ] 放到 README 里，面试官不一定能跑你的代码，但一定会点开看

### Agent 面试八股文速成

以下 10 题是高频题，每条准备 2-3 分钟的口述回答：

1. **Agent 和传统 Chatbot 的本质区别是什么？**
   > Chatbot 是单轮/多轮的"问题→答案"映射。Agent 多了一个"行动→观察→反思"的循环：它能调用外部工具、感知执行结果、根据结果调整下一步计划。本质区别是 Agent 有自主决策和执行能力。

2. **ReAct 模式是什么？为什么比单纯的 CoT 更适合 Agent？**
   > ReAct = Reasoning + Acting 交替进行。CoT 只是思考，不能真的做事。ReAct 让模型一边想一边做，做了之后看到结果再调整思路，形成闭环。例如"帮我查明天北京的天气"→ 思考"需要调用天气工具"→ 调工具得到 25°C → 思考"天热，建议带伞"→ 输出。

3. **Function Calling 的原理？Tool schema 怎么设计？**
   > 本质是 LLM 被 fine-tune 过，能理解 JSON Schema 定义的函数签名。schema 设计原则：函数名用动词、description 写清楚"什么时候应该调用这个函数"而非"这个函数做什么"、参数描述要包含约束条件和示例。

4. **Agent 的记忆有哪几种？怎么实现？**
   > 短期记忆 = 对话历史（直接塞进 messages）。长期记忆 = 向量库存储历史交互的关键信息。工作记忆 = Agent 当前任务的状态（如 LangGraph 的 State）。实现上：短期直接拼接到 context，长期通过检索增强，工作记忆靠框架的 state 管理。

5. **做过的 Agent 项目最大挑战是什么？怎么解决的？**
   > 准备 1-2 个真实踩坑经历，例如：LLM 无限循环调用工具 → 加 max_iterations 限制 + 每次调用记录日志；检索不准 → 调整 chunk size + 加 reranker；多人协作时 Agent 输出不可控 → 引入 structured output + 校验层。

6. **多 Agent 之间怎么通信？上下文爆炸怎么解决？**
   > 通信方式：共享 state / 消息队列 / 直接函数调用。上下文爆炸：每个 Agent 只收到和自己任务相关的上下文（不传完整历史）、汇总时做摘要而非拼接原始输出、用 LangSmith 追踪 token 消耗。

7. **RAG 的完整流程？怎么评估检索质量？**
   > 离线：文档加载→切分（chunk_size+overlap）→Embedding→存向量库。在线：用户 query→可能改写→检索 top_k→重排序→拼接 context→LLM 生成。评估：命中率（答案需要的文档片段是否被检索到）、MRR（倒数排名）。

8. **为什么需要 Human-in-the-loop？在 LangGraph 里怎么实现？**
   > 某些操作需要人类确认（发邮件、执行危险命令）。LangGraph 中把关键节点设为 interrupt point，执行到该节点时暂停，等待人类 approve/reject，然后继续。

9. **设计一个客服 Agent 系统，你会怎么设计？**
   > 意图识别→路由到对应子 Agent（退换货/查询订单/人工客服）→每个子 Agent 有专属工具（查订单 API、退款 API）→无法处理时升级到人工→所有对话存入长期记忆用于优化。

10. **对 AI Agent 未来发展的看法？**
    > 展现你对领域的关注。可以提到：MCP 协议（标准化工具接口）、多模态 Agent、Agent 可靠性（减少幻觉和错误操作）、Agent 之间的标准化通信协议等。

### 简历模板

```
## 项目经历

### 多 Agent 代码审查系统 | LangGraph · FastAPI · Streamlit
[2026.6] 个人项目 | GitHub: github.com/xxx/agent-code-review
- 基于 LangGraph 构建了 4 个 Agent（Planner/Coder/Reviewer/Tester）协作的代码审查系统
- Planner 自动拆解任务并分发给 3 个执行 Agent，最终由 Reporter 汇总生成 Markdown 报告
- 为 Agent 设计了 6 个工具函数（文件读取、代码搜索、Linter 调用等），含路径穿越安全防护
- 使用 Streamlit 实现前端，支持实时展示 Agent 执行过程
- 对 5 个开源项目进行测试，平均能发现 8-15 个有效问题（命名/安全/测试覆盖）

### RAG 知识库问答系统 | LangChain · Chroma · FastAPI · Streamlit
[2026.5] 个人项目 | GitHub: github.com/xxx/rag-knowledge-base
- 实现了完整的 RAG Pipeline：文档加载→切片→Embedding→向量检索→LLM 生成
- 支持 PDF/Markdown/TXT 多格式上传，多轮对话，答案引用来源标注
- 基于 Chroma 向量数据库 + OpenAI text-embedding-3-small，检索延迟 < 200ms
- 使用 RecursiveCharacterTextSplitter 优化切片策略（chunk_size=1000, overlap=200）
```

---

## 第 8 周起：投递 + 面试中迭代

### 投递节奏

- 每天投 3-5 家，不要一次投完
- **先投小厂练手**，面试中你的项目介绍会越来越流畅
- 大厂（字节、腾讯等）留到第 2-3 周再投，此时你已经练过几次面试了

### 投递渠道优先级

1. **内推** > 官网 > Boss 直聘 > 实习僧 > 牛客
2. 找内推的方式：校友群、LinkedIn 搜"公司名 + intern + 内推"、即刻、小红书、V2EX

### 面试中常被要求的 Live Coding

可能会被要求现场写：
- 一个 Function Calling 的完整 loop
- 一个简单的 FastAPI 接口
- 用 LangChain 搭一个 RAG 链
- **你在项目里写的代码就是最好的准备**，项目做扎实了，live coding 不会慌。

---

## 每周检查清单

### 第 1 周
- [ ] 能分别用 OpenAI/Anthropic/DeepSeek API 完成同一个 NLP 任务
- [ ] 写完了 5 种 prompt 的 demo
- [ ] 实现了一个 Function Calling 的完整 loop
- [ ] 跑通了最简 RAG demo

### 第 2-3 周
- [ ] RAG 系统能上传文件、问答、显示引用来源
- [ ] 有 Streamlit/Gradio Web UI
- [ ] GitHub 仓库创建，README 完成初版
- [ ] 跑通了 3 个以上的测试文档，效果可接受

### 第 4-5 周
- [ ] LangGraph 图流程跑通（Planner → Workers → Reporter）
- [ ] 每个 Agent 至少有 2 个以上工具
- [ ] 有 Streamlit UI 展示执行过程
- [ ] 对 2 个以上真实开源项目跑过，产出分析报告

### 第 6-7 周
- [ ] 两个项目 GitHub README 完善（架构图 + 截图/GIF + 安装步骤）
- [ ] 10 道八股文全部能口述回答
- [ ] 1 分钟自我介绍练到脱口而出
- [ ] 简历完成并找 2 个人 review 过

### 第 8 周
- [ ] 已投出 ≥ 10 份简历
- [ ] 至少完成 1 次面试（哪怕是小厂练手）

---

## 不要做的事

- ❌ 从头系统学 ML/DL 数学推导
- ❌ 刷 LeetCode（startup 不考，大厂你 2 个月也刷不够）
- ❌ 学多个 Agent 框架（只学 LangGraph，够用了）
- ❌ 追求完美项目 — 能用就行，面试中迭代
- ❌ 花时间学 Docker/K8s/云原生 — Agent 实习生不需要这些
- ❌ 看论文 — 除了 Anthropic 的 Agent 指南和 Lilian Weng 的博客，别的现阶段不需要

---

## 关键资源（只保留最核心的）

| 资源                                                                                        | 为什么看它                       | 什么时候看         |
| ----------------------------------------------------------------------------------------- | --------------------------- | ------------- |
| [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)    | Function Calling 的官方文档      | 第 1 周         |
| [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | 照着写你的第一个 Agent 图            | 第 4 周         |
| [Anthropic Agent 指南](https://docs.anthropic.com/en/docs/build-with-claude/agent-patterns) | 理解什么场景下用 Agent              | 第 4 周（写项目前读）  |
| [Lilian Weng 的 Agent 博客](https://lilianweng.github.io/posts/2023-06-23-agent/)            | Agent 全景综述                  | 第 6 周（准备面试时读） |
| DeepLearning.AI 的 [LangGraph 短课](https://www.deeplearning.ai/short-courses/)              | 2 小时视频带你写一个 LangGraph Agent | 第 4 周         |

---

> **一句话总结：5 月做项目，6 月打磨投简历，7 月入职。每天至少 4 小时写代码，两个项目 = 一张有竞争力的简历。**
