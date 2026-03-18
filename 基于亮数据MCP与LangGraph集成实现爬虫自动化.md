

之前的案例，我们将亮数据与AI框架结合使用，但需要手动选择MCP工具。

本次案例将亮数据MCP与LangGraph集成，实现爬虫智能体，AI帮我们自动选择最合适的MCP工具，可实现数据自动抓取，高效突破人机验证、验证码识别、IP封禁等各类常见反爬防御壁垒，轻松解决采集过程中的核心阻碍。深度适配企业级快速采集需求，能够支撑大规模数据抓取作业，稳定运行且有效规避封禁风险，兼顾效率与安全性。

接下来，就跟随博主的脚步，一步步开展实战操作，解锁高效数据采集技巧。

# LangGraph 简介
LangGraph 让你可以构建 LLM 应用，其中的控制流是显式且易于检查的，而不是隐藏在提示词或重试逻辑里。每一步都是一个节点，每一次跳转都由你定义。

智能体以循环方式运行。LLM 模型读取当前状态，选择直接回应或请求调用工具。如果它调用了某个工具（例如网页搜索），结果会被加入回状态中，然后模型再次做出决策。当它收集到足够信息时，循环结束。


这就是工作流与智能体的关键区别。工作流遵循固定步骤；智能体则循环：决策、行动、观察，再次决策。这个循环也是 Agentic RAG 系统的基础，在这种系统中，检索是动态发生的，而不是在固定节点执行。

LangGraph 为你提供一种结构化方式来构建这个循环，包含记忆、工具调用，以及明确的停止条件。你可以看到智能体做出的每一个决策，并控制它何时停止。

# 前置条件
要跟随本教程，你需要准备：

●Python 3.11+ 版本
●Bright Data 账号
●OpenAI的密钥

# 步骤1：安装依赖
在cmd执行下面的pip安装命令：
pip install  langgraph langchain langchain-openai langchain-mcp-adapters  python-dotenv


# 步骤2：.env 文件用于存储 API Key
定义两个参数：
●OPENAI_API_KEY：存放OpenAI的密钥
●BRIGHTDATA_TOKEN：存放亮数据密钥

# 步骤3：生成 OpenAI API Key
智能体需要一个 LLM API Key 来进行推理并决定何时使用工具。在该配置中，这个 Key 来自 OpenAI。

# 步骤4：生成 Bright Data API Token

注册完成后，进入账号设置→用户与 API 密钥，即可查看你的专属密钥。
使用该密钥，可调用亮数据的这些 API 工具：
●Web Unlocker API：一键抓取网页，返回纯净 HTML/JSON，自动绕过常规防护。
●SERP API：批量获取 Google、Bing 等搜索引擎数据。
●Web Scraping API：从主流平台直接提取结构化解析数据。

# 步骤5：代码实战

复制代码去编辑器：
```python
import asyncio
import os
from typing import Literal

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState



# 智能体规则
SYSTEM_PROMPT = """
你是一名网络研究助理，根据提示词调用合适的工具采集指定网页

约束条件：
- 最多使用5个来源。
- 首选官方文件或一手资料。
- 保持快速：不要进行深度爬取。
"""



def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
    If the last LLM message requested tool calls,
    continue to the tool execution node.
    Otherwise, end the graph and return the final answer.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool_node"
    return END


# ------------------------------------------------------------
# Node 1: Ask the LLM what to do next
# ------------------------------------------------------------

def make_llm_call_node(llm_with_tools):
    async def llm_call(state: MessagesState):
        """
        Sends the conversation (plus system rules) to the LLM.
        The model can either:
        - return a final answer, or
        - request tool calls (search, scrape, etc.)
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        ai_message = await llm_with_tools.ainvoke(messages)
        return {"messages": [ai_message]}
    return llm_call


# ------------------------------------------------------------
# Node 2: Execute MCP tools requested by the LLM
# ------------------------------------------------------------

def make_tool_node(tools_by_name: dict):
    async def tool_node(state: MessagesState):
        """
        Executes each tool call requested by the LLM and
        returns the results as ToolMessage objects.
        """
        last_ai_msg = state["messages"][-1]
        tool_results = []

        for tool_call in last_ai_msg.tool_calls:
            tool = tools_by_name.get(tool_call["name"])

            if not tool:
                tool_results.append(
                    ToolMessage(
                        content=f"Tool not found: {tool_call['name']}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            # MCP tools are typically async
            observation = (
                await tool.ainvoke(tool_call["args"])
                if hasattr(tool, "ainvoke")
                else tool.invoke(tool_call["args"])
            )

            tool_results.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_results}
    return tool_node


# ------------------------------------------------------------
# Main: wire everything together and run the agent
# ------------------------------------------------------------

async def main():
    # Load env variables
    load_dotenv()

    # Load Bright Data token
    bd_token = os.getenv("BRIGHTDATA_TOKEN")
    if not bd_token:
        raise ValueError("Missing BRIGHTDATA_TOKEN")

    # Connect to Bright Data Web MCP and load available tools
    client = MultiServerMCPClient({
        "bright_data": {
            "url": f"https://mcp.brightdata.com/mcp?token={bd_token}",
            "transport": "streamable_http",
        }
    })

    tools = await client.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    # Create an LLM and allow it to call MCP tools
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build the LangGraph agent
    graph = StateGraph(MessagesState)

    graph.add_node("llm_call", make_llm_call_node(llm_with_tools))
    graph.add_node("tool_node", make_tool_node(tools_by_name))

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")

    agent = graph.compile()

    # 采集提示词
    # 案例1
    topic = "采集crunchbase网站，这家公司信息：https://www.crunchbase.com/organization/apple#predictions_and_insights"

    # 案例2
    # topic = "采集ZoomInfo网站，这个人的信息：https://www.zoominfo.com/p/Jay-Blackmore/11596849948"

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Research this topic:\n{topic}")]}
        , config={"recursion_limit": 12}
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

```





新手用户注册就送25美金：[点击免费体验](https://www.bright.cn/blog/ai/langgraph-with-web-mcp/?utm_source=brand&utm_campaign=brnd-mkt_cn_csdn_man202603&promo=brd25)


亮数据官号：[爬虫技巧/代理IP/粉丝福利](https://bbs.csdn.net/topics/620071800)

