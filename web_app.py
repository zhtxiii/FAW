import asyncio
import json
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# 从 FAW 项目引入核心调度库
import config
from main import setup_logging
from models import TaskRequest
from base_agent import BaseAgent
from reviewer import CompositeReviewer, HardcodeRuleReviewer, LLMReviewer, SchemaReviewer
from skill_manager import create_default_registry

# 初始的全局配置
setup_logging(debug_llm=False, debug_tasks=True, debug_skills=False)
logger = logging.getLogger("web_ui")

app = FastAPI(title="FAW Console", description="Fractal Agent Workflow 操作台")

class TaskInput(BaseModel):
    goal: str
    context: Dict[str, Any] = {}
    max_depth: int = 5
    # LOG LEVELS: "OFF", "SIMPLE" (INFO), "VERBOSE" (DEBUG)
    debug_tasks: str = "VERBOSE"
    debug_llm: str = "OFF"
    debug_skills: str = "OFF"
    save_log: bool = True

class QueueLogHandler(logging.Handler):
    """自定义日志处理器，将日志推送到 asyncio 队列"""
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop
        self.setFormatter(logging.Formatter("%(asctime)s | %(name)-13s | %(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            if not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)
        except Exception:
            pass

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAW Agent Console</title>
    <!-- 引入 Google Fonts: Inter 与 JetBrains Mono 用于代码区 -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0a0c;
            --bg-surface: rgba(22, 24, 29, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-primary: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.4);
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(20, 25, 40, 0.5), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(10, 30, 60, 0.4), transparent 25%);
        }

        header {
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
            background: rgba(10, 10, 12, 0.8);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(to right, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        main {
            flex: 1;
            width: 100%;
            max-width: 1000px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            gap: 2rem;
        }

        .panel {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(16px);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .panel:hover {
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-primary);
            font-size: 0.9rem;
            cursor: pointer;
        }

        .checkbox-item input {
            cursor: pointer;
            width: 1.1rem;
            height: 1.1rem;
            accent-color: var(--accent);
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        input[type="text"], textarea {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-family: inherit;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.2s;
        }

        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        textarea {
            resize: vertical;
            min-height: 100px;
        }

        .row {
            display: flex;
            gap: 1rem;
        }

        .row .form-group {
            flex: 1;
        }

        button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        button:hover {
            background: #2563eb;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px var(--accent-glow);
        }

        button:active {
            transform: translateY(1px);
        }

        button:disabled {
            background: #475569;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-right: 0.75rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .result-container {
            display: none;
            margin-top: 1rem;
        }

        .top-bar-stats {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .status-success {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-failed {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }

        /* 🌲 全新树状视图样式 🌲 */
        .tree-viewport {
            background: rgba(0, 0, 0, 0.5);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.05);
            max-height: 700px;
            overflow-y: auto;
        }

        .tree-node {
            position: relative;
            margin: 0.5rem 0;
            padding: 0.75rem 0;
            border-left: 2px solid rgba(255,255,255,0.05);
            transition: all 0.3s;
        }

        .tree-node::before {
            content: '';
            position: absolute;
            top: 1.5rem;
            left: 0;
            width: 1.5rem;
            height: 2px;
            background: rgba(255,255,255,0.05);
        }

        .node-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding-left: 2rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: #a78bfa;
            cursor: pointer;
        }

        .node-icon {
            font-size: 1.1rem;
        }

        .node-status-badge {
            font-size: 0.65rem;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
            color: white;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: auto;
        }

        .node-content {
            padding-left: 2rem;
            padding-top: 0.75rem;
            font-family: 'JetBrains Mono', 'Menlo', monospace;
            color: #94a3b8;
            white-space: pre-wrap;
            word-wrap: break-word; /* Allows long links/words to wrap */
            word-break: break-all; /* Ensures text does not overflow horizontally */
            font-size: 0.8rem;
            line-height: 1.5;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .log-line {
            padding: 2px 0;
            border-bottom: 1px solid rgba(255,255,255,0.02);
        }
        
        .log-line:hover {
            background: rgba(255,255,255,0.03);
        }

        /* Running State */
        .node-running {
            background: rgba(59, 130, 246, 0.05);
            border-left-color: rgba(59, 130, 246, 0.5);
            border-radius: 0 8px 8px 0;
            animation: pulse-border 2s infinite;
        }
        .node-running .node-status-badge {
            background: var(--accent);
            animation: pulse-bg 1s infinite alternate;
        }

        /* Success State */
        .node-success { border-left-color: var(--success); }
        .node-success .node-header { color: var(--success); }
        .node-success .node-status-badge { background: var(--success); }

        /* Failed State */
        .node-failed { border-left-color: var(--danger); }
        .node-failed .node-header { color: var(--danger); }
        .node-failed .node-status-badge { background: var(--danger); }

        /* Exploring State */
        .node-explore { border-left-color: var(--warning); }
        .node-explore .node-header { color: var(--warning); }
        .node-explore .node-status-badge { background: var(--warning); }

        @keyframes pulse-border {
            0% { border-left-color: rgba(59, 130, 246, 0.5); }
            50% { border-left-color: rgba(59, 130, 246, 1); }
            100% { border-left-color: rgba(59, 130, 246, 0.5); }
        }

        @keyframes pulse-bg {
            from { opacity: 0.7; }
            to { opacity: 1; }
        }

        /* Final Result Formatter */
        .final-result-card {
            background: rgba(16, 185, 129, 0.05);
            border: 1px solid var(--success);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            color: var(--text-primary);
        }
        
        .final-result-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--success);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        pre.json-body {
            background: rgba(0,0,0,0.5);
            padding: 1rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            color: #d8b4fe;
            font-size: 0.85rem;
            white-space: pre-wrap;       /* wrap long lines in formatted JSON */
            word-wrap: break-word;       /* Allow wrapping long strings */
            word-break: break-all;
            overflow-x: hidden;          /* Hide horizontal bar since we wrap */
        }

    </style>
</head>
<body>

    <header>
        <h1>FAW · 繁星战情推演总机</h1>
        <div class="subtitle">Recursive Map-Reduce Agent Workflow Dashboard (Tree Visualization)</div>
    </header>

    <main>
        <section class="panel">
            <form id="taskForm">
                <div class="form-group">
                    <label for="goal">最高战略指令 (Goal)</label>
                    <textarea id="goal" placeholder="输入你需要下发给首脑的复杂自然任务 (例如：盘点三条深度学习新闻标题，并撰写一段分析报告)" required></textarea>
                </div>
                
                <div class="row">
                    <div class="form-group">
                        <label for="context">情报注入 (Context JSON)</label>
                        <input type="text" id="context" value="{}" placeholder='{"key": "value"}'>
                    </div>
                    <div class="form-group">
                        <label for="maxDepth">深度防爆阀 (Max Depth)</label>
                        <input type="number" id="maxDepth" value="5" min="1" max="10">
                    </div>
                </div>

                <div class="checkbox-group" style="flex-direction: column; align-items: stretch;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.5rem;">输出信息详细级别 (Logging Verbosity)</div>
                    
                    <div class="row" style="margin-bottom: 0.5rem; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                        <span style="width: 200px;">🕵️‍♂️ 任务状态树 (TASKS):</span>
                        <div style="display:flex; gap: 1rem;">
                            <label class="checkbox-item"><input type="radio" name="levelTasks" value="OFF"> 隐藏</label>
                            <label class="checkbox-item"><input type="radio" name="levelTasks" value="SIMPLE"> 精简</label>
                            <label class="checkbox-item"><input type="radio" name="levelTasks" value="VERBOSE" checked> 全部</label>
                        </div>
                    </div>
                    
                    <div class="row" style="margin-bottom: 0.5rem; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">
                        <span style="width: 200px;">🧠 大模型指令 (LLM):</span>
                        <div style="display:flex; gap: 1rem;">
                            <label class="checkbox-item"><input type="radio" name="levelLLM" value="OFF" checked> 隐藏</label>
                            <label class="checkbox-item"><input type="radio" name="levelLLM" value="SIMPLE"> 精简</label>
                            <label class="checkbox-item"><input type="radio" name="levelLLM" value="VERBOSE"> 全部</label>
                        </div>
                    </div>

                    <div class="row" style="margin-bottom: 1rem; justify-content: space-between;">
                        <span style="width: 200px;">🔧 技能挂载状态 (SKILLS):</span>
                        <div style="display:flex; gap: 1rem;">
                            <label class="checkbox-item"><input type="radio" name="levelSkills" value="OFF" checked> 隐藏</label>
                            <label class="checkbox-item"><input type="radio" name="levelSkills" value="SIMPLE"> 精简</label>
                            <label class="checkbox-item"><input type="radio" name="levelSkills" value="VERBOSE"> 全部</label>
                        </div>
                    </div>

                    <label class="checkbox-item" style="justify-content: flex-end; width:100%;">
                        <input type="checkbox" id="saveLog" checked>
                        保存运行日志到文件 (Save Log)
                    </label>
                </div>

                <button type="submit" id="submitBtn">
                    <div class="spinner" id="btnSpinner"></div>
                    <span>注入神经元 · 开启推演</span>
                </button>
            </form>
        </section>

        <!-- 全新的树状可视化雷达屏幕 -->
        <section class="panel result-container" id="resultPanel">
            <div class="top-bar-stats">
                <label>实况推演雷达树 (Topology Stream)</label>
                <div id="statusBadge" class="status-badge">系统正在连接...</div>
            </div>
            
            <div id="treeViewport" class="tree-viewport">
                <!-- 动态解析渲染的树状DOM将注入在这里 -->
                <div class="node-content" id="roaming-logs" style="padding-left:0; color:#cbd5e1;"></div>
            </div>
        </section>
    </main>

    <script>
        const form = document.getElementById('taskForm');
        const submitBtn = document.getElementById('submitBtn');
        const btnSpinner = document.getElementById('btnSpinner');
        const btnText = submitBtn.querySelector('span');
        const resultPanel = document.getElementById('resultPanel');
        const statusBadge = document.getElementById('statusBadge');
        const treeViewport = document.getElementById('treeViewport');
        let roamingLogs = document.getElementById('roaming-logs');

        // Map to keep track of created agent nodes
        let nodeCache = {};
        let lastCreatedNodeId = null;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            submitBtn.disabled = true;
            btnSpinner.style.display = 'block';
            btnText.textContent = '神经突触裂变与推演中... 脑电波已同步';
            resultPanel.style.display = 'block';
            
            // 重置面板
            treeViewport.innerHTML = '<div class="node-content" id="roaming-logs" style="padding-left:0; color:#cbd5e1; margin-bottom: 1rem;">>>> 系统主枢纽激活，等待下发通讯...<br/></div>';
            roamingLogs = document.getElementById('roaming-logs');
            nodeCache = {};
            lastCreatedNodeId = null;

            statusBadge.textContent = '任务裂变中 (RUNNING)';
            statusBadge.className = 'status-badge';
            statusBadge.style.color = 'var(--text-primary)';
            statusBadge.style.borderColor = 'var(--border-color)';
            statusBadge.style.background = 'rgba(255,255,255,0.05)';

            let ctxJSON = {};
            try {
                ctxJSON = JSON.parse(document.getElementById('context').value);
            } catch (err) {
                alert("Context JSON 解析失败，请检查格式。将使用空字典 {}");
                ctxJSON = {};
            }
            const depth = parseInt(document.getElementById('maxDepth').value) || 5;
            const debugTasks = document.querySelector('input[name="levelTasks"]:checked').value;
            const debugLLM = document.querySelector('input[name="levelLLM"]:checked').value;
            const debugSkills = document.querySelector('input[name="levelSkills"]:checked').value;
            const saveLog = document.getElementById('saveLog').checked;

            try {
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        goal: document.getElementById('goal').value,
                        context: ctxJSON,
                        max_depth: depth,
                        debug_tasks: debugTasks,
                        debug_llm: debugLLM,
                        debug_skills: debugSkills,
                        save_log: saveLog
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";

                // 辅助函数：提取/创生DOM节点
                const getOrCreateNode = (agentId, depthLevel) => {
                    const key = `node-${agentId}`;
                    if (nodeCache[key]) return nodeCache[key];

                    const el = document.createElement('div');
                    el.id = key;
                    el.className = 'tree-node node-running';
                    const indent = (parseInt(depthLevel) || 0) * 2;
                    el.style.marginLeft = `${indent}rem`;

                    el.innerHTML = `
                        <div class="node-header">
                            <span class="node-icon">🤖</span>
                            [Agent] ${agentId}
                            <span class="node-status-badge" id="badge-${key}">PROCESSING</span>
                        </div>
                        <div class="node-content" id="content-${key}"></div>
                    `;
                    
                    // 将节点加到底部（或加在上一个最后激活的位置下方）
                    treeViewport.appendChild(el);
                    
                    nodeCache[key] = {
                        root: el,
                        content: document.getElementById(`content-${key}`),
                        badge: document.getElementById(`badge-${key}`)
                    };
                    
                    lastCreatedNodeId = key;
                    return nodeCache[key];
                };

                const appendLogToCurrent = (msg, agentKey) => {
                    let targetContent = roamingLogs;
                    if (agentKey && nodeCache[agentKey]) {
                        targetContent = nodeCache[agentKey].content;
                    } else if (lastCreatedNodeId && nodeCache[lastCreatedNodeId]) {
                        targetContent = nodeCache[lastCreatedNodeId].content;
                    }

                    const lineEl = document.createElement('div');
                    lineEl.className = 'log-line';
                    // Strip timestamp prefix visually for cleaner UI:
                    // "2026-03-09 15:20:01,123 | module | message" -> "message"
                    let cleanMsg = msg;
                    const parts = msg.split('|');
                    if(parts.length >= 3) {
                        cleanMsg = parts.slice(2).join('|').trim();
                    }
                    lineEl.textContent = cleanMsg;
                    targetContent.appendChild(lineEl);
                };

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\\n");
                    buffer = lines.pop(); // 保留最后一行未完整的 JSON 弦
                    
                    for (let line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const p = JSON.parse(line);
                            if (p.__type__ === 'log') {
                                // 尝试通过正则捕获特制标签 `[Agent xxx depth=y]`
                                // 提取类似 `[Agent search_news depth=1] Executing ...` 或 `Received task:...`
                                const match = p.message.match(/\\[Agent\\s+(.*?)\\s+depth=(\\d+)\\]\\s+(.*)/);
                                
                                if (match) {
                                    const agentId = match[1];
                                    const dLevel = match[2];
                                    const actualMsg = match[3];
                                    
                                    const nodeRef = getOrCreateNode(agentId, dLevel);
                                    
                                    // 过滤掉日志中的前导格式，直接压入树节点
                                    // Make error messages and critical info easier to spot:
                                    let formattedMsg = p.message;
                                    if(formattedMsg.includes("FAILED") || formattedMsg.includes("failed") || formattedMsg.includes("error")) {
                                        formattedMsg = "<span style='color: var(--danger)'>" + formattedMsg + "</span>";
                                    } else if(formattedMsg.includes("passed review") || formattedMsg.includes("SUCCESS")) {
                                        formattedMsg = "<span style='color: var(--success)'>" + formattedMsg + "</span>";
                                    } else if(formattedMsg.includes("Synthesizing") || formattedMsg.match(/🔧|🟢|🤖/)) {
                                        formattedMsg = "<span style='color: var(--warning)'>" + formattedMsg + "</span>";
                                    }
                                    appendLogToCurrent(formattedMsg, `node-${agentId}`);

                                    // 数据分析决定状态变迁
                                    if (actualMsg.includes("passed review") || actualMsg.includes("SUCCESS")) {
                                        nodeRef.root.classList.remove("node-running", "node-failed", "node-explore");
                                        nodeRef.root.classList.add("node-success");
                                        nodeRef.badge.textContent = "SUCCESS";
                                    } else if (actualMsg.includes("failed") || actualMsg.includes("阵亡") || actualMsg.includes("死锁") || actualMsg.includes("崩溃")) {
                                        nodeRef.root.classList.remove("node-running", "node-success", "node-explore");
                                        nodeRef.root.classList.add("node-failed");
                                        nodeRef.badge.textContent = "FAILED";
                                    } else if (actualMsg.includes("UNKNOWN") || actualMsg.includes("Intel:")) {
                                        nodeRef.root.classList.add("node-explore");
                                        nodeRef.badge.textContent = "EXPLORING";
                                    }
                                } else {
                                    // 游离日志，附加到最新的焦点节点上
                                    let formattedMsg = p.message;
                                    if(formattedMsg.includes("FAILED") || formattedMsg.includes("failed") || formattedMsg.includes("error")) {
                                        formattedMsg = "<span style='color: var(--danger)'>" + formattedMsg + "</span>";
                                    } else if(formattedMsg.includes("passed review") || formattedMsg.includes("SUCCESS")) {
                                        formattedMsg = "<span style='color: var(--success)'>" + formattedMsg + "</span>";
                                    } else if(formattedMsg.includes("Synthesizing") || formattedMsg.match(/🔧|🟢|🤖/)) {
                                        formattedMsg = "<span style='color: var(--warning)'>" + formattedMsg + "</span>";
                                    }
                                    appendLogToCurrent(formattedMsg, null);
                                }
                                
                                treeViewport.scrollTop = treeViewport.scrollHeight;

                            } else if (p.__type__ === 'result') {
                                // 输出完美的最终聚合收敛战报
                                const status = p.data.status;
                                statusBadge.textContent = status === 'SUCCESS' ? '执行大满贯 (SUCCESS)' : '阵线溃败 (FAILED)';
                                statusBadge.className = status === 'SUCCESS' ? 'status-badge status-success' : 'status-badge status-failed';
                                statusBadge.style = '';
                                
                                const finalCard = document.createElement('div');
                                finalCard.className = 'final-result-card';
                                finalCard.innerHTML = `
                                    <div class="final-result-header">
                                        <span>${status === 'SUCCESS' ? '🏆 最终收敛输出完毕' : '☠️ 系统全面宕机'}</span>
                                    </div>
                                    <pre class="json-body">${JSON.stringify(p.data.data, null, 2)}</pre>
                                `;
                                treeViewport.appendChild(finalCard);
                                treeViewport.scrollTop = treeViewport.scrollHeight;

                            } else if (p.__type__ === 'error') {
                                statusBadge.textContent = '系统抛错 (ERROR)';
                                statusBadge.className = 'status-badge status-failed';
                                statusBadge.style = '';
                                
                                const errCard = document.createElement('div');
                                errCard.className = 'final-result-card';
                                errCard.style.borderColor = 'var(--danger)';
                                errCard.innerHTML = `
                                    <div class="final-result-header" style="color:var(--danger)">
                                        <span>🔥 源生守护进程崩溃 (Terminal Exception)</span>
                                    </div>
                                    <pre class="json-body" style="color:#f87171">${p.message}</pre>
                                `;
                                treeViewport.appendChild(errCard);
                                treeViewport.scrollTop = treeViewport.scrollHeight;
                            }
                        } catch (err) {
                            console.error("Parse error on chunk:", line, err);
                        }
                    }
                }
            } catch (err) {
                statusBadge.textContent = '网络超限 (NETWORK_ERR)';
                statusBadge.className = 'status-badge status-failed';
                statusBadge.style = '';
                roamingLogs.textContent += "\\n\\n致命异常: " + String(err);
            } finally {
                submitBtn.disabled = false;
                btnSpinner.style.display = 'none';
                btnText.textContent = '注入神经元 · 重新裂变推演';
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_TEMPLATE

@app.post("/api/run")
async def run_task(req: TaskInput):
    logger.info("📡 Console 请求已收到, 正在通过流式管道分配调度首脑...")
    
    # 动态调配当前会话的 DEBUG 级别与选项
    def _parse_level(level_str: str) -> int:
        if level_str == "VERBOSE": return logging.DEBUG
        if level_str == "SIMPLE": return logging.INFO
        return logging.WARNING # OFF

    config.DEBUG_LLM = (req.debug_llm == "VERBOSE")
    config.DEBUG_TASKS = (req.debug_tasks == "VERBOSE")
    config.DEBUG_SKILLS = (req.debug_skills == "VERBOSE")
    
    logging.getLogger("[LLM_IO]").setLevel(_parse_level(req.debug_llm))
    logging.getLogger("[TASKS]").setLevel(_parse_level(req.debug_tasks))
    logging.getLogger("[SKILLS]").setLevel(_parse_level(req.debug_skills))
    
    q = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    # 将日志切面拦截注入系统流
    log_handler = QueueLogHandler(q, loop)
    root_log = logging.getLogger()
    
    any_verbose = ("VERBOSE" in [req.debug_llm, req.debug_tasks, req.debug_skills])
    if any_verbose or getattr(req, "save_log", False):
        root_log.setLevel(logging.DEBUG)
    else:
        root_log.setLevel(logging.INFO)
        
    root_log.addHandler(log_handler)
    
    file_handler = None
    if getattr(req, "save_log", False):
        import os
        from datetime import datetime
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)-13s | %(message)s"))
        root_log.addHandler(file_handler)


    async def _runner_worker():
        try:
            # 初始化 ROOT 名牌，使得大屏上直接展示
            task = TaskRequest(goal=req.goal, context=req.context, title="ROOT_COMMANDER")
            registry = create_default_registry()
            reviewer = CompositeReviewer(
                schema_reviewer=SchemaReviewer(),
                hardcode_reviewer=HardcodeRuleReviewer(),
                semantic_reviewer=LLMReviewer(),
            )
            agent = BaseAgent(max_depth=req.max_depth, skills=registry, reviewer=reviewer)
            
            result = await agent.solve(task)
            
            logger.info(f"✅ Console 任务退栈归来! Status: {result.status}")
            # 发送结果数据帧
            await q.put(json.dumps({"__type__": "result", "data": result.model_dump()}))
        except Exception as e:
            logger.error(f"❌ Console 任务执行宕机: {str(e)}", exc_info=True)
            await q.put(json.dumps({"__type__": "error", "message": str(e)}))
        finally:
            root_log.removeHandler(log_handler)
            if file_handler:
                root_log.removeHandler(file_handler)
            await q.put(None)  # EOF

    # 启动后台引擎
    asyncio.create_task(_runner_worker())

    async def event_generator():
        while True:
            msg = await q.get()
            if msg is None:
                break
            # 区分结构化指令还是日志透传
            if msg.startswith('{"__type__"'):
                yield msg + "\n"
            else:
                yield json.dumps({"__type__": "log", "message": msg}) + "\n"

    # 以流式 NDJSON 返回
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
