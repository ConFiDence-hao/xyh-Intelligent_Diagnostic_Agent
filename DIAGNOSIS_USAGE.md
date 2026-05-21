# 诊断功能使用说明

## 🎯 功能概述

现在AI Ops诊断支持**两种智能模式**，会根据当前对话状态自动选择：

### 1️⃣ **系统诊断模式**（System Diagnosis）
- **触发条件**：没有对话历史时
- **功能**：检查系统告警、监控数据、日志分析
- **适用场景**：基础设施问题排查

### 2️⃣ **问题诊断模式**（Question Diagnosis）  
- **触发条件**：有对话历史时自动触发
- **功能**：分析用户对话中的具体问题
- **适用场景**：应用层问题、业务逻辑问题

---

## 📝 使用示例

### 场景1：系统诊断
```
1. 打开页面（无对话历史）
2. 点击右上角 "AI Ops" 按钮
3. 自动执行系统诊断
4. 查看告警分析报告
```

### 场景2：问题诊断
```
1. 在对话框中描述问题，例如：
   用户："我的服务响应很慢，CPU使用率很高"
   助手："这可能是由于..."
   
2. 点击 "AI Ops" 按钮
3. 自动分析问题诊断
4. 查看深度分析报告
```

---

## 🔧 技术实现

### 后端改动

#### 1. 请求模型扩展 (`app/models/aiops.py`)
```python
class AIOpsRequest(BaseModel):
    session_id: str = "default"
    diagnosis_type: str = "system"  # 'system' 或 'question'
    question_context: Optional[str] = None  # 问题诊断时的上下文
```

#### 2. 服务层逻辑 (`app/services/aiops_service.py`)
```python
async def diagnose(
    self,
    session_id: str = "default",
    diagnosis_type: str = "system",
    question_context: Optional[str] = None
):
    if diagnosis_type == "question":
        # 问题诊断逻辑
        aiops_task = f"请对以下用户问题进行深度诊断分析：\n{question_context}..."
    else:
        # 系统诊断逻辑（原有）
        aiops_task = "诊断当前系统是否存在告警..."
```

#### 3. API接口 (`app/api/aiops.py`)
```python
@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    # 传递诊断类型和上下文
    async for event in aiops_service.diagnose(
        session_id=request.session_id,
        diagnosis_type=request.diagnosis_type,
        question_context=request.question_context
    ):
        yield event
```

### 前端改动

#### 智能判断逻辑 (`static/app.js`)
```javascript
async triggerAIOps() {
    let diagnosisType = 'system';
    let questionContext = null;
    
    // 如果有对话历史，使用问题诊断
    if (this.currentChatHistory.length > 0) {
        const recentMessages = this.currentChatHistory.slice(-4);
        questionContext = recentMessages.map(msg => 
            `${msg.type === 'user' ? '用户' : '助手'}: ${msg.content}`
        ).join('\n\n');
        diagnosisType = 'question';
    }
    
    await this.sendAIOpsRequest(loadingMessage, diagnosisType, questionContext);
}
```

---

## ✅ 测试步骤

1. **启动服务**
   ```bash
   # 确保 Docker 正在运行
   docker ps
   
   # 启动主服务
   python -m uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload
   ```

2. **测试系统诊断**
   - 打开浏览器访问 http://localhost:9900
   - 直接点击右上角 "AI Ops" 按钮
   - 应该看到"将进行系统诊断..."提示
   - 等待诊断报告生成

3. **测试问题诊断**
   - 在对话框输入问题，例如：
     ```
     用户：数据库连接超时怎么办？
     ```
   - 等待助手回复
   - 点击 "AI Ops" 按钮
   - 应该看到"检测到对话历史，将分析问题..."提示
   - 等待问题分析报告生成

---

## 🎨 用户体验优化

- **自动识别**：无需手动选择，系统智能判断
- **清晰提示**：显示当前使用的诊断模式
- **流式输出**：实时展示诊断进度
- **Markdown渲染**：美观的报告格式

---

## 🚀 后续优化建议

1. **手动切换**：可以添加UI让用户手动选择诊断类型
2. **历史记录**：保存诊断报告供后续查看
3. **诊断模板**：预设常见问题的诊断模板
4. **结果导出**：支持导出诊断报告为PDF/Markdown
