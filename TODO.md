# TODO - stats-cli 改进计划

## 一、增强工作流模板

### 1.1 新增模板
- [ ] `reliability` - 可靠性分析模板
  - clean → descriptive → reliability(weibull) → report
  - 参数: times, status, confidence_level

- [ ] `doe` - 实验设计模板
  - doe(full_factorial) → 回归分析 → 优化
  - 参数: factors, responses

- [ ] `timeseries` - 时间序列模板
  - clean → descriptive → timeseries(exp_smoothing/arima) → report
  - 参数: frequency, n_forecast

- [ ] `regression` - 回归分析模板
  - clean → scatter → regression → residual_analysis
  - 参数: x, y, reg_type

- [ ] `multivariate` - 多变量分析模板
  - clean → correlation_matrix → pca → cluster
  - 参数: n_components, n_clusters

### 1.2 模板增强
- [ ] 添加模板参数验证
- [ ] 添加模板执行进度回调
- [ ] 支持模板嵌套（子工作流）
- [ ] 添加模板缓存机制

---

## 二、增强可视化能力

### 2.1 基础图表增强
- [ ] 散点图矩阵 (scatter matrix)
- [ ] 热力图 (heatmap) - 相关矩阵可视化
- [ ] 箱线图增强 - 支持分组对比
- [ ] 直方图增强 - 支持叠加正态曲线

### 2.2 统计图表
- [ ] Q-Q 图增强 - 支持多种分布
- [ ] 残差图 - 回归诊断
- [ ] 帕累托图 - DOE 因子效应
- [ ] 控制图增强 - 支持多图联动

### 2.3 交互式图表
- [ ] 集成 Plotly 生成交互式图表
- [ ] 支持图表缩放、平移
- [ ] 支持数据点悬停显示
- [ ] 支持图表导出 (PNG/SVG/HTML)

### 2.4 图表模板
- [ ] 预定义制造业图表样式
- [ ] 预定义科研论文图表样式
- [ ] 支持自定义图表主题

---

## 三、解决技术债务

### 3.1 Bug 修复
- [ ] chi_square 测试 numpy 数组转换问题
  - 文件: `stats_engine/nonparametric.py:137`
  - 问题: `r(stat)` 无法处理 numpy 数组

- [ ] 部分 xfail 测试修复
  - Hahn1: 7 参数有理模型收敛问题
  - ENSO: 9 参数谐波模型局部最优
  - BoxBOD: 指数衰减渐近线不明显
  - Bennett5: 数据范围极窄

### 3.2 代码质量
- [ ] 统一命名规范
  - 部分模块使用 snake_case，部分使用 camelCase
  - 统一输出字段命名

- [ ] 添加类型注解
  - 为所有公共函数添加类型注解
  - 使用 mypy 进行类型检查

- [ ] 改进错误处理
  - 统一错误消息格式
  - 添加更多上下文信息

### 3.3 测试改进
- [ ] 提升低覆盖率模块
  - `doe.py`: 75% → 90%
  - `assumptions.py`: 82% → 90%
  - `transform.py`: 81% → 90%
  - `timeseries.py`: 84% → 90%

- [ ] 添加集成测试
  - 端到端工作流测试
  - 大数据集压力测试
  - 并发测试

### 3.4 文档完善
- [ ] API 文档自动生成
  - 使用 Sphinx 或 MkDocs
  - 从 docstring 自动生成

- [ ] 使用示例
  - 添加更多工作流示例
  - 添加制造业实际案例

---

## 四、性能优化

### 4.1 计算优化
- [ ] 使用 NumPy 向量化操作
- [ ] 添加数据分块处理
- [ ] 优化大数据集内存使用

### 4.2 缓存机制
- [ ] 缓存计算结果
- [ ] 缓存文件加载
- [ ] 缓存图表生成

---

## 五、集成功能

### 5.1 MCP Server 支持
- [ ] 实现 MCP 协议
- [ ] 注册工具和资源
- [ ] 支持 Claude Code 直接调用

### 5.2 REST API
- [ ] 添加 HTTP 接口
- [ ] 支持批量请求
- [ ] 添加认证机制

---

## 优先级排序

| 优先级 | 任务 | 预计工作量 |
|--------|------|-----------|
| P0 | 修复 chi_square bug | 1 小时 |
| P1 | 添加 5 个工作流模板 | 3-4 小时 |
| P2 | 增强基础图表 | 2-3 小时 |
| P3 | 添加交互式图表 | 3-4 小时 |
| P4 | 提升测试覆盖率 | 2-3 小时 |
| P5 | 添加类型注解 | 2-3 小时 |
| P6 | API 文档生成 | 2-3 小时 |
| P7 | MCP Server 支持 | 4-5 小时 |

---

## 进度跟踪

| 任务 | 状态 | 开始时间 | 完成时间 |
|------|------|----------|----------|
| chi_square bug | 待开始 | - | - |
| 工作流模板 | 待开始 | - | - |
| 基础图表增强 | 待开始 | - | - |
| 交互式图表 | 待开始 | - | - |
| 测试覆盖率 | 待开始 | - | - |
| 类型注解 | 待开始 | - | - |
| API 文档 | 待开始 | - | - |
| MCP Server | 待开始 | - | - |
