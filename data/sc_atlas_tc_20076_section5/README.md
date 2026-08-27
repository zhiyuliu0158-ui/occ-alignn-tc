# SC-Atlas 20,076 条 Tc 数据 + Section 5 文章字段

本目录以 `OUR_SC_ATLAS_DATA_FINAL_20260810` 中经过三键去重的 20,076 条 SC-Atlas Tc 记录为唯一主表，没有加入 MDR_SuperCon、SuperCon2 或 Hc2 派生记录。

新增字段来自最新版 `all_11200_article_features.xlsx` 的 `文章主表`：

- `competing_orders`
- `gap_symmetry`
- `pairing_mechanism`
- `dopant_defect`

匹配规则为 DOI 精确匹配优先，未命中时使用规范化标题精确回退；若标题对应冲突 DOI，则不自动补充。空白保持缺失，不解释为负标签。

## 文件

- `sc_atlas_tc_20076_with_section5_features.csv`：20,076 条完整公开字段和四个新增字段。
- `section5_feature_match_audit.csv`：逐条匹配方法、冲突和字段可用性审计。
- `SUMMARY.json`：输入输出行数、覆盖率和匹配统计。
