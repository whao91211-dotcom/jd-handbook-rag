"""Bounded handbook vocabulary expansion; contains no evaluation IDs or answers."""

STRATEGIES = ('baseline', 'expanded', 'coverage')


def expand_queries(query: str) -> list[str]:
    query = query.strip()
    if not query:
        return []
    facets = []
    if any(w in query for w in ('电脑', '设备', '财物', '财产')) and any(w in query for w in ('坏', '损', '丢', '赔')):
        facets += ['公司资产 公司财产 故意 过失 毁损 追偿', '经济赔偿 经济损失 扣发工资']
    if any(w in query for w in ('工资', '薪水', '薪酬')) and any(w in query for w in ('到账', '发', '结算', '何时', '几天')):
        facets += ['发薪时间 发薪日 薪酬核算周期']
        if any(w in query for w in ('离职', '辞职', '交接')):
            facets += ['离职工资 交接 结算']
    if '假' in query and any(w in query for w in ('工资', '餐补', '扣', '薪')):
        facets += [('事假 无薪假 薪酬 福利' if '事假' in query else '休假 全薪假 带薪扣减福利假 无薪假')]
        facets += ['餐费补贴 实际出勤 休假 扣减']
    if '病假' in query:
        facets = ['病假 薪酬 全薪 半薪 福利病假', '入职 转正 福利病假天数 折算'] + facets
    if '迟到' in query or '早退' in query:
        facets += ['迟到 早退 连续 纪律处分 记过', '累进式处罚 一般违纪 严重违纪']
    if any(w in query for w in ('转岗', '调动', '换部门', '换到')):
        facets += ['员工发起 公司发起 调动 离任审计', '内部人才流动 原岗位 书面同意']
    if '年假' in query:
        facets += ['法定年假 福利年假 社会工龄 年度上限', '福利年假 转正日 折算']
    return list(dict.fromkeys([query] + facets))[:3]


def fuse_ranks(rank_maps: list[dict[str, int]], constant: int = 60) -> dict[str, float]:
    scores = {}
    for ranks in rank_maps:
        for iid, rank in ranks.items():
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (constant + rank)
    return scores


def select_evidence(fused: list[str], facets: list[list[str]], k: int) -> list[str]:
    """Reserve original-query evidence and at most two sources per extra facet."""
    if k <= 0:
        return []
    if len(facets) <= 1:
        return fused[:k]
    selected = []
    available = set(fused)

    def add(order, limit):
        count = 0
        for iid in order:
            if len(selected) >= k or count >= limit:
                break
            if iid in available and iid not in selected:
                selected.append(iid)
                count += 1

    add(facets[0], min(3, k))
    for facet in facets[1:]:
        add(facet, 2)
    add(fused, k)
    return selected
