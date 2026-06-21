{
  "title": "基于小红书品牌讨论数据的美妆板块另类因子构建与实证研究初稿",
  "document_type": "研究备忘录 / 量化研究初稿",
  "abstract": "本研究基于已构建的A股量化基础因子框架（包含30-50个经典基本面与技术指标），探索小红书（Xiaohongshu）品牌级讨论度数据在大消费及美妆板块（美容护理行业）中的另类Alpha价值。通过建立‘品牌-商品-个股’映射关系，本文构建了声量份额变动（SoV）、互动效率（Engagement）以及大单品热度动量三个维度的另类舆情因子。在剔除传统因子暴露（中性化/正交化）后，实证结果表明，纯净的小红书另类舆情残差因子对美妆板块个股的未来月度超额收益表现出稳定的正向Rank IC（秩相关系数），显示出与传统因子的良好互补性及对基本面数据的超前预测能力。",
  "1_introduction": {
    "background_and_motivation": "在行为金融学框架下，散户投资者的有限关注（Limited Attention）与情绪波动对股票价格具有短期的驱动和反转效应。A股美妆板块具有高Beta、高营销费用占比以及重度依赖单品周期的特征。传统的微博、股吧讨论数据对金融宏观情绪反应敏感，但对于特定行业的基本面边际变化（如产品销售趋势）刻画精度有限。小红书作为高粘性的生活和消费决策平台，其UGC（用户生成内容）数据能直接反映微观层面的产品势能和消费偏好转移。",
    "research_positioning": "遵循量化研究‘先稳固基础、后注入另类’的原则，本研究不将另类数据作为孤立信号，而是将其作为已有传统因子库（P0-P2阶段成果）的增量补充，旨在提炼不被传统动量或财务指标解释的非结构化Alpha。"
  },
  "2_data_and_preprocessing": {
    "sample_universe": "A股美容护理板块主要上市公司（重点覆盖：珀莱雅、贝泰妮、丸美股份、上海家化、水羊股份等），样本期间为历史多期月频数据。",
    "traditional_factor_baseline": "基础因子库包含30~50个传统因子，主要包括：估值因子（PE, PS）、成长因子（单季营收同比、单季净利环比）、盈利质量（毛利率、销售费用率）、技术因子（20日/60日动量、特质波动率）。",
    "alternative_data_processing": {
      "entity_mapping": "利用品牌字典将小红书上的特定品牌词汇（如‘薇诺娜’、‘双抗精华’）精准映射至对应的上市公司代码。",
      "seasonality_adjustment": "美妆行业在‘618’与‘双十一’存在强烈的季节性脉冲，本研究通过跨年度同比（YoY）及行业截面去均值（Demean）方法剔除该季节性噪声。",
      "outlier_handling": "对原始讨论声量进行对数化转换，并进行3倍中位数绝对偏差（MAD）去极值及标准化（Z-Score）处理。"
    }
  },
  "3_factor_construction": {
    "factor_definitions": [
      {
        "factor_name": "声量份额动态变化因子 (Share of Voice, SoV_Dynamics)",
        "logic": "刻画品牌在截面竞争中的相对势能变化，避免因大盘整体季节性上涨导致的信号失真。",
        "formula": "SoV_i_t = Brand_Volume_i_t / Total_Beauty_Volume_t; Factor_i_t = Log(SoV_i_t / SoV_i_t-1)"
      },
      {
        "factor_name": "互动效率因子 (Engagement_Efficiency)",
        "logic": "剥离人工投放水分，衡量真实用户对品牌内容的自发互动（点赞、收藏、评论）深度，预防营销开支导致的利润黑洞。",
        "formula": "Engagement_Efficiency_i_t = Log((Likes_i_t + Collects_i_t + Comments_i_t) / Total_Posts_i_t)"
      },
      {
        "factor_name": "大单品热度动量 (Hero_Product_Momentum)",
        "logic": "美妆公司的业绩高度依赖明星大单品，通过高频监控核心大单品关键词频次，提前锁定单品爆发周期。",
        "formula": "Hero_Product_Momentum_i_t = (Hero_Volume_i_t - Mean(Hero_Volume_i_[t-k, t])) / Std(Hero_Volume_i_[t-k, t])"
      }
    ]
  },
  "4_empirical_methodology": {
    "neutralization_and_orthogonalization": {
      "step_1": "在美妆板块截面上，利用传统的估值、成长、质量及技术因子对个股下期收益率进行回归，估计出基准预期收益率。",
      "step_2": "将构建的小红书另类因子对传统因子暴露进行横截面普通最小二乘回归（OLS），提取残差作为‘纯净舆情因子’，消除与已知风格因子的共线性。",
      "step_3": "对纯净舆情因子进行行业中性化（限制在美妆细分领域内部对比），确保因子提供的是特异性超额收益（Idiosyncratic Alpha）而非行业贝塔。"
    },
    "evaluation_metrics": [
      "月频 Rank IC（秩相关系数）及其均值与标准差",
      "Rank ICIR（信息比率）",
      "多空组合（Long-Short）在美妆细分行业内的超额累积收益率"
    ]
  },
  "5_preliminary_empirical_results": {
    "ic_performance": "初步测算显示，正交化后的小红书声量份额变动（SoV_Dynamics）与互动效率（Engagement_Efficiency）对未来一月收益率（T+1）的 Rank IC 呈现稳定的正向分布，初步验证了非结构化舆情对股价短期定价偏差的解释力。",
    "complementarity": "小红书舆情残差因子与传统‘单季营收增速’及‘销售费用率’因子相关性极低，但对未来的‘财务超预期（Earnings Surprise）’展现出一定的超前指示作用，证明了该另类数据能提供传统基本面因子库之外的‘信息差’。",
    "factor_decay": "该舆情因子的信息衰减速度较快，T+1月信号强度最高，T+2月逐步衰减，不建议用于超长周期的策略配置，更适用于月度换仓的主动管理或风格精细微调。"
  },
  "6_risks_and_limits": {
    "marketing_efficiency_trap": "部分品牌存在‘高营销、低转化’的现象。若公司历史销售费用率极高且利润率持续恶化，单凭舆情声量暴增进行做多可能面临‘利润踩踏’风险。模型需与毛利率等财务基本面指标进行联合风控。",
    "data_compliance_and_stability": "另类数据源在抓取、法律合规、历史回溯的连续性上存在天然脆弱性，需建立Point-in-Time（即时历史点数据）数据库以规避前瞻偏差。"
  },
  "7_next_steps": {
    "step_a": "对美妆板块进一步细分，细化护肤、彩妆等子品类的映射图谱，测试在极细分赛道内的Alpha表现。",
    "step_b": "将模型从单因子验证（IC分析）推进至行业内部多因子优化（如利用Ridge或非线性回归融合），构建小红书优化的美容护理组合策略。",
    "step_c": "进一步引入机器学习特征选择，评估当传统因子的Rank Corr出现阶段性失效时，舆情残差因子的风险对冲效果。"
  }
}