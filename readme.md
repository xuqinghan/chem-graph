# 20200816
    经过对nmr_shift_db2 的数据清洗和统计：
    共 43540个结构式
    非氢原子数量： 25以下的占接近90%， 50个以下的占几乎99%

    峰位置范围: [-45.8 - 333.8]
    主要集中在-10-240 之间
    相邻峰间距范围: [0.01  1.0]
    相邻峰间距范围: [0.01  233.7]
    但主要以0.1 1位小数步进
    
    提示 初步以非氢原子数量50，原子类型9-11
    
    输入：已知分子式（各原子数量+类型）+ 中间谱线编码

    13C_shift编码:  -10 240  -> 2500 维 稀疏向量 0 1 编码

    需要定义损失函数：
    1 code_shift->graph 结构重建误差= 重点是edge的预测！
    2 graph->code_shift code_shift的误差

    
