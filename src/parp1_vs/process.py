"""
@author: qian rong
@date: 2023/10/04
整理sp和xp文件
"""


import pandas as pd

# 读取两个CSV文件
sp_df = pd.read_csv('vsw-parp1-SP_OUT_1.csv')
xp_df = pd.read_csv('vsw-parp1-XP_OUT_1.csv')

# 创建一个新的DataFrame，用于存储结果
result_df = pd.DataFrame(columns=['s_lp_Variant', 'sp_row_indices'])

# 遍历xp_df中的每个值，并查找其在sp_df中的位置
for sp_variant in sp_df['s_lp_Variant']:
    sp_indices = xp_df[xp_df['s_lp_Variant'] == sp_variant].index.tolist()
    result_df = pd.concat([result_df, pd.DataFrame({'s_lp_Variant': [sp_variant], 'sp_row_indices': [sp_indices]})], ignore_index=True)

# 将结果保存到CSV文件
result_df.to_csv('xp_sp_mapping_2.csv', index=False)






# if __name__ == '__main__':
