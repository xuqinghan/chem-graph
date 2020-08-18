''' nmr shift db2 
    用于分子结构式 - 13C碳谱  encoder-decoder
    初步读取 .sd  -> dict yml
    
'''
import math
import re
import u_file
#import yaml
from ruamel import yaml
import numpy as np
import copy
import pickle


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


pattern_mid = re.compile(r'nmrshiftdb2 ([\d]*)')

pattern_sid = re.compile(r'> <Spectrum 13C ([\d]*)>')

# C 4
# O 2
# N 3
# H 1
# P 6
# S 2
# F 1
# Br 1
# Cl 1
# I 1
# Te 2
#Se 2
#Si 4
#B 3



# 'Tl' 铊
# Se 硒
# Ge 锗
# Cs 铯
# Sb 锑
# Bi 铋
# Pd 钯
#atom_name = [ 'C', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'Te']
#all_atom_in_db
atom_name = [ 'C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Te', \
            'F', 'Se', 'Si', 'Na', 'B', 'Sn', 'As', 'Tl', 'Li', \
            'K', 'Al', 'Ge', 'Pb', 'Zn', 'Mg', 'Ti', 'Hg', 'Ag',\
            'Cs', 'Pt', 'Pd', 'Sb', 'Ga', 'Bi']

#只包含这些"常见"原子，才考虑，其他全都删除 20200816 晚增补  Se 2 Si 4 B 3
atom_name_whitelist = [ 'C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Te',\
    'Se', 'Si', 'B']

#原子类型标记编码 无原子的-1 其他从0 开始
atom_codes = {'XXX':-1, **{name:i for i, name in enumerate(atom_name_whitelist)}}



def match_id(line, pattern):
    m = pattern.match(line)
    if m:
        return int(m.group(1))
    else:
        return None

def split_sd2_by_nmrshiftdb2(file):
    '''根据nmrshiftdb2 id 开头的行 切分'''
    id_lines = {}
    id_current = None
    lines1 = []
    while True:
        text_line = file.readline()
        #print(text_line)
        if text_line:
            #是文本
            #print(type(text_line), text_line)
            id_m = match_id(text_line, pattern_mid)
            if id_m is not None:
                #是id
                #老的id_current lines1保存
                if id_current is not None:
                    #id_lines[id_current] = lines1
                    print(id_current)
                    record = parse_lines_m1(lines1)
                    if record is None:
                        #print('含有不常见元素，放弃')
                        pass
                    else:
                        id_lines[id_current] = record
                    
                    #id_lines.append({'id': id_current, **parse_lines_m1(lines1)})
                    #只处理1段测试
                    # if id_current == 2190:
                    #     break
                    #break
                #更新id 清空lines1
                id_current = id_m
                lines1 = []
            else:
                #普通文本行
                if id_current is None:
                    #首行，没有id
                    continue
                else:
                    #正常1行
                    lines1.append(text_line)


        else:
            #是文件结束符
            break

    return id_lines


def parse_row_list_space(line1):
    '''形如  18 19  0  0  0  0  0  0  0  0999 V2000 
        ->['',]
    '''
    return line1.lstrip(' ').rstrip('\n').split()

def round_up(value, num=2):
    '''python的 round有坑
        19.345  num = 2  1934.5 + 0.5 = 1935 / 100
    '''
    p10 = math.pow(10, num)
    return round(value * p10 +0.5) / p10

def parse_spectrum_seg1(seg1):
    '''形如 18.3;0.0T;0 '''
    pos, peak, idx_c = seg1.split(';')
    return [round_up(float(pos), 2), peak[-1], int(idx_c)]

def parse_spectrum(line1):
    '''形如 17.6;0.0Q;10|18.3;0.0T;0|22.6;0.0Q;12|'''
    #去掉最后1个 |的数量和谱线数量相等，结尾多出1个空白，去掉
    spectrum_list = line1.strip(' ').rstrip('\n').split('|')[0:-1]
    #print(spectrum_list)

    spectrum_list = [parse_spectrum_seg1(seg1) for seg1 in spectrum_list]
    return spectrum_list

def reorder_spectrum(spectrum1:list):
    '''去重复，排序'''
    spectrum_pos = list(set([pos for pos, peak, idx_c in spectrum1]))
    return sorted(spectrum_pos)


def reindex_bond(bonds, dict_idx_old_new):
    #print('bonds_old', bonds)
    #每条边替换成新编号
    bonds_new = [[dict_idx_old_new[idx_src], dict_idx_old_new[idx_dst], *res]
                for idx_src, idx_dst, *res in bonds]
    return bonds_new


def remove_H(atoms, bonds):
    '''原始数据有时包含部分H原子，为了统一，去掉
        spectrum 里标注的原子序号同时作废
    '''
    print('去除H原子')
    atoms_removed_H = []
    idxs_H = []
    for idx_atom_old, atom1 in enumerate(atoms, 1):
        #原序号从1开始编，现在要删除，对edge需要代换重新编码，因此需要保留老原子编号
        x,y,z,s, name = atom1
        if name == 'H':
           idxs_H.append(idx_atom_old)
        else:
            #保留的原子需要带上老编号！
            atoms_removed_H.append([idx_atom_old, x,y,z,s, name])
    #atoms 码表
    atoms_new = []
    dict_idx_old_new = {}
    #新变号为了兼容不删除H的，仍然从1开始
    for idx_atom_new, atom1 in enumerate(atoms_removed_H, 1):
        idx_atom_old,x,y,z,s, name = atom1
        #去掉老序号
        atoms_new.append([x,y,z,s, name])
        dict_idx_old_new[idx_atom_old] = idx_atom_new
    # print('atoms_old', atoms)
    # print('atoms reindex', atoms_new)
    # print('idxs_H', idxs_H)
    # print('idx_atom_map', dict_idx_old_new)
    #bonds简单去掉含H的
    bonds_removed_H = [x for x in bonds if (x[0] not in idxs_H) and (x[1] not in idxs_H)]
    #reindex
    bonds_new = reindex_bond(bonds_removed_H, dict_idx_old_new)
    #print('bonds reindex', bonds_new)
    return atoms_new, bonds_new



def reindex_all_atom(atoms, bonds):
    ''' 剩余原子，也应该按照atom_name_whitelist 顺序重新排列，这样才能在新输入分子式的情况下（按原子顺序表排序的情况下，去预测）'''
    #重拍atoms 得到新的原子顺序号
    atoms_reindex = []
    for name_atom in atom_name_whitelist:
        for idx_atom_old, atom1 in enumerate(atoms, 1):
            #对edge需要代换重新编码，因此需要保留老原子编号 编号统一从1开始
            x,y,z,s, name = atom1
            if name_atom == name:
                #保留的原子需要带上老编号！
                atoms_reindex.append([idx_atom_old, x,y,z,s, name])
    #atoms 码表
    atoms_new = []
    dict_idx_old_new = {}
    #新变号为了兼容不删除H的，仍然从1开始
    for idx_atom_new, atom1 in enumerate(atoms_reindex, 1):
        idx_atom_old,x,y,z,s, name = atom1
        #去掉老序号
        atoms_new.append([x,y,z,s, name])
        dict_idx_old_new[idx_atom_old] = idx_atom_new
    #reindx
    bonds_new = reindex_bond(bonds, dict_idx_old_new)
    #print('bonds reindex', bonds_new)
    # print('atoms reindex before', atoms)
    # print('atoms reindex after', atoms_new)
    # print('bonds reindex before', bonds)
    # print('bonds reindex after', bonds_new)

    return atoms_new, bonds_new


def fill_num_atom(id_abs:dict, num_atom_max=50):
    '''不足num_atom_max的，补0，超过的舍弃'''
    id_abs_new = {}
    #构造1行空原子记录
    atoms = list(id_abs.values())[0]['atoms']
    atom_dummy1 = [0 for x in range(len(atoms[0]))]
    #最后1个是原子类型 'XXX' 表示无原子
    atom_dummy1[-1] = 'XXX'

    for idm, record1 in id_abs.items():
        record1_new = copy.deepcopy(record1)
        atoms = record1_new['atoms']
        num_atom = len(atoms)
        if num_atom > num_atom_max:
            #超过的，不保存
            continue
        else:
            #小于等于的都要补足
            #不足的，补足
            while num_atom < num_atom_max:
                record1_new['atoms'].append(atom_dummy1)
                num_atom += 1
  
            id_abs_new[idm] = record1_new
    return id_abs_new
    


def parse_lines_m1(lines):
    '''
    个别行包含错误的行列号码
    nmrshiftdb2 10021716
    133138  0  0  0  0  0  0  0  0999 V2000
    所以不按照 行列数计算，根据atom行，和 edge 行的列数区别先区分这这两种行！

    '''
    # line0 = parse_row_list_space(lines[0])
    # num_atom, num_edge, *res = line0
    # num_atom = int(num_atom)
    # num_edge = int(num_edge)
    #print(num_atom, num_edge)
    #当前行号
    idx_line = 1
    atoms = []
    bonds = []
    need_remove_H = False
    #atom bonds 连续排列，遇到不是的，就跳出
    while True:
        line1 = lines[idx_line]
        list_line1 = parse_row_list_space(line1)
        size_line = len(list_line1)
        #print(size_line, line1)
        if size_line == 16:
            name = list_line1[3]
            atom1 = [*[float(x) for x in list_line1[0:3]], int(list_line1[9]), name]

            if name == 'H':
                need_remove_H = True
            else:
                assert name in atom_name, f'{name} not in {atom_name}'
                if name not in atom_name_whitelist:
                    #如果不是常用原子，则直接退出 舍弃
                    print(f'含有不常见元素 {name}，放弃')
                    return None

            atoms.append(atom1)
        elif size_line == 7:
            bonds.append([int(x) for x in list_line1[0:4]])
        else:
            break
        idx_line += 1

    if need_remove_H:
        atoms, bonds = remove_H(atoms, bonds)

    #按白名单对原子重排序 COCOCO -> CCCCOOO

    atoms, bonds = reindex_all_atom(atoms, bonds)

    num_atom = len(atoms)

    #提取 > <Spectrum 13C 0>
    id_s = None
    dict_spectrum = {}
    while idx_line < len(lines):
        '''提取'''
        id_s = match_id(lines[idx_line], pattern_sid)
        if id_s is not None:
            #当前是碳谱id
            #立即读取下一行，作为碳谱！ 
            idx_line += 1
            spectrum1 = parse_spectrum(lines[idx_line])
            #去重复，重排序
            spectrum1 = reorder_spectrum(spectrum1)
            #谱线根数永远小于等于原子总数
            assert len(spectrum1) <= num_atom
            dict_spectrum[id_s] = spectrum1
        else:
            #不处理
            pass
        idx_line += 1

    if not dict_spectrum:
        print('没有 Spectrum 13C ,舍弃')
        return None

    return {'atoms': atoms, 'bonds': bonds, 'spectrums': dict_spectrum}


def save_ABS(fname_abs:str, id_lines):
    #存储 id atom edge spectrum 
    #u_file.save_yml('id_AES.yml', id_lines)
    with open(fname_abs, 'w', encoding="utf-8") as f:
        yaml.safe_dump(id_lines, f, indent=2, block_seq_indent=3, allow_unicode=True)


def load_ABS(fname_abs:str):
    #存储 id atom edge spectrum 
    #u_file.save_yml('id_AES.yml', id_lines)
    with open(fname_abs, 'r', encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.Loader)

def get_num_atom(id_lines):
    return {id: len(r['atoms']) for id, r in id_lines.items()}


def get_min_shift_spectrum1(spectrum1):
    '''可能有重复！'''
    #
    shifts = list(set([x[0] for x in spectrum1]))
    #默认升序
    shifts = sorted(shifts)
    diff = [round_up(shifts[i]-shifts[i-1]) for i in range(1, len(shifts))]
    return diff, shifts

def get_min_shift_spectrum(id_lines):
    diff, shifts = [], []
    for id, r in id_lines.items():
        for ids, spectrum1 in r['spectrums'].items():
            diff1, shift1 = get_min_shift_spectrum1(spectrum1)
            diff.extend(diff1)
            shifts.extend(shift1)

    return diff, shifts

#---------output scope-----------------

def plot_num_atom(id_num_atoms):


    num_atoms = list(id_num_atoms.values())

    num_atoms_max = max(num_atoms)
    hist, bins = np.histogram(num_atoms, bins=num_atoms_max-1, range = (1, num_atoms_max), density=True)
    cdf = np.cumsum(hist)


    fig = plt.figure()
    fig.suptitle('nmr shift db2 样本的 单分子原子数 分布')
    ax1 = fig.add_subplot(1,1,1)
    #print(hist, bins, cdf)
    ax1.hist(num_atoms, histtype='stepfilled',
         color='steelblue', edgecolor='none')
    ax1.set_ylabel("样本数")
    ax1.set_xlabel("单分子非氢原子总数")
    ax1.set_xticks(range(0, 200, 25))
    #print()
    #ax.bar(hist, bins=bins)
    ax2 = ax1.twinx()
    ax2 = fig.add_subplot(1,1,1)
    ax2.plot(range(1, num_atoms_max), cdf, 'r')
    ax2.set_ylabel(r"cdf")
    ax2.set_xticks(range(0, 200, 25))
    ax2.set_yticks([0, 0.5, 0.9, 0.95, 0.99, 1])
    ax2.set_ylim(0, 1)
    ax2.grid()

    print(f'单结构式最大原子数: {num_atoms_max}')
    plt.savefig(f'dist_num_atom_nmrshift_db2_max{num_atoms_max}.png')

def plot_shift(shifts):

    shift_max = max(shifts)
    shift_min = min(shifts)
    fig = plt.figure()
    fig.suptitle('nmr shift db2 样本的 峰位置ppm 分布')
    ax1 = fig.add_subplot(1,1,1)
    #print(hist, bins, cdf)
    ax1.hist(shifts, bins= 500, histtype='stepfilled',
         color='steelblue', edgecolor='none')
    ax1.set_ylabel("数量")
    ax1.set_xlabel("峰位置(ppm)")
    #ax1.set_xticks(range(0, 200, 25))
    ax1.grid()
    print(f'峰位置范围: [{shift_min} - {shift_max}]')
    plt.savefig(f'shift_nmrshift_db2_min{shift_min}_max{shift_max}.png')

def plot_diff_2shift(diff, bins, ranges):

    diff_max = max(diff)
    diff_min = min(diff)


    fig = plt.figure()
    fig.suptitle('nmr shift db2 样本的 相邻峰间距ppm的分布')
    ax1 = fig.add_subplot(1,1,1)
    #print(hist, bins, cdf)
    ax1.hist(diff, bins= bins, histtype='stepfilled',
         color='steelblue', edgecolor='none')
    ax1.set_ylabel("数量")
    ax1.set_xlabel("相邻峰间距(ppm)")
    if ranges is not None:
        ax1.set_xticks(ranges)
    ax1.set_xlim(diff_min, diff_max)
    ax1.grid()
    print(f'相邻峰间距范围: [{diff_min}  {diff_max}]')
    plt.savefig(f'dist_2shift_nmrshift_db2_min{diff_min}_max{diff_max}.png')

#---------pyTorch Geometric-----------------

def read_edges_to_ptG(record1):
    #to ptG edge_index
    bonds = record1['bonds']
    if not bonds:
        '''甲烷特殊，没有bonds
            但有1个原子， 和谱线 仍然保留
        '''
        return None
    bonds = np.array(bonds)
    #bonds.reshape()
    #print(bonds)
    #原序号1开头
    bonds[:, 0] -= 1
    bonds[:, 1] -= 1
    #print(bonds)
    bonds_2 = bonds.copy()
    bonds_2[:, 0] = bonds[:, 1].copy()
    bonds_2[:, 1] = bonds[:, 0].copy()
    edges = np.vstack((bonds, bonds_2)).T
    return edges


def read_atom_features_to_ptG(atoms):
    '''根据atom_codes 进行编码 为了ptG需要 形如 [num_nodes, num_node_features]'''
    atoms_label = [[atom_codes[x[-1]]] for x in atoms]
    #x
    return atoms_label

def get_spectrum_to_ptG(record1):
    '''谱线 2500维  -10.0 ->240
        向量用 -1 1 编码？  为了HingeLoss？
    '''
    x_beg = -10.0
    x_end =  240.0
    x_beg10 = int(x_beg*10)
    x_end10 = int(x_end*10)
    N = int(x_end10 - x_beg10)
    def get_idx_shift(shift:float):
        '''从位移到量化编码序号'''
        #放大10倍，取整
        v = int(round_up(shift, 1) * 10)
        #print(shift, v)
        if v < x_beg10:
            idx = 0
        elif v >= x_end10:
            idx = N-1
        else:
            idx = v - x_beg10
        return idx

    #第0号谱图
    spectrum1 = record1['spectrums'][0]
    #y = np.zeros((1, N))
    y = np.ones((1, N)) * -1
    #按位设置为1
    for shift, peck, idx_c in spectrum1:
        idx = get_idx_shift(shift)
        #print(shift, idx)
        y[:, idx] = 1
    return y


def get_spectrum_to_ptG_50(record1, num_atom=50):
    '''谱线 50维(原子数)  从低到高的浮点数
        其他位置为0
    '''
    #第0号谱图
    spectrum1 = record1['spectrums'][0]
    y = np.zeros((1, num_atom))

    for i,shift in enumerate(spectrum1):
        y[0, i] = shift
    return y


if __name__ == '__main__':
    fname_abs = 'id_ABS.yml'
    with open('./data/nmrshiftdb2withsignals.sd', 'r', encoding='utf-8') as file:
        id_abs = split_sd2_by_nmrshiftdb2(file)

    #save_ABS(fname_abs:str, id_lines)
    #id_lines = load_ABS(fname_abs)
    print(f'共 {len(id_abs)}个结构式')
    with open('./nmrshiftdb2_orgin.pkl', 'wb') as file:
        pickle.dump(id_abs, file)
    

    record1 = list(id_abs.values())[0]
    print(record1)
    #spectrums = reorder_spectrum(record1['spectrums'][0])
    y = get_spectrum_to_ptG_50(record1, num_atom=50)
    print(y)
    
    #id_abs = fill_num_atom(id_abs, num_atom_max=50)

    #print(id_abs)
    #atoms_label = read_atom_features_to_ptG(record1)
    #print(atoms_label)
    #y = get_spectrum_to_ptG(record1)
    #print(y)
    #print(y.shape)
    # x = 18.25
    # print(x, round(x, 1), round_up(x, 1))
    # x = 18.38
    # print(x, round(x, 1), round_up(x, 1))
    # x = 18.36
    # print(x, round(x, 1), round_up(x, 1))


    #-----------------output-------------------------

    # #每个原子数量 统计最大值和分布
    # id_num_atoms = get_num_atom(id_abs)
    # plot_num_atom(id_num_atoms)
    
    # #每个原子 碳谱线最小间距
    # diff, shifts = get_min_shift_spectrum(id_abs)

    # plot_shift(shifts)
    # #0.01精细
    # diff01 = [x for x in diff if x <= 1]
    # plot_diff_2shift(diff01, bins=100, ranges=np.arange(0.0, 1.0, 0.05))
    # #整体
    # plot_diff_2shift(diff, bins=500, ranges=None)

