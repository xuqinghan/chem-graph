''' nmr shift db2 
    用于分子结构式 - 13C碳谱  encoder-decoder
    初步读取 .sd  -> dict yml
    
'''

import re
import u_file
#import yaml
from ruamel import yaml
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


pattern_mid = re.compile(r'nmrshiftdb2 ([\d]*)')

pattern_sid = re.compile(r'> <Spectrum 13C ([\d]*)>')


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
                    id_lines[id_current] = parse_lines_m1(lines1)
                    #id_lines.append({'id': id_current, **parse_lines_m1(lines1)})
                    #只处理1段测试
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

def round_up(value):
    '''python的 round有坑'''
    return round(value * 100) / 100.0

def parse_spectrum_seg1(seg1):
    '''形如 18.3;0.0T;0 '''
    pos, peak, idx_c = seg1.split(';')
    return [round_up(float(pos)), peak[-1], int(idx_c)]

def parse_spectrum(line1):
    '''形如 17.6;0.0Q;10|18.3;0.0T;0|22.6;0.0Q;12|'''
    #去掉最后1个 |的数量和谱线数量相等，结尾多出1个空白，去掉
    spectrum_list = line1.strip(' ').rstrip('\n').split('|')[0:-1]
    #print(spectrum_list)

    spectrum_list = [parse_spectrum_seg1(seg1) for seg1 in spectrum_list]
    return spectrum_list

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
    #atom bonds 连续排列，遇到不是的，就跳出
    while True:
        line1 = lines[idx_line]
        list_line1 = parse_row_list_space(line1)
        size_line = len(list_line1)
        #print(size_line, line1)
        if size_line == 16:
            atom1 = [*[float(x) for x in list_line1[0:3]], int(list_line1[9]), list_line1[3]]
            atoms.append(atom1)
        elif size_line == 7:
            bonds.append([int(x) for x in list_line1[0:4]])
        else:
            break
        idx_line += 1


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
            dict_spectrum[id_s] = spectrum1
        else:
            #不处理
            pass
        idx_line += 1

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
    #plt.savefig(f'dist_num_atom_nmrshift_db2_max{num_atoms_max}.png')

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



if __name__ == '__main__':
    fname_abs = 'id_ABS.yml'
    with open('./data/nmrshiftdb2withsignals.sd', 'r', encoding='utf-8') as file:
        id_lines = split_sd2_by_nmrshiftdb2(file)

    #save_ABS(fname_abs:str, id_lines)
    #id_lines = load_ABS(fname_abs)
    print(f'共 {len(id_lines)}个结构式')

    #每个原子数量 统计最大值和分布
    id_num_atoms = get_num_atom(id_lines)

    #每个原子 碳谱线最小间距
    diff, shifts = get_min_shift_spectrum(id_lines)

    plot_shift(shifts)
    #0.01精细
    diff01 = [x for x in diff if x <= 1]
    plot_diff_2shift(diff01, bins=100, ranges=np.arange(0.0, 1.0, 0.05))
    #整体
    plot_diff_2shift(diff, bins=500, ranges=None)

    #plt.show()

        # for record1 in id_lines:
        #     for line in record1['atoms']:
        #         s = yaml.dump(line, default_flow_style=True)
        #         print(s)
            # yaml.dump(data, f, default_flow_style=False,\
            #                   encoding = 'utf-8', \
            #                   allow_unicode = True)