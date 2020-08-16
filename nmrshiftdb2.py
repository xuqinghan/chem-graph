''' nmr shift db2 
    用于分子结构式 - 13C碳谱  encoder-decoder
    初步读取 .sd  -> dict yml
    
'''

import re
import u_file
#import yaml
from ruamel import yaml

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

def parse_spectrum_seg1(seg1):
    '''形如 18.3;0.0T;0 '''
    pos, peak, idx_c = seg1.split(';')
    return [float(pos), peak[-1], int(idx_c)]

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

if __name__ == '__main__':
    #line = 'nmrshiftdb2 234'
    with open('./data/nmrshiftdb2withsignals.sd', 'r', encoding='utf-8') as file:
        id_lines = split_sd2_by_nmrshiftdb2(file)
    print(f'共 {len(id_lines)}个结构式')
    #存储 id atom edge spectrum 
    #u_file.save_yml('id_AES.yml', id_lines)
    with open('id_ABS.yml', 'w', encoding="utf-8") as f:
        yaml.safe_dump(id_lines, f, indent=2, block_seq_indent=3, allow_unicode=True)
        # for record1 in id_lines:
        #     for line in record1['atoms']:
        #         s = yaml.dump(line, default_flow_style=True)
        #         print(s)
            # yaml.dump(data, f, default_flow_style=False,\
            #                   encoding = 'utf-8', \
            #                   allow_unicode = True)