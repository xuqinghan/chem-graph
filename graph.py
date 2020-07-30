import networkx as nx
import matplotlib.pyplot as plt

def get_atom_num_dict(str_SMILES:str):
    '''有机物常见原子除去氢的原子数量: SMILES字符串计数'''
    return {name:str_SMILES.count(name) for name in ['C', 'O', 'N', 'S','P']}


if __name__ == '__main__':
    str_SMILES = 'O=C(C)CC=1C=CC=CC=1' #No.5
    atom_num_dict = get_atom_num_dict(str_SMILES)

    G = nx.Graph()                 #建立一个空的无向图G
    #每个原子命名: 每种原子 从1号开始 因为0和O不易区分
    atoms = [f'{name}{i+1}' for name, num in atom_num_dict.items() for i in range(num) ]
    G.add_nodes_from(atoms)
    #用edge 表示chemical bond
    #以str_SMILES的顺序 提取每个键,暂时手工,看SIMLES规则,以后可以自动parse
    #权值表示键: = 2  # 3 没有1
    edges =[('O1','C1', 2), #=2 权值表示键?
        ('C1', 'C2', 1), #遇到括号,说明是分支
        ('C1', 'C3', 1), #括号结束后的C
        ('C3', 'C4', 1), 
        ('C4', 'C5', 2), #C=1
        ('C5', 'C6', 1), 
        ('C6', 'C7', 2), 
        ('C7', 'C8', 1),
        ('C8', 'C9', 2),
        ('C9', 'C4', 1), 
        ] 
    G.add_weighted_edges_from(edges)

    #-----plot--nx的画图每次是随机的,其实应该按化学规则(键的长度)等等去画,也许有别的化学库可以计算长度,角度等等等-------
    #按权重(化学键) 1- 3 归类
    bonds = {num: [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == num] for num in range(1, 4)}

    #节点位置
    pos=nx.spring_layout(G) # positions for all nodes
    #首先画出节点位置
    # nodes
    nx.draw_networkx_nodes(G,pos,node_size=700)
    #根据权重，实线为权值大的边，虚线为权值小的边
    # edges
    nx.draw_networkx_edges(G,pos,edgelist=bonds[1],
                        width=1, edge_color='b',)
    nx.draw_networkx_edges(G,pos,edgelist=bonds[2],
                        width=12,alpha=0.5,edge_color='b')
    nx.draw_networkx_edges(G,pos,edgelist=bonds[3],
                        width=18,alpha=0.5,edge_color='b')
    # labels标签定义
    nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')
     
    plt.axis('off')
    plt.savefig("weighted_graph.png") # save as png

    #nx.draw(G, with_labels=True)

    plt.show()