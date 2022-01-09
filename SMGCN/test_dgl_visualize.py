__author__ = 'dk'
#构建图，添加节点和边
import networkx as nx
import  dgl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


#构建星型图
u=[0,0,0,0,0]
v=[1,2,3,4,5]

#第一种方式,u和v的数组,他们是相同的长度
star1 = dgl.DGLGraph((u,v))
nx.draw(star1.to_networkx(),with_labels=True)#可视化图
plt.savefig("save/star1.png")
plt.close()
# plt.show()

# star2 = dgl.DGLGraph((0,v))
# #对于星型,是可以广播的
# nx.draw(star2.to_networkx(),with_labels=True)
# # plt.show()
# plt.savefig("save/star2.png")
# plt.close()

star3= dgl.DGLGraph([(0,1),(0,2),(0,3),(0,4),(0,5)])
#直接枚举
nx.draw(star3.to_networkx(),with_labels=True)
plt.savefig("save/star3.png")
plt.close()
# plt.show()
