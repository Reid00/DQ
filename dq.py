"""
对csv 数据进行dq 检测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class DQ:
    """
    对csv 文件做data quality 检测，数据的质量进行基本保证
    """
    def __init__(self,path):
        self.path = path
        self.data = pd.read_csv(self.path,sep=',',encoding='utf-8-sig')
        self.columns = self.data.columns
        self.name = self.path.stem
    
    def covert_emtpy_to_nan(self,param):
        """
        将['','','']存储多个空值的列表，转化为np.nan
        """
        if isinstance(param,str):
            if param.find("['")==0:
                param = eval(param)
            if set(param) == {''} or param =="[]":
                return np.nan
        else:
            if param=='':
                return np.nan
        return param 

    def clean_data(self):
        """
        数据清洗，去除数据里面的'',['']等 进行真实有效的coverage统计
        """
        cleaned_data = pd.DataFrame()
        for column in self.columns:
            cleaned_data[column]= self.data[column].apply(lambda x: self.covert_emtpy_to_nan(x))
        return cleaned_data
    
    def coverage(self,data=None):
        """
        显示各个属性的覆盖率
        """
        if data is None:
            data = self.data

        percentage = list()
        for column in self.columns:
            missing = data[column].isnull().sum()
            percentage.append(1 - round(missing / data.shape[0],2))
        coverage = pd.DataFrame({
            'column':self.columns,
            'percentage':percentage
        })
        coverage.to_csv(f'./dq/coverage_{self.name}.csv',index=False,encoding='utf8')

        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        # or 上面两句等价于 fig,ax = plt.subplots(figsize=(16,8))
        sns.barplot(x='column', y='percentage',data=coverage,ax=ax)
        # coverage.plot.bar(ax=ax)
        for p in ax.patches:        # 显示柱状图上的数值
            ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))

        ax.set_title('各个属性覆盖率', backgroundcolor='#3c7f99',fontsize=30, weight='bold',color='white')
        plt.box(False)
        ax.tick_params(labelsize=16,length=0)  #设置x,y tick(刻度) 的字体大小为16，刻度线长度为0
        # ax.legend(loc='best')
        # 设置y轴网格线
        ax.yaxis.grid(linewidth=0.5, color='black')
        ax.set_axisbelow(True) # 将网格线置于底部
        ax.set_xlabel('')      # x轴标签设置为空
        # x轴ticks 设置
        plt.xticks(rotation='45', fontsize=10,color='red')
        plt.subplots_adjust(bottom=0.2) #因为竖着字太长，生成图片中的x轴标签会被截取。因此设置距离底部0.2
        plt.savefig(f'./dq/coverage_{self.name}.png')
        plt.close()
        # plt.show()
        
    def value_counts_property(self,data=None):
        """
        显示每种属性重复值数目最多的前7个
        """
        if data is None:
            data = self.data

        for column in self.columns:
            # 如果数据的值每个都是唯一的则跳过
            if len(data[column].astype('str').unique()) == len(data):
                continue
            # colors=['red','green','yellow','tan','blue','violet','fuchsia']
            #####################
                # 实际数量 #
            ####################
            val_cnts = data[column].value_counts().head(7)
            fig = plt.figure(figsize=(16,7))
            fig.suptitle(f'{column}的value count的统计',backgroundcolor='#3c7f99',fontsize=30, weight='bold',color='red')
            ax1 = fig.add_subplot(121)
            val_cnts.plot.barh(ax=ax1)
            # sns.barplot(x=val_cnts.values, y=val_cnts.index,data=val_cnts)
            # sns.barplot(x=val_cnts.values, y=val_cnts.index.tolist(),ax=ax1)
            for i, v in enumerate(val_cnts.values): # 显示柱状图上的数值
                ax1.text(v,i, str(v), color='r', fontweight='bold')
            plt.box(False)
            # 设置value 的网格线
            ax1.xaxis.grid(linewidth=0.5, color='black')
            ax1.set_axisbelow(True)
            plt.yticks(rotation='45')
            ax1.set_title(f'{column}的value的数量')
            #####################
                # 百分比绘图 #
            ####################
            percentage = round(self.data[column].value_counts().head(7)/ len(self.data),2)
            ax2 = fig.add_subplot(122)
            percentage.plot.barh(ax=ax2) 
            # sns.barplot(x=percentage.values,y=percentage.index,ax=ax2)
            for i, v in enumerate(percentage.values): # 显示柱状图上的数值
                ax2.text(v,i, str(v), color='r', fontweight='bold')
            plt.box(False)
            # 设置value 的网格线
            ax2.xaxis.grid(linewidth=0.5, color='black')
            ax2.set_axisbelow(True)
            ax2.set_title(f'{column}的value百分比')  
            plt.yticks(rotation='45')
            plt.subplots_adjust(left=0.1)
            # plt.show()
            column = column.replace(':','')
            plt.savefig(f'./dq/values_{column}_{self.name}.png')
            val_cnts.to_csv(f'./dq/val_cnts_{column}_{self.name}.csv',encoding='utf-8')
            percentage.to_csv(f'./dq/val_per_{column}_{self.name}.csv',encoding='utf-8')


    def show_column_length(self,data=None):
        """
        显示每列字段的最大长度
        return columns_length
        """
        if data is None:
            data = self.data
        length = pd.DataFrame()
        for column in self.columns:
            length[column] = data[column].astype(str).str.len()  # 取出每个字段的长度生成一个dataframe

        top5 = pd.DataFrame()
        for column in self.columns:
            # 取出length数据表里面每个column 为单位长度前5的数据
            top5[column] = length.nlargest(5,column)[column].values
        top5.to_csv(f'./dq/columns_length_{self.name}.csv',index=False,encoding='utf-8')
        
        
def get_paths_from_folder(path,extension):
    """
    获取某个文件夹路径下的所有后缀为extension 的文件
    """
    folder_path = Path(path)
    file_names = folder_path.glob(f'*.{extension}')
    file_paths = [folder_path / file_name  for file_name in file_names]
    return file_paths

if __name__ == "__main__":
    # paths = get_paths_from_folder(r'E:\GitRepository\KG_Rebuild\ParseLiteratureXML\parsed_res','csv')
    # for path in paths:
    #     exploratory_data_analysis(path)
    path = Path(r'F:\Literature\test_import\pubmed20n0627.csv')
    dq = DQ(path)
    # data = dq.clean_data()
    # dq.value_counts_property(data)

    # dq.show_column_length()
    dq.coverage()
    dq.value_counts_property()
    
