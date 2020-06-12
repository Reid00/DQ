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

        coverage = pd.DataFrame()
        for column in self.columns:
            missing = data[column].isnull().sum()
            percentage = 1 - round(missing / data.shape[0],2)
            coverage[column] = [percentage]
        coverage.to_csv(f'./dq/coverage_{self.name}.csv',index=False,encoding='utf8')

        fig = plt.figure(figsize=(16,7))
        ax = fig.add_subplot(111)
        coverage.plot.bar(ax=ax)
        for p in ax.patches:        # 显示柱状图上的数值
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.set_title('各个属性覆盖率')
        ax.legend(loc='best')
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
            fig.suptitle(f'{column}的value的统计')
            ax1 = fig.add_subplot(121)
            val_cnts.plot.barh(ax=ax1)
            for i, v in enumerate(val_cnts.values): # 显示柱状图上的数值
                ax1.text(v,i, str(v), color='r', fontweight='bold')
            ax1.set_title(f'{column}的value的数量')
            #####################
                # 百分比 #
            ####################
            percentage = round(self.data[column].value_counts().head(7)/ len(self.data),2)
            ax2 = fig.add_subplot(122)
            percentage.plot.barh(ax=ax2) 
            for i, v in enumerate(percentage.values): # 显示柱状图上的数值
                ax2.text(v,i, str(v), color='r', fontweight='bold')
            ax2.set_title(f'{column}的value百分比')  
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
    path = Path(r'E:\GitRepository\KG_Rebuild\ParseLiteratureXML\parsed_res\pubmed20n1015.csv')
    dq = DQ(path)
    data = dq.clean_data()
    # dq.value_counts_property(data)

    # dq.show_column_length()
    dq.coverage()
    dq.value_counts_property()
    