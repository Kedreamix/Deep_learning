# 全球疫情及疫苗接种进度可视化之四--各国疫苗接种进度

[toc]

全国疫情及疫苗接种进度可视化

- [全球疫情及疫苗接种进度可视化之一--全球疫情形势动态地图展示](https://blog.csdn.net/weixin_45508265/article/details/116521226) 
- [全球疫情及疫苗接种进度可视化之二--新冠疫情形势气泡图](https://blog.csdn.net/weixin_45508265/article/details/116568820)
- [全球疫情及疫苗接种进度可视化三--疫苗研发情况](https://blog.csdn.net/weixin_45508265/article/details/116795919)
- [全球疫情及疫苗接种进度可视化之四--各国疫苗接种进度](https://blog.csdn.net/weixin_45508265/article/details/117256608)

<font size=4>如果想了解更多有趣的项目和小玩意，都可以来我这里哦[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)

2020年底以来，欧美、印度、中国、俄罗斯等多国得制药公司纷纷推出了针对新冠肺炎的疫苗，这部分要分析了2020年以来全球疫情形势、各类疫苗在全球的地理分布、疫苗在各国的接种进度进行可视化展示，以期给读者提供当前疫情以及未来疫情防控的直观展示。

## 安装plotly库

因为这部分内容主要是用plotly库进行数据动态展示，所以要先安装**plotly库**

```python
pip install plotly
```

除此之外，我们对数据的处理还用了**numpy**和**pandas**库，如果你没有安装的话，可以用以下命令一行安装

```python
pip install plotly numpy pandas
```
```python
#导入所需库
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```
## 各国疫苗接种进度
### 读取数据
```python
#读取数据
vaccinations=pd.read_csv(r'./data/vaccinations.csv')
```
### 排除全球及大洲数据
我们需要把一些全球和大洲的数据去掉，因为我们是对国家进行分析，要不然会有重复
```python
#排除全球及大洲数据
vaccinations.index=vaccinations.location
vaccinations=vaccinations.drop(['World','Asia','North America', 'Europe','Africa', 'European Union', 'South America'])
vaccinations.reset_index(drop=True,inplace=True)
```

```python
vaccinations
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210525150912942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
我们可以看看现在我们的数据分布是这样的
### 发现某些国家数据不全，需按日期进行补全
```python
#发现某些国家数据不全，需按日期进行补全
for i in vaccinations.location.unique():#遍历国家
    for j in vaccinations.date.unique():#遍历日期
        if vaccinations.loc[(vaccinations.location==i)&(vaccinations.date==j)].empty:#如果该日期没有数据，新增一行空数据
            temp=pd.DataFrame({'location':i,'date':j},index=['new'])
            vaccinations=pd.concat([vaccinations,temp])
```

```python
vaccinations['date_']=pd.to_datetime(vaccinations.date)#创建时间序列
vaccinations=vaccinations.sort_values(by='date_')#按时间排序
vaccinations
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210525151827707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021052515183789.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

```python
temp1=pd.DataFrame()
for i in vaccinations.location.unique():#按国家补全数据
    r=vaccinations.loc[vaccinations.location==i]
    r=r.fillna(method='ffill',axis=0)#先按最近一次数据进行补全
    temp1=pd.concat([temp1,r])
temp1=temp1.fillna(0)#若仍有空值，认为是0
temp1
```
### 取每百人接种量最高的10个国家
```python
#为了减少可视化的数据量，保留总接种量及每百人接种量最高的10个国家
country1=list(temp1.reset_index(drop=True).groupby('location').total_vaccinations.max().sort_values(ascending=False).head(10).index)
country2=list(temp1.reset_index(drop=True).groupby('location').total_vaccinations_per_hundred.max().sort_values(ascending=False).head(10).index)
country1.extend(country2)
country_list=set(country1)
drop_list=list(set(temp1.location.unique())-country_list)
```
## 可视化绘制气泡图
绘制气泡图，横轴为总接种数，纵轴为接种比例，气泡大小为单日接种数

```python
#绘制气泡图
fig=px.scatter(temp1,
                x='total_vaccinations',
                y='total_vaccinations_per_hundred',
                size='daily_vaccinations',
                size_max=50,color='location', 
                animation_frame='date',
                hover_name='location')
fig.update_layout(xaxis={'range':[-1000000,temp1.total_vaccinations.max()*1.2]},
                yaxis={'range':[-10,temp1.total_vaccinations_per_hundred.max()*1.2]})
fig.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210525152453881.gif)



通过上图可以发现：
- 各国的疫苗接种主要开始于20年12月底
- 以色列、阿联酋等国开始接种后，虽然单日接种量不大，但相对其较少的总人口，很快就在接种进度上取得了领先，截止2021年2月16日，分别达到78%、52%
- 从单日接种人数上，中、美两国相对最大，但由于两国热口基数较大，接种进度均比较落后。而中国由于前期（主要是12月底）接种数量少、人口基数大的原因，导致无论是总接种量，还是接种进度，都落后于美国
- 以英国为代表的欧洲各国，虽然人口基数也不小，但归功于其对于疫苗的大力推动，其接种进度正在稳步推进中