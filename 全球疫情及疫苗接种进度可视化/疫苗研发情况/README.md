# 全球疫情及疫苗接种进度可视化三--疫苗研发情况

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

## 疫苗研发情况
## 各国采用的疫苗品牌概览
通过对各国卫生部门确认备案的疫苗品牌，展示各厂商的疫苗在全球的分布

```python
#读取数据
locations=pd.read_csv(r'data/locations.csv')
```

```python
locations
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514162414444.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
>这里我们的loacation中可以看到各个地方的疫苗和数据的来源与数据来源的网页

### 数据处理

```python
#发现数据中vaccines列中包含了多个品牌的情况，将这类数拆为多条
vaccines_by_country=pd.DataFrame()
for i in locations.iterrows():
    df=pd.DataFrame({'Country':i[1].location,'vaccines':i[1].vaccines.split(',')})
    vaccines_by_country=pd.concat([vaccines_by_country,df])
vaccines_by_country['vaccines']=vaccines_by_country.vaccines.str.strip()# 去掉空格
```

```python
vaccines_by_country.vaccines.unique() # 查看疫苗的种类
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514162822212.png)
### 可视化疫苗的分布情况
```python
#绘图
fig=px.choropleth(vaccines_by_country,
                locations='Country',
                locationmode='country names',
                color='vaccines',
                facet_col='vaccines',
                facet_col_wrap=3)
fig.update_layout(width=1200, height=1000)
fig.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514163034103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
各品牌分布：
- Pfizer/BioNTech 主要分布于北美，南美的智利、厄瓜多尔，欧洲、沙特
- Sputnik V 主要分布于俄罗斯、伊朗、巴基斯坦、非洲的阿尔及利亚以及南美的玻利维亚、阿根廷
- Oxford/AstraZeneca 主要分布于欧洲、南亚、巴西
- Moderna 主要分布在北美和欧洲
- Sinopharm/Beijing 主要分布在中国、北非部分国家和南美的秘鲁
- Sinovac 主要分布在中国、南亚、土耳其和南美
- Sinopharm/Wuhan 主要仅分布于中国
- Covaxin 主要分布于印度

综上可以发现，全球采用最广的仍是Pfizer/BioNTech，国产疫苗中Sinovac(北京科兴疫苗)输出到了较多国家

## 各品牌疫苗上市情况（仅部分国家）
根据数据集中提供的部分国家20年12月以来各品牌疫苗接种情况，分析各品牌上市时间及市场占有情况

```python
#读取数据
vacc_by_manu=pd.read_csv(r'data/vaccinations-by-manufacturer.csv')
```

```python
#定义函数，用于从原始数据中组织宽表
def query(df,country,date,vaccine):
    try:
        result=df.loc[(df.location==country)&(df.date==date)&(df.vaccine==vaccine)].total_vaccinations.iloc[0]
    except:
        result=np.nan
    return result
```

```python
vacc_by_manu
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514163206215.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
### 组织宽表
```python
#组织宽表
vacc_combined=pd.DataFrame(columns=['location','date','Pfizer/BioNTech', 'Sinovac', 'Moderna', 'Oxford/AstraZeneca'])
for i in vacc_by_manu.location.unique():
    for j in vacc_by_manu.date.unique():
        for z in vacc_by_manu.vaccine.unique():
            result=query(vacc_by_manu,i,j,z)
            if vacc_combined.loc[(vacc_combined.location==i)&(vacc_combined.date==j)].empty:
                result_df=pd.DataFrame({'location':i,'date':j,z:result},index=['new'])
                vacc_combined=pd.concat([vacc_combined,result_df])
            else:
                vacc_combined.loc[(vacc_combined.location==i)&(vacc_combined.date==j),z]=result
```

```python
vacc_combined
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514163225416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
### 补全缺失数据
```python
#补全缺失数据
temp=pd.DataFrame()
for i in vacc_combined.location.unique():#按国家进行不全
    r=vacc_combined.loc[vacc_combined.location==i]
    r=r.fillna(method='ffill',axis=0)#先按最近一次的数据进行补全
    temp=pd.concat([temp,r])#若没有最近的数据，认为该项为0
temp=temp.fillna(0).reset_index(drop=True)
temp
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021051416324172.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
### 绘制堆叠柱状图
```python
#绘制堆叠柱状图
fig=px.bar(temp,
        x='location',
        y=vacc_by_manu.vaccine.unique(),
        animation_frame='date',
        color_discrete_sequence=['#636efa','#19d3f3','#ab63fa','#00cc96']#为了查看方便，品牌颜色与前一部分对应
        )
fig.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514163632662.gif)

数据中主要涉及Pfizer/BioNTech、Sinovac、Moderna、Oxford/AstraZeneca 4个品牌，其中：
- Pfizer/BioNTech 上市时间最早，20年12月24日时即已经开始在智利接种了，之后在12月底开始在欧洲接种，21年1月12日开始在美国接种
- Sinovac 21年2月2日开始在智利接种
- Moderna  21年1月8日先在意大利开始接种，随后12日即开始在美国大量接种，最终在欧洲及美国均大量接种
- Oxford/AstraZeneca 21年2月2日先在意大利开始接种，随后即在欧洲开始接种
- 整体上看，Pfizer/BioNTech上市最早，且在全球占有份额最大，Moderna 随后上市，主要占据美国和欧洲市场，Sinovac、Oxford/AstraZeneca上市均较晚，其中Sinovac占据了智利的大部分市场份额，而Oxford/AstraZeneca主要分布于欧洲，且占份额很小