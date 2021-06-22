# 全球疫情及疫苗接种进度可视化之二--新冠疫情形势气泡图

- [安装plotly库](#安装plotly库)
- [全球疫情形势](#全球疫情形势)
- [定义工具函数](#定义工具函数)
    - [抽取数据](#抽取数据)
    - [绘制动态图表](#绘制动态图表)
    - [重抽样](#重抽样)
- [数据抽取、整理与可视化展示](#数据抽取、整理与可视化展示)
    - [抽取原始数据](#抽取原始数据)
    - [按周重抽样](#按周重抽样)
- [气泡图可视化](#气泡图可视化)
- [气泡图进阶](#气泡图进阶)

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
## 全球疫情形势
分析2020年以来、全球感染人数、死亡人数、治愈人数的情况，由于涉及时间序列数据，因此拟采用plotly库中动态图表的方式进行直观展示。


## 定义工具函数
### 抽取数据
```python
#抽取数据
def fetch_data(file):
    df=pd.read_csv(file)# 用pd.read_csv读取csv文件
    #由于美国等国数据是按二级行政区划提供的，需按国家进行汇总,并且去掉了Lat,Long两列
    result=df.groupby(['Country/Region']).sum().drop(['Lat','Long'],axis=1).stack()
    # 重新定义索引
    result=result.reset_index()
    result.columns=['Country','date','value']
    result['date_']=pd.to_datetime(result.date)#生成时间索引
    result=result.sort_values(by='date_',axis=0,ascending=True)#按时间排序
    result=result.replace('\*','',regex=True)#有些国家名字中有*，去掉国名中的*
    return result
```
### 绘制动态图表
```python
#绘制动态图表
def draw_data(df, label,color):
    fig=px.choropleth(df,
                    locations='Country',#选择城市为坐标
                    locationmode='country names',
                    animation_frame='date',#以时间为轴
                    color='value',#颜色变化选择人数
                    color_continuous_scale=[[0, 'White'],[1, color]],#按提供的颜色作为最大值的颜色color
                    labels={'value':label},#按提供的label绘制图例
                    range_color=[df.value.min(), df.value.max()])#按全程数据的最大值、最小值进行绘制，不采用autoscale
    fig.show()
```
### 重抽样

```python
#重抽样  源数据为按天展示的数据，为减少数据展示的计算量，需重抽样为周或月
def resample(df,period):
    country_list=df.Country.drop_duplicates()#计算城市,得到城市的列表
    temp=df.copy()
    result=pd.DataFrame()
    for i in country_list: #按国家分别进行重抽样，并合并数据
        r_temp=temp.loc[temp.Country==i]#选择对应的城市的行
        r_temp=r_temp.drop_duplicates(['date_'])#对数据去重
        r_temp=r_temp.set_index('date_')
        r_temp=r_temp.resample(period).asfreq().dropna()#重采样并且删除缺失值
        r_temp=r_temp.reset_index()
        result=pd.concat([result,r_temp])
    return result.sort_values(by='date_',axis=0,ascending=True)
```
## 数据抽取、整理与可视化展示
### 抽取原始数据
```python
#抽取原始数据
confirmed=fetch_data(r'data/time_series_covid19_confirmed_global.csv')
recovered=fetch_data(r'data/time_series_covid19_recovered_global.csv')
deaths=fetch_data(r'data/time_series_covid19_deaths_global.csv')
```
### 按周重抽样
```python
#按周重抽样
confirmed=resample(confirmed,'W')
recovered=resample(recovered,'W')
deaths=resample(deaths,'W')
```
## 气泡图可视化
以下汇总确诊病例、治愈病例、死亡病例进行统一展示，拟采用地理气泡图，在地图上同时展示三种数据，并比较期相对趋势。

```python
#气泡中心点地理数据
geodata=pd.read_csv(r'data/time_series_covid19_confirmed_global.csv',
                    usecols=['Country/Region','Lat','Long']).drop_duplicates(subset=['Country/Region'])
geodata
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509195016141.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)
我们可以得到geodata，在这个数据中有我们的经纬度的位置，也就是我们气泡的位置

```python
#连接各国的气泡坐标 并去掉County/Region这一列
#pd.merge以左边的索引为主
confirmed_loc=pd.merge(confirmed,geodata,how='left',left_on='Country',right_on='Country/Region').drop('Country/Region',axis=1)
recovered_loc=pd.merge(recovered,geodata,how='left',left_on='Country',right_on='Country/Region').drop('Country/Region',axis=1)
deaths_loc=pd.merge(deaths,geodata,how='left',left_on='Country',right_on='Country/Region').drop('Country/Region',axis=1)
```
有关pd.merge的解释可以去这里看[【python】详解pandas库的pd.merge函数](https://blog.csdn.net/brucewong0516/article/details/82707492)，简单来说，顾名思义，就是一个合并函数啦，只不过策略不同
```python
#标记3类数据
confirmed_loc['case_type']='confirmed'
recovered_loc['case_type']='recovered'
deaths_loc['case_type']='deaths'
```

```python
#组合数据
total_loc=pd.concat([confirmed_loc,recovered_loc,deaths_loc])
total_loc
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210509195145373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTUwODI2NQ==,size_16,color_FFFFFF,t_70)

```python
#绘制气泡图
fig=px.scatter_geo(total_loc,
                    lat='Lat',#纬度位置
                    lon='Long',#经度位置
                    size='value',#以value作为气泡的大小
                    size_max=50,#气泡最大是50
                    animation_frame='date',#随着时间变换
                    color='case_type',#颜色依据case_type
                    color_discrete_sequence=['Red','Green','Black'],
                    hover_name='Country')
fig.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514161507694.gif)

我们可以得到以上的气泡图，我们可以比较气泡的大小来判断，三种情况，死亡，痊愈，确诊的情况，由上图可以发现，3类数据基本均与确诊病例数呈正相关，但就治愈人数与确诊人数比例而言，美国、加勒比海各国、欧洲各国表现较差，而印度、俄罗斯、巴西、阿根廷相对较好。

## 气泡图进阶
为了更清楚地展示数据，我们换一种方式来绘制气泡图：
- 横坐标为治愈人数
- 纵坐标为死亡人数
- 气泡大小为确诊人数

```python
#整理数据宽表
#得到新的数据表
total_loc_combined=total_loc.set_index(['date_','Country','case_type']).unstack().droplevel(0,axis=1).iloc[:,[0,3,4,5,6,9]]
total_loc_combined=total_loc_combined.reset_index()
#重新设置索引
total_loc_combined.columns=['date_','Country','date','confirmed','deaths','recovered','Lat','Long']
total_loc_combined
```
如果不太明白droplevel()的可以看看[Python pandas.DataFrame.droplevel函数方法的使用](https://www.cjavapy.com/article/433/)，其实只是去掉多了一个索引，因为在我们unstack以后会出现多个行索引
```python
#绘制气泡图：
fig=px.scatter(total_loc_combined,
                x='recovered',#x为治愈人数
                y='deaths',#y为死亡人数
                size='confirmed',#气泡的大小由确诊人数定
                size_max=50,
                color='Country', 
                animation_frame='date',
                hover_name='Country')

#设置x和y的范围，以期能较好表示更多的数据
fig.update_layout(xaxis={'range':[-1500000,total_loc_combined.recovered.max()*1.2]},
                yaxis={'range':[-20000,total_loc_combined.deaths.max()*1.2]})
fig.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514161522139.gif)



同样可以看到我们之前发现的结论：

- 各个国家大致分布在一个向右上方倾斜的直线上，且越向右上，气泡越大，表示对大多数国家而言，确诊人数、治愈人数、死亡人数呈一定比例
- 印度、巴西、俄罗斯、土耳其等国分布在直线以下，表示其治愈人数较多，死亡较少
- 美国、欧洲各国分布在直线以上，表示其治愈人数较少，死亡较多
