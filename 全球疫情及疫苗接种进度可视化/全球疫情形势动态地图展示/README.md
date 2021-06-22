# 全球疫情及疫苗接种进度可视化之一--全球疫情形势动态地图展示)

- [安装plotly库](#安装plotly库)
- [全球疫情形势](#全球疫情形势)
- [定义工具函数](#定义工具函数)
    - [抽取数据](#抽取数据)
    - [绘制动态图表](#绘制动态图表)
    - [重抽样](#重抽样)
- [数据抽取、整理与可视化展示](#数据抽取、整理与可视化展示)
    - [抽取原始数据](#抽取原始数据)
    - [按周重抽样](#按周重抽样)
    - [确诊病例](#确诊病例)
    - [治愈病例](#治愈病例)
    - [死亡病例](#死亡病例)


全国疫情及疫苗接种进度可视化
- [全球疫情及疫苗接种进度可视化之一--全球疫情形势动态地图展示](https://blog.csdn.net/weixin_45508265/article/details/116521226) 
- [全球疫情及疫苗接种进度可视化之二--新冠疫情形势气泡图](https://blog.csdn.net/weixin_45508265/article/details/116568820)
- [全球疫情及疫苗接种进度可视化三--疫苗研发情况](https://blog.csdn.net/weixin_45508265/article/details/116795919)
- [全球疫情及疫苗接种进度可视化之四--各国疫苗接种进度](https://blog.csdn.net/weixin_45508265/article/details/117256608)



如果想了解更多有趣的项目和小玩意，都可以来我这里哦[通道](https://blog.csdn.net/weixin_45508265/article/details/114663239)

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
### 确诊病例

```python
#确诊病例
draw_data(confirmed,'确诊病例数','Red')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514160639575.gif)

>这是一个动态的图，可以看到每个时期的变化，这里我给出图片，详细可以实现代码观测

由上图可以看到：

- 美国疫情大规模爆发大致始于2020年4月初
- 随后是俄罗斯、印度、南美各国，于2020年6月后先后爆发了大规模疫情
- 最后是欧洲，在2020年底发生了疫情爆发
- 从全球目前累计感染人数来看，美国占绝大多数，另外印度、巴西、俄罗斯、欧洲也较多
### 治愈病例

```python
#治愈病例
draw_data(recovered,'治愈病例数','Green')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/202105141608506.gif)

由上图可以看到：

- 2020年6月中旬后，美国、巴西、印度、俄罗斯开始有较多患者被治愈
- 印度在2020年下半年有较多治愈病例
- 2020年12月底以后，美国治愈病例数据有缺失，造成图像失真(我打算找机会造点数据)

### 死亡病例

```python
#死亡病例
draw_data(deaths,'死亡病例数','Black')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210514161034812.gif)

由上图可以看到，死亡病例趋势与确诊病例大致相同

数据来源：
[COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)
