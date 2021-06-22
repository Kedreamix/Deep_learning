#导入所需库
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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


#抽取原始数据
confirmed=fetch_data(r'data/time_series_covid19_confirmed_global.csv')
recovered=fetch_data(r'data/time_series_covid19_recovered_global.csv')
deaths=fetch_data(r'data/time_series_covid19_deaths_global.csv')

#按周重抽样
confirmed=resample(confirmed,'W')
recovered=resample(recovered,'W')
deaths=resample(deaths,'W')

#确诊病例
draw_data(confirmed,'确诊病例数','Red')

#治愈病例
draw_data(recovered,'治愈病例数','Green')

#死亡病例
draw_data(deaths,'死亡病例数','Black')
