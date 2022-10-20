import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from lang import lang_dict

#ページ設定
st.set_page_config(
page_title="Regression Analysis App", page_icon="📊")

if st.checkbox('日本語(Japanese)'):
    select_lang='ja'
else:
    select_lang='en'


#回帰の結果出力
def result_regress(Y,X,data):
    if log_trans:
        y_data=np.log(data[Y])
        x_data=np.log(data[X])
    else:
        y_data=data[Y]
        x_data=data[X]
    X_values=sm.add_constant(x_data)
    
    if robust:
        res=sm.RLM(y_data,X_values, M=sm.robust.norms.HuberT()).fit()
    else:
        res=sm.OLS(y_data,X_values).fit()
    return res

#相関行列表示
def show_heatmap(df):
    fig, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(df.corr(), annot=True, ax=ax,cmap='RdBu_r',square=True,vmax=1, vmin=-1, center=0)
    st.pyplot(fig)

#回帰式表示
def func():
    if log_trans:
        a=[f'log({box1})','=',round(result_regress(box1,box2,df).params[0],2)]
    else:
        a=[box1,'=',round(result_regress(box1,box2,df).params[0],2)]
    for i in range(1,len(box2)+1):
        if result_regress(box1,box2,df).params[i]>0:
            a.append("+")
            a.append(round(result_regress(box1,box2,df).params[i],2))
            if log_trans:
                a.append(f'log({box2[i-1]})')
            else:    
                a.append(box2[i-1])
    #a.append("+u")
    a=' '.join(map(str,a))
    return a


st.title(lang_dict[select_lang]['Streamlit Data Analysis --β Version--'])
st.header(lang_dict[select_lang]['Regression Analysis'])


uploaded_file=st.file_uploader(lang_dict[select_lang]["CSV file upload (please use organized data)"], type='csv')
use_example_file=st.checkbox(lang_dict[select_lang]['Use sample data : wooldridge(wage1)'],False,help="These are data from the 1976 Current Population Survey, collected by Henry Farber when he and I were colleagues at MIT in 1988.'https://rdrr.io/cran/wooldridge/man/wage1.html'")

if use_example_file:
    uploaded_file="wooldridge_wage1.csv"

if uploaded_file:
    df=pd.read_csv(uploaded_file,encoding="cp932")

    columns_list=list(df.columns)
    
    #データ表示
    if st.checkbox(lang_dict[select_lang]['Data preview']):
        st.dataframe(df,height=300)
        st.write(df.shape)

    st.sidebar.subheader(lang_dict[select_lang]['Select variable for analysis\n(Numerical data only)'])


    #被説明変数を選ぶ
    box1=st.sidebar.selectbox(lang_dict[select_lang]["explained variable"],columns_list)

    #説明変数を選ぶ
    box2=st.sidebar.multiselect(lang_dict[select_lang]["Explanatory variable"],columns_list)
    st.sidebar.write(lang_dict[select_lang]['Number of parameters : ']+str(len(box2)))

    robust=st.sidebar.checkbox(lang_dict[select_lang]["use robust standard errors"])
    log_trans=st.sidebar.checkbox(lang_dict[select_lang]["Logarithmic transfomation"])

    st.subheader(lang_dict[select_lang]['Result'])

    if box2:
        try:
            if st.checkbox(lang_dict[select_lang]['Correlation coefficient matrix display']):
                show_heatmap(df[box2])

            st.subheader(func())

            if st.checkbox(lang_dict[select_lang]['Estimation Result Details']):
                st.write(result_regress(box1,box2,df).summary())
                if robust:
                    st.info(lang_dict[select_lang]["Using robust standard errors"])
                st.write(lang_dict[select_lang]["About statsmodels:\nhttps://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html"])
        except:
            st.warning(lang_dict[select_lang]["Valid only for numerical data or Missing values and zeros cannot be logarithmized"])

    else:
        st.sidebar.info(lang_dict[select_lang]['Please enter variables'])
else:
    st.info(lang_dict[select_lang]['Please upload your data.']) 

st.subheader(lang_dict[select_lang]["Feedback and Opinion"])
st.write("https://forms.gle/zEdDaNrSQ4NpPcdi8")
