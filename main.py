import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

#ページ設定
st.set_page_config(
page_title="回帰分析app", page_icon="📊")

#回帰の結果出力
def result_regress(Y,X,data):
    X_values=[]
    for i in range(len(X)):
        X_values.append(X[i])

    X_values=sm.add_constant(data[X_values])
    res=sm.OLS(data[Y],X_values).fit(cov_type="HC3",use_t=True)

    return res

#相関行列表示
def show_heatmap(df):
    fig, ax = plt.subplots(figsize=(12,9))
    sns.heatmap(df.corr(), annot=True, ax=ax,cmap='RdBu_r',square=True,vmax=1, vmin=-1, center=0)
    st.pyplot(fig)

#回帰式表示
def func():
    a=[box1,'=',round(result_regress(box1,box2,df).params[0],2)]
    for i in range(1,len(box2)+1):
        if result_regress(box1,box2,df).params[i]>0:
            a.append("+")
        a.append(round(result_regress(box1,box2,df).params[i],2))
        a.append(box2[i-1])
    a.append("+u")
    a=' '.join(map(str,a))
    return a


st.title('Streamlit データ分析')
st.header('回帰分析')


uploaded_file=st.file_uploader("csvファイルアップロード", type='csv')
use_example_file=st.checkbox('サンプルデータを使用 : wooldridge(wage1)',False,help="These are data from the 1976 Current Population Survey, collected by Henry Farber when he and I were colleagues at MIT in 1988. Data loads lazily.'https://rdrr.io/cran/wooldridge/man/wage1.html'")

if use_example_file:
    uploaded_file="wooldridge_wage1.csv"

if uploaded_file:
    df=pd.read_csv(uploaded_file,encoding="cp932")

    columns_list=list(df.columns)
    
    #データ表示
    if st.checkbox('Data preview'):
        st.dataframe(df,height=300)
        st.write(df.shape)

    st.sidebar.subheader('Select columns for analysis\n(数値データのみ)')


    #被説明変数を選ぶ
    box1=st.sidebar.selectbox("被説明変数",columns_list)

    #説明変数を選ぶ
    box2=st.sidebar.multiselect("説明変数(複数可)",columns_list)
    st.sidebar.write(f'number of explanatory variable : {len(box2)}')


    st.subheader('Result')

    if box2:
        try:
            if st.checkbox('相関行列表示'):
                show_heatmap(df[box2])
        

            st.subheader(func())

        ###st.subheader(f'N={len(df)} , Adj_R^2={result_regress(box1,box2,df).rsquared_adj:.3f}')

            if st.checkbox('detail'):
                st.write(result_regress(box1,box2,df).summary())
                st.info("頑健標準誤差を使用")
        else:
            st.write("数値データのみ")

    else:
        st.sidebar.info('変数を入力してください')
else:
    st.info('データをアップロードしてください。') 


        

