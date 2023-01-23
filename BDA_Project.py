# Porject By:
# Priyanka Marathe (826133692)
# Pooja Vaidya (826523588)

import pandas as pd
import plotly
import plotly.express as px
import matplotlib.pyplot as plot
import streamlit as st
import plotly.graph_objects as go
import altair as alt
import numpy as np
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

path=''
st.image('https://static.amazon.jobs/teams/53/images/IMDb_Header_Page.jpg?1501027252', caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
df = pd.read_csv (path+'imdb_top_1000.csv')

### Setting Navigation menu###
with st.sidebar:
    option = st.selectbox('Select your Choice!',('Explore Data', 'Make Prediction'))
    st.write('You selected:', option)
    if option=='Explore Data':
        selected=option_menu(menu_title='Explore Data',options=['Yearly Movies Count','Movie of the year','Successful movies of 2020','Progress of Alfred Hitchcock','Top Stars with Most Movies','Number of Movies by Genre','Movies by Certificate','Relation between Popularity and Revenue','Correlations of features','Top 10 Rated Movies','Top genres by certificate','Top 5 directors with most/least movies','Filter movies by duration','Tradeoff between Revenue and Features','Boxplots'])
    else:
        selected=option_menu(menu_title='Predict Rating',options=['Random Forest','Decision Tree'])
if option=='Explore Data':
    st.markdown("<h2 style='text-align: center; color: grey;'> Welcome to IMDB Data Analysis </h2>", unsafe_allow_html=True)

### Data Preprocessing ###
df_copy = df.copy()
df['Gross'] = df['Gross'].str.replace(',', '')
df=df.fillna(0)
df["Gross"]=df["Gross"].astype(int)
df['Runtime'] = df['Runtime'].str.split(' ').str[0].astype(int)
df.drop(df[df['Released_Year']== "PG"].index, inplace = True)
df['Released_Year'] = df['Released_Year'].astype(int)
df["Meta_score"]=df["Meta_score"].astype(int)

### Movie release count every year ###
if selected=='Yearly Movies Count':
    movie_df=df.groupby('Released_Year')['Series_Title'].count().reset_index().rename(columns={'Series_Title':'Movies_count'})
    movie_df_plot = px.scatter(movie_df, x="Released_Year", y="Movies_count",color='Movies_count',log_x=True, size_max=60,labels={
                     "Released_Year": "Released Year",
                     "Movies_count": "Movies Count"
                 })
    movie_df_plot.update_layout(
        font_family="Courier New",
        font_color="#0d5447"
    )
    st.markdown("<h3 style='text-align: center; color: black;'> Yearly Movies Count </h3>", unsafe_allow_html=True)
    st.plotly_chart(movie_df_plot, caption='Movie relese count by year')

### Hit Movie Every Year ###
movie_every_year = df.groupby(['Released_Year'])['IMDB_Rating'].transform(max) == df['IMDB_Rating']
hit_movies=df[movie_every_year]
if selected=='Movie of the year':
    years=[]
    years=hit_movies['Released_Year'].tolist()
    years.sort()
    value = st.select_slider('Select Year',options=years)
    hit_movies=hit_movies.set_index('Released_Year')
    if isinstance(hit_movies.at[value,'Poster_Link'], pd.Series)==True:
        for i, v in hit_movies.at[value,'Poster_Link'].items():
            st.image(v, width = 100)
    else:
            st.image(hit_movies.at[value,'Poster_Link'], width = 80)
    st.write('Movie(s) of the year is', hit_movies.at[value,'Series_Title'])

### Most successful movies of 2020 ###
df_2020 = df.loc[df['Released_Year']==2020]
df_temp=df_2020.sort_values(by='IMDB_Rating',ascending = False)
movies_successful=px.bar(df_temp,y='IMDB_Rating',x='Series_Title',labels={
                     "IMDB_Rating": "IMDB Rating",
                     "Series_Title": "Series Title"
                 },color='Series_Title',title='Most successful Movies of Year:2020')
movies_successful.update_layout(yaxis_range=[7,9],font_family="Courier New",font_color="#0d5447")
if selected=='Successful movies of 2020':
    st.markdown("<h3 style='text-align: center; color: black;'> Most successful movies of 2020 </h3>", unsafe_allow_html=True)
    st.plotly_chart(movies_successful, caption='Most successful movies of 2020')

### Track progress of Alfred Hitchcock ###
df['Released_Year'] = df['Released_Year'].astype(str)
director = df.loc[df['Director']=='Alfred Hitchcock']
director_line_plot = px.line(director, x="Released_Year", y="No_of_Votes", labels={
                     "Released_Year": "Released Year",
                     "No_of_Votes": "No. of Votes"
                 }, title='Popularity of Alfred Hitchcock Movies')
director_line_plot.update_layout(
    font_family="Courier New",
    font_color="#0d5447"
)
if selected=='Progress of Alfred Hitchcock':
    st.markdown("<h3 style='text-align: center; color: black;'> Track progress of Alfred Hitchcock </h3>", unsafe_allow_html=True)
    st.plotly_chart(director_line_plot, caption='Track progress of Alfred Hitchcock')
df['Released_Year'] = df['Released_Year'].astype(int)

### Top 10 stars with most movies ###
count_of_movies=df['Star1'].value_counts().reset_index()
top_10_stars=count_of_movies.rename(columns={'index':'Star_Name','Star1':'Count_of_Movies'})
top_10=pd.DataFrame()
top_10=top_10_stars.head(10)
pie_plot=px.pie(top_10, values='Count_of_Movies', names='Star_Name', title='Top Stars Movies Distribution')
pie_plot.update_layout(
    font_family="Courier New",
    font_color="#0d5447"
)
if selected=='Top Stars with Most Movies':
    st.markdown("<h3 style='text-align: center; color: black;'> Top 10 Stars with Most Movies </h3>", unsafe_allow_html=True)
    st.plotly_chart(pie_plot, caption='Top 10 stars with most movies')

### Group movies by genre ###
genres=pd.DataFrame()
genres['genre'] = df['Genre'].apply(lambda text: text.split(',')[0])
if selected=='Number of Movies by Genre':
    st.markdown("<h3 style='text-align: center; color: black;'> Number of Movies by Genre </h3>", unsafe_allow_html=True)
    genre_hist = px.histogram(genres, x="genre",labels={
                     "genre": "Genre"
                 })
    genre_hist.update_layout(
        font_family="Courier New",
        font_color="#0d5447"
    )
    st.plotly_chart(genre_hist, caption='Histogram of Movies by Genre')

### Count movies by certificate ###
count_by_Certificate=df['Certificate'].value_counts().reset_index()
certificate_Count=count_by_Certificate.rename(columns={'index':'Certificate_Type','Certificate':'Count_of_Movies'})
if selected=='Movies by Certificate':
    certificate_bar_plot=px.bar(certificate_Count,x='Count_of_Movies',y='Certificate_Type',labels={
                     "Count_of_Movies": "Count of Movies",
                     "Certificate_Type":"Certificate Type"
                 },color='Certificate_Type',title='Movies count by Genre')
    certificate_bar_plot.update_layout(
        font_family="Courier New",
        font_color="#0d5447"
    )
    st.markdown("<h3 style='text-align: center; color: black;'> Movies by Certificate </h3>", unsafe_allow_html=True)
    st.plotly_chart(certificate_bar_plot, caption='Movies by Certificate')

### Relation between Popularity and Revenue ###
if selected=='Relation between Popularity and Revenue':
    st.markdown("<h3 style='text-align: center; color: black;'> Relation between Popularity and Revenue </h3>", unsafe_allow_html=True)
    scatter_gross_rating = px.scatter(df, x="No_of_Votes", y="Gross",labels={
                     "No_of_Votes": "No. of Votes",
                     "Gross":"Gross"
                 })
    scatter_gross_rating.update_layout(
        font_family="Courier New",
        font_color="#0d5447"
    )
    st.plotly_chart(scatter_gross_rating)

### Correlations of features ###
if selected=='Correlations of features':
    st.markdown("<h3 style='text-align: center; color: black;'> Correlations of features </h3>", unsafe_allow_html=True)
    correlation_matrix = df.corr(method='pearson')
    heatmap_corr, ax = plot.subplots()
    sns.heatmap(correlation_matrix, 
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,ax=ax,annot=True)
    st.write(heatmap_corr)

### Top 10 Rated Movies ###
if selected=='Top 10 Rated Movies':
    st.markdown("<h3 style='text-align: center; color: black;'> Top 10 Rated Movies </h3>", unsafe_allow_html=True)
    top_ten_movies_df = pd.DataFrame()
    top_ten_movies_df["Movie Name"] = df.nlargest(10,['IMDB_Rating'])["Series_Title"]
    top_ten_movies_df["Rating"] = df.nlargest(10,['IMDB_Rating'])["IMDB_Rating"]
    top_10_movies = px.bar(top_ten_movies_df, x="Movie Name", y="Rating", color="Rating", range_y=[8,10])
    top_10_movies.update_layout(
        font_family="Courier New",
        font_color="#0d5447"
    )
    st.plotly_chart(top_10_movies)

### Top genres by certificate ###
if selected=='Top genres by certificate':
    st.markdown("<h3 style='text-align: center; color: black;'> Top genres by certificate </h3>", unsafe_allow_html=True)
    top_three_genre = pd.DataFrame()
    top_three_genre["Genre"] = df["Genre"]
    top_three_genre["Certificate"] = df["Certificate"]
    top_three_genre["No_of_Votes"] = df["No_of_Votes"]

    certificate_list =  set(df["Certificate"].tolist())
    selected_option = st.selectbox('Select Certificate', certificate_list)
    st.write('You selected:', selected_option)

    text = top_three_genre.set_index(['Certificate','No_of_Votes']).Genre.str.split(',', expand=True).stack().reset_index(['Certificate','No_of_Votes']).rename(columns={0:'Genre'})[['Certificate','Genre','No_of_Votes']].reset_index(drop=True)
    text_fil = text.loc[text['Certificate'] == selected_option]
    text_fil['Genre'] = text_fil['Genre'].str.strip()
    temp = text_fil.groupby('Genre',as_index=False).count()
    top_ten_genres = temp.nlargest(10,['No_of_Votes'])

    genre_chart = alt.Chart(top_ten_genres).mark_circle().encode(
        x='Genre', y='No_of_Votes',size='No_of_Votes',color=alt.Color('No_of_Votes', scale=alt.Scale(scheme='dark2')))
    st.altair_chart(genre_chart,use_container_width=True)

### Top 5 directors with most/least movies ###
if selected=='Top 5 directors with most/least movies':
    st.markdown("<h3 style='text-align: center; color: black;'> Top 5 directors with most/least movies </h3>", unsafe_allow_html=True)
    movies_count = df['Director'].value_counts().reset_index()

    selected_opt = st.radio("Select range",('Top 5', 'Bottom 5'))
    if selected_opt=='Top 5':
        result_df = movies_count.head(5)
    if selected_opt=='Bottom 5':
        result_df = movies_count.tail(5)
    labels = result_df["index"].tolist()
    values = result_df["Director"].tolist()
    director_movies_plot = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    st.plotly_chart(director_movies_plot)

### Filter movies by duration ###
if selected=='Filter movies by duration':
    st.markdown("<h3 style='text-align: center; color: black;'> Filter movies by duration </h3>", unsafe_allow_html=True)
    duration_list =  set(df["Runtime"].tolist())
    start_mins, end_mins = st.select_slider(
        'Select a range of duration(in mins) of the movie',
        options=duration_list,
        value=(min(duration_list), max(duration_list)))
    st.write('You selected duration between', start_mins, 'and', end_mins)

    filtered_df = df.loc[(df['Runtime'] >= start_mins) & (df['Runtime'] <= end_mins)]
    st.write('Movie(s) with selected duration are',filtered_df['Series_Title'])

### Trade off between revenue and features ###
if selected=='Tradeoff between Revenue and Features':
    st.markdown("<h3 style='text-align: center; color: black;'> Tradeoff between Revenue and Features </h3>", unsafe_allow_html=True)
    regression_plot, ax = plot.subplots(figsize=(5,3))
    p_color = dict(color="C0")
    l_color = dict(color="C1")

    ax=sns.regplot(x="Runtime", y="Gross", data=df, fit_reg=True, scatter_kws=p_color, line_kws=l_color)
    st.pyplot(regression_plot)

    regression_plot, ax = plot.subplots(figsize=(5,3))
    ax=sns.regplot(x="IMDB_Rating", y="Gross", data=df, fit_reg=True, scatter_kws=p_color, line_kws=l_color)
    st.pyplot(regression_plot)

    regression_plot, ax = plot.subplots(figsize=(20,6))
    ax=sns.boxplot(x="Meta_score", y="Gross", data=df)
    st.pyplot(regression_plot)

    regression_plot, ax = plot.subplots(figsize=(5,3))
    ax=sns.regplot(x="No_of_Votes", y="Gross", data=df, fit_reg=True, scatter_kws=p_color, line_kws=l_color)
    st.pyplot(regression_plot)

    regression_plot, ax = plot.subplots(figsize=(5,3))
    ax=sns.regplot(x="Released_Year", y="Gross", data=df, fit_reg=True, scatter_kws=p_color, line_kws=l_color)
    st.pyplot(regression_plot)

### Box Plots ###
if selected=='Boxplots':
    st.markdown("<h3 style='text-align: center; color: black;'> Boxplots </h3>", unsafe_allow_html=True)
    parameters=["Runtime","IMDB_Rating","Meta_score","No_of_Votes","Released_Year"]
    for param in parameters:
        boxplot, ax = plot.subplots(figsize=(5,1))
        ax=sns.boxplot(x=param,data=df)
        st.pyplot(boxplot)

### Data Predictions ###
bins = [7,7.5,8,8.5,9,9.5]
df["imdb_binned_score"]=pd.cut(df['IMDB_Rating'], bins=bins, right=True, labels=False)+1
X=pd.DataFrame(columns=['Meta_score','No_of_Votes','Gross','Runtime','Released_Year'],data=df)
y=pd.DataFrame(columns=['imdb_binned_score','Series_Title','IMDB_Rating'],data=df)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)

df_movie_actual=y_test.copy()
df_movie_actual=df_movie_actual.set_index('Series_Title')

y_test.drop('IMDB_Rating',axis=1,inplace=True)
y_test.drop('Series_Title',axis=1,inplace=True)
y_train.drop('Series_Title',axis=1,inplace=True)
y_train.drop('IMDB_Rating',axis=1,inplace=True)

if selected=='Random Forest':
    st.markdown("<h2 style='text-align: center; color: grey;'> Prediction by Random Forest </h2>", unsafe_allow_html=True)
    #menu options
    option = st.selectbox('Select Movie to Predict Rating',df_movie_actual.index.tolist())
    movies_list=[]
    movies_list=df_movie_actual.index.tolist()
    ind=movies_list.index(option)

    rfc = RandomForestClassifier(n_estimators = 200)
    rfc.fit(X_train, np.ravel(y_train,order='C'))
    rfcpred = rfc.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, rfcpred)
    acc=metrics.accuracy_score(y_test, rfcpred)
    val=rfcpred[ind]
    Result=pd.DataFrame()
    Result['Actual Rating']=''
    Result['Predicted Rating']=''
    Result = Result.append({'Actual Rating' : df_movie_actual.at[option,'IMDB_Rating'], 'Predicted Rating' : bins[val]},ignore_index=True)
    st.write(Result)
    st.write("Accuracy is:",acc)

if selected=='Decision Tree':
    st.markdown("<h2 style='text-align: center; color: grey;'> Prediction by Decision Tree </h2>", unsafe_allow_html=True)
    option = st.selectbox('Select Movie to Predict Rating',df_movie_actual.index.tolist())
    movies_list=[]
    movies_list=df_movie_actual.index.tolist()
    ind=movies_list.index(option)

    dtree = DecisionTreeClassifier(criterion='gini')
    dtree.fit(X_train, np.ravel(y_train,order='C'))
    dtreepred = dtree.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, dtreepred)
    acc=metrics.accuracy_score(y_test, dtreepred)
    val=dtreepred[ind]
    Result=pd.DataFrame()
    Result['Actual Rating']=''
    Result['Predicted Rating']=''
    Result = Result.append({'Actual Rating' : df_movie_actual.at[option,'IMDB_Rating'], 'Predicted Rating' : bins[val]},ignore_index=True)
    st.write(Result)
    st.write("Accuracy is:",acc)

