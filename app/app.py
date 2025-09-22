import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

@st.cache_data
def load_data():
    df = pd.read_csv("data/metadata_clean.csv")
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors="coerce")
    df['year'] = df['publish_time'].dt.year
    return df

df = load_data()

st.title("CORD-19 Data Explorer")
st.write("A simple interactive app for exploring COVID-19 research metadata.")

year_range = st.slider("Select year range", 2019, 2022, (2020, 2021))
filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.subheader("Publications by Year")
year_counts = filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax)
st.pyplot(fig)

st.subheader("Top Journals")
top_journals = filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax)
st.pyplot(fig)

st.subheader("Word Cloud of Titles")
text = " ".join(filtered['title'].dropna())
wc = WordCloud(width=800, height=400).generate(text)
fig, ax = plt.subplots()
ax.imshow(wc, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

st.subheader("Sample Data")
st.dataframe(filtered.head(20))
