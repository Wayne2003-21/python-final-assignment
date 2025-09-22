import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide"
)

# Load the cleaned data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('metadata.csv')
        
        # Create a cleaned version
        df_cleaned = df.dropna(subset=['title', 'publish_time']).copy()
        df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
        df_cleaned['year'] = df_cleaned['publish_time'].dt.year
        df_cleaned['abstract'] = df_cleaned['abstract'].fillna('')
        df_cleaned['journal'] = df_cleaned['journal'].fillna('Unknown')
        
        # Create new columns
        df_cleaned['abstract_word_count'] = df_cleaned['abstract'].apply(lambda x: len(str(x).split()))
        df_cleaned['title_word_count'] = df_cleaned['title'].apply(lambda x: len(str(x).split()))
        
        return df_cleaned
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df_cleaned = load_data()

# Check if data loaded successfully
if df_cleaned.empty:
    st.error("Failed to load data. Please make sure metadata.csv is in the same directory.")
    st.stop()

# Streamlit app layout
st.title("📊 CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers metadata")

# Add interactive elements in sidebar
st.sidebar.header("Filters")

# Year range filter
min_year = int(df_cleaned['year'].min())
max_year = int(df_cleaned['year'].max())
year_range = st.sidebar.slider(
    "Select year range", 
    min_year, 
    max_year, 
    (min_year, max_year)
)

# Journal filter
journal_options = ['All'] + list(df_cleaned['journal'].value_counts().head(20).index)
selected_journal = st.sidebar.selectbox("Select journal", journal_options)

# Filter data based on selections
filtered_df = df_cleaned[
    (df_cleaned['year'] >= year_range[0]) & 
    (df_cleaned['year'] <= year_range[1])
]

if selected_journal != 'All':
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

# Display basic stats
st.header("Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Papers", len(filtered_df))
col2.metric("Date Range", f"{year_range[0]} - {year_range[1]}")
if selected_journal != 'All':
    col3.metric("Selected Journal", selected_journal)
else:
    col3.metric("Journals", f"{filtered_df['journal'].nunique()} total")

# Show a sample of the data
if st.checkbox("Show sample data"):
    st.subheader("Sample Data")
    st.dataframe(filtered_df[['title', 'journal', 'year', 'publish_time']].head(10))

# Create visualizations
st.header("Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Publications Over Time", "Top Journals", "Word Analysis", "Data Summary"])

with tab1:
    # Publications by year
    st.subheader("Publications by Year")
    year_counts = filtered_df['year'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(year_counts.index, year_counts.values, color='skyblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.set_title('Number of Publications by Year')
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    # Top journals
    st.subheader("Top Journals")
    top_journals = filtered_df['journal'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_journals)), top_journals.values, color='lightgreen')
    ax.set_yticks(range(len(top_journals)))
    ax.set_yticklabels(top_journals.index)
    ax.set_xlabel('Count')
    ax.set_title('Top 10 Journals by Publication Count')
    st.pyplot(fig)

with tab3:
    # Word cloud
    st.subheader("Word Cloud of Titles")
    if not filtered_df['title'].empty:
        all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Paper Titles')
        st.pyplot(fig)
    else:
        st.write("No titles available for word cloud.")

    # Most frequent words
    st.subheader("Most Frequent Words in Titles")
    def get_word_frequencies(text_series, n=20):
        all_text = ' '.join(text_series.dropna().astype(str).str.lower())
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
        word_freq = pd.Series(words).value_counts()
        stop_words = ['the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 'as', 'an', 'at', 'from', 
                     'that', 'is', 'are', 'this', 'be', 'we', 'using', 'which', 'their', 'has', 'have', 'was']
        word_freq = word_freq[~word_freq.index.isin(stop_words)]
        return word_freq.head(n)

    if not filtered_df['title'].empty:
        title_word_freq = get_word_frequencies(filtered_df['title'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(title_word_freq)), title_word_freq.values, color='salmon')
        ax.set_yticks(range(len(title_word_freq)))
        ax.set_yticklabels(title_word_freq.index)
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Words in Titles')
        st.pyplot(fig)
    else:
        st.write("No titles available for word frequency analysis.")

with tab4:
    # Data summary
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Abstract Length Distribution**")
        if not filtered_df.empty and 'abstract_word_count' in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(filtered_df['abstract_word_count'], bins=30, color='purple', alpha=0.7)
            ax.set_xlabel('Word Count')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Abstract Word Counts')
            st.pyplot(fig)
        else:
            st.write("Abstract word count data not available.")
    
    with col2:
        st.write("**Publication Sources**")
        if 'source_x' in filtered_df.columns and not filtered_df['source_x'].empty:
            source_counts = filtered_df['source_x'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
            ax.set_title('Top 10 Sources by Publication Count')
            st.pyplot(fig)
        else:
            st.write("Source data not available.")

# Add some metrics
st.sidebar.header("Metrics")
st.sidebar.write(f"**Total papers:** {len(filtered_df)}")
if not filtered_df.empty and 'abstract_word_count' in filtered_df.columns:
    st.sidebar.write(f"**Average abstract length:** {filtered_df['abstract_word_count'].mean():.1f} words")
if not filtered_df.empty and 'title_word_count' in filtered_df.columns:
    st.sidebar.write(f"**Average title length:** {filtered_df['title_word_count'].mean():.1f} words")
if not filtered_df.empty:
    st.sidebar.write(f"**Earliest publication:** {int(filtered_df['year'].min())}")
    st.sidebar.write(f"**Latest publication:** {int(filtered_df['year'].max())}")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app explores the CORD-19 dataset containing metadata about COVID-19 research papers. "
    "Use the filters to explore specific subsets of the data."
)