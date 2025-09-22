import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# --- Part 1 & 2: Data Loading, Cleaning, and Preparation ---

# Load the data and cache it for efficiency.
# The `st.cache_data` decorator tells Streamlit to run the function
# only once and store the result in a local cache.
@st.cache_data
def load_and_prepare_data(url="metadata.csv"):
    """
    Loads, cleans, and prepares the CORD-19 metadata.csv file.
    """
    try:
        df = pd.read_csv(url)
    except FileNotFoundError:
        st.error("metadata.csv not found. Please download it from the CORD-19 dataset and place it in the same directory.")
        return pd.DataFrame() # Return an empty DataFrame to avoid errors

    # Handle missing data
    # Drop rows where 'publish_time' is missing as it's crucial for time-based analysis.
    df_cleaned = df.dropna(subset=['publish_time']).copy()
    # Fill missing 'abstract' values with a placeholder.
    df_cleaned['abstract'] = df_cleaned['abstract'].fillna('No abstract provided.')
    # Fill missing 'journal' values with a placeholder.
    df_cleaned['journal'] = df_cleaned['journal'].fillna('Unknown Journal')
    
    # Prepare data for analysis
    # Convert 'publish_time' to datetime format. The 'coerce' error handling
    # will turn any unparseable dates into NaT (Not a Time), which we'll then drop.
    df_cleaned['publish_time'] = pd.to_datetime(df_cleaned['publish_time'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['publish_time'])
    
    # Extract the year from the publication date.
    df_cleaned['year'] = df_cleaned['publish_time'].dt.year
    
    return df_cleaned

# Load the data
df = load_and_prepare_data()

# Check if the DataFrame is empty before proceeding
if df.empty:
    st.stop()

# Get the min and max year from the cleaned data
min_year = int(df['year'].min())
max_year = int(df['year'].max())

# --- Part 4: Streamlit Application Layout ---

st.title("CORD-19 Research Paper Explorer 📊")
st.markdown("A simple data exploration of the COVID-19 Open Research Dataset (CORD-19) focusing on the metadata.")

st.sidebar.header("Filter Options")
st.sidebar.markdown("Use the controls below to customize the data displayed.")

# Interactive widgets
year_range = st.sidebar.slider(
    "Select a year range for analysis",
    min_value=min_year,
    max_value=max_year,
    value=(2020, max_year)
)

# Filter the DataFrame based on the selected year range
df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

# Display a brief summary of the filtered data
st.info(f"Showing **{len(df_filtered)}** papers published between **{year_range[0]}** and **{year_range[1]}**.")

# --- Part 3: Data Analysis and Visualization ---

st.header("Analysis & Visualizations")
st.markdown("Here are some key insights from the dataset within the selected time frame.")

# --- Plot 1: Publications Over Time ---
st.subheader("1. Publications by Year")
st.markdown("This bar chart shows the total number of papers published each year.")

year_counts = df_filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(year_counts.index.astype(int), year_counts.values, color='skyblue')
ax.set_title('Number of Publications by Year', fontsize=16)
ax.set_xlabel('Publication Year', fontsize=12)
ax.set_ylabel('Number of Papers', fontsize=12)
ax.set_xticks(year_counts.index.astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot(fig)


st.markdown("The chart clearly illustrates the rapid growth in COVID-19 research from 2020 onwards, as the pandemic unfolded.")

# --- Plot 2: Top Publishing Journals ---
st.subheader("2. Top Publishing Journals")
st.markdown("This chart highlights the journals that have published the most papers on the subject.")

top_journals = df_filtered['journal'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(12, 8))
top_journals.sort_values().plot(kind='barh', ax=ax, color='lightgreen')
ax.set_title('Top 10 Journals by Publication Count', fontsize=16)
ax.set_xlabel('Number of Publications', fontsize=12)
ax.set_ylabel('Journal', fontsize=12)
plt.tight_layout()
st.pyplot(fig)


st.markdown("We can see that a few key journals have been at the forefront of disseminating research during this period.")

# --- Plot 3: Word Cloud of Paper Titles ---
st.subheader("3. Most Frequent Words in Titles")
st.markdown("This word cloud visualizes the most common words found in the paper titles, giving a quick overview of the research topics.")

# Combine all titles into a single string
title_text = " ".join(title for title in df_filtered['title'].dropna())

# Create a set of custom stopwords to remove uninformative words
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['covid', '19', '2019', 'study', 'case', 'report', 'disease', 'new', 'clinical', 'data', 'sars', 'cov', '2'])

if title_text:
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        stopwords=custom_stopwords
    ).generate(title_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
else:
    st.warning("No titles found for the selected year range to generate a word cloud.")

st.markdown("The most prominent words often represent the central themes of the research, such as 'virus', 'pandemic', and 'patients'.")

# --- Part 5: Sample of the Data ---
st.header("Sample Data")
st.markdown("Below is a sample of the cleaned data to give you a sense of its structure.")
st.dataframe(df_filtered[['title', 'authors', 'journal', 'year', 'abstract']].head(10))

# --- Part 6: Documentation and Reflection ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This application was built as a data science project to demonstrate "
    "the end-to-end workflow using **pandas** for data manipulation, "
    "**matplotlib** and **wordcloud** for visualization, and **Streamlit** "
    "to create an interactive web application. It showcases basic exploratory "
    "data analysis (EDA) techniques on a real-world dataset."
)