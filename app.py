import streamlit as st
import pandas as pd
import pickle
import requests

#Fetching the movies_list for VECTORIZATION
movies_dict = pickle.load(open('movie_dict_Vectorization.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity_Vectorization.pkl', 'rb'))

#Fetching the movies_list for KMEANS
movies_dict_kmeans = pickle.load(open('movie_dict_kmeans.pkl', 'rb'))
movies_kmeans = pd.DataFrame(movies_dict_kmeans)

#Fetching the movies_list for LSA
movies_dict_LSA = pickle.load(open('movie_dict_LSA.pkl', 'rb'))
movies_LSA = pd.DataFrame(movies_dict_LSA)
similarity_LSA = pickle.load(open('similarity_LSA.pkl', 'rb'))

#Fetch poster
def fetch_poster(movie_id):
  # Make the API request to get movie details
  response = requests.get(
    f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US')
  # Check if the request was successful (status code 200)
  if response.status_code == 200:
    data = response.json()
    # Check if 'poster_path' is available in the response
    if 'poster_path' in data and data['poster_path'] is not None:
      return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    else:
      # Return a default image if no poster is available
      return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"
  else:
    # If the request failed, return a default image
    return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

#Vectorization recommend movie
def recommend_movie_vectorization(movie_title):
  movie_index = movies[movies['title'] == movie_title].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
  recommended_movies_names = []
  recommended_movies_posters = []
  for i in movies_list:
    movie_id = movies.iloc[i[0]].movie_id
    recommended_movies_names.append(movies.iloc[i[0]].title)
    recommended_movies_posters.append(fetch_poster(movie_id))
  return recommended_movies_names, recommended_movies_posters

#KMeans recommend movie
def recommend_movie_kmeans(movie_title):
  movie_index = movies_kmeans[movies_kmeans['title'] == movie_title].index[0]
  cluster_label = movies_kmeans.loc[movie_index, 'cluster']
  # Get all movies in the same cluster and create a clean copy
  cluster_movies = movies_kmeans[movies_kmeans['cluster'] == cluster_label].copy()
  movie_tags = set(movies_kmeans.loc[movie_index, 'tags'].split())
  cluster_movies = cluster_movies.assign(
    similarity=cluster_movies['tags'].apply(
      lambda tags: len(movie_tags.intersection(set(tags.split()))) / len(movie_tags.union(set(tags.split())))
    )
  )
  recommendations = (
    cluster_movies.sort_values(by='similarity', ascending=False)
    .drop(index=movie_index)
    .head(5)
  )
  recommended_movies_names = []
  recommended_movies_posters = []
  for index, row in recommendations.iterrows():  # Iterate over rows
    movie_id = row['movie_id']
    recommended_movies_names.append(row['title'])
    recommended_movies_posters.append(fetch_poster(movie_id))
  return recommended_movies_names, recommended_movies_posters




#LSA recommend movie
def recommend_movie_LSA(movie_title):
  movie_index = movies_LSA[movies_LSA['title'] == movie_title].index[0]
  distances = similarity_LSA[movie_index]
  movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
  recommended_movies_names = []
  recommended_movies_posters = []
  for i in movies_list:
    movie_id = movies_LSA.iloc[i[0]].movie_id
    recommended_movies_names.append(movies_LSA.iloc[i[0]].title)
    recommended_movies_posters.append(fetch_poster(movie_id))
  return recommended_movies_names, recommended_movies_posters

#Title of the project
st.title("Movie Recommender System")

#Choose Model
selected_model = st.selectbox(
    'Select a model', ['Vectorization', 'KMeans', 'LSA [Ensemble]']
)

#Create option box
selected_movie_name = st.selectbox(
'Select a movie', movies['title'].values
)

recommended_movies_names, recommended_movies_posters = [], []

#Create button
if st.button('Recommend') :
  if selected_model == 'Vectorization':
    recommended_movies_names, recommended_movies_posters = recommend_movie_vectorization(selected_movie_name)

  elif selected_model == 'KMeans':
    recommended_movies_names, recommended_movies_posters = recommend_movie_kmeans(selected_movie_name)

  if selected_model == 'LSA [Ensemble]':
    recommended_movies_names, recommended_movies_posters = recommend_movie_LSA(selected_movie_name)

  col1, col2, col3, col4, col5 = st.columns(5)
  with col1:
    st.text(recommended_movies_names[0])
    st.image(recommended_movies_posters[0])
  with col2:
    st.text(recommended_movies_names[1])
    st.image(recommended_movies_posters[1])
  with col3:
    st.text(recommended_movies_names[2])
    st.image(recommended_movies_posters[2])
  with col4:
    st.text(recommended_movies_names[3])
    st.image(recommended_movies_posters[3])
  with col5:
    st.text(recommended_movies_names[4])
    st.image(recommended_movies_posters[4])
