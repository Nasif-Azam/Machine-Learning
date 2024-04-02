import pickle
import streamlit as st
import requests

movies = pickle.load(open('Models/movies.pkl', 'rb'))
similarity = pickle.load(open('Models/similarity.pkl', 'rb'))
movie_list = movies['title'].values


def recommend(selected_movie):
    movie_index = movies[movies['title'] == selected_movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:8]

    recommended_movies = []
    recommended_movies_poster = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # Fetch Poster From API
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_poster


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=5dc569a32ec52b70d17cea7fe0bc1dd3'.format(movie_id))
    data = response.json()
    return 'https://image.tmdb.org/t/p/original' + data['poster_path']


st.set_page_config(layout="wide")
st.header('Movie **:green[Recommendation]** System')
selected_movie_name = st.selectbox(
    'Type or select your **:green[preferable movie]** from the dropdown :sunglasses:',
    movie_list
)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.markdown(names[0])
        st.image(posters[0])
    with col2:
        st.markdown(names[1])
        st.image(posters[1])
    with col3:
        st.markdown(names[2])
        st.image(posters[2])
    with col4:
        st.markdown(names[3])
        st.image(posters[3])
    with col5:
        st.markdown(names[4])
        st.image(posters[4])
    with col6:
        st.markdown(names[5])
        st.image(posters[5])
    with col7:
        st.markdown(names[6])
        st.image(posters[6])
