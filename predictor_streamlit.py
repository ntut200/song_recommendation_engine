import matplotlib.pyplot as plt
from sqlalchemy import except_all, null
import streamlit as st
import pickle
import pandas as pd

# import required libraries for the classifier models
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sympy import separatevars

st.set_page_config(page_title=None, page_icon=None, layout="wide",
                   initial_sidebar_state="auto", menu_items=None)

st.header("Spotify Song Recommendation engine")
st.text('\n')
st.text('\n')


@st.cache(suppress_st_warning=True)
def knn_Decision_tree_refined(genre='Country'):
    feature_cols = ['Celebrities', 'Music', 'Slow songs or fast songs',
                    'Spending on healthy eating', 'Spending on looks', 'Small - big dogs',

                    'Life struggles', 'Getting angry', 'Funniness', 'Internet', 'Politics',
                    'Shopping', 'Cars', 'Countryside, outdoors', 'Adrenaline sports',
                    'Daily events', 'Friends versus money', 'Knowing the right people',
                    'Energy levels', 'Workaholism', 'Thinking ahead', 'Assertiveness']

    study_data = pd.read_csv('responses.csv')
    study_data = pd.get_dummies(data=study_data, columns=[
                                'Education', 'Gender', 'Village - town'], drop_first=True)

    study_data_random = study_data
    for k in range(0, len(study_data)):
        if study_data[genre].loc[k] < 3:
            study_data_random[genre].loc[k] = 0
        else:
            study_data_random[genre].loc[k] = 1

    study_data_random = study_data_random.select_dtypes(include='number')
    study_data_random.dropna(inplace=True)

    # Set up X and y
    X = study_data_random[feature_cols]
    y = study_data_random[genre]

    # Set up test and train data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,  random_state=42)

    # Scale the test features
    ss = StandardScaler()
    Z_train = ss.fit_transform(X_train)
    Z_test = ss.transform(X_test)
    Z = ss.transform(X)

    # Calculate TRAINING ERROR and TESTING ERROR for K=1 through 100.

    k_range = list(range(1, 101))
    training_error = []
    testing_error = []

    # Find test accuracy for all values of K between 1 and 100 (inclusive).
    for k in k_range:

        # Instantiate the model with the current K value.
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Z_train, y_train)

        # Calculate training error (error = 1 - accuracy).
        y_pred_class = knn.predict(Z)
        training_accuracy = metrics.accuracy_score(y, y_pred_class)
        training_error.append(1 - training_accuracy)

        # Calculate testing error.
        y_pred_class = knn.predict(Z_test)
        testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
        testing_error.append(1 - testing_accuracy)

    # Create a DataFrame of K, training error, and testing error.
    column_dict = {'K': k_range, 'training error': training_error,
                   'testing error': testing_error}
    df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)

    k_value = min(list(zip(testing_error, k_range)))[1]
    knn = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
    knn.fit(Z_train, y_train)
    model = knn
    st.write(genre, 'model test score is ', model.score(Z_test, y_test))
    return model


country_model = knn_Decision_tree_refined()
pop_model = knn_Decision_tree_refined('Pop')
rock_model = knn_Decision_tree_refined("Rock")
folk_model = knn_Decision_tree_refined('Folk')
alternative_model = knn_Decision_tree_refined('Alternative')
rap_model = knn_Decision_tree_refined('Hiphop, Rap')
punk_model = knn_Decision_tree_refined('Punk')
classical_model = knn_Decision_tree_refined('Classical music')
musical_model = knn_Decision_tree_refined('Musical')
dance_model = knn_Decision_tree_refined('Dance')

genre_df_merged = pd.read_csv('genre_df_merged.csv')

with st.expander("Expand to see a random selection of songs from the database"):
    st.write(genre_df_merged[['name', 'artists']].sample(30))

artist_choice = st.text_input(
    "Input an artist's name (spelling matters!) ").lower()
songs_list = genre_df_merged[genre_df_merged['artists'].str.contains(
    artist_choice)].reset_index()
# st.write(songs_list)
songs_list2 = songs_list.name.str.cat(sep=', ')
# st.write(songs_list2)
song_choice = st.selectbox("Select Song ", options=[songs_list2.strip()
                                                    for songs_list2 in songs_list2.split(", ")])


def get_cosim_user_inputs(df=genre_df_merged, n=200):

    # Get user inputs. must be spelled correctly but can be upper or lower case
    artist_request = artist_choice.lower()
    song_request = song_choice.lower()
    # st.text_input(
    #    "what artist would you like similar recommendations for? ", key='form_value')
    # if st.session_state.form_value:
    # st.write("Received input from user")
    #    artist_request = str(st.session_state.form_value).lower()
    # st.write("Your input value was", artist_request)

    # st.text_input(
    #    "what song would you like similar recommendations for ", key='form_value')

    # if st.session_state.form_value:
    # st.write("Received input from user")
    #    song_request = str(st.session_state.form_value).lower()
    # st.write("Your input value was", song_request)'''

    genre_request = st.selectbox(
        "Do you want the music to be in the same genre? ", options=('Yes', 'No')).lower()
    # if st.session_state.form_value:
    # st.write("Received input from user")
    #    genre_request = str(st.session_state.form_value).lower()
    # st.write("Your input value was", genre_request)

    try:
        # creates a DF of the full request (artist and song) from the million song dataset
        full_request = genre_df_merged[genre_df_merged['artists'].str.contains(
            artist_request) & genre_df_merged['name'].str.contains(song_request)].reset_index()
        # st.write(full_request)

    # Pulls out the song into an array for ease of use
        request2 = full_request.iloc[[0]]
        # Just the requested song ID
        song_id = request2.iat[0, 3]
        # Genre code of the requested song
        genre_code = request2.iat[0, 38]

        genre_df_merged3 = genre_df_merged[[
            'id', 'artists', 'name', 'genre_code']]
        df = genre_df_merged
        ss = StandardScaler()
        df2 = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'popularity', 'id']]
        df2.set_index('id', inplace=True)
        df_scaled = ss.fit_transform(df2)
        df = pd.DataFrame(data=df_scaled, index=df2.index)
        # st.write(song_id, genre_code)

        artist_names = []
        song_array = np.array(df.T[song_id]).reshape(1, -1)
        dataset_array = df.drop(index=song_id).values

        cosim_scores = cosine_similarity(song_array, dataset_array).flatten()

        song_names_array = df2.drop(index=song_id).index.values

        df_result = pd.DataFrame(
            data={
                'id': song_names_array,
                'CoSim': cosim_scores,

            }
        )

        df_result = df_result.sort_values(by='CoSim', ascending=False).head(n)
        merged_df = df_result.merge(genre_df_merged3, how='inner', on=['id'])
        # Little if statement that produces songs only in the same genre as the user input song, if the
        # user prefers
        if genre_request == "yes":
            st.write(
                merged_df.loc[merged_df['genre_code'] == genre_code].head(10))
        else:
            st.write(merged_df.reset_index(drop=True).head(10))
    except:
        st.write('waiting on user inputs')


get_cosim_user_inputs()


def get_personality():
    Celebrities = st.selectbox(
        'I enjoy learning about celebrities: Strongly disagree 1-2-3-4-5 Strongly agree ?',
        (1, 2, 3, 4, 5))
    # st.write('You selected:', Celebrities, type(Celebrities))
    # Celebrities = st.text_input(
    #    "Celebrities: Not interested 1-2-3-4-5 Very interested  ", key=int)
    music = st.selectbox(
        "I enjoy listening to music.: Strongly disagree 1-2-3-4-5 Strongly agree   ", (1, 2, 3, 4, 5))
    Slow_songs_or_fast_songs = st.selectbox(
        "I prefer.: Slow paced music 1-2-3-4-5 Fast paced music  ", (1, 2, 3, 4, 5))
    Spending_on_healthy_eating = st.selectbox(
        "I will hapilly pay more money for good, quality or healthy food.: Strongly disagree 1-2-3-4-5 Strongly agree  ", (1, 2, 3, 4, 5))
    spending_on_looks = st.selectbox(
        "I spend a lot of money on my appearance.: Strongly disagree 1-2-3-4-5 Strongly agree  ", (1, 2, 3, 4, 5))
    Small_big_dogs = st.selectbox(
        "I prefer big dangerous dogs to smaller, calmer dogs.: Strongly disagree 1-2-3-4-5 Strongly agree  ", (1, 2, 3, 4, 5))
    Life_struggles = st.selectbox(
        "I cry when I feel down or things don't go the right way from 1-5  ", (1, 2, 3, 4, 5))
    Getting_angry = st.selectbox(
        "I can get angry very easily from 1-5  ", (1, 2, 3, 4, 5))
    Funniness = st.selectbox(
        "I always try to be the funniest one from 1-5  ", (1, 2, 3, 4, 5))
    Internet = st.selectbox(
        "Internet: Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    Politics = st.selectbox(
        "Politics - Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    Shopping = st.selectbox(
        "Shopping - Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    Cars = st.selectbox(
        "Cars: Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    Countryside_outdoors = st.selectbox(
        "Outdoor activities: Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    adrenaline_sports = st.selectbox(
        "Adrenaline sports: Not interested 1-2-3-4-5 Very interested  ", (1, 2, 3, 4, 5))
    daily_events = st.selectbox(
        "I take notice of what goes on around me from 1-5  ", (1, 2, 3, 4, 5))
    Friends_versus_money = st.selectbox(
        "I would rather have lots of friends than lots of money  ", (1, 2, 3, 4, 5))
    Knowing_the_right_people = st.selectbox(
        "I always make sure I connect with the right people  ", (1, 2, 3, 4, 5))
    energy_levels = st.selectbox(
        "I am always full of life and energy  ", (1, 2, 3, 4, 5))
    workaholism = st.selectbox(
        "I often study or work even in my spare time  ", (1, 2, 3, 4, 5))
    assertiveness = st.selectbox(
        "I am not afraid to give my opinion if I feel strongly about something  ", (1, 2, 3, 4, 5))
    thinking_ahead = st.selectbox(
        "I look at things from all different angles before I go ahead  ", (1, 2, 3, 4, 5))
    try:
        data = {'Celebrities': Celebrities,
                'Music': music,
                'Slow songs or fast songs': Slow_songs_or_fast_songs,
                'Spending on healthy eating': Spending_on_healthy_eating,
                'Spending on looks': spending_on_looks,
                'Small - big dogs': Small_big_dogs,
                'Life struggles': Life_struggles,
                'Getting angry': Getting_angry,
                'Funniness': Funniness,
                'Internet': Internet,
                'Politics': Politics,
                'Shopping': Shopping,
                'Cars': Cars,
                'Countryside, outdoors': Countryside_outdoors,
                'Adrenaline sports': adrenaline_sports,
                'Daily events': daily_events,
                'Friends versus money': Friends_versus_money,
                'Knowing the right people': Knowing_the_right_people,
                'Energy levels': energy_levels,
                'Workaholism': workaholism,
                'Assertivenesss': assertiveness,
                'Thinking ahead': thinking_ahead
                }

        personality = pd.DataFrame(data=data, index=[0])
        X = personality
        X = np.array(X).reshape(1, -1)

        # Predict interest based on the KNN models made previously for each genre
        rock_interest = rock_model.predict(X)
        country_interest = country_model.predict(X)
        pop_interest = pop_model.predict(X)
        folk_interest = folk_model.predict(X)
        rap_interest = rap_model.predict(X)
        punk_interest = punk_model.predict(X)
        classical_interest = classical_model.predict(X)
        musical_interest = musical_model.predict(X)
        alternative_interest = alternative_model.predict(X)
        dance_interest = dance_model.predict(X)

        # create dictionary containing predictions and labels
        predictions = {'Rock': rock_interest[0],
                       'Country': country_interest[0],
                       'Pop': pop_interest[0],
                       'folk': folk_interest[0],
                       'Rap': rap_interest[0],
                       'Punk': punk_interest[0],
                       'Classical': classical_interest[0],
                       'Musical': musical_interest[0],
                       'Alternative': alternative_interest[0],
                       'Dance': dance_interest[0]

                       }

        personality = pd.DataFrame(data=predictions, index=[0])

        return(personality)
    except:
        st.write('Please finish filling out the personality survey')


# This function takes genre predictions based on user answers to the survey and then grabs 3 songs at random from
# the million song database that match each of the genres the user is predicted to like, and concatenates that
# into a data frame. This function is nested inside the next function, but does the heavy lifting of creating a
# dataframe of recommended songs for the user


def personality_recs():
    rec_df = genre_df_merged[['id', 'genre_code', 'artists', 'name']]
    genre_coded = []
    rock_recs = pd.DataFrame()
    pop_recs = pd.DataFrame()
    folk_recs = pd.DataFrame()
    country_recs = pd.DataFrame()
    rap_recs = pd.DataFrame()
    musical_recs = pd.DataFrame()
    dance_recs = pd.DataFrame()
    punk_recs = pd.DataFrame()

    if personality['Rock'].values == 1:
        genre_coded.append(0)
        rock_recs = rec_df[rec_df['genre_code'] == 0].sample(3)
        rock_recs['genre'] = 'rock'
    if personality['Pop'].values == 1:
        genre_coded.append(1)
        pop_recs = rec_df[rec_df['genre_code'] == 1].sample(3)
        pop_recs['genre'] = 'pop'
    if personality['folk'].values == 1:
        genre_coded.append(2)
        folk_recs = rec_df[rec_df['genre_code'] == 2].sample(3)
        folk_recs['genre'] = 'folk'
    if personality['Country'].values == 1:
        genre_coded.append(3)
        country_recs = rec_df[rec_df['genre_code'] == 3].sample(3)
        country_recs['genre'] = 'country'
    if personality['Rap'].values == 1:
        genre_coded.append(4)
        rap_recs = rec_df[rec_df['genre_code'] == 4].sample(3)
        rap_recs['genre'] = 'rap'
    if personality['Musical'].values == 1:
        genre_coded.append(5)
        musical_recs = rec_df[rec_df['genre_code'] == 5].sample(3)
        musical_recs['genre'] = 'musical'
    if personality['Dance'].values == 1:
        genre_coded.append(6)
        dance_recs = rec_df[rec_df['genre_code'] == 6].sample(3)
        dance_recs['genre'] = 'dance'
    if personality['Punk'].values == 1:
        genre_coded.append(6)
        punk_recs = rec_df[rec_df['genre_code'] == 6].sample(3)
        punk_recs['genre'] = 'punk'

    recommendations = pd.concat([rock_recs, pop_recs, folk_recs,
                                 country_recs, rap_recs, punk_recs, musical_recs, dance_recs])
    st.write(recommendations[['artists', 'name', 'genre', 'id']])


st.text('\n')
st.text('\n')
st.text('\n')
st.subheader("Personality Based Genre Recommendation System")
st.text('\n')


with st.expander(label="Expand for personality based genre recommendations", ):
    personality = get_personality()
    st.text('\n')
    st.write(personality)

    try:
        personality_recs()
    except:
        st.write("waiting on personality results")


#st.write("END OF FILE")
