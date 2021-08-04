# SpotifyApiExploration
In this project, I had 2 goals:
1. Build a machine learning model to predict a Spotify track's popularity be based on its audio qualities.
2. Build a machine learning model to predict a Spotify track's genre based on its audio qualities.

## 0. Data Preprocessing
I started by scraping about 80,000 songs from Spotify's API and cleaning them up to explore. This included basic things like dropping duplicates, and more complex munging tasks like summarizing sub-genres of songs into more generalized genres that I could work with. My code for calling this API and cleaning the data can be found in [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Generate_Tracks.ipynb) and [Spotify_Clean_Data.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Clean_Data.ipynb).

Here is a sample of what the data looked like once it was scraped from the API and cleaned up:

| track_pop | track_name     | track_id               | track_year | track_spotify_genre | art_name                            | art_id                 | alb_name   | alb_id                 | art_genre                                           | duration_ms | time_signature | key | loudness | energy | speechiness | acousticness | mode | tempo   | valence | danceability | instrumentalness | liveness | genre_words                                         | master_popular_genre |
|-----------|----------------|------------------------|------------|---------------------|-------------------------------------|------------------------|------------|------------------------|-----------------------------------------------------|-------------|----------------|-----|----------|--------|-------------|--------------|------|---------|---------|--------------|------------------|----------|-----------------------------------------------------|----------------------|
| 50        | Clusterhug     | 65yrKuYdBYsMA1ItIngmjc | 2020       | rock                | I DONT KNOW HOW   BUT THEY FOUND ME | 0Raaw7kr1Vzat4ZvHzjsJR | RAZZMATAZZ | 7q8hYYZgsIQCXibLzwiPll | ['alt z', 'indie   pop', 'modern alternative roc... | 192066      | 4              | 0   | -4.067   | 0.837  | 0.052       | 0.00324      | 1    | 121.901 | 0.23    | 0.476        | 0                | 0.105    | {'alt': 1, 'z':   1, 'indie': 1, 'pop': 1, 'mode... | rock                 |
| 57        | Fare Thee Well | 1fzw0qGcB6xs4IBXhdfAkj | 2020       | rock                | Stone Temple   Pilots               | 2UazAtjfzqBF0Nho2awK4z | Perdida    | 27evZfDFySSv4dcje8afMI | ['alternative   metal', 'alternative rock', 'gru... | 261880      | 4              | 7   | -8.709   | 0.371  | 0.0272      | 0.547        | 1    | 66.853  | 0.196   | 0.431        | 0.0154           | 0.0921   | {'alternative':   2, 'metal': 2, 'rock': 4, 'gru... | rock                 |

## 1. Predicting Track Popularity
Do you ever hear a song on the radio or Top 100 and think, "How did *this* song get *this* popular?". Numerous complex factors like promotion, word of mouth, fortunate timing, and countless other qualities contribute to a song blowing up. But is there something numeric and calculable about the audio track itself that increases the chance that a song charts at the top? Is it the tempo, the danceability, the acousticness? Through this analysis, I hope to find some answer to the question: What makes a song a hit?

In this section, I explored what insights machine learning algorithms could provide in predicting song popularity using Spotify's API. This API provides a ton of useful attributes about each track like the song's tempo, energy, instrumentalness, and much more. Obvious factors of a song like the artist and genre will hugely influence the song's popularity, but I wanted to attempt to predict what song would be big based on these audible features alone.

To start, I wanted to see what features I had for each song and what track qualities would be worth dropping, either because it was TOO predictive and would overpower all other features (artist name), because they were redundant and covariant (energy and loudness), or because they simply weren't predictive of popularity at all (mode). I used a few visuals like the following distributions and correlation heatmap to make these decisions:

![image](https://user-images.githubusercontent.com/25894069/122130209-29ab4c00-cdec-11eb-8fda-e765e54cd7c0.png)

![image](https://user-images.githubusercontent.com/25894069/122130230-2d3ed300-cdec-11eb-86e2-d10e675808f7.png)

Once I had nailed down what features I wanted to explore, I started by conducting a basic multiple linear regression to get a general idea of which variables were predictive of popularity. This provided some interesting insights into what factors contributed to a track's popularity:

| Dep.   Variable:    | track_pop        | R-squared   (uncentered):      | 0.928     |
|---------------------|------------------|--------------------------------|-----------|
| Model:              | OLS              | Adj. R-squared   (uncentered): | 0.928     |
| Method:             | Least Squares    | F-statistic:                   | 1.48E+04  |
| Date:               | Tue, 15 Jun 2021 | Prob (F-statistic):            | 0         |
| Time:               | 7:11:13          | Log-Likelihood:                | -1.79E+05 |
| No.   Observations: | 44678            | AIC:                           | 3.59E+05  |
| Df Residuals:       | 44639            | BIC:                           | 3.59E+05  |
| Df Model:           | 39               |                                |           |
| Covariance   Type:  | nonrobust        |                                |           |

|                             | coef      | std err  | t       | P>\|t\| | [0.025    | 0.975]    |
|-----------------------------|-----------|----------|---------|---------|-----------|-----------|
| track_year                  | 0.0489    | 0.003    | 14.806  | 0       | 0.042     | 0.055     |
| duration_ms                 | -7.26E-06 | 9.28E-07 | -7.823  | 0       | -9.08E-06 | -5.44E-06 |
| loudness                    | 0.4966    | 0.028    | 17.862  | 0       | 0.442     | 0.551     |
| speechiness                 | 1.7271    | 0.746    | 2.314   | 0.021   | 0.264     | 3.19      |
| acousticness                | 5.1878    | 0.302    | 17.197  | 0       | 4.597     | 5.779     |
| tempo                       | 0.0008    | 0.002    | 0.361   | 0.718   | -0.004    | 0.005     |
| valence                     | -3.5675   | 0.322    | -11.09  | 0       | -4.198    | -2.937    |
| danceability                | 17.5364   | 0.539    | 32.555  | 0       | 16.481    | 18.592    |
| instrumentalness            | -9.862    | 0.346    | -28.528 | 0       | -10.54    | -9.184    |
| liveness                    | -10.4296  | 0.412    | -25.326 | 0       | -11.237   | -9.622    |
| mode_1                      | -0.3075   | 0.142    | -2.163  | 0.031   | -0.586    | -0.029    |
| key_1                       | -0.054    | 0.267    | -0.203  | 0.839   | -0.577    | 0.469     |
| key_2                       | -0.4519   | 0.281    | -1.606  | 0.108   | -1.003    | 0.1       |
| key_3                       | 0.7739    | 0.41     | 1.887   | 0.059   | -0.03     | 1.578     |
| key_4                       | -0.3939   | 0.304    | -1.296  | 0.195   | -0.99     | 0.202     |
| key_5                       | 0.8387    | 0.296    | 2.833   | 0.005   | 0.259     | 1.419     |
| key_6                       | 0.0367    | 0.3      | 0.123   | 0.902   | -0.551    | 0.624     |
| key_7                       | -0.3977   | 0.273    | -1.456  | 0.145   | -0.933    | 0.138     |
| key_8                       | 0.4151    | 0.301    | 1.378   | 0.168   | -0.175    | 1.006     |
| key_9                       | -1.0577   | 0.283    | -3.733  | 0       | -1.613    | -0.502    |
| key_10                      | 0.2087    | 0.319    | 0.655   | 0.512   | -0.416    | 0.833     |
| key_11                      | -0.2323   | 0.293    | -0.794  | 0.427   | -0.806    | 0.341     |
| time_signature_1            | -61.3608  | 6.713    | -9.141  | 0       | -74.518   | -48.204   |
| time_signature_3            | -61.1911  | 6.649    | -9.203  | 0       | -74.223   | -48.159   |
| time_signature_4            | -61.8123  | 6.645    | -9.303  | 0       | -74.836   | -48.789   |
| time_signature_5            | -61.373   | 6.667    | -9.205  | 0       | -74.441   | -48.305   |
| track_spotify_genre_country | 2.5872    | 0.734    | 3.526   | 0       | 1.149     | 4.025     |
| track_spotify_genre_hip-hop | 8.4786    | 0.735    | 11.543  | 0       | 7.039     | 9.918     |
| track_spotify_genre_house   | 8.9183    | 0.727    | 12.275  | 0       | 7.494     | 10.342    |
| track_spotify_genre_indie   | 13.9114   | 0.68     | 20.467  | 0       | 12.579    | 15.244    |
| track_spotify_genre_pop     | 12.7292   | 0.687    | 18.537  | 0       | 11.383    | 14.075    |
| track_spotify_genre_r&b     | 0.7398    | 0.702    | 1.054   | 0.292   | -0.636    | 2.116     |
| track_spotify_genre_rock    | 15.3225   | 0.671    | 22.835  | 0       | 14.007    | 16.638    |
| master_popular_genre_hiphop | 2.4432    | 0.456    | 5.363   | 0       | 1.55      | 3.336     |
| master_popular_genre_house  | -3.3727   | 0.511    | -6.602  | 0       | -4.374    | -2.371    |
| master_popular_genre_indie  | -0.1607   | 0.406    | -0.396  | 0.692   | -0.956    | 0.634     |
| master_popular_genre_pop    | -0.3584   | 0.386    | -0.93   | 0.353   | -1.114    | 0.397     |
| master_popular_genre_r&b    | 1.4465    | 0.469    | 3.086   | 0.002   | 0.528     | 2.365     |
| master_popular_genre_rock   | -3.0339   | 0.362    | -8.372  | 0       | -3.744    | -2.324    |

Features like danceability, liveness, time signature, and genre classification were huge determinants of popularity.

After this, I ran the data through a number of learning methods to try to best predict popularity given these qualities. Much to the chagrin of my PC, I trained a number of Machine Learning algorithms on the data using GridSearches to tune hyperparameters and was left with the following results for each model:

![image](https://user-images.githubusercontent.com/25894069/122119268-d4683e00-cddd-11eb-8099-683732aff735.png)

I was proud to achieve a mean absolute error of only ~7.5 for my best model, meaning that on a scale of 1-100 in terms of track popularity, the best model predicted within +/- 7.5 of the true popularity, even without potentially useful features like track artist. While this wasn't dramatically better than the 10 MSE produced by the linear regression, a 25% improvement was still good to see.

Finally, I produced a number of error visualizations to see where the model was having trouble. 

![image](https://user-images.githubusercontent.com/25894069/122130734-ee5d4d00-cdec-11eb-91e8-8df882c49d81.png)

![image](https://user-images.githubusercontent.com/25894069/122130681-de456d80-cdec-11eb-9cc8-a9eff9630464.png)

![image](https://user-images.githubusercontent.com/25894069/122130701-e30a2180-cdec-11eb-8556-b0a3273136f8.png)

One interesting obeservation was how the error varied for different genres of music:

![image](https://user-images.githubusercontent.com/25894069/121953776-d618ff80-cd12-11eb-81aa-ee4c8e0a2281.png)

It seems that its much easier to predict whether an Indie or classical song will succeed compared to predicting whether or not a pop song will. This may be because of how broad the classification of pop is, as patterns may be harder to nail down.

Check out the [Spotify_Analyze_Popularity.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Popularity.ipynb) notebook to see my full code for what qualities of a song are best used to predict popularity. 

## 2. Predicting Track Genre
From there, I took on another challenge of trying to predict a song's genre based on these audio features alone, again excluding artist, as this essentially gave the genre away. See the full notebook of exploration here: [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Genre.ipynb). 

The goal was to build a multi-class classifier that could predict a track's genre based on its audible features. I used the same dataset produced in part 0, but filtered to only tracks within the 7 most common genres. Those were the following:



Overall, I found this to be an intriguing project that made me rethink what subtle factors contribute to make a song blow up on the charts. In terms of future exploration, I plan on investigating whether these trends have changed over the years, i.e do the qualities that made a song popular in 2010 still apply in 2021?

To do your own explorations using the Spotify API, check out the documentation [here](https://developer.spotify.com/documentation/web-api/).
