# SpotifyApiExploration - In Progress

Do you ever hear a song on the radio or Top 100 and think, "How did *this* song get *this* popular?". Numerous complex factors like promotion, word of mouth, fortunate timing, and countless other qualities contribute to a song blowing up. But is there something numeric and calculable about the audio track itself that increases the chance that a song charts at the top? Is it the tempo, the danceability, the acousticness? Through this analysis, I hope to find some answer to the question: What makes a song a hit?

In this project, I explored what insights machine learning algorithms could provide in predicting song popularity using Spotify's API. This API provides a ton of useful attributes about each track like the song's tempo, energy, instrumentalness, and much more. Obvious factors of a song like the artist and genre will hugely influence the song's popularity, but I wanted to attempt to predict what song would be big based on these audible features alone.

I started by scraping about 80,000 songs from Spotify's API and cleaning them up to explore. This included basic things like dropping duplicates, and more complex munging tasks like summarizing sub-genres of songs into more generalized genres that I could work with. My code for calling this API and cleaning the data can be found in [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Generate_Tracks.ipynb) and [Spotify_Clean_Data.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Clean_Data.ipynb). Here is a sample of what the data looked like:

`
track_pop                                                              30
track_name                                            Tragedy and Harmony
track_id                                           1MLDzI1sCN0cr9UmTwyDd0
track_year                                                           2012
track_spotify_genre                                             classical
art_name                                                      Jeff Loomis
art_id                                             1bqlyIXRTCilCmvAr1newg
alb_name                                               Plains Of Oblivion
alb_id                                             337UZEcZWgax9G9xYIDN7Z
art_genre               ['djent', 'instrumental rock', 'jazz metal', '...
duration_ms                                                        301013
time_signature                                                          4
key                                                                     1
loudness                                                           -4.799
energy                                                              0.972
speechiness                                                         0.143
acousticness                                                     4.12e-06
mode                                                                    1
tempo                                                              135.05
valence                                                             0.409
danceability                                                        0.474
instrumentalness                                                   0.0222
liveness                                                            0.098
genre_words             {'djent': 1, 'instrumental': 1, 'rock': 1, 'ja...
master_popular_genre                                                 rock
`

From there I wanted to see what features I had for each song and what track qualities would be worth dropping, either because it was TOO predictive and would overpower all other features (artist name), because they were redundant and covariant(energy and loudness), or because they simply weren't predictive of popularity at all (mode).

Once I had nailed down what features I wanted to explore, I started by conducting a basic multiple linear regression to get a general idea of which variables were predictive of popularity.

After this, I ran the data through a number of learning methods to try to best predict popularity given these qualities. Much to the chagrin of my PC, I trained a number of Machine Learning algorithms on the data using GridSearches to tune hyperparameters and was left with the following results for each model:


I was proud to achieve a mean absolute error of only ~7.5 for my best model, meaning that on a scale of 1-100 in terms of track popularity, the best model predicted within +/- 7.5 of the true popularity, even without potentially useful features like track artist. While this wasn't dramatically better than the 10 MSE produced by the linear regression, a 25% improvement was still good to see.

![image](https://user-images.githubusercontent.com/25894069/121592529-a8207c00-c9ef-11eb-8654-da783bb67c75.png)

Check out the [Spotify_Analyze_Popularity.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Popularity.ipynb) notebook to see my full code for what qualities of a song are best used to predict popularity. 

One interesting takeaway was reflecting how the error varied for different genres of music:

![image](https://user-images.githubusercontent.com/25894069/121953776-d618ff80-cd12-11eb-81aa-ee4c8e0a2281.png)

It seems that its much easier to predict whether a classical song will succeed compared to predicting whether or not a pop song will. This may be because of how broad the classification of pop is, as patterns may be harder to nail down.

From there, I took on another challenge of trying to predict a song's genre based on these audio features alone, again excluding artist, as this essentially gave the genre away. See the full notebook of exploration here: [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Genre.ipynb). This one took significantly more data massaging, and unfortunately wasn't as accurate given it's multiclass classification challenge. Given 1 guess to predict the genre amongst the top 7 most popular genres (alternative, country, hip-hop, house, indie, pop, r&b and rock), it was only correct about 53% of the time. But hey, for an 8-way classification, this wasn't too bad. For fun, I also checked how often we would predict the genre if given 2 changes to guess corectly, and that raised the success rate to 73%, so at least we were in the right overall song-type.

```
Accuracy predicting Alternative: 0.4980237154150198
Accuracy predicting Country: 0.6268781302170284
Accuracy predicting Hip-Hop: 0.637233259749816
Accuracy predicting House: 0.6541666666666667
Accuracy predicting Indie: 0.37308622078968573
Accuracy predicting Pop: 0.49044585987261147
Accuracy predicting R&B: 0.43853820598006643
Accuracy predicting Rock: 0.37142857142857144
```

Overall, I found this to be an intriguing project that made me rethink what subtle factors contribute to make a song blow up on the charts. In terms of future exploration, I plan on investigating whether these trends have changed over the years, i.e do the qualities that made a song popular in 2010 still apply in 2021?

To do your own explorations using the Spotify API, check out the documentation [here](https://developer.spotify.com/documentation/web-api/).
