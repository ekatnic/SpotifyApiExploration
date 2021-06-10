# SpotifyApiExploration

Do you ever hear a song on the radio or Top 100 and think, "How did this song get this popular?". Promotion, word of mouth, fortunate timing, and a number of other complex factors contribute to a song blowing up. But is there something numeric and calculable about the actual track itself that can answer the question "What makes a song popular?". 

In this project, I explored what insights machine learning algorithms could provide in predicting song popularity using Spotify's API. This API provides a ton of useful attributes about each track like the song's tempo, energy, instrumentalness, and much more. Obvious factors of a song like the artist and genre will hugely influence the song's popularity, but I wanted to attempt to predict what song would be big based on these audible features alone.

This started with scraping about 50,000 songs from Spotify's API and cleaning them up to explore. This included basic things like dropping duplicates, and more complex munging tasks like summarizing sub-genres of songs into more generalized genres that I could work with.  My code for calling this API and cleaning the data can be found in [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Generate_Tracks.ipynb) and [Spotify_Clean_Data.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Clean_Data.ipynb).

From there I wanted to see what features I had for each song and what track qualities would be worth dropping, either because it was TOO predictive and would overpower all other features (artist_name), because they were redundant and covariant(energy and loudness), or because they simply weren't predictive of popularity at all (mode).

![image](https://user-images.githubusercontent.com/25894069/121595524-377b5e80-c9f3-11eb-963b-0b7a8cc44362.png)
![image](https://user-images.githubusercontent.com/25894069/121595546-3e09d600-c9f3-11eb-9c34-53bade362faa.png)

Once I had nailed down what features I wanted to explore, I ran the data through a number of learning methods to try to best predict popularity given these qualities. After a number of grid searches for several algorithms, my PC was able to finally rest. 

I was proud to achieve a mean absolute error of only ~6, meaning that on a scale of 1-100 in terms of track popularity, the best model predicted within +/- 6 to the true popularity, even without potentially useful features like track artist. 
![image](https://user-images.githubusercontent.com/25894069/121592529-a8207c00-c9ef-11eb-8654-da783bb67c75.png)

Check out the [Spotify_Analyze_Popularity.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Popularity.ipynb) notebook to see my findings on what qualities of a song are best used to predict popularity. 

From there, I took on another challenge of trying to predict a song's genre based on these audio features alone, again excluding artist, as this essentially gave the genre away. See the full notebook of exploration here: [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Genre.ipynb). This one took significantly more data massaging, and unfortunately wasn't as accurate given it's multiclass classification challenge. Given 1 guess to predict the genre amongst the top 8 most popular genres (alternative, country, hip-hop, house, indie, pop, r&b and rock), it was only correct about 53% of the time. But hey, for an 8-way classification, this wasn't too bad. For fun, I also checked how often we would predict the genre if given 2 changes to guess corectly, and that raised the success rate to 73%, so at least we were in the right overall song-type.

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

Overall, I found this to be an intriguing project that made me rethink what subtle factors contribute to make a song blow up on the charts. To do your own explorations using the Spotify API, check out the documentation [here](https://developer.spotify.com/documentation/web-api/).
