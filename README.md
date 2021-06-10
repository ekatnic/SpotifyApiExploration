# SpotifyApiExploration

In this project, I explored what insights that Machine Learning algorithms could provide in predicting song popularity and genre using Spotify's API. My code for calling this API and cleaning the data can be found in [Spotify_Generate_Tracks.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Generate_Tracks.ipynb) and [Spotify_Clean_Data.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Clean_Data.ipynb). This API provides a ton of useful attributes about each track like the song's tempo, energy, instrumentalness, and much more. 



Check out the [Spotify_Analyze_Popularity.ipynb](https://github.com/ekatnic/SpotifyApiExploration/blob/master/Spotify_Analyze_Popularity.ipynb) notebook to see my findings on what qualities of a song are best used to predict popularity. I was proud to achieve a mean absolute error of only ~6, meaning that on a scale of 1-100 in terms of track popularity, the best model predicted within +/- 6 to the true popularity, even without potentially useful features like track artist.
![image](https://user-images.githubusercontent.com/25894069/121592529-a8207c00-c9ef-11eb-8654-da783bb67c75.png)

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
