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

The goal was to build a multi-class classifier that could predict a track's genre based on its audible features. I used the same dataset produced in part 0, but filtered to only tracks within the 7 most common genres. Those genres were the following (along with the number of tracks we had for each genre:

| Genre   | TrackCount |
|---------|------------|
| pop     | 17277      |
| country | 10005      |
| hiphop  | 10728      |
| rock    | 8967       |
| indie   | 7673       |
| house   | 6529       |
| r&b     | 5505       |

![image](https://user-images.githubusercontent.com/25894069/128584060-b9097333-2d4a-4c92-b135-b1a38d0eb123.png)

The imbalance of classes in the above list of genre counts was a major piece of this section of my project. I tried to build a few different models based on separate sampling techniques on the data. I performed the following 3 approaches to try to improve the model:

### Building a model based on the imbalanced classes above
#### A.
 First, I built a basic random forest model using the data above with no changes to the class distribution. The results of that model were the following:

  **Model Accuracy: 0.6806779969099337**
  
  Accuracy predicting country : 0.7844257366205653
  
  Accuracy predicting hiphop : 0.7706289671090595
  
  Accuracy predicting house : 0.6566682048915552
  
  Accuracy predicting indie : 0.6363636363636364
  
  Accuracy predicting pop : 0.6996659046949182
  
  Accuracy predicting r&b : 0.42412451361867703
  
  Accuracy predicting rock : 0.6347264695888549
  
![image](https://user-images.githubusercontent.com/25894069/128584091-26f42a6f-a723-4d47-9346-042dbe28a77c.png)

  While this model performed relatively well, it was clear that the model was learning to favor pop tracks heavily. As visualize by the verticle column where `x = pop`, the model defaulted to classifying the sample as pop whenever there was doubt. Because of this, the accuracy in predicting other genres suffered as it overcompensated toward pop. 

#### B.  From here, I modified the dataset by undersampling the number of pop tracks. Instead of having 7,000 more pop tracks than any other, I reduced the number of pop tracks to create a more even balance.

| Genre   | TrackCount |
|---------|------------|
| pop     | 8000       |
| hiphop  | 7962       |
| country | 7585       |
| rock    | 6785       |
| indie   | 5740       |
| house   | 4879       |
| r&b     | 4161       |

The overall accuracy of this model was about the same, but the change to the training data impacted where the model was accurate and where it was not:

**Model Accuracy: 0.6788518008030203**

Accuracy predicting country : 0.8107438016528926

Accuracy predicting hiphop : 0.8380332610267535

Accuracy predicting house : 0.7351515151515151

Accuracy predicting indie : 0.6652871184687015

Accuracy predicting pop : 0.5277777777777778

Accuracy predicting r&b : 0.5476190476190477

Accuracy predicting rock : 0.6851512373968836

![image](https://user-images.githubusercontent.com/25894069/128584230-49412205-e6a0-4bcc-b9c0-46c98b7577a6.png)

While accuracy in predicting pop dropped by over 15%, other genres like hiphop, house, and country, increased by 5-10% each. This makes sense, as the undersampling reduced the likelihood that the model predicted overpredicted pop. 

#### C. Finally, I chose to try out using the SMOTE oversampling technique, which generates synthetic data of the underrepresented classes in order to balance all classes evenly. For this example, I had to make sure not to validate or test on this oversampled dataset. The results of this model are as follows:

**Model Accuracy: 0.6776788148686722**

Accuracy predicting country : 0.796452194828623

Accuracy predicting hiphop : 0.7830351990767456

Accuracy predicting house : 0.7000461467466543

Accuracy predicting indie : 0.6382734912146677

Accuracy predicting pop : 0.5987339546333743

Accuracy predicting r&b : 0.5491939966648138

Accuracy predicting rock : 0.6690451919809718

![image](https://user-images.githubusercontent.com/25894069/128584329-f8e36532-c356-4757-8b63-695e05518ef9.png)

This model produced results that were a bit of a blend of the above 2 results. Overall, accuracy was the same. However, it had worse pop accuracy than approach A, but better pop accuracy than approach B. Conversely, it had better accuracy for the minority classes than approach A, but worse accuracy on these for approach B.

The accuracy of each different approach was summarized as follows:
|          |         |                  | Data to Build Model |                    |
|----------|---------|------------------|---------------------|--------------------|
|          |         | Original Dataset | Undersample Pop     | SMOTE Oversampling |
|          | overall | **0.681**            | 0.679               | 0.678              |
|          | country | 0.784            | **0.811**              | 0.796              |
|          | hiphop  | 0.771            | **0.838**               | 0.783              |
| Accuracy | house   | 0.657            | **0.735 **              | 0.700              |
|          | indie   | 0.636            | **0.665**               | 0.638              |
|          | pop     | **0.700**            | 0.528               | 0.599              |
|          | r&b     | 0.424            | 0.548               | **0.549**              |
|          | rock    | 0.635            | **0.685**               | 0.669              |

In the notebook, I detail a longer discussion of the "why" , but I chose to explore the results of approach C for the conclusion of the project.

### Results and Listening to the Predictions
Once this model was hypertuned, I took a look at the specific predictions it made to see how *I* would classify these spotify tracks. These are some of the tracks that the model was **most** confident in predicting:

| Prediction   Confidence | Predicted Correctly | True Genre | Predicted Genre | Track Name              | Artist Name              |
|-------------------------|---------------------|------------|-----------------|-------------------------|--------------------------|
| 1                       | 1                   | r&b        | r&b             | Can We (with Kacy Hill) | Jim-E Stack              |
| 0.99938                 | 1                   | r&b        | r&b             | Polaroids               | Jay Prince               |
| 0.99875                 | 1                   | indie      | indie           | Song For Zula           | Phosphorescent           |
| 0.9975                  | 1                   | r&b        | r&b             | John Redcorn            | SiR                      |
| 0.9975                  | 1                   | rock       | rock            | Out of My League        | Fitz and The Tantrums    |
| 0.99688                 | 1                   | rock       | rock            | Too Late                | The Happy Fits           |
| 0.99688                 | 1                   | rock       | rock            | Pompeii                 | Bastille                 |
| 0.99625                 | 1                   | hiphop     | hiphop          | I Mean It (feat. Remo)  | G-Eazy                   |

* Here is the track by Jim-E Stack that the model correctly predicted as R&B: [Can We (with Kacy Hill) - Jim-E Stack](https://open.spotify.com/track/5mVWKI0OgsFIXF8aJccfO8?si=67ac936034124432)

* Here is a track by Fitz and The Tantrums that the model correctly predicted as Rock: [Out of My League	Fitz and The Tantrums](https://open.spotify.com/track/2AYEOC02WLhUiOoaig2SEH?si=5abe7f31926b4f06)

* Here is a track by Phosphorescent that the model correctly predicted as Indie: [Song For Zula - Phosphorescent](https://open.spotify.com/track/3zr2s3o2Ye1j6t0ZMdoUYi?si=c32ad6ac6c644a23)

These all make sense, as they sound quintessentially R&B, Rock and Indie.

More interesting will be looking at what tracks the model predicted *incorrectly*. Let's take a look at a few of those.

| Prediction   Confidence | Predicted Correctly | True Genre | Predicted Genre | Track Name                                        | Artist Name         |
|-------------------------|---------------------|------------|-----------------|---------------------------------------------------|---------------------|
| 0.95938                 | 0                   | pop        | country         | Suitcase                                          | Steve Moakler       |
| 0.92063                 | 0                   | pop        | house           | Dechorro                                          | Deorro              |
| 0.91875                 | 0                   | pop        | hiphop          | Press                                             | Cardi B             |
| 0.91375                 | 0                   | pop        | house           | Flying Blind                                      | Cosmic Gate         |
| 0.90875                 | 0                   | pop        | house           | Raise Your Head                                   | Alesso              |
| 0.90313                 | 0                   | house      | country         | Gonna Be Alright - Man Cub Remix                  | Tritonal            |
| 0.9025                  | 0                   | pop        | house           | Lights                                            | Swedish House Mafia |
| 0.89875                 | 0                   | pop        | house           | This Is What It Feels Like - W&W Remix            | Armin van Buuren    |

Unsurprisingly, most of these missed predictions are true-genre pop. This is clearly the model's worst category at predicting, thought this was expected after we went with the Method #2 approach. Let's listen to a few of these in particular:

* The first one in the list sounds a lot like country to me, despite its genre label being pop: [Suitcase -	Steve Moakler](https://open.spotify.com/track/0uhxXyG4Eb5sIIt3GZxJcn?si=f3c81e3fc9004886) The line between country, pop, and country-pop is certainly a blurry one, and this may be more of a failure of my labeling system rather than one by the model.

*  Let's look at a track that isn't true-genre pop. This one is true-genre house: [Gonna Be Alright - Man Cub Remix	Tritonal](https://open.spotify.com/track/3wJYhSVsFnfKr46ufuZgjA?si=6b74d547dfc34868) This is a surprising one. This is a fairly obvious pop or house track, so it's quite surprising that the model labeled this country. This is a pretty bad miss.

* Finally, this Cardi B song is a bit of a surprise: [Press - Cardi B](https://open.spotify.com/track/6dPyzkyZwoj9LqjQXOFdVv?si=ab252432ee5f4544). I would definitely have labeled this as hiphop, so this is another understandable mistake by the model. This seems to be more of a case of an unclear distinction between pop/hiphop or a mistake on the genre labeling.

### Second Chances
As a final exploration, I wanted to see how the model would do with not just one prediction, but 2 predictions to get the true genre correct. Taking the genres with the two highest predictive probabilities, the model's accuracy increased to about 82%, a 14% improvement from the model with only 1 prediction. 

Let's look at which predictions this got wrong with both guesses. These are tracks that the model not only misclassified, but wasn't even close to classifying correctly:
| Predicted   Correctly | True Genre | Predicted Genres | Track Name                                        | Artist Name    |
|-----------------------|------------|------------------|---------------------------------------------------|----------------|
| 0                     | country    | [hiphop, pop]    | Smoke Stack                                       | The Lacs       |
| 0                     | house      | [country, pop]   | Mono in Love - Radio Edit                         | Edward Maya    |
| 0                     | hiphop     | [house, rock]    | Tattoo                                            | Kevin Abstract |
| 0                     | r&b        | [pop, hiphop]    | Dingo X BIBI - she got it                         | BIBI           |
| 0                     | pop        | [house, rock]    | Gooey Rework                                      | Glass Animals  |
| 0                     | country    | [r&b, pop]       | Telescope                                         | Nashville Cast |
| 0                     | country    | [r&b, pop]       | Mother                                            | Sugarland      |
| 0                     | hiphop     | [pop, house]     | I've Been Waiting (feat. ILOVEMAKONNEN) - Orig... | Lil Peep       |
| 0                     | indie      | [pop, rock]      | We Must Be Killers                                | Mikky Ekko     |
| 0                     | country    | [pop, hiphop]    | Snow White                                        | Katie Noel     |

Let's check some of these out:
*   [Smoke Stack - The Lacs](https://open.spotify.com/track/2XBYAG0RgFgzONthKkuaT5?si=d00724b4fac74469) - This one makes sense as a hiphop/pop track, as it's a pretty strange rap country fusion. 
*   [Tattoo - Kevin Abstrac](https://open.spotify.com/track/2jEkD23p3LVw2D19OiFvMK?si=460fea7143af495b) - This example again makes sense why the model would miss it. The sound features an acoustic guitar and pretty traditional rock drums, although it's strange that the model predicted house and not something like Indie/Rock.
*   [Boogieman - Childish Gambino](https://open.spotify.com/track/0SunFlwqT44E0BU0yrgM7u?si=5c52a9b7638143e7) - This one makes sense as just an extremely difficult song to predict. I would personally call this funk, perhaps? Of our genre's, I think hiphop is a fair true labell, but I'm not surprised the model missed this, given its unconventional sound.

### How to improve the genre model in the future
It seems that the best place to focus on for future improvement of the genre predicting model would be to improve the true labeling of the tracks. Detailed in part 1, every track / artist outputted by the Spotify API has a number of genre tags, but there is not a single definitive genre for one track. To allow us to build a genre-predicting algorithm, the track genre tags were pooled into a dictionary and I chose the most occuring gender to determine the True Genre for that track. This is somewhat imperfect and likely lead to some faulty labeling of true genre. However, this was the best option available until Spotify labels the genre of the track themselves, or I come up with some other NLP improvements for the tag text.

Overall, I found this to be an intriguing project that made me rethink what subtle factors contribute to make a song blow up on the charts. In terms of future exploration, I plan on investigating whether these trends have changed over the years, i.e do the qualities that made a song popular in 2010 still apply in 2021?

To do your own explorations using the Spotify API, check out the documentation [here](https://developer.spotify.com/documentation/web-api/).
