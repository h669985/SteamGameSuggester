Okay I think I know how we're going about this...

So the idea is that the user will give as input a small set of data
in the form of games they've selected.
With these games, they've also potentially given their own user score.
Also they've said what platform they use. (Mac, Linux, Windows)

Based on this the moodel is going to give a reccomendation based on the user's
provided data.

The best model for this in my opinion is Nearest Neighbors:
https://scikit-learn.org/stable/modules/neighbors.html

"The principle behind nearest neighbor methods is to find a predefined number of
training samples closest in distance to the new point,
and predict the label from these."

This pretty much describes exactly what we are looking for - a set of games
similar to the games the user gave as input.

So I believe this is the best model for our vision.

So what is there to do?
We have to:
 - Parse the data (done? - might need to be adjusted)
 - Train the model on the data
 - Provide the user a means of giving input data (partially done)
 - Have the model make a prediction based on the input data and provide the result
   back to the user

Parsing the data:
    We need to edit the parse to remove anything that is strictly not interesting
    for our needs. For example there is no need to provide a detailed description as a variable in each game.
    These are usually a whole paragraph of text and there is no way the model can figure out anything useful from
    this. You would need a whole LLM to do something with that.

Training the model on the data:
    I'm not yet sure exactly how to go about this - being the most difficult part of this whole project after all.

Provide the user a means of giving input data: https://docs.streamlit.io/get-started
    This is the frontend. It's going to be a web application hosted on Streamlit.
    I've heard this can all be done in python. So we should probably stick with that and
    avoid javascript/html/css because I hate javascript with a passion still. 
    The user should be met with the option to input up to 10 games. This list has to be created automatically
    with our parsed data. I'm not sure thumbnails for each game is possible - although it is not necessary.
    Once they have selected up to ten games. The can provide each game with a rating from 0 to 100.
    It's important to note that giving 0 score means not giving a score at all, while giving 1 is the lowest score.
    So the default score is 0 and the user should be made aware of this distinction.
    Lasty there is a checkbox they can check for each platform (Windows, Linux, Mac) so that we don't recommend
    games they cannot play. Not all games are supported for mac and linux on Steam.

Have the model make a prediction based on the input data and provide the result back to the user:
    Once the user has sent the data. We should preferably not redirect them, and simply have an output box on the
    right side of the screen that shows the recommendations the model gave back based on the data the user provided.

