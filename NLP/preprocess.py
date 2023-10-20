# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a pandas dataframe
data = pd.read_csv('hate_speech_dataset.csv', names=['tweet', 'sentiment'], header=0)

# Perform a descriptive statistical analysis of the data
print(data.describe(include='all'))

# Handle missing values
data = data.dropna()

# Remove user @'s and non alphabetical characters
data['tweet'] = data['tweet'].replace(to_replace=r'@\w+', value='', regex=True)
data['tweet'] = data['tweet'].replace(to_replace='[^A-Za-z0-9\s]+', value='', regex=True)

# Count the number of positive, negative, and neutral text items
positive_tweets = len(data[data['sentiment'] == 1])
negative_tweets = len(data[data['sentiment'] == -1])
neutral_tweets = len(data[data['sentiment'] == 0])

# Display findings in a plot
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_tweets, negative_tweets, neutral_tweets]
colors = ['gold', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()
