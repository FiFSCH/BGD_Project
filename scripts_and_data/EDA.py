import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('full_dataset.csv')

# Labels to emotions mapping
label_translation = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

df['label'] = df['label'].map(label_translation)

# Word count per label
label_counts = df['label'].value_counts()

plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color='lightcoral', edgecolor='black')

plt.title('Emotion Label Distribution', fontsize=16)
plt.xlabel('Emotions', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('.././images/label_dist.png')
plt.show()

# Word Clouds
def generate_wordcloud(data, title, sentiment):
    wordcloud = WordCloud(width=800, height=400, background_color='white', mode="RGB").generate(' '.join(data))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'.././images/word_cloud_{sentiment}.png')
    plt.show()

for sentiment in df['label'].unique():
    sentiment_text = df[df['label'] == sentiment]['text']
    generate_wordcloud(sentiment_text, f'Common Words for {sentiment}', sentiment)

# Text length for each label
df['text_length'] = df['text'].apply(len)
avg_text_length = df.groupby('label')['text_length'].mean()

plt.figure(figsize=(8, 6))
avg_text_length.plot(kind='bar', color='lightgreen', edgecolor='black')

plt.title('Average Text Length by Emotion', fontsize=16)
plt.xlabel('Emotions', fontsize=14)
plt.ylabel('Average Text Length', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'.././images/length_per_label.png')
plt.show()

