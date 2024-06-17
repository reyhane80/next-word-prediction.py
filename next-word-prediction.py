import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Embedding 

# Sample text data
text = '''Machine learning is a branch of artificial intelligence that focuses on the development of algorithms and models that enable computers to learn from data and make predictions or decisions based on that data. In essence, machine learning algorithms are designed to identify patterns in data and use those patterns to make informed decisions or predictions without the need for explicit programming for each task.

One key concept in machine learning is the idea of training a model on a dataset, which involves presenting the model with data samples and their corresponding correct outputs. Through this process, the model learns to generalize from the training data and can then make predictions on new and unseen data.

There are different types of machine learning algorithms, including supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm is trained on labeled data, where each sample is paired with its corresponding correct output. This allows the algorithm to learn to map input data to output labels.

On the other hand, unsupervised learning trains the algorithm on unlabeled data and tasks it with finding patterns or structures present in the data, such as clustering or dimensionality reduction.

Reinforcement learning is another type of machine learning that teaches an agent to interact with an environment by performing actions and receiving rewards or penalties based on those actions. The goal of reinforcement learning is to maximize cumulative rewards over time by learning an optimal policy for decision-making.

Machine learning has wide-ranging applications in various fields such as healthcare, finance, business, and more. It is used for tasks such as image recognition, natural language processing, recommendation systems, and predictive analysis.

With the advancement of machine learning, researchers are constantly working on developing new algorithms and techniques to improve the performance and capabilities of machine learning models. Deep learning, a subset of machine learning that utilizes neural networks with multiple layers, has shown promising results in solving complex problems and achieving state-of-the-art results in various domains.

Overall, machine learning is a rapidly evolving field with vast potential for innovation and impact. By harnessing the power of data and algorithms, researchers and practitioners are striving to push the boundaries of what computers can learn and achieve, paving the way for a future where intelligent systems help us tackle some of society's most challenging issues.
The capabilities of machine learning include:

1. Pattern Recognition: Machine learning algorithms are capable of identifying patterns and relationships within large datasets, allowing them to make predictions or decisions based on these patterns.

2. Data Mining: Machine learning can be used to extract valuable insights and knowledge from vast amounts of data, helping organizations make informed decisions and drive business growth.

3. Predictive Analysis: Machine learning models can analyze historical data to predict future trends or outcomes, enabling businesses to anticipate customer behavior, market changes, and more.

4. Anomaly Detection: Machine learning algorithms can detect unusual patterns or outliers in data, which can be useful for identifying fraudulent activities, system failures, or other anomalies.

5. Natural Language Processing: Machine learning techniques enable computers to understand, interpret, and generate human language, facilitating applications such as chatbots, sentiment analysis, and language translation.

6. Image Recognition: Machine learning algorithms can analyze and interpret visual data, allowing computers to recognize objects, faces, scenes, and other visual patterns.

7. Recommendation Systems: Machine learning is used to build recommendation systems that suggest products, services, or content to users based on their preferences and behavior.

8. Autonomous Decision-Making: Machine learning enables autonomous systems to make decisions and take actions without human intervention, such as self-driving cars or automated trading systems.

9. Personalization: Machine learning algorithms can tailor experiences and recommendations to individual users based on their preferences, behavior, and feedback.

10. Continuous Learning: Machine learning models can adapt and improve over time as they are exposed to new data, allowing them to continuously learn and evolve their capabilities.

These are just a few of the many capabilities of machine learning that are transforming industries and driving innovation across various sectors.'''

# Tokenization using tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
encoded = tokenizer.texts_to_sequences([text])[0]

# Creating sequences
sequences = []
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)
sequences = np.array(sequences)

X, y = sequences[:,0], sequences[:,1]
X = to_categorical(X, num_classes=len(tokenizer.word_index)+1)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=1))
model.add(LSTM(50))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=3, verbose=2)

# Generate predictions
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = tokenizer.index_word[predicted[0]]
        seed_text += " " + output_word
    return seed_text

# Example usage
generated_text = generate_text("build", 5)
print(generated_text)