# AI-SW-Engineering-Week-3
PART 3
1. Ethical Considerations
Potential Biases:
•	MNIST: While MNIST may seem neutral, bias can arise in expanded handwriting datasets that skew toward particular writing styles (e.g., age or region-specific stroke patterns), potentially disadvantaging certain populations in handwriting recognition.
•	Amazon Reviews: This dataset can carry language and sentiment biases—for instance, disproportionately negative reviews from certain demographics or skewed sentiment labeling due to sarcasm or cultural differences.
Mitigation Tools:
•	TensorFlow Fairness Indicators: These help you measure model performance across slices of your data (like gender, product category, or reviewer location). You could identify whether your sentiment classifier is less accurate for specific reviewer demographics, and retrain or rebalance accordingly.
•	spaCy Rule-Based Systems: For NLP tasks like review analysis, spaCy can be used to override or supplement biased learned patterns. For example, if a model misinterprets idioms or sarcastic remarks as negative sentiment, spaCy’s rule-based components can add pattern recognition to handle such edge cases.

2. Troubleshooting Challenge

Here’s a common example of buggy TensorFlow code and how it might be fixed:
Buggy Snippet (Python)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
Common Issues:
•	Missing activation functions, which hampers learning.
•	No input shape defined in the first layer.
•	Mismatch between output layer and loss function if y_train is sparse.

•	Corrected Version:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

This version adds an activation function, corrects the input shape for MNIST-style 784-pixel inputs, and switches the loss function for sparse labels.
