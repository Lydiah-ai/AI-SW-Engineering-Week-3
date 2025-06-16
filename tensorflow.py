model = tf.keras.Sequential([
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
