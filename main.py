# Split the data into input (x) training and testing data, and ouput (y) training and testing data, 
# with training data being 80% of the data, and testing data being the remaining 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#, shuffle=True)

# Scale both training and testing input data
X_train = preprocessing.maxabs_scale(X_train)
X_test = preprocessing.maxabs_scale(X_test)

model = Sequential()
model.add(Dense(4, input_shape=(4,)))
model.add(Dense(4, input_shape=(4,)))
model.add(Dense(1, input_shape=(4,)))

model.compile(optimizer="adam", loss="msle", metrics=['mean_squared_logarithmic_error','accuracy'])

# Pass several parameters to 'EarlyStopping' function and assign it to 'earlystopper'
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

model.summary()
history = model.fit(X_train, y_train, epochs = 2000, validation_split = 0.3, verbose = 2, callbacks = [earlystopper])

# Runs model (the one with the activation function, although this doesn't really matter as they perform the same) 
# with its current weights on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

df = pd.read_csv('test_two_data.csv')
df = df[['rpm','iq','uq','udc','idc']]
X = df[df.columns[:-1]]
Y = df.idc
X_validate = preprocessing.maxabs_scale(X)

y_pred = model.predict(X_validate)

plt.plot(Y)
plt.plot(y_pred)
plt.show()

(weight_0,bias_0) = model.layers[0].get_weights()
(weight_1,bias_1) = model.layers[1].get_weights()
