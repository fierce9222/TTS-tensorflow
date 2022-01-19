import tensorflow as tf
class Encoder(tf.keras.Model):
  def __init__(self, vocabularySize, e, enc_units, batch_sz, timing, inp_len):
    super(Encoder, self).__init__()

    self.batch_sz = batch_sz
    self.inp_len = inp_len
    self.enc_units = enc_units
    self.timing = timing
    self.embedding = tf.keras.layers.Embedding(vocabularySize, e)
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.inp = tf.keras.layers.InputLayer(input_shape=(inp_len,))
    self.out3 = tf.keras.layers.Dense(80)
    self.flat = tf.keras.layers.Flatten()
    self.den_time = tf.keras.layers.Dense(timing)
  def call(self, x): #при обращении к экземпляру класса как к функции, будет вызываться этот метод:
    x = self.inp(x)
    x = self.embedding(x) # входящие тензоры преобразовываются в эмбеддинг
    output, state = self.gru(x)
    var = self.den_time(output)
    var = tf.transpose(var, perm=[0, 2, 1])
    var = self.out3(var)
    return output, state, var #вызов метода/функции вернёт выход из сети GRU и состояние на выходе


  def initialize_hidden_state(self, train = True): #создаем метод инициализации состояний на скрытых слоях
        if train == True:
            return tf.zeros((self.batch_sz,  self.enc_units))
        else:
            return tf.zeros((1, self.enc_units))



class BahdanauAttention(tf.keras.Model): #название класса именем создателя механизма Дмитрия Богданова(Bahdanau)
  def __init__(self, units): # создаем слой внимания из указанного кол-ва нейронов
    super(BahdanauAttention, self).__init__() #даем возможность использовать и исполнять методы класса-родителя в классе потомке
    self.W1 = tf.keras.layers.Dense(units) #атрибут: получаем веса, пропуская через полносвязный слой
    self.W2 = tf.keras.layers.Dense(units) #атрибут: получаем веса, пропуская через полносвязный слой
    self.V =  tf.keras.layers.Dense(1) #атрибут: пропускаем через Dense с одним нейроном, получаем отдельный вес на каждый такт

  def call(self, hidden_state, values):
    hidden_with_time_axis = tf.expand_dims(hidden_state, 1)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, dec_units, batch_sz):
    super(Decoder, self).__init__() #даем возможность использовать и исполнять методы класса-родителя в классе потомке
    self.batch_sz = batch_sz #атрибут возвращает размер батча
    self.dec_units = dec_units #атрибут возвращает размер слоя в декодере(кол-во нейронов)
    self.gru = tf.keras.layers.GRU(self.dec_units, return_state=True, recurrent_initializer='glorot_uniform')
    self.attention = BahdanauAttention(self.dec_units) #атрибут подключит механизм внимания, описанный ранее
    self.conv1 = tf.keras.layers.Conv1D(10, 3, padding='same', activation= 'sigmoid')
    self.conv_out = tf.keras.layers.Conv1D(1, 3, padding='same')
    self.drop = tf.keras.layers.Dropout(0.2)
  def call(self, x, hidden, enc_output):
    x = tf.reshape(x, (-1,1,80))
    context_vector, attention_weights = self.attention(hidden, enc_output)
    out, state = self.gru(x, initial_state = context_vector)
    state = tf.expand_dims(state, axis=-1)
    state = self.conv1(state)
    state = self.drop(state)
    state = self.conv_out(state)
    state = tf.reshape(state, (-1, 80))
    return state

#class Mel_to_mag(tf.keras.Model):
#  def __init__(self, enc_unit, dec_unit, time_stamp):
#    super(Mel_to_mag, self).__init__() #даем возможность использовать и исполнять методы класса-родителя в классе потомке
#
#    self.enc_unit = enc_unit
#    self.dec_unit = dec_unit
#    self.time_stamp = time_stamp
#    self.inp = tf.keras.layers.InputLayer(input_shape=(int(self.time_stamp/4),self.enc_unit))
#    self.conv1 = tf.keras.layers.Conv1D(dec_unit, 3, padding='same', activation = 'sigmoid')
#    self.batch = tf.keras.layers.BatchNormalization()
#    #self.den = tf.keras.layers.Dense(int(self.dec_unit/2), activation = 'sigmoid')
#    #self.den2 = tf.keras.layers.Dense(self.enc_unit, activation = 'relu')
#    self.conv2 = tf.keras.layers.Conv2D(8, 3, padding='same', data_format='channels_first', activation = 'sigmoid')
#    self.conv3 = tf.keras.layers.Conv2D(1, 3, padding='same', data_format='channels_first')
#    self.up = tf.keras.layers.UpSampling1D(size=4)
#  def call(self, x):
#    x = self.inp(x)
#    x = self.conv1(x)
#    x = self.batch(x)
#    #x = self.up(x)
#    x = tf.expand_dims(x, axis=1)
#    x = self.conv2(x)
#    x = self.conv3(x)
#    x = tf.reshape(x, (-1, self.time_stamp, self.dec_unit))
#    #x = self.den(x)
#    #x = self.conv1(x)
#    #x = tf.reshape(x, (-1, 4, 1025, 1))
#    #x = tf.keras.activations.hard_sigmoid(x)
#    #x = tf.keras.layers.ReLU(max_value=1)(x)
#    return x
class Mel_to_mag(tf.keras.Model):
  def __init__(self):
    super(Mel_to_mag, self).__init__() #даем возможность использовать и исполнять методы класса-родителя в классе потомке
    self.inp = tf.keras.layers.InputLayer(input_shape=(80,))
    self.conv1 = tf.keras.layers.Conv1D(513, 3, padding='same')
    self.conv2 = tf.keras.layers.Conv1D(10, 3, padding='same', activation= 'sigmoid')
    self.conv_out = tf.keras.layers.Conv1D(1, 3, padding='same')

    self.up = tf.keras.layers.UpSampling1D(size=4)

  def call(self, x):
    x = self.inp(x)
    x = tf.reshape(x, (-1, 1, 80))
    x = self.conv1(x)
    #x = self.up(x)
    x = tf.reshape(x, (-1, 513, 1))
    x = self.conv2(x)
    x = self.conv_out(x)
    x = tf.reshape(x, (-1, 513))
    return x

def create_m_m(dec_unit, enc_unit):
  dropaut_rate = 0.1
  query_input = tf.keras.Input(shape=(enc_unit,))
  #x = tf.reshape(query_input, (-1, 1, enc_unit))
  #x = tf.keras.layers.Conv1D(dec_unit, 3, padding='same')(x)
  #x = tf.reshape(x, (-1, dec_unit, 1))
  #x = tf.keras.layers.Conv1D(10, 3, padding='same', activation= 'sigmoid')(x)
  #x = tf.keras.layers.Conv1D(1, 3, padding='same')(x)
  #x = tf.reshape(x, (-1, dec_unit))

  x = tf.keras.layers.Dense(int(dec_unit*2), activation = 'relu')(query_input)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.layers.Dense(int(dec_unit), activation = 'sigmoid')(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.layers.Dense(dec_unit)(x)
  x = tf.reshape(x, (-1, dec_unit, 1))
  x = tf.keras.layers.Conv1D(10, 3, padding='same')(x)
  x = tf.keras.activations.hard_sigmoid(x)
  x = tf.keras.layers.Conv1D(10, 5, padding='same')(x)
  x = tf.keras.layers.Conv1D(1, 3, padding='same')(x)
  x = tf.reshape(x, (-1, dec_unit))


  m_m = tf.keras.Model(query_input, x)
  return m_m

def create_auto(vocabularySize, e, enc_units, timestamp, input_len, dec_unit):
  dropaut_rate = 0.2
  query_input = tf.keras.Input(shape=(input_len,))
  query_embeddings = tf.keras.layers.Embedding(vocabularySize, e, mask_zero=True)(query_input)
  x = tf.keras.layers.LSTM(e, return_sequences = True, dropout=0.3)(query_embeddings)

  x = tf.keras.layers.Dense(timestamp)(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.layers.Conv1D(dec_unit, 3,  padding="same")(x)
  enc = tf.keras.Model(query_input, x)

  return enc


def create_encoder(vocabularySize, e, enc_units, timestamp, input_len):
  dropaut_rate = 0.2
  query_input = tf.keras.Input(shape=(input_len,))
  query_embeddings = tf.keras.layers.Embedding(vocabularySize, e)(query_input)

  x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(e, return_sequences=True, dropout=0.3))(query_embeddings)
  x = tf.keras.layers.Dense(timestamp*3)(x)
  x = tf.keras.layers.SpatialDropout1D(dropaut_rate)(x)
  x = tf.keras.layers.Dense(timestamp)(x)
  x = tf.keras.layers.SpatialDropout1D(dropaut_rate)(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x1 = tf.keras.layers.Dense(enc_units)(x)
  x1 = tf.keras.layers.Softmax(axis = -1)(x1)
  x1 = tf.keras.activations.sigmoid(x1)
  #x = tf.keras.layers.Flatten()(x)

  x = tf.keras.layers.Dense(enc_units*3, activation= 'relu')(x)
  x = tf.keras.layers.SpatialDropout1D(dropaut_rate)(x)
  #x = tf.keras.layers.SeparableConv1D(timestamp,3, data_format='channels_first', padding='same', activation='sigmoid')(x)
  x = tf.keras.layers.Dense(enc_units)(x)
  x = tf.keras.layers.SpatialDropout1D(dropaut_rate)(x)
  x = tf.expand_dims(x, axis=1)
  x = tf.keras.layers.Conv2D(20, 3, padding='same', data_format='channels_first', activation = 'sigmoid')(x)
  x = tf.keras.layers.Conv2D(1, 3, padding='same', data_format='channels_first', activation = 'sigmoid')(x)
  x = tf.reshape(x, (-1, timestamp, enc_units))
  x = x * x1
  x = tf.keras.layers.Dense(enc_units)(x)

  x = tf.keras.activations.sigmoid(x)
  #x = tf.keras.layers.ReLU(max_value=1.)(x)

  enc = tf.keras.Model(query_input, x)
  return enc


def create_encoder2(vocabularySize, e, enc_units, timestamp, input_len):
  dropaut_rate = 0.1
  query_input = tf.keras.Input(shape=(input_len,))
  x = tf.keras.layers.Embedding(vocabularySize, e, mask_zero=True)(query_input)
  x = tf.keras.layers.Conv1D(timestamp, 3, padding = 'same', activation = 'sigmoid', data_format='channels_first')(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.layers.GRU(enc_units, return_sequences=True, dropout = dropaut_rate)(x)
  x = tf.expand_dims(x, axis=1)
  x = tf.keras.layers.Conv2D(20, 3, padding='same', data_format='channels_first' , activation = 'relu')(x)
  x = tf.keras.activations.hard_sigmoid(x)
  x = tf.keras.layers.Conv2D(20, 5, padding='same', data_format='channels_first' , activation = 'relu')(x)
  x = tf.keras.layers.Conv2D(1, 3, padding='same', data_format='channels_first')(x)

  x = tf.reshape(x, (-1, timestamp, enc_units))
  enc = tf.keras.Model(query_input, x)
  return enc
def create_encoder3(vocabularySize, e, enc_units, timestamp, input_len):
  dropaut_rate = 0.02
  query_input = tf.keras.Input(shape=(input_len,))

  query_embeddings = tf.keras.layers.Embedding(vocabularySize, e, mask_zero= True)(query_input)
  x = tf.keras.layers.SeparableConv1D(timestamp, 3, strides=1, padding='same', data_format='channels_first')(query_embeddings)
  x = tf.keras.layers.Dense(enc_units*2, activation = 'sigmoid')(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x = tf.keras.layers.Dense(timestamp*2)(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.activations.sigmoid(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x = tf.keras.layers.Dense(enc_units, activation = 'sigmoid')(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x = tf.keras.layers.Dense(timestamp)(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.keras.activations.sigmoid(x)
  x = tf.transpose(x, perm=[0, 2, 1])
  x = tf.keras.layers.Dense(enc_units)(x)
  #x = tf.keras.layers.ReLU(max_value=1)(x)
  enc = tf.keras.Model(query_input, x)
  return enc


def create_decoder(enc_units, timestamp,  dec_unit):
  query_input = tf.keras.Input(shape=(timestamp, enc_units))
  x = tf.keras.layers.UpSampling1D(size=4)(query_input)
  x = tf.keras.layers.Conv1D(timestamp*4, 3, padding = 'same', data_format = 'channels_first')(x)
  x = tf.keras.layers.Conv1D(dec_unit, 3,  padding="same")(x)




  dec = tf.keras.Model(query_input, x)
  return dec

def create_net(vocabularySize, e, dec_unit, timestamp, input_len):
  dropaut_rate = 0.2
  query_input = tf.keras.Input(shape=(input_len,))
  query_embeddings = tf.keras.layers.Embedding(vocabularySize, e, mask_zero=True)(query_input)
  x = tf.keras.layers.LSTM(e, return_sequences = True, dropout=0.3)(query_embeddings)
  x = tf.keras.layers.Dense(timestamp*4, activation = 'sigmoid')(x)
  x = tf.keras.layers.Dropout(dropaut_rate)(x)
  x = tf.transpose(x, perm=[0, 2, 1])

  x = tf.keras.layers.Dense(dec_unit, activation = 'sigmoid')(x)



  net = tf.keras.Model(query_input, x)
  return net


class MaskedLoss(tf.keras.losses.Loss):

  def __call__(self, y_true, y_pred, sample_weight = None):
    loss = self.loss(y_true, y_pred, sample_weight)
    mask = tf.cast(y_true != 0, tf.float32)
    loss *= mask
    return tf.reduce_sum(loss)