import librosa # audio
import numpy as np
from Constant import *
from scipy import signal
import copy, time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import tensorflow as tf

mel_basis = librosa.filters.mel(sr, n_fft, n_mels)
inv_mel_filtr = np.linalg.pinv(mel_basis)

'''             read path to files          '''
def get_path(path = ''):
    aud_path = []
    txt_path = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-3:] == 'txt':
                txt_path.append(os.path.join(dirpath, filename))
            elif filename[-3:] == 'wav':
                aud_path.append(os.path.join(dirpath, filename))
    return aud_path, txt_path


'''                 load text                   '''


def path_txt_to_txt(path_txt):
    with open(path_txt, "r", encoding="utf-8") as file:
        txt = file.read()
        txt = txt.replace(',', ' ,')
        txt = txt.replace('!', ' !')
        txt = txt.replace('?', ' ?')
        txt = '<start> ' + txt[:-1] + ' <end>'

    return txt
def len_of_inp(txts):
    lens = []
    for txt in txts:
        lens.append(len(txt.split(' ')))
    return np.amax(lens)


def text_to_token(x, input_len):
    cur_time = time.time()
    tokenizer = Tokenizer(filters='"#$%&()*+-–/…:;<=>@[\\]^`{|}~«»\t\n\ufeff', lower=True,
                          split=' ', oov_token='unknown', char_level=False)
    tokenizer.fit_on_texts(x)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
    items = list(tokenizer.word_index.items())  # Вытаскиваем индексы слов для просмотра
    vocabularySize = len(items) + 1
    padded = pad_sequences(tokenizer.texts_to_sequences(x), maxlen=input_len, padding='post')
    print('Время обработки: ', round(time.time() - cur_time, 2), 'c', sep='')
    print(items[:20])  # Посмотрим 20 самых часто встречающихся слов
    print("Размер словаря", len(items))  # Длина словаря
    print(x[100])
    print(padded[100])
    print("Размер предложения", len)  # Длина словаря
    return tokenizer, np.array(padded, dtype = np.float32), vocabularySize

'''             load audio              '''


def my_griffin_lim(spectrogram, n_iter=3):
    #копируем спектр
    x_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        # получаем сигнал
        x_t = librosa.istft(x_best, hop_length, window="hann")
        # получаем спектр
        spec = librosa.stft(x_t, n_fft, hop_length)
        # берём фазу
        phase = np.angle(spec)
        # Получаем полный спектр исходного сигнала
        x_best =  spectrogram*np.cos(phase) + 1j*spectrogram*np.sin(phase)
    # Итоговый сигнал
    x_t = librosa.istft(x_best, hop_length, window="hann")
    return np.real(x_t)
def spectrogram2wav(mag,  mags_max):
    # Транспонируем
    mag = mag.T
    # Денормализуем
    #mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db
    mag = (mag* (100+mags_max)) - 100
    # to amplitude
    mag = np.power(5.0, mag * 0.05)
    # Реконструкция
    wav = my_griffin_lim(mag**power)
    # Убираем префазного фильтра
    wav = signal.lfilter([1], [1, -preemphasis], wav)
    # обрезка по краям
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)

def get_mel_mag(y, max_len):

    # Обрезка тишины по краям
    y, _ = librosa.effects.trim(y)
    # Добиваем до одинаковых размеров
    y = np.array(np.append(y, np.random.normal(0, 1e-05, (max_len - len(y), 1))))
    y = np.where((y < 1e-05) & (y > -1e-05), 0, y)
    # Добавление префазного фильтра
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    # Оконное преобразование Фурье
    spectrogram = librosa.stft(y=y,
                               n_fft=n_fft,
                               hop_length=hop_length,
                               win_length =win_length)
    # Амплитуда
    mag = np.abs(spectrogram)
    # Мел-спектр
    mel = np.dot(mel_basis, mag).astype(np.float32)
    # Переводим в децибелы
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    # Нормализуем
    #mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    #mag = (mag  - ref_db + max_db) / max_db
    # Транспонируем
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    # Добиваем нулями до правильных размерностей
    t = mel.shape[0]
    num_paddings = r - (t % r) if t % r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    # Понижаем частоту дискретизации для мел-спектра

    #mel = tf.keras.layers.AveragePooling1D(pool_size=4, strides=4, padding='valid')(mel)
    return mel, mag
def load_audio(path, sr):
  x_train = []
  y_train = []
  curr_time = time.time()
  len_of_y = []
  mel_out = []
  mag_out = []
  for i in path[0:9000]:
    y, sr = librosa.load(i, sr = sr) # Загрузга аудио
    if len(y) <= 5 * sr:
        y_train.append(np.array(y))
        x_train.append(path_txt_to_txt(i.replace('.wav', '.txt')))
        #out.append(get_mel_mag(y))
        len_of_y.append(len(y))
  max_len = np.amax(len_of_y)
  for i in range(len(y_train)):
    mel,mag = get_mel_mag(y_train[i], max_len)
    mel_out.append(mel)
    mag_out.append(mag)
  print("Времени ушло " + str(round(time.time() - curr_time)))
  return np.array(x_train), np.array(mel_out),  np.array(mag_out)

def sintezis(text, input_len, tokenizer, enc, dec, m_m, dec_unit, mags_max): #, enc,  m_m, enc_units, dec_unit
    inp = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=input_len, padding='post')
    #enc_hidden = enc.initialize_hidden_state()
    enc_output, enc_hidden, var = enc(np.reshape(inp, (1,input_len)), training = False)
    dec_hidden = enc_hidden
    mels = []
    for t in range(var.shape[1]):
        dec_input = var[:, t, :]
        dec_hidden = dec(dec_input, dec_hidden, enc_output, training = False)
        mels.append(dec_hidden)
    #mels = m_m.predict(np.reshape(inp, (1,input_len)))
    print((np.array(mels).shape))
    mags = []
    for i in range(len(mels)):
        mags.append((m_m.predict(mels[i])))
    #print(mags.shape)
   #with tf.GradientTape() as tape:

   #    enc_output = enc(np.reshape(inp, (1,input_len)))

   #    enc_output = np.reshape(enc_output,(1,-1,enc_units))

   #    predict = m_m(enc_output)


    out =spectrogram2wav(np.reshape(np.array(mags), (-1, dec_unit)), mags_max)
        
    return out
def sintezis2(text, input_len, tokenizer, enc, m_m, dec_unit, mags_max): #, enc,  m_m, enc_units, dec_unit
    inp = pad_sequences(tokenizer.texts_to_sequences(text), maxlen=input_len, padding='post')
    mels = enc.predict(np.reshape(inp, (1,input_len)))
    mags = []
    for i in range(len(mels)):
        mags.append((m_m(mels[i])))
    #print(mags.shape)
    out =spectrogram2wav(np.reshape(np.array(mags), (-1, dec_unit)), mags_max)

    return out


