

maxWordsCount = 18000 # @param {type:"integer"}
frame_shift = 0.0125
frame_length = 0.05
sr = 16000 # @param {type:"integer"}
batch_size = 50 # @param {type:"integer"} размер батча
latent_dim = 256  # @param {type:"integer"} размер скрытого слоя/пространства
dropout_rate = 0.1 # @param {type:"integer"} размер слоя регуляризации, "выключим" указанное количество нейронов, во избежание переобучения
start_lr = 0.0001  # @param {type:"integer"} шаг обучения

n_fft = 1024 # @param {type:"integer"} Длина кадра
hop_length = 256 # @param {type:"integer"} На сколько выборок сдвигать окно
n_mels = 80 # @param {type:"integer"} Колличество мел фильтров
win_length = 1024   # @param {type:"integer"} Размер окна
max_db = 120 # @param {type:"integer"}
ref_db = 20 # @param {type:"integer"}
preemphasis = .97
r = 4
e = 128 # @param {type:"integer"} embedding

power = 1.5