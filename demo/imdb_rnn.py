from sunyata.dataset.imdb import load_imdb


data, tf = load_imdb()
print(data[0][0].shape)
print(data[0][1].shape)
