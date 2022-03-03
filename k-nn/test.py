from knn import K_nearest_neighbour as knn

data = knn.GenerateData((
    (1, 2, 3, 4, 5),
    (10, 11, 12),
    ("hehe", "haha", "hihi", "hoho", "huhu")
), 1000)

k = knn(data)
results = k.Get_knn(3, [1, 10, "haha"])
print(results)