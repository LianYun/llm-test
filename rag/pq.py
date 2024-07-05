from random import randint
# From https://www.pinecone.io/learn/series/faiss/product-quantization/
x = [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]
m = 4
D = len(x)
D_ = int(D / m)
u = [x[row:row+D_] for row in range(0, D, D_)]
print(f"{m=}, {D=}, {D_=}, \n{u=}")

k = 2**5
k_ = int(k / m)
print(f"{k=}, {k_=}")

c = []
for j in range(m):
    c_j = []
    for i in range(k_):
        c_ji = [randint(0, 9) for _ in range(D_)]
        c_j.append(c_ji)
    c.append(c_j)
print(f"c dimensions: {len(c)}x{len(c[0])}x{len(c[0][0])}")

def euclidean(v, u):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance

def nearest(c_j, u_j):
    distance = 9e9
    for i in range(k_):
        new_dist = euclidean(c_j[i], u_j)
        if new_dist < distance:
            nearest_idx = i
            distance = new_dist
    return nearest_idx


ids = []
for j in range(m):
    i = nearest(c[j], u[j])
    ids.append(i)
print(f"{ids=}")

q = []
for j in range(m):
    c_ji = c[j][ids[j]]
    q.extend(c_ji)
print(f"{q=}")