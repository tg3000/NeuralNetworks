from nn import MLP

my_nn = MLP(3, [4, 20, 1], [False, True, False])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for i in range(200):
    ypred = [my_nn(x)[0] for x in xs]
    loss = sum([(yx - yhat) ** 2 for yhat, yx in zip(ys, ypred)])
    if i == 199:
        print(loss)

    loss.zero_grad()
    loss.grad = 1
    loss.backward()
    my_nn.apply_grad(alpha=0.01)
    loss = 0

for i, j in zip(xs, ys):
    print("Got: %0.4f      Expected: %0.4f" % (my_nn(i)[0].data, j))
