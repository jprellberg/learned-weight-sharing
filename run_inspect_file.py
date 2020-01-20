import numpy as np

np.set_printoptions(threshold=2000, linewidth=1000, precision=3)
file = 'results/cifar100-learned_sharing-20191226T113759Z.VOES.npz'
data = np.load(file, allow_pickle=True)
print(list(data.keys()))
print(data['args'])

print()
print("metrics per task")
print("----------------")
print(data['test/metric_per_task'])

rp = data['test/routing_probs']
for i, p in enumerate(rp):
    print()
    print("routing probabilities @", i)
    print("---------------------")
    for j, mod in enumerate(p):
        print("module", j)
        arr = np.stack([x for x in mod])
        print(arr)
        print(arr.argmax(-1))
