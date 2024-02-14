import sys
import numpy as np
import matplotlib.pyplot as plt

for f in sys.argv[1:]:
    d = np.loadtxt(f, delimiter=',', usecols=(1, 5))
    snr = d[:, 0]
    ber = d[:, 1]
    plt.plot(snr, ber, label=f)

plt.semilogy()
plt.legend()
plt.grid()
plt.show()
