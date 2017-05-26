__author__ = 'Maxim Vedernikov'

import math

def parse(n, m):
    with open('result.txt', 'r') as f:
        for i in range(n):
            minimum = 1e10
            sum = 0
            for j in range(m):
                p = float(f.readline().split()[2])
                sum += p
                minimum = min(p, minimum)
            avg = sum / m
            print(i, avg, minimum)

def count_precise(i):
    t = math.ceil(math.pow(i, 1/3.))
    ans = 1 - 2*((t - 1)/i) - ((math.pi ** 2) / (4 * (t**2)))
    print (i, ans)

if __name__ == '__main__':
    parse(7, 20)
    for i in [10, 20, 50, 100, 200, 500, 1000]:
        count_precise(i)