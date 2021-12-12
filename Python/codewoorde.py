import numpy as np
import itertools
from kanaalcodering import Kanaalcodering

obj = Kanaalcodering()
def gewicht(bitvec):
    g = 0
    for bit in bitvec:
        if bit == 1: g += 1
    return g

def codewoorden():
    G = np.array([
            [1,1,0,0,0,0,0,0,1,0,0,0,0,0],
            [0,1,0,0,0,1,0,0,1,0,1,0,0,0],
            [0,0,1,0,0,1,1,0,0,0,0,0,0,0],
            [0,1,0,1,0,1,1,0,0,0,0,0,0,0],
            [0,1,0,0,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,0,1,0,0,1,0,0],
            [0,0,0,0,0,1,0,0,1,0,0,0,1,0],
            [0,1,0,0,0,1,1,1,1,0,0,0,0,0],
            [0,1,0,0,0,0,1,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,1,0,1,1,0,0,0,0]
        ])
    b = [np.array(bits) for bits in itertools.product([0, 1], repeat=10)]
    
    cw = []
    for bits in b:
        ci = np.mod(np.matmul(bits, G),2)
        cw.append(ci)
    return cw

def lengte3(woorden):
    N = 0
    for woord in woorden:
        if gewicht(woord) == 3:
            N += 1
    return N

def pm(woorden ,p):
    sum = 0
    for c in woorden:
        sum += np.power(p, gewicht(c)) * np.power(1-p, 14-gewicht(c))
    return sum

def crc5_codewoorden():
    min_gewicht = 100000
    
    b = [np.array(bits) for bits in itertools.product([0, 1], repeat=5)]
    c = np.array([obj.encodeer_inwendig(woord, g_x=[1,1,0,1,0,1]) for woord in b[1:]])
    print(c)
    for woord in c:
        if(gewicht(woord) < min_gewicht): min_gewicht = gewicht(woord)

    return min_gewicht
def pm_crc5():
    b = [np.array(bits) for bits in itertools.product([0, 1], repeat=5)]
    c = np.array([obj.encodeer_inwendig(woord, g_x=[1,1,0,1,0,1]) for woord in b[1:]])
    print(c)
    sum = 0
    for woord in c:
        sum += (0.05)**(gewicht(woord)) * (0.95)**(10-gewicht(woord))
        print((0.05)**(gewicht(woord)) * (0.95)**(10-gewicht(woord)))
    return sum

def crc8_codewoorden():
    min_gewicht = 100000
    
    b = [np.array(bits) for bits in itertools.product([0, 1], repeat=2)]
    c = np.array([obj.encodeer_inwendig(woord, g_x=[1,1,0,0,1,1,0,1,1]) for woord in b[1:]])
    for woord in c:
        if(gewicht(woord) < min_gewicht): min_gewicht = gewicht(woord)

    return min_gewicht
def pm_crc8():
    b = [np.array(bits) for bits in itertools.product([0, 1], repeat=2)]
    c = np.array([obj.encodeer_inwendig(woord, g_x=[1,1,0,0,1,1,0,1,1]) for woord in b[1:]])
    sum = 0
    for woord in c:
        sum += (0.05)**(gewicht(woord)) * (0.95)**(10-gewicht(woord))
    return sum
# def pf(error, all_error, woorden, p):
#     sum = 0
#     for e in all_error:
#         if e.tolist() not in error.tolist():
#             sumi = 0
#             for c in woorden:
#                 ec = np.mod(np.add(e,c), 2)
#                 sumi += np.power(p, gewicht(ec)) * np.power(1-p, 14-gewicht(ec))
#             sum += sumi
#     return sum

# err = np.array([
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [1,0,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,1,0,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,1,0,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,1,0,0,0,0,0,0],
# [0,0,0,0,0,0,0,0,1,0,0,0,0,0],
# [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
# [0,0,0,0,0,0,0,0,0,0,1,0,0,0],
# [0,0,0,0,0,0,0,0,0,0,0,1,0,0],
# [0,0,0,0,0,0,0,0,0,0,0,0,1,0],
# [0,0,0,0,0,0,0,0,0,0,0,0,0,1],
# [1,0,0,0,0,0,1,0,0,0,0,0,0,0]
# ])

# e = [np.array(bits) for bits in itertools.product([0, 1], repeat=14)]

# cw = codewoorden()
# print(lengte3(cw))
# print(pm(cw[1:], 0.05))
# #print(pf(err[1:], e[1:], cw[1:], 0.05))

print(pm_crc5())
print(pm_crc8())
print(crc8_codewoorden())