def add1(x):
    if x>10:
        return x
    else:
        x += 1
        return add1(x)

print(add1(8))