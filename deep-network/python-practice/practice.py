bigx=10

def yeki_bala(x=bigx):
    x=x+1
    print('x is: %s'%x)

yeki_bala()
bigx=100
yeki_bala()

def add_to_dict(args={'a':1,'b':2}):
    for i in args.keys():
        args[i]+=1
    print(args)
add_to_dict()
add_to_dict()
add_to_dict()

dict1={'a':1,'b':2,'c':3}
print(id(dict1))

dict1['a'] =5
print(id(dict1))

def try_to_modify(x,y,z):
    x=23
    y.append(42)
    z=[99]
    print(x)
    print(y)
    print(z)


a=77
b=[99]
c=[28]
try_to_modify(a, b, c)
print('a=%s'%a)
print('b=%s'%b)
print('c=%s'%c)
