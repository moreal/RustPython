# Instance attributes of heap types are stored inline until something asks
# for the instance __dict__; the loops run long enough to get specialized.
import copy
import pickle


class C:
    def __init__(self, x):
        self.x = x
        self.y = x + 1

    def m(self):
        return self.x


def bump(o, n):
    for _ in range(n):
        o.x = o.x + 1


c = C(1)
bump(c, 1000)
assert c.x == 1001
assert c.__dict__ == {"x": 1001, "y": 2}

# After the dict exists, the same specialized code keeps it in sync.
c.__dict__["x"] = 5
assert c.x == 5
bump(c, 1000)
assert c.x == 1005 and c.__dict__["x"] == 1005


class O:
    pass


# vars() keeps per-instance insertion order, not the shared key order.
a = O()
a.p = 1
a.q = 2
b = O()
b.q = 1
b.p = 2
assert list(vars(a)) == ["p", "q"]
assert list(vars(b)) == ["q", "p"]
b2 = O()
b2.p = 1
b2.q = 2
del b2.p
b2.p = 3
assert list(vars(b2)) == ["q", "p"]

d = O()
d.p = 1
del d.p
assert not hasattr(d, "p")
try:
    del d.p
except AttributeError:
    pass
else:
    assert False, "deleting a missing attribute must raise"
assert vars(d) == {}

e = O()
e.p = 1
e.__dict__ = {"z": 5}
assert e.z == 5 and not hasattr(e, "p")
e.w = 3
assert e.__dict__ == {"z": 5, "w": 3}


class P:
    pass


f = O()
f.p = 1
f.q = 2
f.__class__ = P
assert vars(f) == {"p": 1, "q": 2}
g = P()
g.q = "g"
assert g.q == "g" and vars(g) == {"q": "g"}


class S(str):
    pass


h = O()
setattr(h, S("p"), 4)
assert h.p == 4
assert type(next(iter(vars(h)))) is S

# More names than the shared keys hold.
k = O()
for i in range(100):
    setattr(k, f"a{i}", i)
for i in range(100):
    assert getattr(k, f"a{i}") == i
assert len(vars(k)) == 100

assert vars(copy.copy(C(3))) == {"x": 3, "y": 4}
assert vars(copy.deepcopy(C(3))) == {"x": 3, "y": 4}
assert vars(pickle.loads(pickle.dumps(C(5)))) == {"x": 5, "y": 6}


class D(C):
    pass


assert D(10).m() == 10 and vars(D(10)) == {"x": 10, "y": 11}


def call_m(o, n):
    r = None
    for _ in range(n):
        r = o.m()
    return r


o1 = C(1)
assert call_m(o1, 1000) == 1
o1.m = lambda: "shadow"
assert call_m(o1, 10) == "shadow"
assert call_m(C(2), 10) == 2


class Slotted:
    __slots__ = ("s", "__dict__")


sd = Slotted()
sd.s = 1
sd.x = 2
assert sd.s == 1 and sd.x == 2 and vars(sd) == {"x": 2}


# A finalizer run by replacing an attribute may touch the same object.
class Finalizer:
    def __init__(self, o):
        self.o = o

    def __del__(self):
        self.o.seen = self.o.v


t = O()
t.v = Finalizer(t)
t.v = 2
assert t.seen == 2
