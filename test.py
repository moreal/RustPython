import dataclasses

print(dict.__new__)
print(dict.__new__)

# @dataclasses.dataclass
class A:
    ...

print(A.__init__)
print(object.__init__)
print(A.__new__)
print(object.__new__)
