#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Exercises to understand bitcoin from the basics**

# **First step - finite fields (mathematics)**

# In[ ]:


class FieldElement:
    def __init__ (self, num, prime):
        if num >= prime or num < 0 :
            error = 'Num {} not in the field range 0 to {}'.format (num, prime - 1)
            raise ValueError (error)
        self.num = num
        self.prime = prime

    def __repr__ (self):
        return 'FieldElement_{}({})'.format (self.prime, self.num)
    
    def __eq__ (self, other):
        if other is None:
            return False
        return self.num == other.num and self.prime == other.prime
    
    def __add__ (self, other):
        if self.prime != other.prime:
            raise TypeError ('Cannot add two numbers in different Fields')
        num = (self.num + other.num) % self.prime
        return self.__class__ (num, self.prime)
    
    def __sub__ (self, other):
        if self.prime != other.prime:
            raise TypeError ('Cannot subtract two numbers in different Fields')
        num = (self.num - other.num) % self.prime
        return self.__class__ (num, self.prime)
    
    def __pow__ (self, exponent):
        num = (self.num ** exponent) % self.prime
        return self.__class__ (num, self.prime)
        n = exponent
        while n < 0:
            n += self.prime - 1
        num = pow (self.num, n, self.prime)
        return self.__class__ (num, self.prime)
        n = exponent % (self.prime - 1)
        num = pow (self.num, n, self.prime)
        return self.__class__ (num, self.prime)
    
    def __ne__ (self, other):
        return not (self == other)
    
    def __mul__ (self, other):
        if self.prime != other.prime:
            raise TypeError ("Cannot multiply two numbers in different Fields")
        num = (self.num * other.num) % self.prime
        return self.__class__ (num, self.prime)
    
    def __truediv__ (self, other):
        if self.prime != other.prime:
            raise TypeError ('Cannot divide two numbers in different Fields')
        num = self.num * pow (other.num, self.prime - 2, self.prime) % self.prime
        return self.__class__ (num, self.prime)


# In[ ]:


a = FieldElement (5, 19)
b = FieldElement (8, 25)
print (a == b)


# In[ ]:


print (a == a)


# In[ ]:


a = FieldElement (5, 19)
b = FieldElement (13, 19)
c = FieldElement (6, 19)
print(a + b == c)


# In[ ]:


a = FieldElement (12, 15)
b = FieldElement (5, 16)
print (a ** 3 == b)


# In[ ]:


a = FieldElement (8, 19)
b = FieldElement (4, 12)
print (a ** - 5 == b)


# **Elliptic curves**

# In[ ]:


class Point:
    def __init__ (self, x, y, a, b):
        self.a = a
        self.b = b
        self.x = x
        self.y = y
        if self.x is None and self.y is None:
            return
        if self.y ** 2 != self.x ** 3 + a * x + b:
            raise ValueError ('({}, {}) is not on the curve'.format (x, y))
        
    def __eq__ (self, other):
        if self.x == other.x and self.y != other.y:
            return self.__class__ (None, None, self.a, self.b)
        return self.x == other.x and self.y == other.y \
            and self.a == other.a and self.b == other.b
        
        if self.y ** 2 != self.x ** 3 + a * x + b:
            raise ValueError ('({}, {}) is not on the curve'.format (x, y))
        
    def __add__ (self, other):
        if self.a != other.a or self.b != other.b:
            raise TypeError ('Points {}, {} are not on same curve'.format (self, other))
        if self.x is None:
            return other
        if other.x is None:
            return self
        if self.x != other.x:
            s = (other.y - self.y) / (other.x - self.x)
            x = s**2 - self.x - other.x
            y = s * (self.x - x) - self.y
            return self.__class__ (x, y, self.a, self.b)
        if self == other:
            s = (3 * self.x**2 + self.a) / (2 * self.y)
            x = s**2 - 2 * self.x
            y = s * (self.x - x) - self.y
            return self.__class__ (x, y, self.a, self.b)
        
    def __rmul__ (self, coefficient):
        product = self.__class__ (None, None, self.a, self.b)
        for _ in range (coefficient):
            product += self
        return product
        coef = coefficient
        current = self
        result = self.__class__ (None, None, self.a, self.b)
        while coef:
            if coef & 1:
                result += current
            current += current
            coef >>= 1
        return result
    
    def __ne__ (self, other):
        return not (self == other)
        


# In[ ]:


p1 = Point (-1, -1, 3, 5)
p2 = Point (-1, 1, 3, 5)
inf = Point (None, None, 3, 5)
print (p1 + inf)


# In[ ]:


print (inf + p2)


# In[ ]:


print (p1 + p2)


# **Elliptic curve over Finite Fields**

# In[ ]:


a = FieldElement (num = 0, prime = 223)
b = FieldElement (num = 7, prime = 223)
x = FieldElement (num = 192, prime = 223)
y = FieldElement (num = 105, prime = 223)
p1 = Point (x, y, a, b)
print (p1)


# **Point addition over Finite Fields**

# **Scalar multiplicationn Redux**

# In[ ]:


prime = 223
a = FieldElement (num = 0, prime = prime)
b = FieldElement (num = 7, prime = prime)
x1 = FieldElement (num = 192, prime = prime)
y1 = FieldElement (num = 105, prime = prime)
x2 = FieldElement (num = 17, prime = prime)
y2 = FieldElement (num = 56, prime = prime)
p1 = Point (x1, y1, a, b)
p2 = Point (x2, y2, a, b)
print (p1 + p2)


# In[ ]:


prime = 223
a = FieldElement (0, prime)
b = FieldElement (7, prime)
def on_curve (x, y):
    return y**2 == x**3 + a*x + b
print (on_curve (FieldElement (192, prime), FieldElement (105, prime)))

print (on_curve (FieldElement (17, prime), FieldElement (56, prime)))

print (on_curve (FieldElement (200, prime), FieldElement (119, prime)))

print (on_curve (FieldElement (1, prime), FieldElement (193, prime)))

print (on_curve (FieldElement (42, prime), FieldElement (99, prime)))


# **Coding scalar multiplication**

# In[ ]:


prime = 223
a = FieldElement (0, prime)
b = FieldElement (7, prime)
def on_curve (x, y):
    return y**2 == x**3 + a*x + b
print (on_curve (FieldElement (192, prime), FieldElement (105, prime)))

print (on_curve (FieldElement (17, prime), FieldElement (56, prime)))

print (on_curve (FieldElement (200, prime), FieldElement (119, prime)))

print (on_curve (FieldElement (42, prime), FieldElement (99, prime)))


# **Working with secp256k1**

# In[ ]:


gx = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
gy = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
p = 2**256 - 2**32 - 977
print (gy**2 % p == (gx**3 +7) % p)


# In[ ]:


P = 2**256 - 2**32 - 977

class S256Field (FieldElement):
    def __init__ (self, num, prime = None):
        super ().__init__ (num = num, prime = P)
        
    def __repr__ (self):
        return '{:x}'.format (self.num).zfill (64)


# In[ ]:


A = 0
B = 7
N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141

class S256Point (Point):
    def __init__ (self, x, y, a = None, b = None):
        a, b = S256Field (A), S256Field (B)
        if type (x) == int:
            super ().__init__(x = S256Field (x), y = S256Field (y), a = a, b = b)
        else:
            super ().__init__ (x = x, y = y, a = a, b = b)
            
    def __rmul__ (self, coefficient):
        coef = coefficient % N
        return super ().__rmul__(coef)
    
    def verify (self, z, sig):
        s_inv = pow (sig.s, N - 2, N)
        u = z * s_inv % N
        v = sig.r * s_inv & N
        total = u * G + v * self
        return total.x.num == sig.r
    
    def verify (self, z, sig):
        s_inv = pow (sig.s, N - 2, N)
        u = z * s_inv % N
        v = sig.r * s_inv % N
        total = u * G + v * self
        return total.x.num == sig.r
    
    def sec (self):
        """return the binary version of the SEC format"""
        return b'\x04' + self.x.num.to_bytes (32, 'big') + self.y.num.to_bytes (32, 'big')
    
    def sec (self, compressed = True):
        """returns the binary versiom of the SEC format"""
        if compressed:
            if self.y.num % 2 == 0:
                return b'\x02' + self.x.num.to_bytes (32, 'big')
            else:
                return b'\x03' + self.x.num.to_bytes (32, 'big')
        else:
            return b'\x04' + self.x.num.to_bytes (32, 'big') + self.y.num.to_bytes (32, 'big')


# In[ ]:


G = S256Point (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
               0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
print (N * G)


# **Verifying a signature**

# In[ ]:


class Signature:
    def __init__ (self, r, s):
        self.r = r
        self.s = s
        
    def __repr__ (self):
        return "Signature ({:x}, {:x})".format (self.r, self.s)


# In[ ]:


class PrivateKey:
    def __init__ (self, secret):
        self.secret = secret
        self.point = secret * G
        
    def hex (self):
        return '{:x}'.format (self.secret).zfill (64)
    
    def sign (self, z):
        k = randint (0, N)
        r = (k * G).x.num
        k_inv = pow (k, N - 2, N)
        s = (z + r * self.secret) * k_inv % N
        if s > N/2:
            s = N - s
        return Signature (r, S)
        k = self.deterministic_k (z)
        r = (k * G).x.num
        k_inv = pow (k, N - 2, N)
        s = (z + r * self.secret) * k_inv % N
        if s > N / 2:
            s = N - s
        return Signature (r, s)
    
    def deterministic_k (self, z):
        k = b'\x00' * 32
        v = b'\x01' * 32
        if z > N:
            z -= N
        z_bytes = z.to_bytes (32, 'big')
        secret_bytes = self.secret.to_bytes (32, 'big')
        s256 = hashlib.sha256
        k = hmac.new (k, v + b'\x00' + secret_bytes + z_bytes, s256).digest ()
        v = hmac.new (k, v, s256).digest ()
        k = hmac.new (k, v + b'\x01' + secret_bytes + z_bytes, s256).digest ()
        v = hmac.new (k, v, s256).digest ()
        while True:
            v = hmac.new (k, v, s256).digest ()
            candidate = int.from_bytes (v, 'big')
            if candidate >= 1 and candidate < N:
                return candidate
            k = hmac.new (k, v + b'\x00', s256).digest ()
            v = hmac.new (k, v, s256).digest ()

