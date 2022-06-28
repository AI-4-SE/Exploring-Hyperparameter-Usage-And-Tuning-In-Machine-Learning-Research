#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import unicodedata

from rich import print


# ## It looks the same! But is it the same?
# 
# Here is a question. Will `kāśī == kāśī` return `True` or `False`? Stop and think about it for a moment.

# In[ ]:


import urllib

import matplotlib.pyplot as plt
import numpy as np
import PIL


# In[ ]:


plt.figure(figsize=(60, 60))
plt.axis(False)
plt.imshow(
    np.array(
        PIL.Image.open(
            urllib.request.urlopen(
                "https://user-images.githubusercontent.com/43331416/169664045-5c0b8f98-e63a-4775-8558-97b9d30f7811.png"
            )
        )
    )
)


# Well, they both look the same so the answer must be True and well it is.

# In[ ]:


x = "kāśī"
y = "kāśī"

print(x == y)


# But there’s a catch.

# In[ ]:


x = "kāśī"
y = "kāśī"

print(x == y)


# So what just happened? Examining both the strings, we get an interesting insight.

# In[ ]:


x = "kāśī"
print(len(x))


# In[ ]:


y = "kāśī"
print(len(y))


# Let's make things more concrete by considering just a single glyph.

# In[ ]:


x = "ā"
print(len(x))


# In[ ]:


y = "ā"
print(len(y))


# Both the strings have different lengths. But how is it possible?
# 
# To answer this question, we need to dive into character encodings, Unicode, and UTF-8.

# ## Strings to Bytes and Bytes to Strings
# 
# We need to convert the strings to bytes to see the real difference. Python `str` have `encode` method to convert `str` to `bytes`. Similarly, `bytes` have `decode` method to convert `bytes` to `str`.

# In[ ]:


print(x.encode())


# In[ ]:


print(y.encode())


# So here's where the fun begins. The bytes differ in the two. So when we loop over the strings, we find that the actual characters in the strings differ.

# In[ ]:


for i in x:
    print(i)


# In[ ]:


for i in y:
    print(i)


# But when we decode the `bytes` back to `str`, we get the strings as-is.

# In[ ]:


print(b"\xc4\x81".decode())


# In[ ]:


print(b"a\xcc\x84".decode())


# Okay, it's confusing and not making any sense. Let's jot down the confusing points:
# 
# 1. Both strings *look* the same.
# 
# 1. Both strings have different underlying representations.
# 
# 1. Why are we looking at the `bytes` instead of directly working with `str`?
# 
# 1. Why do we get different bytes for the same string?
# 
# 1. How can two strings be different but look the same?
# 
# 1. Why did we get `True` earlier but `False` later?

# ## Character Encodings
# 
# The answer to all the above questions lies in what we call character encoding. Fundamentally, computers just deal with numbers. More specifically, they deal just with binary numbers $0$ and $1$ known as bits grouped together into a byte of 8 bits. So we are left with the simple task of assigning a unique number to each character essentially creating a one-to-one mapping between each character and a number.
# 
# ASCII (American Standard Code for Information Interchange) was developed by ANSI (American National Standards Institute) which encoded 128 characters using a 7-bit number from $0x00$ to $0x7F$. "A" is assigned $65\,(0x41)$. "0" is assigned $48\,(0x30)$. In ASCII, each byte represents one character. A string of length $n$ will require $n$ bytes of space. Both `str.encode` and `bytes.decode` methods have encoding parameter in order to faciliate working.

# In[ ]:


help(str.encode)


# In[ ]:


help(bytes.decode)


# In[ ]:


print(" ".join((f"{hex(i)}" for i in "PYTHON".encode())))


# In[ ]:


print(" ".join((f"{i:08b}" for i in "PYTHON".encode())))


# This works very elegantly. But it will only work as long as there are no characters beyond the English alphabet. There was no way of representing characters of other languages in that system. Similarly, other countries would develop their own encodings. The mappings are more or less arbitrary. German systems can assign "Ä" to $65\,(0x41)$ and it works as expected on their machines.
# 
# A stream of bytes does not mean anything in isolation. It requires an encoding to be actually parsed correctly. $65$ for ASCII would be completely different from $65$ for CP1258. Matters worsen when languages like Chinese are considered which have over $70000$ characters and can't be stored using $1$ byte.

# In[ ]:


print(0xe3)


# In[ ]:


print(b"\xe3".decode("cp720")) # Arabic


# In[ ]:


print(b"\xe3".decode("cp1255")) # Hebrew


# In[ ]:


print(b"\xe3".decode("cp1258")) # Vietnamese


# In[ ]:


print(b"\xe3".decode("latin_1")) # Western Europe


# The full list of codecs supported by Python is listed here: https://docs.python.org/3/library/codecs.html

# ## Unicode
# 
# The problems listed above were solved by the formation of **Unicode** standard which would go on to incorporate characters of a lot of languages over the years. To be precise, Unicode encodes scripts of languages rather than languages themselves thus making it agnostic to the underlying language.
# 
# In order to really appreciate the genius and simplicity of the system, we need to take a closer look.
# 
# The first task is to gather all the characters and assign unique numbers sequentially. This number assigned is known as **Unicode Code Point**. The code point of a character can be queried with `ord` function. The interesting thing to note here is that Unicode maintains compatibility with ASCII since ASCII was used almost ubiquitously in the English speaking countries. Thus the ASCII value is equal to the Unicode Code Point for those characters. ASCII ends at $127$ while Unicode can have over $1$ million characters.

# In[ ]:


help(ord)


# In[ ]:


print(ord("A"))


# In[ ]:


print(ord("ã"))


# In[ ]:


print(ord("ñ"))


# Similarly, `chr` is the inverse function of `ord` and it returns the character associated with the particular code point.

# In[ ]:


help(chr)


# In[ ]:


print(chr(3746))


# In[ ]:


print(chr(22472))


# In[ ]:


print(chr(65))


# In[ ]:


print(chr(2325))


# This solves a major problem of portability. Now the systems that support Unicode do not have to worry about wrong interpretation and parsing. Each code point is uniquely associated with a character so there are no ambiguities. A thing to be very specifically noted here is that the code point just refers to a unique integer assigned to a character. It is not concerned with how the bytes are arranged in memory. Unicode supports variety of characters including accents, signs, and emojis.
# 
# So in the above case, "ā" can be input in two ways:
# 
# 1. A single character "ā"
# 
# 1. A single character "a" followed by an overbar character to form "ā"
# 
# Both of them are different strings but the fonts render the same character. This abstract character formed by a sequence of one or more code points is known as a **Unicode glyph**. Thus `kāśī` can be formed with a combination of a lot of different code points. They all represent the same glyph but are different strings completely. To make things more concrete, here is an example in *devanāgarī* script. "की" appears to be a single character but it is infact formed by addition of two characters "क" $(2325)$ and "ी" $(2368)$.
# 
# > **So the answer to the question 'Is `kāśī` == `kāśī`?' is neither True nor False but "Maybe!"**
# 
# There is a way to solve this problem. One must always first use `unicodedata.normalize` function to normalize the text. There are $4$ available options. The ones that end in $C$ result in a composed form while the ones that end in $D$ result in a decomposed form. It doesn't matter which one we use as long it is used uniformly in the entire process.

# In[ ]:


help(unicodedata.normalize)


# In[ ]:


x = "kāśī"
y = "kāśī"


# In[ ]:


print(x == y)


# In[ ]:


print(unicodedata.normalize("NFC", x) == unicodedata.normalize("NFC", y))


# In[ ]:


print(unicodedata.normalize("NFD", x) == unicodedata.normalize("NFD", y))


# In[ ]:


print(len(unicodedata.normalize("NFC", y)))


# In[ ]:


print(len(unicodedata.normalize("NFD", x)))


# In[ ]:


plt.figure(figsize=(60, 60))
plt.axis(False)
plt.imshow(
    np.array(
        PIL.Image.open(
            urllib.request.urlopen(
                "https://user-images.githubusercontent.com/43331416/169764515-f01d15f4-4860-43cf-97bc-97374b117b4c.png"
            )
        )
    )
)


# ## UTF-8
# 
# The real engineering starts here. Once we assign a number to a character, how do we represent it in bytes? Going back to ASCII which is an $8$-bit encoding, every character takes up $1$ byte. Since Unicode stores has code point as high as a million, we require around $4$ bytes. Now one easy way would be to encode everything using $32$-bits. This would be an absolute waste of space. Now suddenly, all the files are $4$ times the original size. Also there would be a lot of long sequences of zeros. For example, ASCII representation of "A" is $01000001$. If the earlier mentioned system was to be followed, the Unicode representation would be $00000000\,00000000\,00000000\,01000001$. Many systems work in a way where receiving an ASCII NUL $(00000000)$ would signal the end of transmission. Also just like ASCII, we cannot directly use the binary representation of the code point because since a number can be arbitrarily small or large, there would be no way to know the boundary of a character as the system would not know how many bytes to read before actually parsing them into a character.
# 
# So a system was needed which would be able to store binary values of code point using the minimum number of bytes required and would also store how much further to read without needing any extra mechanism which would be a waste of both space and time. At the same time, it had to be compatible with ASCII encoding. Along with this, there should never be a series of NUL bytes unless explicitly required. To tackle all these issues, a new encoding was created known as the **Unicode Translation Format - 8-bit** or simply **UTF-8**.
# 
# The UTF-8 encoding works as follows:
# 
# - If the character has only one byte, the first bit is set to $0$.
# - If the character has more than one byte, the first byte starts with the number of $1$ equal to the number of bytes required to encode the code point followed by a $0$. The remaining bytes all start with bits $10$.
# - All the remaining bits are set to the binary representation of the code point and padding them with the necessary number of $0$.
# 
# | First code point | Last code point | Byte 0     | Byte 1     | Byte 2     | Byte 3     |
# | ---------------: | --------------: | ---------- | ---------- | ---------- | ---------- |
# |             U+00 |            U+7F | `0xxxxxxx` |            |            |            |
# |            U+080 |           U+7FF | `110xxxxx` | `10xxxxxx` |            |            |
# |           U+0800 |          U+FFFF | `1110xxxx` | `10xxxxxx` | `10xxxxxx` |            |
# |          U+10000 |        U+10FFFF | `11110xxx` | `10xxxxxx` | `10xxxxxx` | `10xxxxxx` |
# 
# This simple yet genius trick means that one simply needs to find just the last header byte to know which byte they are reading and find the next header byte to know the word boundary. Nowhere in this sequence will we ever have a NUL byte unless sent explicitly. This is backward compatible with ASCII and thus all the ASCII text is automatically compatible with Unicode. The examples of how code points are converted to bytes are enlisted in [UTF-8 Wikipedia](https://en.wikipedia.org/wiki/UTF-8). If one knows the hexadecimal code point, we can get the character.

# In[ ]:


print("\u00fe") # U+FE


# In[ ]:


print("\u0915") # U+915


# In[ ]:


print("\u5e8f") # U+5E8F


# In[ ]:


print("\U0001f925") # U+1f925


# One can get `bytes` from `str` and get `str` from `bytes`.

# In[ ]:


print("序".encode("utf-8"))


# In[ ]:


print(b"\xe5\xba\x8f".decode("utf-8"))


# In[ ]:


print("पूर्व्यांश".encode("utf-8"))


# In[ ]:


print(
    b"\xe0\xa4\xaa\xe0\xa5\x82\xe0\xa4\xb0\xe0\xa5\x8d\xe0\xa4\xb5\xe0\xa5\x8d\xe0\xa4\xaf\xe0\xa4\xbe\xe0\xa4\x82\xe0\xa4\xb6".decode(
        "utf-8"
    )
)


# In[ ]:


print("🌹".encode("utf-8"))


# In[ ]:


print(b"\xf0\x9f\x8c\xb9".decode("utf-8"))


# In[ ]:


print("Python3".encode("ascii").decode("utf-8"))


# In[ ]:


print("Python3".encode("utf-8").decode("ascii"))


# ***

# In[ ]:


from datetime import datetime

print(f"Published at {datetime.now()} by https://www.kaggle.com/dhruvildave")


# ***
