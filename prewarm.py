from app import get_arcface
print("prewarm: starting")
fa = get_arcface()
print("prewarm: ready =", bool(fa))
