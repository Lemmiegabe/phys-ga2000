import math

a,b,c = map(float, input("Enter the coefficients of a, b and c spaced out: ").split())

d = b**2-4*a*c # discriminant

if d < 0:
    print ("This equation has no real solution")
elif d == 0:
    x = 2*c/(-b+math.sqrt(d))
    print ("This equation has one solutions: ", x)
else:
    x1 = 2*c/(-b+math.sqrt(d))
    x2 = 2*c/(-b-math.sqrt(d))
    print ("This equation has two solutions: ", x1, " and", x2)