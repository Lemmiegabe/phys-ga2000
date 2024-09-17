import math

def solve_quadratic(a, b, c):
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        print("The equation has complex solutions.")
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(-discriminant) / (2*a)
        print(f"The solutions are: {real_part} ± {imaginary_part}i")
    else:
        # Calculate the two solutions
        solution1 = (-b + math.sqrt(discriminant)) / (2*a)
        solution2 = (-b - math.sqrt(discriminant)) / (2*a)
        print(f"The solutions are: {solution1} and {solution2}")

# Take input from the user
a = float(input("Enter the coefficient a: "))
b = float(input("Enter the coefficient b: "))
c = float(input("Enter the coefficient c: "))

# Solve the quadratic equation
solve_quadratic(a, b, c)