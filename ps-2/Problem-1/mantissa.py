def Mantissa(n):
    # Continue until the number is less than 2
    while n >= 2:
        remainder = n % 2  # Calculate the remainder when divided by 2
        print(f"Number: {n}, Remainder: {remainder}")
        n //= 2  # Divide the number by 2 (integer division)

    # Print the final value less than 2
    print(f"Final Number: {n}")


number = 100  
Mantissa(number)