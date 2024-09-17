def fractional(decimal_number):
    # Extract the fractional part of the decimal number
    fractional_part = decimal_number - int(decimal_number)
    
    # Initialize an empty string to store the binary representation
    binary_fraction = ''
    
    # Continue multiplying the fractional part by 2 until it equals 1 or a loop limit is reached
    while fractional_part != 0:
        fractional_part *= 2
        if fractional_part >= 1:
            binary_fraction += '1'
            fractional_part -= 1  # Remove the integer part
        else:
            binary_fraction += '0'
        
        # Break if the fractional part becomes exactly 1
        if fractional_part == 1:
            binary_fraction += '1'
            break
    
    print(f"Binary fractional part: 0.{binary_fraction}")

# Example usage
decimal_number = float(input("Enter a decimal number: "))
fractional(decimal_number)

