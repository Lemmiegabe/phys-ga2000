import struct
Value=100.98763
def binary_to_float32(binary_string):
    # Ensure the binary string is 32 bits long
    if len(binary_string) != 32:
        raise ValueError('Binary string must be 32 bits long.')
    
    # Convert binary string to an integer
    int_representation = int(binary_string, 2)
    
    # Pack the integer as 32-bit binary and unpack it as a float
    float_value = struct.unpack('!f', struct.pack('!I', int_representation))[0]
    
    return float_value

# Example usage
binary_input = '01000010110010011111100110101010'  
float_value = binary_to_float32(binary_input)
print(f'The acutal value is:       {Value}')
print(f'The 32-bit float value is: {float_value}')
print(f'The difference is:         {Value-float_value}')