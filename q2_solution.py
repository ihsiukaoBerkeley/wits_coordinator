def convert_to_decimal(number_str, base_name):
    #map base names to their numeric values
    base_dict = {
        "binary": 2,
        "octal": 8,
        "decimal": 10,
        "hex": 16
    }
    
    base = base_dict.get(base_name, None)
    if base is None:
        raise ValueError(f"Unsupported base")
    
    base = base_dict[base_name]
    
    #split into integer and fractional parts
    if "." in number_str:
        integer_part, fractional_part = number_str.split(".")
    else:
        integer_part = number_str
        fractional_part = ""
    
    result = 0.0
    
    try:
        #convert integer part (from right to left)
        for i, digit in enumerate(reversed(integer_part)):
            digit_value = int(digit, base)
            result += digit_value * (base ** i)
        
        #convert fractional part (from left to right)
        for i, digit in enumerate(fractional_part):
            digit_value = int(digit, base)
            result += digit_value * (base ** -(i + 1))
            
    except ValueError:
        raise ValueError(f"Invalid digit for base")
    
    return result
