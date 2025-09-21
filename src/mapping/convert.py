def decimal_to_binary(num):
    """Converts decimal to binary"""
    if num < 0:
        return "Error"
    if num >= 1:
        raise ValueError("Number must be less than 1")

    integer_part = int(num)
    fractional_part = num - integer_part

    # binary_integer = ""
    # if integer_part == 0:
    #     binary_integer = ""
    # else:
    #     while integer_part > 0:
    #         binary_integer = str(integer_part % 2) + binary_integer
    #         integer_part //= 2

    binary_fractional = ""
    while fractional_part > 0 and len(binary_fractional) < 10:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional += str(bit)
        fractional_part -= bit

    # return binary_integer + binary_fractional
    return binary_fractional


def binary_to_decimal(binary_str):
    """Converts binary to decimal"""
    decimal = 0.0
    for i, bit in enumerate(binary_str, 1):
        # print(f"{bit} -- {i}")
        if bit == '1':
            decimal += 1.0 / (2 ** i)
    return decimal


def decimal_to_quaternary(num):
    """Converts decimal to quaternary"""
    if num < 0:
        return "Error"
    if num >= 1:
        raise ValueError("Number must be less than 1")

    integer_part = int(num)
    fractional_part = num - integer_part

    binary_fractional = ""
    while fractional_part > 0 and len(binary_fractional) < 10:
        fractional_part *= 4
        bit = int(fractional_part)
        binary_fractional += str(bit)
        fractional_part -= bit

    return binary_fractional