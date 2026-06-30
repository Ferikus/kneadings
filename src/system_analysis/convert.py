def decimal_to_number_system(num, system_num):
    """Converts decimal to a certain number system"""
    if num < 0:
        return "Error"
    if num >= 1:
        raise ValueError("Number must be less than 1")
    if num == 0:
        return "0"

    sequence = ""
    while num > 0:
        num *= system_num
        bit = int(num)
        sequence += str(bit)
        num -= bit

    return sequence


def convert_heavy_tail_to_sequence(num, system_num, kneading_len):
    """Converts heavy-tailed kneading invariant to kneading sequence"""
    sequence = decimal_to_number_system(num, system_num)
    if sequence.isdigit():
        sequence = sequence[::-1]
        if len(sequence) < kneading_len:
            sequence = '0' * (kneading_len - len(sequence)) + sequence
    return sequence


def binary_to_decimal(binary_str):
    """Converts binary to decimal"""
    decimal = 0.0
    for i, bit in enumerate(binary_str, 1):
        # print(f"{bit} -- {i}")
        if bit == '1':
            decimal += 1.0 / (2 ** i)
    return decimal