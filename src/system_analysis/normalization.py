def correct_sequence_period(seq):
    """Periodicity Correction algorithm"""
    n = len(seq)
    seq_double = seq + seq  # для проверки всевозможных сдвигов
    seq_min = seq
    for i in range(n):
        seq_min_temp = seq_double[i:(i + n)]
        seq_min = min(seq_min, seq_min_temp)
    return seq_min


def compress_sequence_lz76(seq):
    """Lempel-Ziv-76 compression algorithm"""
    if seq == "":
        return seq

    n = len(seq)
    i = 0  # текущая позиция
    k = 1  # длина подстроки
    complexity = 1  # общая сложность

    while i + k < n:
        sub = seq[(i + 1):(i + k + 1)]
        source = seq[0:(i + k)]

        if sub in source:
            k += 1
        else:
            complexity += 1
            i += k
            k = 1

    complexity /= len(seq)
    return complexity


if __name__ == "__main__":
    seq = "012301230"
    print(compress_sequence_lz76(seq))
    print(correct_sequence_period(seq))

    # match_index = (seq + seq).find(seq, 1)
    # if match_index < len(seq):
    #     # Период найден
    #     print(f"REGULAR (p={match_index}, norm={seq})")