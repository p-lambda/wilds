import math

replicates = 3
best = (54.2, 0.8)
numbers_to_check = [
    (53.3, 0),
    (53.3, 0),
    (52.3, 1.1),
    (53.9, 0.7),
]


def calculate_standard_error(number_to_check):
    _, std_best = best
    _, std = number_to_check
    return (std_best + std) / math.sqrt(replicates)

def calculate_gap(number_to_check):
    result_best, _ = best
    result, _ = number_to_check
    return result_best - result

for number_to_check in numbers_to_check:
    error = calculate_standard_error(number_to_check)
    gap = calculate_gap(number_to_check)
    # print(f"error={error}, gap={gap}")
    if gap <= error:
        print(f"Bold: {number_to_check}")
