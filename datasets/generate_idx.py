def generate_numbers_to_file(filename):
    with open(filename, 'w') as file:
        for number in range(1000):
            file.write(f"{number}\n")

# Example usage
generate_numbers_to_file("numbers.txt")