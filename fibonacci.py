def fibonacci_series(n_terms):

    """Generate a Fibonacci series up to n_terms."""

    # Initialize the first two Fibonacci numbers and the series list

    a, b = 0, 1

    fib_series = []


    # Loop to generate the Fibonacci series

    for _ in range(n_terms):

        fib_series.append(a)

        a, b = b, a + b


    return fib_series


# Example usage:

if __name__ == "__main__":

    num_terms = 10

    print(f"Fibonacci series up to {num_terms} terms: {fibonacci_series(num_terms)}")