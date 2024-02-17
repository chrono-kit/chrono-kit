from chronokit.preprocessing._dataloader import DataLoader
import numpy as np

class AutoCorrelation:
    def __init__(self, data):
        self.data = DataLoader(data).to_numpy()

    def __p(self, k):
        """
        Calculates the pearson correlation with lag k (for ACF) with the
        data passed in the constructor.

        Arguments:

        *k (int): lag

        Returns:

        *p (float): Pearson correlation at lag k.
        """

        # Return 1 if lag is 0
        if k == 0:
            return 1.0

        # Get the length and mean of data
        n = len(self.data)
        mean = np.mean(self.data)

        # Calculate the pearson correlation
        p = np.sum((self.data[: n - k] - mean) * (self.data[k:] - mean)) / np.sum(
            (self.data - mean) ** 2
        )

        return p

    def acf(self, lag):
        """
        Calculates the autocorrelation function (ACF).

        Arguments:

        *lag (int): Lag to calculate autocorrelation for.

        Returns:

        *acfs (array_like): Autocorrelation function with length (lag+1)
        """

        # Initialize empty array
        acfs = np.array([])

        # Calculate autocorrelation for each lag and append it to acfs array
        for i in range(lag + 1):
            acfs = np.append(acfs, self.__p(i))

        return acfs

    def __phi(self, k):
        """
        Calculates the phi coefficients (for PACF) with the data passed in
        the constructor using coefficient matrix method.

        Arguments:

        *k (int): lag

        Returns:

        *phi (array_like): Phi coefficients for calculation
        of PACF with length (k+1)
        """

        # If lag is 0, return [1.](as a list for proper usage in pacf function)
        if k == 0:
            return [1.0]

        # Initialize autocorrelation matrix as a zero kxk zero matrix
        ac_matrix = np.zeros((k, k))

        # Initialize solution vector as a zero vector of length k
        sol_vector = np.zeros(k)

        for i in range(k):
            # Assign each entry of the solution vector
            # as the pearson coefficient
            sol_vector[i] = self.__p(i + 1)

            # Assign the entries in the autocorrelation matrix
            # as the pearson coefficient of lag |i-j|
            for j in range(k):
                ac_matrix[i, j] = self.__p(np.abs(i - j))

        # Take the solution for x of ac_matrix(x) = solution_vector
        # as the phi coefficient
        phi = np.linalg.solve(ac_matrix, sol_vector)

        return phi

    def pacf(self, lag):
        """
        Calculates the partial autocorrelation function (PACF).

        Arguments:

        *lag (int): Lag to calculate autocorrelation for.

        Returns:

        *pacfs (array_like): Partial Autocorrelation function
        with length (lag+1)
        """

        # Initialize pacfs as an empty array
        pacfs = np.array([])

        for i in range(lag + 1):
            # Append the phi coefficient to the pacfs array
            pacfs = np.append(pacfs, self.__phi(i)[i - 1])

        return pacfs