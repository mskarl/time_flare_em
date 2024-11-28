# Example for fitting a time series with expectation maximization

In this example, we use the IceCat-1 catalog with IceCube neutrino alert events and look for temporal clustering at predefined positions. The positions are identified with the multiplet search (see the alert multiplet repository), but can of course be customized. 

I show an example of how to apply the fit and evaluate significances in the jupyter notebook. Expectation Maximization itself is implemented in the `utils.py` file. 

For the background simulations, run `run_flare_signalness.py` once you calculated the background weights. You can reuse the background trials from the multiplet search (see the alert multiplet repository). 
