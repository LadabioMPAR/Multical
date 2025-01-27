Examples
========

This section provides examples to help you understand how to use MultiCal effectively.

Example 1: Basic Usage
----------------------

This example demonstrates how to use MultiCal to adjust parameters in an ODE system.

.. code-block:: python

   from MultiCal import fit_parameters

   # Define your ODE system
   def ode_system(y, t, params):
       # Your ODE equations here
       return dydt

   # Data and initial parameters
   data = [...]  # Your data
   initial_params = [...]

   # Fit parameters
   best_params = fit_parameters(ode_system, data, initial_params)
   print(best_params)

Example 2: Custom Optimization
------------------------------

In this example, we use a custom optimization method for parameter fitting.

.. code-block:: python

   from MultiCal import fit_parameters

   # Custom optimization setup
   method = 'Nelder-Mead'
   best_params = fit_parameters(ode_system, data, initial_params, method=method)
   print(best_params)

Example 3: Visualizing Results
------------------------------

Here's how to visualize the fitted curve:

.. code-block:: python

   from MultiCal import fit_parameters, plot_fit

   best_params = fit_parameters(ode_system, data, initial_params)
   plot_fit(ode_system, data, best_params)
